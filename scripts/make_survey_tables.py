"""Tổng hợp E1–E6 thành bảng (CSV + LaTeX booktabs) và hình cho bản thảo survey v2; kèm manifest truy vết.

Đọc results/survey_v2/{E1,E2_E3,E4_E5}/... (phần nào chưa có thì bỏ qua và báo), ghi vào results/survey_v2/tables/
và results/survey_v2/figures/. Mọi bảng ghi kèm `provenance_manifest.json`: tệp đầu vào, SHA-256, commit hash của
từng config.json — người viết trích số liệu từ đây và ghi [source: ...] vào working/03_claims.md.

Thống kê (KBTN v2.1; Review 24/10 Bảng 10):
* E1: mean, SD và CI 95% (phương sai Nadeau–Bengio) qua J = 25 fold (wrapper J = 15, ghép cặp theo (repeat, fold)).
* Kiểm định vs random-k: d_j = AUROC(method, k, fold j) − trung bình 100 lần rút random-k trên fold j; t-test
  Nadeau–Bengio một phía (greater); Holm trong họ = một phương pháp × 5 mức k; thêm p_random_empirical kiểu P0.
* Kiểm định vs B0: d_j = AUROC(method, k, j) − AUROC(B0, j); hai phía; Holm trong cùng họ.
* Bayesian correlated t-test (ROPE ±0,01 AUROC) cho các cặp dẫn đầu trên GSE68465 (phương pháp tốt nhất tại mỗi k
  so với từng phương pháp khác) và ma trận P(right) đầy đủ tại k = 50.
* E6: ghép AUROC (E1) – Nogueira (E4, tỉ lệ --e6-fraction) – thời gian (E2, k = 50); đánh dấu mặt Pareto.

    python scripts/make_survey_tables.py
    python scripts/make_survey_tables.py --e6-fraction 0.9 --no-figures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.runlog import DATA_ROOT, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stats import bayesian_correlated_ttest, holm, mean_ci, nadeau_bengio_ttest  # noqa: E402

METHOD_LABEL = {"f_test": "F-test", "mutual_info": "MI", "mrmr": "mRMR", "relieff": "ReliefF", "l1_lr": "L1-LR",
                "elastic_net": "Elastic Net", "rf_importance": "RF importance", "svm_rfe": "SVM-RFE", "ga": "GA",
                "bpso": "BPSO", "b0": "B0 (all genes)", "random_k": "random-k"}
METHOD_ORDER = list(C.FILTER_METHODS + C.EMBEDDED_METHODS + C.WRAPPER_METHODS) + ["random_k", "b0"]
# Màu theo HỌ (3 màu đã kiểm tra an toàn mù màu theo mọi cặp), tham chiếu màu trung tính; phương pháp phân biệt
# bằng marker + kiểu nét + nhãn trực tiếp (mã hóa kép, không dựa vào màu đơn thuần).
FAMILY_COLOR = {"filter": "#2a78d6", "embedded": "#eb6834", "wrapper": "#1baf7a", "reference": "#6b6b6b"}
METHOD_MARKER = {"f_test": "o", "mutual_info": "s", "mrmr": "^", "relieff": "D", "l1_lr": "o", "elastic_net": "s",
                 "rf_importance": "^", "svm_rfe": "D", "ga": "o", "bpso": "s", "b0": "x", "random_k": "+"}
METHOD_LS = {"f_test": "-", "mutual_info": "--", "mrmr": "-.", "relieff": ":", "l1_lr": "-", "elastic_net": "--",
             "rf_importance": "-.", "svm_rfe": ":", "ga": "-", "bpso": "--", "b0": "-", "random_k": "--"}
# Số mũ lý thuyết theo Bảng 5 của bản v1 (draft/05_time_complexity.tex, tab:fs_complexity), quy về (n, p) với k cố định.
TABLE5_EXPONENTS = {
    "f_test": (1, 1, r"O(np)"), "mutual_info": (1, 1, r"O(np)"), "relieff": (2, 1, r"O(n^2 p)"),
    "mrmr": (1, 2, r"O(p^2 n) [cài đặt tham lam k gen: O(k n p) -> kỳ vọng alpha_p = 1]"),
    "l1_lr": (1, 1, r"O(n p t)"), "elastic_net": (1, 1, r"O(n p t)"), "rf_importance": (1, 1, r"O(B n p log n)"),
    "svm_rfe": (2, 2, r"O(n^2 p^2) [loại 10%/vòng: ~O(n^2 p) hoặc O(n p^2) tùy n, p]"),
    "ga": (None, None, r"O(G·C(k))"), "bpso": (None, None, r"O(I·C(k))"),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1-dir", default="E1")
    ap.add_argument("--e2e3-dir", default="E2_E3")
    ap.add_argument("--e4e5-dir", default="E4_E5")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--fig-out", default="figures")
    ap.add_argument("--classifier", default="lr_l2")
    ap.add_argument("--e6-fraction", type=float, default=C.REDUNDANCY_FRACTION)
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------------------------------
class Manifest:
    def __init__(self):
        self.inputs: dict[str, dict] = {}
        self.outputs: list[str] = []

    def read_csv(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        cfg = path.parent / "config.json"
        commit = None
        if cfg.exists():
            try:
                commit = json.loads(cfg.read_text(encoding="utf-8")).get("commit_hash")
            except Exception:
                commit = None
        self.inputs[str(path)] = {"sha256": sha256(path), "rows": int(len(df)), "commit_hash": commit}
        return df

    def save(self, out_dir: Path, prov: dict):
        payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_commit": prov.get("commit_hash"),
                   "inputs": self.inputs, "outputs": sorted(set(self.outputs)),
                   "input_commit_hashes": sorted({str(v["commit_hash"]) for v in self.inputs.values()})}
        (out_dir / "provenance_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_pm(m, s, nd=3):
    return "" if pd.isna(m) else (f"{m:.{nd}f}" if pd.isna(s) else f"{m:.{nd}f} ± {s:.{nd}f}")


def to_booktabs(df: pd.DataFrame, path: Path, caption: str, label: str, float_fmt: str = "{:.3f}") -> None:
    """Ghi bảng LaTeX booktabs đơn giản (không phụ thuộc DataFrame.to_latex)."""
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else float_fmt.format(v)
        return str(v).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    cols = [cell(c) for c in df.columns]
    lines = [r"\begin{table*}[t]", r"  \centering", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", r"  \small",
             "  \\begin{tabular}{" + "l" * len(cols) + "}", r"    \toprule", "    " + " & ".join(cols) + r" \\", r"    \midrule"]
    for row in df.itertuples(index=False):
        lines.append("    " + " & ".join(cell(v) for v in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save(df: pd.DataFrame, out: Path, name: str, man: Manifest, tex: tuple | None = None):
    p = out / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    man.outputs.append(str(p))
    if tex is not None:
        caption, label = tex
        to_booktabs(df, out / f"{name}.tex", caption, label)
        man.outputs.append(str(out / f"{name}.tex"))
    print(f"  ghi {name}: {len(df)} dòng", flush=True)


# ---------------------------------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------------------------------
def e1_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (co, m, k, clf), g in per_fold.groupby(["cohort", "method", "k", "classifier"]):
        row = {"cohort": co, "method": m, "k": int(k), "classifier": clf, "n_folds": len(g)}
        for metric in ("auroc", "bacc", "mcc"):
            s = mean_ci(g[metric].to_numpy())
            row.update({f"{metric}_mean": s["mean"], f"{metric}_sd": s["sd"], f"{metric}_ci_low": s["ci_low"],
                        f"{metric}_ci_high": s["ci_high"]})
        for extra in ("select_time_s", "n_filled", "n_eval", "best_fitness"):
            if extra in g.columns:
                row[f"{extra}_mean"] = float(pd.to_numeric(g[extra], errors="coerce").mean())
        if "exact_k" in g.columns:
            row["exact_k_rate"] = float(g["exact_k"].astype(str).str.lower().eq("true").mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
    return df.sort_values(["cohort", "classifier", "k", "method_order"]).drop(columns="method_order")


def random_per_fold(random_k: pd.DataFrame) -> pd.DataFrame:
    """Kỳ vọng random-k trên từng fold: trung bình (và SD) qua các lần rút; kèm số lần rút."""
    g = random_k.groupby(["cohort", "repeat", "fold", "k"], as_index=False).agg(
        random_auroc_mean=("auroc", "mean"), random_auroc_sd=("auroc", "std"), random_auroc_q95=("auroc", lambda s: s.quantile(0.95)),
        random_bacc_mean=("bacc", "mean"), n_draws=("auroc", "size"))
    return g


def e1_tests(per_fold: pd.DataFrame, random_k: pd.DataFrame | None, classifier: str) -> pd.DataFrame:
    pf = per_fold[per_fold.classifier == classifier]
    methods = [m for m in pf.method.unique() if m not in ("b0", "random_k")]
    b0 = pf[pf.method == "b0"][["cohort", "repeat", "fold", "auroc"]].rename(columns={"auroc": "auroc_b0"})
    rnd = random_per_fold(random_k) if random_k is not None and len(random_k) else None
    rows = []
    for co in pf.cohort.unique():
        for m in methods:
            for k, g in pf[(pf.cohort == co) & (pf.method == m)].groupby("k"):
                base = {"cohort": co, "method": m, "k": int(k), "classifier": classifier, "n_folds": len(g),
                        "auroc_mean": float(g.auroc.mean())}
                if rnd is not None:
                    j = g.merge(rnd[(rnd.cohort == co) & (rnd.k == k)], on=["cohort", "repeat", "fold", "k"], how="inner")
                    if len(j) >= 2:
                        t = nadeau_bengio_ttest((j.auroc - j.random_auroc_mean).to_numpy(), alternative="greater")
                        # p kiểu P0: phân bố trung bình-qua-fold của từng lần rút random-k so với trung bình của phương pháp
                        rk = random_k[(random_k.cohort == co) & (random_k.k == k)]
                        rk = rk.merge(g[["repeat", "fold"]].drop_duplicates(), on=["repeat", "fold"])
                        draw_means = rk.groupby("draw")["auroc"].mean().to_numpy()
                        p_emp = (np.sum(draw_means >= j.auroc.mean()) + 1) / (len(draw_means) + 1) if len(draw_means) else np.nan
                        base.update({"random_mean": float(j.random_auroc_mean.mean()), "gain_vs_random": t["mean"],
                                     "gain_ci_low": t["ci_low"], "gain_ci_high": t["ci_high"], "t_nb_random": t["t"],
                                     "p_nb_random": t["p"], "n_pairs_random": t["n"], "p_random_empirical": p_emp,
                                     "n_random_draws": int(len(draw_means))})
                if len(b0):
                    j = g.merge(b0[b0.cohort == co], on=["cohort", "repeat", "fold"], how="inner")
                    if len(j) >= 2:
                        t = nadeau_bengio_ttest((j.auroc - j.auroc_b0).to_numpy(), alternative="two-sided")
                        base.update({"b0_mean": float(j.auroc_b0.mean()), "diff_vs_b0": t["mean"], "diff_b0_ci_low": t["ci_low"],
                                     "diff_b0_ci_high": t["ci_high"], "t_nb_b0": t["t"], "p_nb_b0": t["p"], "n_pairs_b0": t["n"]})
                rows.append(base)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Holm trong họ = (cohort, method, loại so sánh) qua các mức k
    for pcol in ("p_nb_random", "p_nb_b0"):
        if pcol in df.columns:
            df[f"{pcol}_holm"] = np.nan
            for _, idx in df.groupby(["cohort", "method"]).groups.items():
                df.loc[idx, f"{pcol}_holm"] = holm(df.loc[idx, pcol].to_numpy())
    if "p_nb_random_holm" in df.columns:
        df["beats_random_holm"] = df["p_nb_random_holm"] < C.ALPHA
        df["useful_rule_p0"] = df["beats_random_holm"] & (df["gain_vs_random"] >= 0.02)
    if "p_nb_b0_holm" in df.columns:
        df["differs_from_b0_holm"] = df["p_nb_b0_holm"] < C.ALPHA
    return df


def e1_bayes(per_fold: pd.DataFrame, cohort: str, classifier: str, rope: float = C.ROPE_AUROC):
    pf = per_fold[(per_fold.cohort == cohort) & (per_fold.classifier == classifier) & (~per_fold.method.isin(["b0", "random_k"]))]
    if pf.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, matrix_rows = [], []
    for k, g in pf.groupby("k"):
        means = g.groupby("method")["auroc"].mean().sort_values(ascending=False)
        best = means.index[0]
        gb = g[g.method == best][["repeat", "fold", "auroc"]]
        for other in means.index[1:]:
            go = g[g.method == other][["repeat", "fold", "auroc"]]
            j = gb.merge(go, on=["repeat", "fold"], suffixes=("_best", "_other"))
            b = bayesian_correlated_ttest((j.auroc_best - j.auroc_other).to_numpy(), rope=rope)
            rows.append({"cohort": cohort, "k": int(k), "best": best, "other": other, "n_pairs": b["n"], "mean_diff": b["mean"],
                         "p_best_better": b["p_right"], "p_rope": b["p_rope"], "p_other_better": b["p_left"], "rope": rope})
        if int(k) == C.OVERLAP_K:
            ms = list(means.index)
            for a in ms:
                ga = g[g.method == a][["repeat", "fold", "auroc"]]
                for b_ in ms:
                    gb_ = g[g.method == b_][["repeat", "fold", "auroc"]]
                    j = ga.merge(gb_, on=["repeat", "fold"], suffixes=("_a", "_b"))
                    r = bayesian_correlated_ttest((j.auroc_a - j.auroc_b).to_numpy(), rope=rope) if a != b_ else None
                    matrix_rows.append({"cohort": cohort, "k": int(k), "method_row": a, "method_col": b_,
                                        "p_row_better": np.nan if r is None else r["p_right"],
                                        "p_rope": np.nan if r is None else r["p_rope"], "n_pairs": len(j)})
    return pd.DataFrame(rows), pd.DataFrame(matrix_rows)


# ---------------------------------------------------------------------------------------------------
# E2–E3
# ---------------------------------------------------------------------------------------------------
def e3_exponent_table(exps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in exps.itertuples(index=False):
        th = TABLE5_EXPONENTS.get(r.method, (None, None, ""))
        rows.append({"method": r.method, "family": r.family, "alpha_n_measured": r.alpha_n, "alpha_n_table5": th[0],
                     "alpha_p_measured": r.alpha_p, "alpha_p_large_measured": getattr(r, "alpha_p_large", np.nan),
                     "alpha_p_table5": th[1], "complexity_table5": th[2], "n_range": r.n_range, "p_range": r.p_range,
                     "wall_spread_max_rel": getattr(r, "wall_spread_max_rel", np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------------------------------
def pareto_flags(df: pd.DataFrame, maximize=("auroc_mean", "nogueira"), minimize=("wall_s_e2_k50",)) -> np.ndarray:
    vals = np.column_stack([df[c].to_numpy(float) for c in maximize] + [-df[c].to_numpy(float) for c in minimize])
    ok = ~np.isnan(vals).any(axis=1)
    flags = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(ok)
    for i in idx:
        dominated = False
        for j in idx:
            if i != j and np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated = True
                break
        flags[i] = not dominated
    return flags


def e6_tradeoff(summary: pd.DataFrame, stab: pd.DataFrame | None, cost: pd.DataFrame | None, fraction: float, classifier: str):
    s = summary[(summary.classifier == classifier) & (~summary.method.isin(["b0"]))][
        ["cohort", "method", "k", "n_folds", "auroc_mean", "auroc_sd", "auroc_ci_low", "auroc_ci_high", "select_time_s_mean"]]
    out = s.copy()
    if stab is not None and len(stab):
        st = stab[np.isclose(stab.fraction, fraction)][["cohort", "method", "k", "nogueira", "nogueira_ci_low", "nogueira_ci_high",
                                                        "kuncheva", "jaccard_mean", "M"]]
        out = out.merge(st, on=["cohort", "method", "k"], how="left")
        out["stability_fraction"] = fraction
    if cost is not None and len(cost):
        c = cost[["method", "wall_s", "cpu_s", "peak_mb"]].rename(columns={"wall_s": "wall_s_e2_k50", "cpu_s": "cpu_s_e2_k50",
                                                                            "peak_mb": "peak_mb_e2_k50"})
        out = out.merge(c, on="method", how="left")
    if "nogueira" in out.columns and "wall_s_e2_k50" in out.columns:
        out["pareto_optimal"] = False
        for (co, k), idx in out.groupby(["cohort", "k"]).groups.items():
            sub = out.loc[idx]
            out.loc[idx, "pareto_optimal"] = pareto_flags(sub)
    return out


# ---------------------------------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4.5, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    return plt


def _savefig(fig, fig_out: Path, name: str, man: Manifest):
    for ext in ("pdf", "png"):
        p = fig_out / f"{name}.{ext}"
        fig.savefig(p)
        man.outputs.append(str(p))
    import matplotlib.pyplot as plt
    plt.close(fig)


def fig_e1(summary: pd.DataFrame, tests: pd.DataFrame, cohort: str, classifier: str, fig_out: Path, man: Manifest):
    plt = _style()
    s = summary[(summary.cohort == cohort) & (summary.classifier == classifier)]
    if s.empty:
        return
    fams = [("filter", C.FILTER_METHODS), ("embedded", C.EMBEDDED_METHODS), ("wrapper", C.WRAPPER_METHODS)]
    fams = [(f, [m for m in ms if m in set(s.method)]) for f, ms in fams]
    fams = [(f, ms) for f, ms in fams if ms]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.0), sharey=True, squeeze=False, constrained_layout=True)
    b0 = s[s.method == "b0"]
    rnd = tests[(tests.cohort == cohort)].groupby("k")["random_mean"].mean() if len(tests) and "random_mean" in tests else None
    ks = sorted(s[s.method != "b0"].k.unique())
    for i, (ax, (fam, ms)) in enumerate(zip(axes[0], fams)):
        if len(b0):
            ax.axhline(b0.auroc_mean.iloc[0], color=FAMILY_COLOR["reference"], ls="-", lw=1, label="B0 (all genes)" if i == 0 else None)
        if rnd is not None and len(rnd):
            r = rnd.reindex(ks)
            ax.plot(range(len(ks)), r.values, ls="--", color=FAMILY_COLOR["reference"], marker="+",
                    label="random-k (mean of draws)" if i == 0 else None)
        for m in ms:
            g = s[s.method == m].set_index("k").reindex(ks)
            x = np.arange(len(ks))
            ax.errorbar(x, g.auroc_mean, yerr=[g.auroc_mean - g.auroc_ci_low, g.auroc_ci_high - g.auroc_mean],
                        color=FAMILY_COLOR[fam], ls=METHOD_LS[m], marker=METHOD_MARKER[m], capsize=2, elinewidth=0.7,
                        label=METHOD_LABEL[m])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_xlabel("k (genes)")
        ax.set_title(fam)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.legend(loc="lower left", frameon=False, ncol=1)
    axes[0][0].set_ylabel(f"AUROC ({classifier}), mean ± 95% CI")
    fig.suptitle(f"E1 · {cohort}", fontsize=9)
    _savefig(fig, fig_out, f"fig_e1_auroc_vs_k_{cohort}", man)


def fig_e3(dn: pd.DataFrame, dp: pd.DataFrame, fig_out: Path, man: Manifest):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
    p_fixed = int(dn.p.iloc[0]) if len(dn) else C.PREFILTER_P0
    n_all = int(dp.n.iloc[0]) if len(dp) else None
    for ax, d, var, xl in ((axes[0], dn, "n", f"n (train samples), p = {p_fixed}"), (axes[1], dp, "p", f"p (genes), n = {n_all}")):
        for m, g in d.groupby("method"):
            g = g.sort_values(var)
            fam = C.METHOD_FAMILY.get(m, "reference")
            ax.plot(g[var], g.wall_s, color=FAMILY_COLOR[fam], ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"),
                    label=METHOD_LABEL.get(m, m))
            ax.annotate(METHOD_LABEL.get(m, m), (g[var].iloc[-1], g.wall_s.iloc[-1]), fontsize=6, color=FAMILY_COLOR[fam],
                        xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("wall time (s), median of 3")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("E3 · scaling of the selection step (BLAS 1 thread)", fontsize=9)
    _savefig(fig, fig_out, "fig_e3_scaling", man)


def fig_e4(stab: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest, k: int = C.OVERLAP_K):
    plt = _style()
    s = stab[(stab.cohort == cohort) & (stab.k == k)]
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)
    for m, g in s.groupby("method"):
        g = g.sort_values("fraction")
        fam = C.METHOD_FAMILY.get(m, "reference")
        ax.errorbar(g.fraction, g.nogueira, yerr=[g.nogueira - g.nogueira_ci_low, g.nogueira_ci_high - g.nogueira],
                    color=FAMILY_COLOR[fam], ls=METHOD_LS.get(m, "-"), marker=METHOD_MARKER.get(m, "o"), capsize=2,
                    elinewidth=0.7, label=METHOD_LABEL.get(m, m))
        ax.annotate(METHOD_LABEL.get(m, m), (g.fraction.iloc[-1], g.nogueira.iloc[-1]), fontsize=6, color=FAMILY_COLOR[fam],
                    xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xlabel("subsample fraction of patients")
    ax.set_ylabel(f"Nogueira stability (k = {k}), ± 95% CI")
    ax.set_title(f"E4 · {cohort}")
    ax.legend(frameon=False, ncol=2)
    _savefig(fig, fig_out, f"fig_e4_stability_{cohort}", man)


def fig_e5c(mat: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    names = [m for m in METHOD_ORDER if m in mat.index]
    M = mat.loc[names, names].to_numpy(float)
    fig, ax = plt.subplots(figsize=(3.9, 3.4), constrained_layout=True)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([METHOD_LABEL.get(m, m) for m in names], rotation=60, ha="right")
    ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in names])
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="mean Jaccard (k = 50)", shrink=0.8)
    ax.set_title(f"E5c · {cohort}")
    _savefig(fig, fig_out, f"fig_e5c_overlap_{cohort}", man)


def fig_e6(trade: pd.DataFrame, cohort: str, fig_out: Path, man: Manifest):
    plt = _style()
    t = trade[(trade.cohort == cohort) & trade.nogueira.notna()]
    if t.empty:
        return
    ks = sorted(t.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(2.7 * len(ks), 3.1), sharey=True, sharex=True, squeeze=False,
                             constrained_layout=True)
    has_cost = "wall_s_e2_k50" in t.columns and t.wall_s_e2_k50.notna().any()
    for ax, k in zip(axes[0], ks):
        g = t[t.k == k]
        for r in g.itertuples():
            fam = C.METHOD_FAMILY.get(r.method, "reference")
            size = 12 + 10 * np.log10(max(r.wall_s_e2_k50, 1e-2) + 1) * 4 if has_cost and not pd.isna(r.wall_s_e2_k50) else 20
            ax.errorbar(r.nogueira, r.auroc_mean, xerr=[[r.nogueira - r.nogueira_ci_low], [r.nogueira_ci_high - r.nogueira]],
                        yerr=[[r.auroc_mean - r.auroc_ci_low], [r.auroc_ci_high - r.auroc_mean]], fmt="none",
                        ecolor=FAMILY_COLOR[fam], elinewidth=0.6, alpha=0.7)
            ax.scatter([r.nogueira], [r.auroc_mean], s=size, color=FAMILY_COLOR[fam], marker=METHOD_MARKER.get(r.method, "o"),
                       edgecolor="black" if getattr(r, "pareto_optimal", False) else "none", linewidth=0.8, zorder=3)
            ax.annotate(METHOD_LABEL.get(r.method, r.method), (r.nogueira, r.auroc_mean), fontsize=5.5, xytext=(3, 3),
                        textcoords="offset points", color=FAMILY_COLOR[fam])
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Nogueira stability")
    axes[0][0].set_ylabel("AUROC (mean ± 95% CI)")
    note = "marker size ∝ log wall time (E2, k = 50); black edge = Pareto-optimal" if has_cost else "black edge = Pareto-optimal"
    fig.suptitle(f"E6 · {cohort} · stability – performance – cost ({note})", fontsize=8)
    _savefig(fig, fig_out, f"fig_e6_tradeoff_{cohort}", man)


# ---------------------------------------------------------------------------------------------------
def main(argv=None, script=__file__) -> None:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    root = DATA_ROOT.joinpath(*C.RESULT_DIR)
    out = results_dir(args.out)
    fig_out = results_dir(args.fig_out)
    man = Manifest()
    status = {}

    # ---------------- E1 ----------------
    e1_root = root / args.e1_dir
    per_fold_all, random_all, genes_all = [], [], []
    if e1_root.exists():
        for d in sorted(p for p in e1_root.iterdir() if p.is_dir()):
            pf = man.read_csv(d / "e1_per_fold.csv")
            if pf is not None:
                per_fold_all.append(pf)
            rk = man.read_csv(d / "e1_random_k.csv")
            if rk is not None:
                random_all.append(rk)
    summary = tests = pd.DataFrame()
    if per_fold_all:
        per_fold = pd.concat(per_fold_all, ignore_index=True)
        random_k = pd.concat(random_all, ignore_index=True) if random_all else None
        summary = e1_summary(per_fold)
        save(summary, out, "e1_summary", man)
        main_tab = summary[summary.classifier == args.classifier].copy()
        main_tab["AUROC"] = [fmt_pm(m, s) for m, s in zip(main_tab.auroc_mean, main_tab.auroc_sd)]
        wide = main_tab[main_tab.method != "b0"].pivot_table(index=["cohort", "method"], columns="k", values="AUROC",
                                                             aggfunc="first").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        save(wide, out, "e1_auroc_table", man, tex=("E1: AUROC (mean $\\pm$ SD over repeated patient-level CV) by method and budget $k$.", "tab:e1_auroc"))
        b0_tab = main_tab[main_tab.method == "b0"][["cohort", "k", "n_folds", "AUROC", "bacc_mean", "mcc_mean"]].rename(columns={"k": "n_genes"})
        if len(b0_tab):
            save(b0_tab, out, "e1_b0_table", man, tex=("E1: reference B0 (all genes, no selection) with the same classifier and CV protocol.", "tab:e1_b0"))
        tests = e1_tests(per_fold, random_k, args.classifier)
        if len(tests):
            save(tests, out, "e1_tests_vs_random_b0", man)
        bayes, bayes_mat = e1_bayes(per_fold, C.RANKING_DATASET, args.classifier)
        if len(bayes):
            save(bayes, out, "e1_bayes_rope_best_vs_others", man)
            save(bayes_mat, out, f"e1_bayes_rope_matrix_k{C.OVERLAP_K}", man)
        if random_k is not None:
            save(random_per_fold(random_k), out, "e1_random_k_per_fold", man)
        status["E1"] = {"cohorts": sorted(per_fold.cohort.unique()), "methods": sorted(per_fold.method.unique()),
                        "n_rows": int(len(per_fold))}
        if not args.no_figures:
            for co in per_fold.cohort.unique():
                fig_e1(summary, tests, co, args.classifier, fig_out, man)
    else:
        status["E1"] = "chưa có"
        print("E1: chưa có kết quả", flush=True)

    # ---------------- E2–E3 ----------------
    e23 = root / args.e2e3_dir
    cost = man.read_csv(e23 / "e2_cost_point.csv") if e23.exists() else None
    if cost is not None:
        c = cost.copy()
        c["wall_s"] = c["wall_s"].astype(float)
        save(c[[x for x in ("method", "family", "n", "p", "k", "wall_s", "cpu_s", "peak_mb", "maxrss_mb", "repeats", "pop", "generations")
                if x in c.columns]], out, "e2_cost_table", man,
             tex=("E2: cost of the selection step at a fixed point ($n$ = training part, $p$ = 5000, $k$ = 50; median of 3 runs, one machine, BLAS single-threaded).", "tab:e2_cost"))
        dn, dp = man.read_csv(e23 / "e3_scaling_n.csv"), man.read_csv(e23 / "e3_scaling_p.csv")
        exps = man.read_csv(e23 / "e3_exponents.csv")
        if exps is not None:
            save(e3_exponent_table(exps), out, "e3_exponents_vs_table5", man,
                 tex=("E3: empirical log--log exponents of wall time in $n$ and $p$ versus the theoretical orders of Table~\\ref{tab:fs_complexity}.", "tab:e3_exponents"))
        if dn is not None and dp is not None and not args.no_figures:
            fig_e3(dn, dp, fig_out, man)
        status["E2_E3"] = {"methods": sorted(cost.method.unique())}
    else:
        status["E2_E3"] = "chưa có"
        print("E2/E3: chưa có kết quả", flush=True)

    # ---------------- E4–E5 ----------------
    e45 = root / args.e4e5_dir
    stab_all, e5a_all, e5b_all = [], [], []
    overlaps = {}
    if e45.exists():
        for d in sorted(p for p in e45.iterdir() if p.is_dir()):
            s = man.read_csv(d / "e4_stability.csv")
            if s is not None:
                stab_all.append(s)
            a = man.read_csv(d / "e5a_redundancy.csv")
            if a is not None:
                e5a_all.append(a)
            b = man.read_csv(d / "e5b_seed_stability.csv")
            if b is not None:
                e5b_all.append(b)
            o = d / "e5c_overlap_k50.csv"
            if o.exists():
                man.read_csv(o)
                mat = pd.read_csv(o, index_col=0)
                if len(mat):
                    overlaps[d.name] = mat
    stab = pd.concat(stab_all, ignore_index=True) if stab_all else None
    if stab is not None:
        save(stab, out, "e4_stability", man)
        wide = stab.assign(val=[fmt_pm(m, (h - l) / 2) for m, l, h in zip(stab.nogueira, stab.nogueira_ci_low, stab.nogueira_ci_high)])
        wide = wide.pivot_table(index=["cohort", "fraction", "method"], columns="k", values="val", aggfunc="first").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        save(wide, out, "e4_nogueira_table", man,
             tex=("E4: Nogueira stability ($\\pm$ half-width of the 95\\% CI) over $M$ = 30 independent patient subsamples, by subsample fraction and budget $k$.", "tab:e4_nogueira"))
        if not args.no_figures:
            for co in stab.cohort.unique():
                fig_e4(stab, co, fig_out, man)
        status["E4"] = {"cohorts": sorted(stab.cohort.unique())}
    else:
        status["E4"] = "chưa có"
    if e5a_all:
        e5a = pd.concat(e5a_all, ignore_index=True)
        save(e5a, out, "e5a_redundancy", man)
        cols = ["cohort", "rho", "method", "k", "nogueira", "nogueira_merged", "delta_nogueira_vs_e4", "pairs_both_selected", "n_copy_selected"]
        save(e5a[[c for c in cols if c in e5a.columns]], out, "e5a_redundancy_table", man,
             tex=("E5a: stability under injected redundancy (50 correlated copies, $\\rho \\in \\{0.5, 0.8, 0.95\\}$): raw and copy-merged Nogueira, change versus E4 at the same fraction, mean number of (source, copy) pairs selected together.", "tab:e5a_redundancy"))
        status["E5a"] = "có"
    if e5b_all:
        e5b = pd.concat(e5b_all, ignore_index=True)
        save(e5b, out, "e5b_seed_variability", man,
             tex=("E5b: seed sensitivity of GA, BPSO and RF importance on one training fold of GSE68465 (5 seeds): Nogueira/Jaccard between the five selected panels and the spread of test AUROC.", "tab:e5b_seed"))
        status["E5b"] = "có"
    for co, mat in overlaps.items():
        if not args.no_figures:
            fig_e5c(mat, co, fig_out, man)
        m2 = mat.copy()
        m2.insert(0, "method", m2.index)
        save(m2.reset_index(drop=True), out, f"e5c_overlap_k50_{co}", man)
    if overlaps:
        status["E5c"] = sorted(overlaps)

    # ---------------- E6 ----------------
    if len(summary):
        trade = e6_tradeoff(summary, stab, cost, args.e6_fraction, args.classifier)
        save(trade, out, "e6_tradeoff", man)
        if "pareto_optimal" in trade.columns:
            save(trade[trade.pareto_optimal], out, "e6_pareto_front", man,
                 tex=("E6: Pareto-optimal (method, $k$) points in the AUROC -- stability -- cost space.", "tab:e6_pareto"))
        if not args.no_figures and "nogueira" in trade.columns:
            for co in trade.cohort.unique():
                fig_e6(trade, co, fig_out, man)
        status["E6"] = "có" if "nogueira" in trade.columns else "thiếu E4"

    man.save(out, prov)
    write_config(out, prov, {"experiment": "tables_figures", "status": status, "classifier": args.classifier,
                             "e6_fraction": args.e6_fraction, "n_inputs": len(man.inputs)})
    print(json.dumps(status, ensure_ascii=False, indent=1))
    print("xong:", out, "|", fig_out, flush=True)


if __name__ == "__main__":
    main()
