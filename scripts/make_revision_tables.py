"""Bảng và quy tắc đọc của đợt hiệu chỉnh DMKD (B1–B7), đúng quy tắc chốt 12/11/2026 (YEU-CAU-DOI-CODE-2026-11-12-GVHD-B1-B7 §3).

Mỗi lệnh con đọc kết quả mới trong ``$KLTN_DATA_ROOT/results/survey_v2/B*`` và kết quả đã công bố của tag survey-v3-dmkd.1
(``results/survey_v2/`` của kho mã, chỉ đọc), ghi bảng vào ``results/survey_v2/<B…>_tables/`` kèm config.json (commit, SHA-256 đầu
vào). Không có số nào tính tay; mọi nhãn đọc gán bằng quy tắc trong mã. Mọi bảng là MÔ TẢ, trừ các cổng bằng nhau.

    python scripts/make_revision_tables.py b1      # cổng E1b/E7 5 lần chia; nhãn cấp lần chia của E7 (20 lần chia)
    python scripts/make_revision_tables.py b2 b3 b5
    python scripts/make_revision_tables.py b4      # cổng B0; bảng biến thể; quy tắc GSE42127 ≥ 0,935; k* độ nhạy
    python scripts/make_revision_tables.py b6gate  # cổng B6: 5 lần chia trùng rq2_r_all_7f097da (mã 3 nếu không đạt)
    python scripts/make_revision_tables.py b6      # nhãn cấp lần chia của ptLasso, Trans-GLM so với target-only (cần B1_E7_20)
    python scripts/make_revision_tables.py b7      # R1–R4 (R5 do người viết đọc trên khung Fig. 13)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.data import sha256  # noqa: E402
from survey_v2.kstar import kstar_table, per_cohort, per_cohort_kstar, pooled  # noqa: E402
from survey_v2.revision import (GATE_TOL_AUROC, PUBLISHED_ROOT, PUBLISHED_TAG, compare_rows, split_level_labels)  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.stats import holm, mean_ci, nadeau_bengio_ttest  # noqa: E402

GPL570 = list(C.GPL570_COHORTS)       # k* gộp; đổi bằng --kstar-cohorts (chỉ dùng cho test trên dữ liệu giả)
FEW = list(C.FEWSHOT_COHORTS)
MAIN = list(C.MAIN_DATASETS)


class Inputs:
    """Ghi lại mọi tệp đầu vào (đường dẫn, SHA-256) để đưa vào config.json."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def csv(self, path: Path, **kw) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"thiếu đầu vào: {path}")
        self.files[str(path)] = sha256(path)
        return pd.read_csv(path, **kw)

    def maybe(self, path: Path, **kw) -> pd.DataFrame | None:
        return self.csv(path, **kw) if Path(path).exists() else None


def _save(df: pd.DataFrame, out: Path, name: str) -> Path:
    path = out / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  {name}: {len(df)} dòng", flush=True)
    return path


def _finish(out: Path, prov: dict, task: str, inp: Inputs, summary: dict, choices=()):
    cfg = {"experiment": f"{task}_tables", "spec": "YEU-CAU-DOI-CODE-2026-11-12-GVHD-B1-B7.md §3 (reading rules fixed 12/11/2026)",
           "published_root": str(PUBLISHED_ROOT), "published_tag": PUBLISHED_TAG, "inputs_sha256": inp.files, "summary": summary,
           "choices_to_confirm": list(choices), "versions": all_versions()}
    write_config(out, prov, cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str), flush=True)


def _res(*parts) -> Path:
    return DATA_ROOT.joinpath(*C.RESULT_DIR, *parts)


# ==========================================================================================================================
# B1: 20 lần chia few-shot của E1b và E7
# ==========================================================================================================================
def task_b1(args, prov):
    inp, out = Inputs(), results_dir("B1_tables")
    gates, labels, e1b_rows = [], [], []
    for co in FEW:
        e7 = inp.csv(_res(args.b1_e7, co, "e7_per_run.csv"), dtype={"n_t": str})
        e7_pub = inp.csv(PUBLISHED_ROOT / "E7" / co / "e7_per_run.csv", dtype={"n_t": str})
        g = compare_rows(e7[e7.split < C.FEWSHOT_SPLITS], e7_pub, ["split", "subset", "n_t", "strategy", "k"], cols=("auroc",))
        gates.append({"experiment": "E7", "cohort": co, **g})
        e1b = inp.csv(_res(args.b1_e1b, co, "e1b_per_run.csv"), dtype={"n_t": str})
        e1b_pub = inp.csv(PUBLISHED_ROOT / "E1b" / co / "e1b_per_run.csv", dtype={"n_t": str})
        e1b_pub = e1b_pub[e1b_pub.method.isin(e1b.method.unique()) & e1b_pub.n_t.isin(e1b.n_t.unique())]
        g = compare_rows(e1b[e1b.split < C.FEWSHOT_SPLITS], e1b_pub, ["split", "subset", "n_t", "method", "k"], cols=("auroc",))
        gates.append({"experiment": "E1b", "cohort": co, **g})
        labels.append(split_level_labels(e7, "strategy", "target_only", ["cohort", "n_t", "k"], seed_tag="b1"))
        lvl = e1b.groupby(["cohort", "n_t", "method", "k", "split"], as_index=False)["auroc"].mean()
        for (n_t, m, k), x in lvl.groupby(["n_t", "method", "k"]):
            ci = mean_ci(x.auroc.to_numpy())
            e1b_rows.append({"cohort": co, "n_t": n_t, "method": m, "k": int(k), "n_splits": int(len(x)), "auroc_mean": ci["mean"],
                             "auroc_sd_split": ci["sd"], "auroc_q25": float(x.auroc.quantile(0.25)), "auroc_q75": float(x.auroc.quantile(0.75))})
    gates = pd.DataFrame(gates)
    _save(gates, out, "b1_gates.csv")
    lab = pd.concat(labels, ignore_index=True)
    _save(lab, out, "b1_e7_split_labels.csv")
    _save(pd.DataFrame(e1b_rows), out, "b1_e1b_split_summary.csv")
    counts = lab.groupby(["strategy", "label"]).size().unstack(fill_value=0)
    _save(counts.reset_index(), out, "b1_e7_label_counts.csv")
    _finish(out, prov, "B1", inp, {"gates_pass": bool(gates["pass"].all()), "gate_tol": GATE_TOL_AUROC,
                                   "label_counts": {s: r.to_dict() for s, r in counts.iterrows()}},
            choices=["label precedence: positive/negative transfer before practically equivalent when both hold",
                     "Wilcoxon exact two-sided; scipy falls back to the normal approximation only when |Δ| has ties (column p_wilcoxon)",
                     "E1b with 20 splits: descriptive per-split summary only (no reading rule was specified for E1b)"])


# ==========================================================================================================================
# B2: k-TSP bỏ phiếu
# ==========================================================================================================================
def task_b2(args, prov):
    inp, out = Inputs(), results_dir("B2_tables")
    e1 = pd.concat([inp.csv(_res("B2_ktsp_vote", d, "b2_e1_per_fold.csv")) for d in MAIN], ignore_index=True)
    e1t = e1.groupby(["cohort", "k"], as_index=False).agg(n=("delta_auroc", "size"), auroc_vote=("auroc_vote", "mean"),
                                                          auroc_lr=("auroc_lr", "mean"), delta_mean=("delta_auroc", "mean"),
                                                          delta_sd=("delta_auroc", "std"))
    _save(e1t, out, "b2_e1_delta.csv")
    e1b = pd.concat([inp.csv(_res("B2_ktsp_vote", d, "b2_e1b_per_run.csv"), dtype={"n_t": str}) for d in FEW], ignore_index=True)
    e1bt = e1b.groupby(["cohort", "n_t", "k"], as_index=False).agg(n=("delta_auroc", "size"), auroc_vote=("auroc_vote", "mean"),
                                                                    auroc_lr=("auroc_lr", "mean"), delta_mean=("delta_auroc", "mean"))
    _save(e1bt, out, "b2_e1b_delta.csv")
    e7 = pd.concat([inp.csv(_res("B2_ktsp_vote", d, "b2_e7_per_run.csv"), dtype={"n_t": str}) for d in FEW], ignore_index=True)
    e7t = e7.groupby(["cohort", "strategy", "n_t", "k"], as_index=False).agg(n=("delta_auroc", "size"), auroc_vote=("auroc_vote", "mean"),
                                                                              auroc_lr=("auroc_lr", "mean"), delta_mean=("delta_auroc", "mean"))
    _save(e7t, out, "b2_e7_delta.csv")
    # câu đọc: source-only (không phụ thuộc n_T, tập con) — trung bình theo lần chia
    so = e7[e7.strategy == "source_only"].drop_duplicates(["cohort", "split", "k"])
    so = so.groupby(["cohort", "k"], as_index=False).agg(auroc_vote=("auroc_vote", "mean"), auroc_lr=("auroc_lr", "mean"))
    so["vote_below_05"], so["lr_below_05"] = so.auroc_vote < 0.5, so.auroc_lr < 0.5
    _save(so, out, "b2_source_only_reversal.csv")
    avoided = bool(not so.vote_below_05.any())
    sentence = ("The voting rule avoids source-only AUROC < 0.5 in every cohort and every k." if avoided else
                "The voting rule does not avoid source-only AUROC < 0.5: " +
                "; ".join(f"{r.cohort} k={int(r.k)} ({r.auroc_vote:.3f})" for r in so[so.vote_below_05].itertuples()) + ".")
    _finish(out, prov, "B2", inp, {"source_only_vote_avoids_reversal_everywhere": avoided, "sentence": sentence,
                                   "cells_lr_below_05": int(so.lr_below_05.sum()), "cells_vote_below_05": int(so.vote_below_05.sum())},
            choices=["source-only cell value = mean over the 5 published splits (source-only does not depend on subset or n_T)"])


# ==========================================================================================================================
# B3: hai biến thể zero-shot
# ==========================================================================================================================
def task_b3(args, prov):
    inp, out = Inputs(), results_dir("B3_tables")
    df = pd.concat([inp.csv(_res("B3_zero_shot", d, "b3_per_split.csv")) for d in FEW], ignore_index=True)
    t = df.groupby(["cohort", "variant", "k"], as_index=False).agg(n_splits=("auroc", "size"), auroc_mean=("auroc", "mean"),
                                                                   auroc_sd=("auroc", "std"), bacc_mean=("bacc", "mean"))
    t["reversal"] = t.auroc_mean < 0.5
    _save(t, out, "b3_zero_shot_table.csv")
    gates = []
    for co in FEW:
        pub = inp.csv(PUBLISHED_ROOT / "E7" / co / "e7_per_run.csv", dtype={"n_t": str})
        pub = pub[pub.strategy == "source_only"].drop_duplicates(["split", "k"])
        new = df[(df.cohort == co) & (df.variant == "e7_phi")]
        gates.append({"cohort": co, **compare_rows(new, pub, ["split", "k"], cols=("auroc",), tol=1e-9)})
    gates = pd.DataFrame(gates)
    _save(gates, out, "b3_gate_e7_source_only.csv")
    rev = {v: [f"{r.cohort} k={int(r.k)}" for r in t[(t.variant == v) & t.reversal].itertuples()] for v in t.variant.unique()}
    sentence = "; ".join(f"{v}: reversal (mean AUROC < 0.5) remains at " + (", ".join(c) if c else "no cohort/k") for v, c in rev.items()
                         if v != "e7_phi")
    _finish(out, prov, "B3", inp, {"reversal_cells": rev, "sentence": sentence, "gate_e7_phi_pass": bool(gates["pass"].all())},
            choices=["gate tolerance for e7_phi vs published E7 source_only: 1e-9 (same code path, same seed)",
                     "variant (ii) uses unlabelled target data (all samples of the target cohort) — must be stated in the paper"])


# ==========================================================================================================================
# B4: biến thể B0
# ==========================================================================================================================
def _published_lr_per_fold(inp: Inputs, cohorts) -> pd.DataFrame:
    frames = []
    for d in cohorts:
        pf = inp.csv(PUBLISHED_ROOT / "E1_classification" / d / "classification_per_fold.csv")
        frames.append(pf[pf.model == "lr_l2"][["cohort", "repeat", "fold", "method", "k", "auroc", "bacc", "mcc", "logloss", "brier"]])
    return pd.concat(frames, ignore_index=True)


def task_b4(args, prov):
    inp, out = Inputs(), results_dir("B4_tables")
    b4 = pd.concat([inp.csv(_res("B4_b0_variants", d, "b4_per_fold.csv")) for d in MAIN], ignore_index=True)
    pub = _published_lr_per_fold(inp, MAIN)
    gates = []
    for d in MAIN:
        gates.append({"cohort": d, **compare_rows(b4[(b4.cohort == d) & (b4.method == "b0")], pub[(pub.cohort == d) & (pub.method == "b0")],
                                                   ["repeat", "fold"], cols=("auroc",))})
    gates = pd.DataFrame(gates)
    _save(gates, out, "b4_gate_b0.csv")
    rows = []
    for (d, v), g in b4.groupby(["cohort", "method"]):
        ci = mean_ci(g.auroc.to_numpy())
        rows.append({"cohort": d, "variant": v, "n_folds": ci["n"], "auroc_mean": ci["mean"], "auroc_sd": ci["sd"], "ci_low": ci["ci_low"],
                     "ci_high": ci["ci_high"], "C_mode": g.C.mode().iloc[0] if len(g) else np.nan, "n_features": int(g.n_features.iloc[0])})
    tab = pd.DataFrame(rows)
    _save(tab, out, "b4_b0_variants.csv")
    g42 = tab[(tab.cohort == "gse42127") & (tab.variant != "b0")]
    under = bool((g42.auroc_mean >= args.b4_threshold).any())
    # k* độ nhạy: thay B0 bằng biến thể tốt nhất (trung bình AUROC trên 3 cohort GPL570)
    var_mean = tab[tab.cohort.isin(GPL570) & (tab.variant != "b0")].groupby("variant").auroc_mean.mean()
    best = str(var_mean.idxmax())
    methods = [m for m in pub.method.unique() if m not in ("b0", "random_k")]
    k_orig_tab, k_orig = kstar_table(pub, methods, cohorts=GPL570)
    alt = pd.concat([pub[pub.method != "b0"], b4[b4.method == best].assign(method="b0")[pub.columns.intersection(b4.columns)]], ignore_index=True)
    k_alt_tab, k_alt = kstar_table(alt, methods, cohorts=GPL570)
    k_cmp = k_orig[["method", "k_star_ivw", "k_star_both", "verdict_vs_b0"]].merge(
        k_alt[["method", "k_star_ivw", "k_star_both", "verdict_vs_b0"]], on="method", suffixes=("_b0", f"_{best}"))
    _save(k_cmp, out, "b4_kstar_sensitivity.csv")
    _save(k_alt_tab, out, "b4_kstar_sensitivity_by_k.csv")
    sentence = (f"A B0 variant reaches >= {args.b4_threshold} AUROC on GSE42127 "
                f"({', '.join(f'{r.variant} {r.auroc_mean:.3f}' for r in g42[g42.auroc_mean >= args.b4_threshold].itertuples())}): "
                "the original B0 is under-regularised on that cohort." if under else
                f"No B0 variant reaches {args.b4_threshold} AUROC on GSE42127 (best "
                f"{g42.auroc_mean.max():.3f}); the original B0 is kept as reference.")
    _finish(out, prov, "B4", inp, {"gate_b0_pass": bool(gates["pass"].all()), "gse42127_threshold": args.b4_threshold,
                                   "b0_under_regularised_on_gse42127": under, "sentence": sentence, "best_variant_for_kstar": best,
                                   "kstar_changed_methods": k_cmp[(k_cmp.k_star_both_b0.astype(str) != k_cmp[f"k_star_both_{best}"].astype(str))].method.tolist()},
            choices=["best B0 variant for the k* sensitivity analysis = highest mean AUROC averaged over the 3 GPL570 cohorts",
                     "threshold 0.935 = mRMR AUROC on GSE42127 quoted in the spec (not recomputed)"])


# ==========================================================================================================================
# B5: chữ ký 10 gen
# ==========================================================================================================================
def task_b5(args, prov):
    inp, out = Inputs(), results_dir("B5_tables")
    b5 = pd.concat([inp.csv(_res("B5_signatures10", d, "b5_e1_per_fold.csv")) for d in MAIN], ignore_index=True)
    pub = _published_lr_per_fold(inp, MAIN)
    selectors = [m for m in C.ALL_SELECTORS if m in set(pub.method)]
    k10 = pub[(pub.k == 10) & pub.method.isin(selectors)]
    best = k10.groupby(["cohort", "method"]).auroc.mean().reset_index().sort_values("auroc", ascending=False).drop_duplicates("cohort")
    best = dict(zip(best.cohort, best.method))
    rows = []
    idx = ["cohort", "repeat", "fold"]
    for sig, g in b5.groupby("method"):
        for d in MAIN:
            a = g[g.cohort == d].set_index(idx).auroc
            for ref_name, ref in (("best_selector_k10", pub[(pub.cohort == d) & (pub.method == best[d]) & (pub.k == 10)]),
                                  ("ihc_4", pub[(pub.cohort == d) & (pub.method == "ihc_4")]),
                                  ("b0", pub[(pub.cohort == d) & (pub.method == "b0")])):
                b = ref.set_index(idx).auroc
                dd = (a - b).dropna()
                t = nadeau_bengio_ttest(dd.to_numpy(), alternative="two-sided")
                rows.append({"signature": sig, "cohort": d, "reference": ref_name, "reference_method": best[d] if ref_name.startswith("best") else ref_name,
                             "auroc_signature": float(a.mean()), "auroc_reference": float(b.mean()), "diff": t["mean"], "ci_low": t["ci_low"],
                             "ci_high": t["ci_high"], "p_nb_two_sided": t["p"], "n_folds": t["n"]})
    cmp = pd.DataFrame(rows)
    cmp["p_holm"] = np.nan
    for _, ix in cmp.groupby(["signature", "reference"]).groups.items():
        cmp.loc[ix, "p_holm"] = holm(cmp.loc[ix, "p_nb_two_sided"].to_numpy())
    _save(cmp, out, "b5_comparisons.csv")
    # không thua kém B0 với biên 0,02 như k* (một mức k = 10; gộp qua 3 cohort GPL570 theo IVW và trọng số bằng nhau)
    ni_rows = []
    both = pd.concat([pub, b5[pub.columns.intersection(b5.columns)]], ignore_index=True)
    for sig in b5.method.unique():
        pc = per_cohort(both[both.cohort.isin(GPL570)], sig, 10)
        p = pooled(pc)
        ni_rows.append({"signature": sig, "scope": "GPL570 pooled", **{k: v for k, v in p.items() if not k.startswith("w_")},
                        "non_inferior_ivw": bool(p["lb_ivw"] > -C.KSTAR_DELTA), "non_inferior_both": bool(p["lb_ivw"] > -C.KSTAR_DELTA and p["lb_eq"] > -C.KSTAR_DELTA)})
        for r in per_cohort(both, sig, 10).itertuples():
            ni_rows.append({"signature": sig, "scope": r.cohort, "diff_ivw": r.diff, "lb_ivw": r.lb, "non_inferior_ivw": bool(r.lb > -C.KSTAR_DELTA)})
    _save(pd.DataFrame(ni_rows), out, "b5_noninferiority_b0.csv")
    # E1b ở k = 10: mô tả
    e1b_rows = []
    for d in FEW:
        nb = inp.csv(_res("B5_signatures10", d, "b5_e1b_per_run.csv"), dtype={"n_t": str})
        pb = inp.csv(PUBLISHED_ROOT / "E1b" / d / "e1b_per_run.csv", dtype={"n_t": str})
        pb = pb[((pb.method == best[d]) & (pb.k == 10)) | (pb.method == "ihc_4")]
        allr = pd.concat([nb[["n_t", "method", "auroc"]], pb[["n_t", "method", "auroc"]]], ignore_index=True)
        for (n_t, m), x in allr.groupby(["n_t", "method"]):
            e1b_rows.append({"cohort": d, "n_t": n_t, "method": m, "role": "best_selector_k10" if m == best[d] else m,
                             "n_runs": int(len(x)), "auroc_mean": float(x.auroc.mean())})
    _save(pd.DataFrame(e1b_rows), out, "b5_e1b_k10.csv")
    _finish(out, prov, "B5", inp, {"best_selector_k10": best,
                                   "non_inferior_to_b0_pooled": {r["signature"]: r["non_inferior_both"] for r in ni_rows if r["scope"] == "GPL570 pooled"}},
            choices=["best data-driven selector at k = 10 per cohort = highest mean published E1 AUROC at k = 10 among the 13 selectors present",
                     "comparisons vs best selector and IHC-4: Nadeau–Bengio paired two-sided; Holm across the 5 cohorts within (signature, reference)",
                     "non-inferiority vs B0: one level k = 10 of the k* rule (lb one-sided 95 % > −0.02, IVW and equal weights, 3 GPL570 cohorts)"])


# ==========================================================================================================================
# B7: EGFR
# ==========================================================================================================================
def task_b7(args, prov):
    sys.path.insert(0, str(REPO / "scripts"))
    from make_survey_tables import e1_tests  # noqa: E402
    inp, out = Inputs(), results_dir("B7_tables")
    k = C.B7_GATE_K
    summ, tests, kst, aup = [], [], [], []
    for d in C.B7_DATASETS:
        pf = inp.maybe(_res("E1_classification_b7", d, "classification_per_fold.csv"))
        if pf is None:
            continue
        pf = pf[pf.model == "lr_l2"].copy()
        rk = inp.csv(_res("E1_classification_b7", d, "e1_random_k.csv"))
        t = e1_tests(pf.rename(columns={"model": "classifier"}), rk, "lr_l2")
        tests.append(t)
        sc = inp.maybe(_res("E1_classification_b7", d, "e1_scores.csv"))
        if sc is not None:
            ap = sc.groupby(["repeat", "fold", "method", "k"]).apply(lambda g: average_precision_score(g.label, g.score)).rename("auprc").reset_index()
            aup.append(ap.assign(cohort=d))
            pf = pf.merge(ap, on=["repeat", "fold", "method", "k"], how="left")
        s = pf.groupby(["cohort", "method", "k"], as_index=False)[[c for c in ("auroc", "auprc", "bacc", "mcc", "logloss", "brier") if c in pf]].mean()
        summ.append(s)
        methods = [m for m in pf.method.unique() if m not in ("b0", "random_k")]
        for delta in (C.KSTAR_DELTA, 0.01):
            kk = per_cohort_kstar(pf, methods, [d], delta=delta)
            kst.append(kk.assign(delta=delta))
    if not summ:
        raise SystemExit("chưa có E1_classification_b7 đủ (pha B)")
    summ, tests, kst = pd.concat(summ), pd.concat(tests), pd.concat(kst)
    _save(summ, out, "b7_e1_summary.csv")
    _save(tests, out, "b7_e1_tests_vs_random_b0.csv")
    k_wide = kst.pivot_table(index=["cohort", "method"], columns="delta", values="k_star_cohort", aggfunc="first").reset_index()
    k_wide.columns = [c if not isinstance(c, float) else f"k_star_delta_{c:g}" for c in k_wide.columns]
    _save(k_wide, out, "b7_kstar.csv")
    # R1
    r1 = tests[(tests.k == k) & tests.method.isin(C.NON_WRAPPER_METHODS)].groupby("cohort").beats_random_holm.agg(["sum", "size"]).reset_index()
    r1.columns = ["cohort", "n_beat_random_k10_holm", "n_selectors"]
    _save(r1, out, "b7_R1.csv")
    # R3: Kendall giữa thứ hạng AUROC ở k = 10 (EGFR) và LUAD/LUSC TCGA (đã công bố)
    pub = _published_lr_per_fold(inp, [C.SOURCE_DATASET])
    pub10 = pub[(pub.k == k) & pub.method.isin(C.NON_WRAPPER_METHODS)].groupby("method").auroc.mean()
    r3 = []
    for d in summ.cohort.unique():
        e = summ[(summ.cohort == d) & (summ.k == k) & summ.method.isin(C.NON_WRAPPER_METHODS)].set_index("method").auroc
        common = sorted(set(e.index) & set(pub10.index))
        tau, p = st.kendalltau(e[common], pub10[common])
        r3.append({"cohort": d, "reference": f"{C.SOURCE_DATASET} (published E1, LUAD vs LUSC)", "k": k, "n_methods": len(common),
                   "kendall_tau": float(tau), "p_value": float(p)})
    _save(pd.DataFrame(r3), out, "b7_R3_kendall_auroc.csv")
    # R4: Kendall của Nogueira ở k = 10, f = 0,7
    r4 = []
    st_new = inp.maybe(_res("E4_E5_b7", "tcga_luad_egfr", "e4_stability.csv"))
    if st_new is not None:
        st_pub = inp.csv(PUBLISHED_ROOT / "E4_E5" / C.SOURCE_DATASET / "e4_stability.csv")
        f = C.REDUNDANCY_FRACTION
        a = st_new[(st_new.k == k) & np.isclose(st_new.fraction, f) & st_new.method.isin(C.NON_WRAPPER_METHODS)].set_index("method").nogueira
        b = st_pub[(st_pub.k == k) & np.isclose(st_pub.fraction, f) & st_pub.method.isin(C.NON_WRAPPER_METHODS)].set_index("method").nogueira
        common = sorted(set(a.index) & set(b.index))
        tau, p = st.kendalltau(a[common], b[common])
        r4.append({"cohort": "tcga_luad_egfr", "reference": C.SOURCE_DATASET, "k": k, "fraction": f, "n_methods": len(common),
                   "kendall_tau": float(tau), "p_value": float(p)})
        _save(st_new, out, "b7_e4_stability.csv")
    _save(pd.DataFrame(r4), out, "b7_R4_kendall_stability.csv")
    if aup:
        _save(pd.concat(aup), out, "b7_auprc_per_fold.csv")
    _finish(out, prov, "B7", inp, {"R1": r1.to_dict("records"), "R3": r3, "R4": r4,
                                   "R5": "not computed: the writer reads the Fig. 13 decision framework against b7_e1_summary, b7_kstar, b7_e4_stability"},
            choices=["R1 uses the main E1 rule: Nadeau–Bengio one-sided vs random-k, Holm within (cohort, method) across k, read at k = 10",
                     "R2: per-cohort k* (single cohort; fixed sequence, lb one-sided 95 % > −Δ) with Δ = 0.02 and Δ = 0.01",
                     "R3/R4: Kendall tau-b over the 11 non-wrapper selectors present in both tables",
                     "AUPRC = average precision per fold from per-sample test scores (lr_l2)"])


# ==========================================================================================================================
# B6: ptLasso, Trans-GLM trong E7
# ==========================================================================================================================
B6_REFERENCE = PUBLISHED_ROOT / "B6_reference_rq2_r_7f097da"
B6_GATE_TOL = 1e-9
B6_KEY = ["split", "subset", "n_t", "method", "k"]


def task_b6gate(args, prov):
    """Cổng B6: trên 5 lần chia đã công bố, panel và AUROC từng lần chạy trùng rq2_r_all_7f097da (sai khác < 1e-9). Không đạt -> mã 3."""
    inp, out = Inputs(), results_dir("B6_gate")
    rows = []
    for co in FEW:
        new = inp.csv(_res(args.b6_gate_dir, co, "e7_r_per_run.csv"), dtype={"n_t": str})
        ref = inp.csv(B6_REFERENCE / co / "rq2_r_per_run.csv", dtype={"n_t": str})
        ref = ref[ref.n_t.isin(new.n_t.unique()) & ref.k.isin(new.k.unique())]
        g = compare_rows(new, ref, B6_KEY, cols=("auroc",), tol=B6_GATE_TOL)
        gn = inp.csv(_res(args.b6_gate_dir, co, "e7_r_selected_genes.csv"), dtype={"n_t": str})
        gr = inp.csv(B6_REFERENCE / co / "rq2_r_selected_genes.csv", dtype={"n_t": str})
        gr = gr[gr.n_t.isin(new.n_t.unique()) & gr.k.isin(new.k.unique())]
        pn = gn.sort_values("rank").groupby(B6_KEY).gene.apply(tuple)
        pr = gr.sort_values("rank").groupby(B6_KEY).gene.apply(tuple)
        both = pr.index.intersection(pn.index)
        n_panel_diff = int((pn.loc[both] != pr.loc[both]).sum()) + int(len(pr.index.difference(pn.index)))
        rows.append({"cohort": co, **g, "n_panels_reference": int(len(pr)), "n_panels_differ": n_panel_diff,
                     "pass_all": bool(g["pass"] and n_panel_diff == 0)})
    gate = pd.DataFrame(rows)
    _save(gate, out, "b6_gate.csv")
    ok = bool(gate.pass_all.all())
    _finish(out, prov, "B6_gate", inp, {"pass": ok, "tol": B6_GATE_TOL, "reference": str(B6_REFERENCE)})
    if not ok:
        raise SystemExit(3)


def task_b6(args, prov):
    inp, out = Inputs(), results_dir("B6_tables")
    labels, desc, degen = [], [], []
    for co in FEW:
        r = inp.csv(_res(args.b6_dir, co, "e7_r_per_run.csv"), dtype={"n_t": str})
        e7 = inp.csv(_res(args.b1_e7, co, "e7_per_run.csv"), dtype={"n_t": str})
        r = r[r.n_t.isin(e7.n_t.unique())].rename(columns={"method": "strategy"})
        cols = ["cohort", "split", "subset", "n_t", "strategy", "k", "auroc"]
        both = pd.concat([r[cols], e7[cols]], ignore_index=True)
        lab = split_level_labels(both[both.strategy.isin(["ptlasso", "transglm", "target_only"])], "strategy", "target_only",
                                 ["cohort", "n_t", "k"], seed_tag="b6")
        labels.append(lab)
        for ref in ("select_on_source", "pooled"):
            d = split_level_labels(both[both.strategy.isin(["ptlasso", "transglm", ref])], "strategy", ref, ["cohort", "n_t", "k"], seed_tag=f"b6{ref}")
            desc.append(d[["strategy", "cohort", "n_t", "k", "reference", "n_splits", "mean_delta", "ci_low", "ci_high"]])
        dg = inp.csv(_res(args.b6_dir, co, "e7_r_diagnostics.csv"), dtype={"n_t": str})
        dg = dg[dg.n_t.isin(e7.n_t.unique())]
        for (m, n_t, k), x in dg.groupby(["method", "n_t", "k"]):
            degen.append({"cohort": co, "method": m, "n_t": n_t, "k": int(k), "n_runs": int(len(x)),
                          "error_rate": float(x["error"].notna().mean()) if "error" in x else 0.0,
                          "filled_rate": float((x["n_filled"].fillna(0) > 0).mean()) if "n_filled" in x else np.nan,
                          "same_as_stage1_topk_rate": float(x["same_as_stage1_topk"].mean()) if "same_as_stage1_topk" in x and m == "ptlasso" else np.nan})
    lab = pd.concat(labels, ignore_index=True)
    _save(lab, out, "b6_split_labels_vs_target_only.csv")
    _save(pd.concat(desc, ignore_index=True), out, "b6_delta_vs_naive.csv")
    _save(pd.DataFrame(degen), out, "b6_degeneracy.csv")
    counts = lab.groupby(["strategy", "label"]).size().unstack(fill_value=0)
    _finish(out, prov, "B6", inp, {"label_counts": {s: r.to_dict() for s, r in counts.iterrows()}},
            choices=["same split-level rule as B1 (Wilcoxon exact two-sided, Holm within strategy, bootstrap CI over splits, margin 0.01)",
                     "deltas vs select-on-source and pooled are descriptive (no Holm)",
                     "target-only, select-on-source, pooled from B1_E7_20 (same 20 splits, same subsets)"])


TASKS = {"b1": task_b1, "b2": task_b2, "b3": task_b3, "b4": task_b4, "b5": task_b5, "b6gate": task_b6gate, "b6": task_b6, "b7": task_b7}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="+", choices=list(TASKS))
    ap.add_argument("--b1-e7", default="B1_E7_20")
    ap.add_argument("--b1-e1b", default="B1_E1b_20")
    ap.add_argument("--b4-threshold", type=float, default=0.935)
    ap.add_argument("--b6-gate-dir", default="B6_E7_gate5")
    ap.add_argument("--b6-dir", default="B6_E7_principled")
    ap.add_argument("--fewshot-cohorts", nargs="+", default=list(C.FEWSHOT_COHORTS), help=argparse.SUPPRESS)
    ap.add_argument("--main-datasets", nargs="+", default=list(C.MAIN_DATASETS), help=argparse.SUPPRESS)
    ap.add_argument("--kstar-cohorts", nargs="+", default=list(C.GPL570_COHORTS), help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None, script=__file__):
    global FEW, MAIN, GPL570
    args = parse_args(argv)
    FEW, MAIN, GPL570 = list(args.fewshot_cohorts), list(args.main_datasets), list(args.kstar_cohorts)
    prov = start_run(script, params={k: str(v) for k, v in vars(args).items()})
    for t in args.tasks:
        print(f"== {t}", flush=True)
        TASKS[t](args, prov)


if __name__ == "__main__":
    main()
