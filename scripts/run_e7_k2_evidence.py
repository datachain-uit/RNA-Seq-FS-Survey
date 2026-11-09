"""Y62 phần 1 (yêu cầu 09/11/2026, Việc 1) — tệp số liệu cho đoạn zero-shot ở k = 2 (Mục 7.4 bài survey).

Không chạy lại E7 và không sửa mã sinh số liệu [v03-11]. Script chỉ đọc dữ liệu sau QC và kết quả E7 (commit 9b2d6a5) rồi ghi:

* e7_k2_gene_directions.csv : mỗi (cohort, gene): n, trung bình, độ lệch chuẩn theo lớp, shift = mean_LUSC − mean_LUAD,
  AUROC khi dùng giá trị gen làm điểm cho LUSC (nhãn 1 = LUSC). Thêm một dòng φ = gen 1 − gen 2 của panel nguồn k = 2
  (đúng thứ tự cột E7 dùng trong survey_v2.transfer.panel_transform). Tính trên toàn bộ cohort, cùng tập mẫu và thang đo E7.
* e7_k2_zero_shot_per_run.csv, e7_k2_zero_shot_summary.csv (+ _by_n_t) : các dòng strategy = source_only, k = 2 của
  E7/<cohort>/e7_per_run.csv, và tóm tắt (n_runs_below_0_5 = số lần chạy có AUROC < 0,5).
* e7_k2_source_coefficient.csv : fit lại đúng mô hình source_only ở k = 2 (TransferContext.source_only: evaluate_panel(...,
  classifier="lr_l2", seed = seed của context, small_n=False) trên φ của TCGA) rồi ghi C, hệ số và hằng số. E7 dùng seed
  riêng theo cohort đích cho CV nội bộ, nên có một mô hình nguồn cho mỗi cohort đích. Kiểm tra tái lập: AUROC của mô hình
  fit lại trên từng tập test few-shot phải trùng cột auroc của e7_per_run.csv.

    python scripts/run_e7_k2_evidence.py
    python scripts/run_e7_k2_evidence.py --e7-dir /path/results/survey_v2/E7 --transfer-table /path/e7_transfer_table.csv

Đầu ra: $KLTN_DATA_ROOT/results/survey_v2/E7_k2_evidence/{4 CSV, e7_k2_checks.json, config.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.fewshot import fewshot_runs  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from survey_v2.transfer import panel_transform  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402

K = 2
STRATEGY = "source_only"
EXTRA_GENES = ("DSC3", "HNF1B", "ARHGEF38")          # các gen vào panel nguồn ở k = 3, 4, 5
PER_RUN_COLUMNS = ["cohort", "split", "seed", "subset", "n_t", "k", "strategy", "auroc"]
# Thang đo của dữ liệu E7 (ma trận expression.h5 sau QC); GEO: theo !Sample_data_processing của series matrix.
SCALE = {
    "tcga_luad_lusc_primary_tumor": "log2(TPM+1)",
    "gse10245": "log2 gcRMA",
    "gse30219": "log2 RMA",
    "gse37745": "log2 RMA",
    "gse42127": "log2 MBCB (Illumina GPL6884)",
}
# Kỳ vọng trong yêu cầu 09/11 — chỉ dùng để ghi cờ khớp/lệch vào e7_k2_checks.json, không chỉnh số.
EXPECTED_ZERO_SHOT_MEAN = {"gse10245": 0.800, "gse30219": 0.607, "gse37745": 0.343, "gse42127": 0.828}
EXPECTED_PHI_AUROC_TCGA = 0.155


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.FEWSHOT_COHORTS))
    ap.add_argument("--source", default=C.SOURCE_DATASET)
    ap.add_argument("--e7-dir", type=Path, default=None, help="thư mục E7 (mặc định $KLTN_DATA_ROOT/results/survey_v2/E7)")
    ap.add_argument("--transfer-table", type=Path, default=None, help="e7_transfer_table.csv để đối chiếu trung bình (tuỳ chọn)")
    ap.add_argument("--extra-genes", nargs="*", default=list(EXTRA_GENES))
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="E7_k2_evidence")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------------------------------
# Hàm thuần (test được không cần dữ liệu thật)
# ---------------------------------------------------------------------------------------------------
def direction_row(cohort: str, gene: str, scale: str, x: np.ndarray, y: np.ndarray) -> dict:
    """Thống kê theo lớp của một biến x (nhãn 0 = LUAD, 1 = LUSC); AUROC dùng x làm điểm cho LUSC."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    a, b = x[y == 0], x[y == 1]
    return {"cohort": cohort, "gene": gene, "scale": scale, "n_luad": int(len(a)), "n_lusc": int(len(b)),
            "mean_luad": float(a.mean()), "mean_lusc": float(b.mean()),
            "sd_luad": float(a.std(ddof=1)), "sd_lusc": float(b.std(ddof=1)),
            "shift": float(b.mean() - a.mean()), "auroc_as_lusc_score": float(roc_auc_score(y, x))}


def phi_name(panel) -> str:
    return f"phi_{panel[0]}_minus_{panel[1]}"


def gene_direction_rows(cohort: str, scale: str, X: np.ndarray, genes, y: np.ndarray, panel, extra_genes=()) -> list[dict]:
    """Dòng cho các gen của panel (+ gen thêm có trong cohort) và một dòng cho φ = panel[0] − panel[1]."""
    index = {g: j for j, g in enumerate(np.asarray(genes).astype(str))}
    rows = []
    for g in list(panel) + [g for g in extra_genes if g not in panel]:
        if g in index:
            rows.append(direction_row(cohort, g, scale, X[:, index[g]], y))
    phi = panel_transform(X[:, [index[panel[0]], index[panel[1]]]])[:, 0]
    rows.append(direction_row(cohort, phi_name(panel), scale, phi, y))
    return rows


def zero_shot_rows(per_run: pd.DataFrame, k: int = K, strategy: str = STRATEGY) -> pd.DataFrame:
    sel = per_run[(per_run["strategy"] == strategy) & (per_run["k"].astype(int) == k)]
    return sel[PER_RUN_COLUMNS].reset_index(drop=True)


def zero_shot_summary(rows: pd.DataFrame, by=("cohort",)) -> pd.DataFrame:
    g = rows.groupby(list(by), sort=False)["auroc"]
    out = pd.DataFrame({"n_runs": g.size(), "auroc_mean": g.mean(), "auroc_sd": g.std(ddof=1),
                        "auroc_min": g.min(), "auroc_max": g.max(),
                        "n_runs_below_0_5": g.apply(lambda s: int((s < 0.5).sum()))})
    return out.reset_index()


def fit_source_model(Xs_phi: np.ndarray, ys: np.ndarray, Xt_phi: np.ndarray, yt: np.ndarray, seed: int) -> dict:
    """evaluate_panel đúng như TransferContext.source_only, rồi fit lại StandardScaler + LR-L2 cùng C/seed để đọc hệ số.

    Mô hình fit lại giống hệt _fit_predict trong survey_v2.classify (cùng tham số, cùng dữ liệu), nên điểm trên Xt_phi phải
    trùng res.scores_test; độ lệch lớn nhất được trả về để kiểm tra.
    """
    res = evaluate_panel(Xs_phi, ys, Xt_phi, yt, classifier="lr_l2", seed=seed, small_n=False)
    sc = StandardScaler().fit(Xs_phi)
    lr = LogisticRegression(C=res.hyperparam, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(sc.transform(Xs_phi), ys)
    p = lr.predict_proba(sc.transform(Xt_phi))[:, 1]
    coef, icpt = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    mu, sd = float(sc.mean_[0]), float(sc.scale_[0])
    return {"C": float(res.hyperparam), "coef_phi": coef, "intercept": icpt,
            "coef_phi_raw_scale": coef / sd, "intercept_raw_scale": icpt - coef * mu / sd,
            "scaler_mean_phi": mu, "scaler_sd_phi": sd, "inner_cv_seed": int(seed),
            "auroc_target_full_cohort": float(res.auroc), "max_abs_diff_scores_vs_evaluate_panel": float(np.max(np.abs(p - res.scores_test))),
            "_scaler": sc, "_model": lr}


# ---------------------------------------------------------------------------------------------------
def _read_e7_config(e7_dir: Path, name: str) -> dict:
    return json.loads((e7_dir / name / "config.json").read_text(encoding="utf-8"))


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    e7_dir = args.e7_dir or DATA_ROOT.joinpath(*C.RESULT_DIR, "E7")
    if args.transfer_table is None:
        default_tt = DATA_ROOT.joinpath(*C.RESULT_DIR, "tables", "e7_transfer_table.csv")
        args.transfer_table = default_tt if default_tt.exists() else None
    out = results_dir(args.out)
    checks: dict = {"expected": {"zero_shot_mean": EXPECTED_ZERO_SHOT_MEAN, "phi_auroc_tcga": EXPECTED_PHI_AUROC_TCGA}}

    # --- panel nguồn k = 2 theo từng cohort đích (đọc từ config E7, không chọn lại) --------------------------------
    e7_cfg = {name: _read_e7_config(e7_dir, name) for name in args.datasets}
    panels = {name: list(cfg["source_panels"][str(K)]) for name, cfg in e7_cfg.items()}
    distinct = sorted({tuple(p) for p in panels.values()})
    checks["source_panel_k2"] = panels
    checks["source_panel_k2_same_for_all_cohorts"] = len(distinct) == 1
    checks["e7_commit_hash"] = {name: cfg.get("commit_hash") for name, cfg in e7_cfg.items()}
    checks["e7_traceable"] = {name: cfg.get("traceable") for name, cfg in e7_cfg.items()}

    source = load_dataset(DATA_ROOT, args.source, gene_space=None)          # như run_survey_e7_transfer_naive.py
    src_sha = {f: source.files_sha256[f] for f in source.files_sha256}
    checks["source_sha256_matches_e7"] = {name: cfg.get("source_sha256") == src_sha for name, cfg in e7_cfg.items()}

    # --- Tệp 1: hướng thay đổi của gen ----------------------------------------------------------------------------
    dir_rows = []
    for p in distinct:
        dir_rows += gene_direction_rows(source.name, SCALE.get(source.name, "?"), source.X, source.genes, source.y, p, args.extra_genes)
    targets = {}
    for name in args.datasets:
        ds = load_dataset(DATA_ROOT, name)
        targets[name] = ds
        dir_rows += gene_direction_rows(name, SCALE.get(name, "?"), ds.X, ds.genes, ds.y, panels[name], args.extra_genes)
    directions = pd.DataFrame(dir_rows)
    directions.to_csv(out / "e7_k2_gene_directions.csv", index=False)
    genes_main = [g for p in distinct for g in p]
    sub = directions[directions["gene"].isin(genes_main)]
    checks["panel_genes_higher_in_lusc_all_cohorts"] = bool((sub["shift"] > 0).all())
    phi_tcga = directions[(directions["cohort"] == source.name) & directions["gene"].str.startswith("phi_")]["auroc_as_lusc_score"]
    checks["phi_auroc_tcga"] = [float(v) for v in phi_tcga]

    # --- Tệp 2: zero-shot k = 2 theo lần chạy -----------------------------------------------------------------------
    per_run_files = {}
    zs = []
    for name in args.datasets:
        f = e7_dir / name / "e7_per_run.csv"
        per_run_files[name] = f
        zs.append(zero_shot_rows(pd.read_csv(f, dtype={"n_t": str})))
    zs = pd.concat(zs, ignore_index=True)
    zs.to_csv(out / "e7_k2_zero_shot_per_run.csv", index=False)
    summary = zero_shot_summary(zs)
    summary.to_csv(out / "e7_k2_zero_shot_summary.csv", index=False)
    zero_shot_summary(zs, by=("cohort", "n_t")).to_csv(out / "e7_k2_zero_shot_summary_by_n_t.csv", index=False)
    checks["zero_shot_mean_matches_expected_3dp"] = {
        r.cohort: bool(round(r.auroc_mean, 3) == EXPECTED_ZERO_SHOT_MEAN.get(r.cohort)) for r in summary.itertuples()}
    checks["n_runs_below_0_5"] = {r.cohort: int(r.n_runs_below_0_5) for r in summary.itertuples()}
    if args.transfer_table is not None:
        tt = pd.read_csv(args.transfer_table, dtype={"n_t": str})
        tt = tt[(tt["strategy"] == STRATEGY) & (tt["k"].astype(int) == K)]
        by_nt = zero_shot_summary(zs, by=("cohort", "n_t")).merge(tt[["cohort", "n_t", "auroc_strategy"]], on=["cohort", "n_t"])
        checks["max_abs_diff_vs_transfer_table"] = float((by_nt["auroc_mean"] - by_nt["auroc_strategy"]).abs().max())
        checks["transfer_table_sha256"] = sha256(args.transfer_table)

    # --- Tệp 3: hệ số của φ trong LR-L2 nguồn -----------------------------------------------------------------------
    coef_rows = []
    split_files = []
    for name in args.datasets:
        ds, panel = targets[name], panels[name]
        seed = C.seed_for(C.cohort_id(name), 0, 0, 7)                        # seed của TransferContext trong E7
        s_idx = [int(np.flatnonzero(source.genes == g)[0]) for g in panel]
        t_idx = [int(np.flatnonzero(ds.genes == g)[0]) for g in panel]
        Xs_phi = panel_transform(source.X[:, s_idx])
        Xt_phi = panel_transform(ds.X[:, t_idx])
        fit = fit_source_model(Xs_phi, source.y, Xt_phi, ds.y, seed)
        sc, lr = fit.pop("_scaler"), fit.pop("_model")
        # Tái lập: AUROC trên từng tập test few-shot phải trùng e7_per_run.csv
        path = fewshot_path(ds, args.splits_root)
        split_files.append(path)
        man = pd.read_csv(path, dtype={"n_t": str})
        index = {s: i for i, s in enumerate(ds.sample_ids)}
        ref = zs[zs["cohort"] == name].set_index(["split", "subset", "n_t"])["auroc"]
        diffs = []
        for split, _seed, subset, n_t, _tr, te_ids in fewshot_runs(man):
            key = (split, subset, n_t)
            if key not in ref.index:
                continue
            te = np.array([index[s] for s in te_ids])
            p = lr.predict_proba(sc.transform(Xt_phi[te]))[:, 1]
            diffs.append(abs(float(roc_auc_score(ds.y[te], p)) - float(ref.loc[key])))
        e7_C = sorted(pd.read_csv(per_run_files[name], dtype={"n_t": str})
                      .query("strategy == @STRATEGY and k == @K")["C"].astype(float).unique().tolist())
        coef_rows.append({"cohort_source": source.name, "cohort_target": name, "panel": "+".join(panel), "phi": phi_name(panel),
                          **fit, "C_in_e7_per_run": ";".join(f"{c:g}" for c in e7_C),
                          "C_matches_e7": e7_C == [fit["C"]], "n_runs_checked": len(diffs),
                          "max_abs_diff_auroc_vs_e7_per_run": float(max(diffs)) if diffs else float("nan")})
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(out / "e7_k2_source_coefficient.csv", index=False)
    checks["coef_phi_negative_all"] = bool((coef["coef_phi"] < 0).all())
    checks["reproduces_e7_per_run"] = bool((coef["max_abs_diff_auroc_vs_e7_per_run"] < 1e-9).all() and coef["C_matches_e7"].all())
    (out / "e7_k2_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    cfg = {"experiment": "E7_k2_evidence", "request": "YEU-CAU-DOI-CODE-2026-11-09 Việc 1 (Y62 phần 1)", "source": args.source,
           "datasets": args.datasets, "k": K, "strategy": STRATEGY, "extra_genes": args.extra_genes, "scale": SCALE,
           "e7_dir": str(e7_dir), "e7_per_run_sha256": {n: sha256(f) for n, f in per_run_files.items()},
           "e7_config_commit": checks["e7_commit_hash"], "source_sha256": src_sha,
           "fewshot_splits_sha256": {str(f): sha256(f) for f in split_files},
           "outputs": sorted(p.name for p in out.glob("*.csv")) + ["e7_k2_checks.json"],
           "checks": checks, "elapsed_s": time.time() - t_all, "versions": all_versions()}
    data_files = [source.data_dir / "expression.h5", source.data_dir / "metadata.csv", *per_run_files.values(), *split_files]
    for ds in targets.values():
        data_files += [ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"]
    write_config(out, prov, cfg, data_files=data_files)
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str))
    print(f"[E7_k2_evidence] xong: {out} ({time.time() - t_all:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
