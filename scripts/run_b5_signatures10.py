"""B5 (QĐ175, đặc tả 12/11): chữ ký Girard 2016 và Wilkerson 2013 rút về 10 gen TRONG từng fold train.

Trong mỗi phần train: thống kê F của từng gen của chữ ký có trong không gian gen G, giữ 10 gen có F lớn nhất, phân loại bằng
LR-L2 như thường (CV nội bộ chọn C; quy tắc M5 cho n_T nhỏ ở E1b). Chữ ký có ít hơn 10 gen trong G thì giữ mọi gen có mặt và
ghi `exact_k = False` (không xảy ra với hai chữ ký này: 41/42 và ≥ 50/52 gen có mặt).

* e1  : 5 bộ × 25 fold (bản chia 5 × 5 đã công bố).
* e1b : 4 cohort GEO, 5 lần chia đã công bố, mọi (tập con, n_T), k = 10.

    python scripts/run_b5_signatures10.py --datasets gse37745

Đầu ra: results/survey_v2/B5_signatures10/<bộ>/{b5_e1_per_fold.csv, b5_e1b_per_run.csv, b5_selected_genes.csv, config.json}.
Phân tích (so với bộ chọn tốt nhất ở k = 10 và IHC-4, không thua kém B0 với biên 0,02): scripts/make_revision_tables.py b5.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256  # noqa: E402
from survey_v2.fewshot import fewshot_runs, validate_fewshot_manifest  # noqa: E402
from survey_v2.methods import ftest_order  # noqa: E402
from survey_v2.panels import PANELS_CSV, panel_indices, published_panels  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from scripts.build_fewshot_splits import fewshot_path  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402

B5_K = 10
SIGNATURES = ("panel_girard2016", "panel_wilkerson2013")
SEED_TAG = 91                     # phần tử seed của B5 (panel cố định công bố dùng 90)


def method_name(sig: str) -> str:
    return "sig10_" + sig.removeprefix("panel_")


def top_f_genes(Xtr: np.ndarray, ytr: np.ndarray, cols: np.ndarray, k: int = B5_K) -> np.ndarray:
    """k gen của chữ ký (chỉ số cột trong G) có F lớn nhất trên phần train; ít hơn k gen thì giữ tất cả."""
    cols = np.asarray(cols, dtype=int)
    if len(cols) <= k:
        return cols
    order = ftest_order(Xtr[:, cols], ytr)
    return cols[order[:k]]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN_DATASETS))
    ap.add_argument("--parts", nargs="+", default=["e1", "e1b"], choices=["e1", "e1b"])
    ap.add_argument("--signatures", nargs="+", default=list(SIGNATURES))
    ap.add_argument("--panels-csv", type=Path, default=PANELS_CSV, help="tệp chữ ký (mặc định survey_v2/panels/published_panels.csv)")
    ap.add_argument("--k", type=int, default=B5_K)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="B5_signatures10")
    return ap.parse_args(argv)


def _row(res, cols):
    return {"n_selected": len(cols), "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss,
            "brier": res.brier, "C": res.hyperparam}


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    sigs = published_panels(args.panels_csv)
    missing_sig = [s for s in args.signatures if s not in sigs]
    if missing_sig:
        raise SystemExit(f"thiếu chữ ký trong published_panels.csv: {missing_sig}")
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        avail = {}
        for s in args.signatures:
            cols, missing = panel_indices(ds.genes, sigs[s])
            avail[s] = (cols, missing)
        e1_rows, e1b_rows, gene_rows, info = [], [], [], {}
        if "e1" in args.parts:
            manifest_path = split_path(ds, args.splits_root)
            manifest = pd.read_csv(manifest_path)
            runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
            for run in runs.itertuples(index=False):
                repeat, seed, fold = int(run.repeat), int(run.seed), int(run.fold)
                tr, te = split_indices(ds, manifest, repeat, fold)
                with threadpool_limits(1):
                    for s in args.signatures:
                        cols_all = avail[s][0]
                        if len(cols_all) == 0:
                            continue
                        cols = top_f_genes(ds.X[tr], ds.y[tr], cols_all, args.k)
                        res = evaluate_panel(ds.X[tr][:, cols], ds.y[tr], ds.X[te][:, cols], ds.y[te], classifier="lr_l2",
                                             seed=C.seed_for(cid, seed, fold, args.k, SEED_TAG))
                        m = method_name(s)
                        e1_rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "method": m, "k": args.k,
                                        "model": "lr_l2", "n_signature_in_G": len(cols_all), "exact_k": len(cols) == args.k, **_row(res, cols)})
                        gene_rows += [{"part": "e1", "cohort": name, "repeat": repeat, "fold": fold, "method": m, "rank": r + 1,
                                       "gene": str(ds.genes[c])} for r, c in enumerate(cols)]
            pd.DataFrame(e1_rows).to_csv(out / "b5_e1_per_fold.csv", index=False, encoding="utf-8-sig")
            info["e1"] = {"split_file": str(manifest_path), "split_sha256": sha256(manifest_path), "rows": len(e1_rows)}
        if "e1b" in args.parts and name in C.FEWSHOT_COHORTS:
            path = fewshot_path(ds, args.splits_root)
            man = pd.read_csv(path, dtype={"n_t": str})
            errors = validate_fewshot_manifest(man)
            if errors:
                raise ValueError(f"{name}: fewshot_splits.csv không hợp lệ: {errors}")
            index = {s: i for i, s in enumerate(ds.sample_ids)}
            for split, seed, subset, n_t, train_ids, test_ids in fewshot_runs(man):
                tr = np.array([index[s] for s in train_ids]); te = np.array([index[s] for s in test_ids])
                n_t_int = 10**6 if n_t == "pool" else int(n_t)
                with threadpool_limits(1):
                    for s in args.signatures:
                        cols_all = avail[s][0]
                        if len(cols_all) == 0:
                            continue
                        cols = top_f_genes(ds.X[tr], ds.y[tr], cols_all, args.k)
                        res = evaluate_panel(ds.X[tr][:, cols], ds.y[tr], ds.X[te][:, cols], ds.y[te], classifier="lr_l2",
                                             seed=C.seed_for(cid, seed, subset, n_t_int, args.k, SEED_TAG))
                        m = method_name(s)
                        e1b_rows.append({"cohort": name, "split": split, "seed": seed, "subset": subset, "n_t": n_t, "n_train": len(tr),
                                         "n_test": len(te), "method": m, "k": args.k, "n_signature_in_G": len(cols_all),
                                         "exact_k": len(cols) == args.k, **_row(res, cols)})
            pd.DataFrame(e1b_rows).to_csv(out / "b5_e1b_per_run.csv", index=False, encoding="utf-8-sig")
            info["e1b"] = {"fewshot_splits_file": str(path), "fewshot_splits_sha256": sha256(path), "rows": len(e1b_rows)}
        pd.DataFrame(gene_rows).to_csv(out / "b5_selected_genes.csv", index=False, encoding="utf-8-sig")
        cfg = {"experiment": "B5_signatures10", "decision": "QĐ175 (Review 12/11), spec §B5", "dataset": name, "k": args.k,
               "rule": "per outer-train fold: F statistic of each signature gene present in G, keep the top 10, LR-L2 as usual",
               "panels_csv": str(args.panels_csv), "panels_csv_sha256": sha256(args.panels_csv),
               "signatures": {s: {"n_genes": len(sigs[s]), "n_in_G": int(len(avail[s][0])), "missing": avail[s][1]} for s in args.signatures},
               **info, "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
        print(f"[{name}] B5 xong: E1 {len(e1_rows)} dòng, E1b {len(e1b_rows)} dòng", flush=True)


if __name__ == "__main__":
    main()
