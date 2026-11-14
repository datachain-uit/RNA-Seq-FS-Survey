"""B4 (QĐ174, đặc tả 12/11): ba biến thể của B0 trên đúng 25 fold của E1, năm bộ; C chọn bằng CV nội bộ như giao thức chính.

* b0                : B0 gốc chạy lại (toàn bộ G, lưới C gốc {0,01; 0,1; 1; 10}) — CỔNG: phải trùng E1 đã công bố.
* b0_full_wide      : toàn bộ G, lưới rộng {1e-4; 1e-3; 0,01; 0,1; 1; 10}.
* b0_p5000          : 5.000 gen lọc phương sai trong fold (không nhãn, trên phần train), lưới gốc.
* b0_p5000_wide     : 5.000 gen lọc trong fold, lưới rộng.

Cùng bộ phân loại (LR-L2, StandardScaler, balanced, liblinear), cùng seed với B0 của E1 (seed_for(cohort, seed, fold, 0)).

    python scripts/run_b4_b0_variants.py --datasets gse42127

Đầu ra: results/survey_v2/B4_b0_variants/<bộ>/{b4_per_fold.csv, config.json}. Bảng và k* độ nhạy: make_revision_tables.py b4.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from survey_v2.classify import evaluate_panel  # noqa: E402
from survey_v2.data import load_dataset, sha256, variance_prefilter  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402
from scripts.run_e1_classification import append_csv_aligned  # noqa: E402
from scripts.run_e1_feature_selection import split_indices, split_path  # noqa: E402

WIDE_C_GRID = (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)
VARIANTS = {"b0": (False, C.C_GRID), "b0_full_wide": (False, WIDE_C_GRID),
            "b0_p5000": (True, C.C_GRID), "b0_p5000_wide": (True, WIDE_C_GRID)}
KEY = ["repeat", "fold", "method"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.MAIN_DATASETS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--splits-root", type=Path, default=None)
    ap.add_argument("--out", default="B4_b0_variants")
    return ap.parse_args(argv)


def main(argv=None, script=__file__):
    args = parse_args(argv)
    prov = start_run(script, params={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    t_all = time.time()
    for name in args.datasets:
        ds, out = load_dataset(DATA_ROOT, name), results_dir(args.out, name)
        cid = C.cohort_id(name)
        manifest_path = split_path(ds, args.splits_root)
        manifest = pd.read_csv(manifest_path)
        per_fold = out / "b4_per_fold.csv"
        done = set()
        if per_fold.exists():
            done = set(map(tuple, pd.read_csv(per_fold)[KEY].itertuples(index=False, name=None)))
        runs = manifest[["repeat", "seed", "fold"]].drop_duplicates().sort_values(["repeat", "fold"])
        for run in runs.itertuples(index=False):
            repeat, seed, fold = int(run.repeat), int(run.seed), int(run.fold)
            if repeat not in args.repeats:
                continue
            tr, te = split_indices(ds, manifest, repeat, fold)
            Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
            pre = None
            rows = []
            with threadpool_limits(1):
                for v in args.variants:
                    if (repeat, fold, v) in done:
                        continue
                    prefilter, grid = VARIANTS[v]
                    cols = None
                    if prefilter:
                        pre = variance_prefilter(Xtr, args.prefilter) if pre is None else pre
                        cols = pre
                    t0 = time.perf_counter()
                    a, b = (Xtr, Xte) if cols is None else (Xtr[:, cols], Xte[:, cols])
                    res = evaluate_panel(a, ytr, b, yte, classifier="lr_l2", seed=C.seed_for(cid, seed, fold, 0), c_grid=grid)
                    rows.append({"cohort": name, "repeat": repeat, "seed": seed, "fold": fold, "method": v, "k": int(a.shape[1]),
                                 "model": "lr_l2", "n_features": int(a.shape[1]), "c_grid": ";".join(f"{c:g}" for c in grid),
                                 "auroc": res.auroc, "bacc": res.bacc, "mcc": res.mcc, "logloss": res.logloss, "brier": res.brier,
                                 "threshold": res.threshold, "C": res.hyperparam, "fit_time_s": time.perf_counter() - t0})
                    print(f"[{name}] repeat {repeat} fold {fold} {v}: AUROC={res.auroc:.4f} C={res.hyperparam}", flush=True)
            if rows:
                append_csv_aligned(pd.DataFrame(rows), per_fold)
        df = pd.read_csv(per_fold) if per_fold.exists() else pd.DataFrame()
        cfg = {"experiment": "B4_b0_variants", "decision": "QĐ174 (Review 12/11), spec §B4", "dataset": name,
               "variants": {v: {"prefilter_p0": args.prefilter if VARIANTS[v][0] else None, "c_grid": list(VARIANTS[v][1])} for v in args.variants},
               "protocol": "same 25 outer folds as E1; C and threshold by inner 3-fold CV on outer train; seed = seed_for(cohort, seed, fold, 0) as B0",
               "gate": "b0 must reproduce published E1 b0 (|ΔAUROC| < 1e-12), checked by make_revision_tables.py b4",
               "split_file": str(manifest_path), "split_sha256": sha256(manifest_path), "n_rows": int(len(df)),
               "gene_space": ds.notes, "n_genes": ds.p, "elapsed_s": time.time() - t_all, "versions": all_versions()}
        write_config(out, prov, cfg, data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv", manifest_path])
        print(f"[{name}] B4 xong: {len(df)} dòng", flush=True)


if __name__ == "__main__":
    main()
