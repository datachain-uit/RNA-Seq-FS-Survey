"""Y10 — Sanity check (đối chứng dương) của pipeline survey v2 trên tcga_normal_vs_tumor_sanity.

Mục đích: chứng minh pipeline đánh giá (CV theo bệnh nhân, lọc sơ bộ trong fold, chọn k = 10 gen, LR-L2 với C chọn
bằng CV nội bộ) phát hiện được tín hiệu MẠNH (normal vs tumor), để kết quả yếu ở bài toán giai đoạn không phải do
pipeline hỏng. Tiêu chí đặt trước: AUROC trung bình ≥ 0,99 cho cả ba pipeline (f_test, l1_lr, rf_importance) ở k = 10.

    python scripts/run_sanity_auroc.py                    # 5×5 CV
    python scripts/run_sanity_auroc.py --repeats 0        # 1×5 CV (nhanh)

Đầu ra: results/survey_v2/sanity/{sanity_per_fold.csv, sanity_summary.csv, config.json}; mã thoát 1 nếu không đạt.
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
from survey_v2.data import load_dataset, repeated_patient_splits, variance_prefilter  # noqa: E402
from survey_v2.methods import select  # noqa: E402
from survey_v2.runlog import DATA_ROOT, all_versions, results_dir, start_run, write_config  # noqa: E402

SANITY_METHODS = ("f_test", "l1_lr", "rf_importance")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=C.SANITY_DATASET)
    ap.add_argument("--methods", nargs="+", default=list(SANITY_METHODS))
    ap.add_argument("--k", type=int, default=C.SANITY_K)
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(len(C.REPEAT_SEEDS))))
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--threshold", type=float, default=C.SANITY_AUROC_THRESHOLD)
    ap.add_argument("--out", default="sanity")
    return ap.parse_args(argv)


def main(argv=None, script=__file__) -> int:
    args = parse_args(argv)
    prov = start_run(script, params=vars(args))
    out = results_dir(args.out)
    ds = load_dataset(DATA_ROOT, args.dataset)
    splits = [s for s in repeated_patient_splits(ds.y, ds.groups) if s[0] in set(args.repeats)]
    print(f"[{ds.name}] {ds.n} mẫu × {ds.p} gen | {len(set(ds.groups))} bệnh nhân | {len(splits)} fold", flush=True)
    rows = []
    t0 = time.time()
    with threadpool_limits(1):
        for repeat, seed, fold, tr, te in splits:
            pre = variance_prefilter(ds.X[tr], args.prefilter)
            Xtr, Xte = ds.X[tr][:, pre], ds.X[te][:, pre]
            for m in args.methods:
                sel = select(m, Xtr, ds.y[tr], [args.k], seed=C.seed_for(9, seed, fold, 1))
                idx = sel.panels[args.k]
                r = evaluate_panel(Xtr[:, idx], ds.y[tr], Xte[:, idx], ds.y[te], classifier="lr_l2", seed=C.seed_for(9, seed, fold, 2))
                rows.append({"cohort": ds.name, "repeat": repeat, "seed": seed, "fold": fold, "method": m, "k": args.k,
                             "auroc": r.auroc, "bacc": r.bacc, "mcc": r.mcc, "n_train": len(tr), "n_test": len(te),
                             "genes": ";".join(ds.genes[pre[idx]])})
            print(f"  repeat {repeat} fold {fold}: " + ", ".join(f"{x['method']} {x['auroc']:.3f}" for x in rows[-len(args.methods):]), flush=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "sanity_per_fold.csv", index=False, encoding="utf-8-sig")
    summary = per_fold.groupby(["cohort", "method", "k"], as_index=False).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), auroc_min=("auroc", "min"),
        bacc_mean=("bacc", "mean"), mcc_mean=("mcc", "mean"), n_folds=("auroc", "size"))
    summary["pass"] = summary["auroc_mean"] >= args.threshold
    summary.to_csv(out / "sanity_summary.csv", index=False, encoding="utf-8-sig")
    ok = bool(summary["pass"].all())
    print(summary.to_string(index=False))
    print("SANITY", "ĐẠT" if ok else "KHÔNG ĐẠT", f"(ngưỡng AUROC trung bình ≥ {args.threshold})", flush=True)
    write_config(out, prov, {"experiment": "sanity_Y10", "dataset": ds.name, "n_samples": ds.n, "n_genes": ds.p,
                             "n_patients": int(len(set(ds.groups))), "methods": args.methods, "k": args.k,
                             "repeats": args.repeats, "threshold": args.threshold, "pass": ok,
                             "versions": all_versions(), "elapsed_s": time.time() - t0},
                 data_files=[ds.data_dir / "expression.h5", ds.data_dir / "metadata.csv"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
