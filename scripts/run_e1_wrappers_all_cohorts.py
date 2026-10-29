"""E1 pha A cho wrapper (GA, BPSO) — survey v3: chỉ các cohort WRAPPER_DATASETS (3 cohort GPL570), 3 lần lặp đầu (3×5),
k ∈ WRAPPER_K = {5, 10, 50}; quần thể 30 × 50 thế hệ; fitness = AUROC CV 3 fold của LR-L2 trong phần train (O4).

Đây là điểm vào riêng cho pha A của wrapper (Kaggle: một Save Version = một cohort). Sau đó chạy pha B:
    python scripts/run_e1_classification.py --datasets <cohort> --methods ga bpso
(pha B mặc định gộp panel từ E1_feature_selection và E1_wrapper_feature_selection).

    python scripts/run_e1_wrappers_all_cohorts.py                       # 3 cohort, GA + BPSO
    python scripts/run_e1_wrappers_all_cohorts.py --datasets gse37745 --methods ga --repeats 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
from scripts.run_e1_feature_selection import main  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(C.WRAPPER_DATASETS))
    ap.add_argument("--methods", nargs="+", default=list(C.WRAPPER_METHODS))
    ap.add_argument("--repeats", nargs="+", type=int, default=list(range(C.WRAPPER_REPEATS)))
    ap.add_argument("--k", nargs="+", type=int, default=list(C.WRAPPER_K))
    ap.add_argument("--splits-root")
    ap.add_argument("--pop", type=int, default=C.WRAPPER_POP)
    ap.add_argument("--generations", type=int, default=C.WRAPPER_GENERATIONS)
    ap.add_argument("--prefilter", type=int, default=C.PREFILTER_P0)
    ap.add_argument("--out", default="E1_wrapper_feature_selection")
    return ap.parse_args(argv)


def run(argv=None, script=__file__):
    args = parse_args(argv)
    bad = [m for m in args.methods if m not in C.WRAPPER_METHODS]
    if bad:
        raise SystemExit(f"chỉ nhận wrapper {C.WRAPPER_METHODS}, gặp {bad}")
    outside = [d for d in args.datasets if d not in C.WRAPPER_DATASETS]
    if outside:
        print(f"CẢNH BÁO: {outside} không thuộc WRAPPER_DATASETS {C.WRAPPER_DATASETS} (QĐ15: wrapper chỉ 3 cohort GPL570)", file=sys.stderr)
    forwarded = ["--datasets", *args.datasets, "--methods", *args.methods, "--repeats", *map(str, args.repeats),
                 "--k", *map(str, args.k), "--pop", str(args.pop), "--generations", str(args.generations),
                 "--prefilter", str(args.prefilter), "--out", args.out]
    if args.splits_root:
        forwarded += ["--splits-root", args.splits_root]
    main(forwarded, script=script)


if __name__ == "__main__":
    run()
