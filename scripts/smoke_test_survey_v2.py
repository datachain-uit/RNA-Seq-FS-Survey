"""Chạy thử toàn bộ chuỗi survey v3 trên DỮ LIỆU TỔNG HỢP (≈ 5–15 phút) để kiểm tra môi trường trước khi chạy thật.

Tạo KLTN_DATA_ROOT tạm (hoặc --root), sinh nguồn TCGA giả (perm/), 4 cohort GEO giả (geo-target/) và bộ sanity, rồi chạy
lần lượt: build_e1_repeated_splits → build_fewshot_splits → run_survey_e0_signal_check → run_e1_feature_selection (pha A)
→ run_e1_wrappers_all_cohorts → run_e1_classification (pha B) → run_survey_e1b_learning_curve → run_survey_e7_transfer_naive
→ run_survey_e2_e3_timing → run_survey_e4_e5_stability → make_survey_tables → run_sanity_auroc. In thời gian từng bước.
Kết quả chạy thử KHÔNG được dùng cho bài (dữ liệu giả); xóa thư mục tạm sau khi xem.

    python scripts/smoke_test_survey_v2.py                 # thư mục tạm, cấu hình nhỏ
    python scripts/smoke_test_survey_v2.py --root /tmp/kltn_smoke --n 160 --p 1500 --full-methods --n-jobs 4
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="KLTN_DATA_ROOT tạm (mặc định: thư mục tạm của hệ thống)")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--p", type=int, default=800)
    ap.add_argument("--full-methods", action="store_true", help="chạy đủ 11 phương pháp không wrapper (chậm hơn)")
    ap.add_argument("--n-jobs", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="kltn_smoke_"))
    os.environ["KLTN_DATA_ROOT"] = str(root)
    os.environ.setdefault("KLTN_ALLOW_UNPUSHED", "1")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import config as C
    from survey_v2 import synthetic

    print(f"KLTN_DATA_ROOT (tạm) = {root}")
    t0 = time.time()
    synthetic.make_all(root, n=args.n, p=args.p, effect=0.8, n_target=max(60, args.n // 2))
    timings = {"synthetic data": time.time() - t0}
    methods = list(C.NON_WRAPPER_METHODS) if args.full_methods else ["f_test", "mrmr", "l1_lr", "stability_selection", "ktsp"]
    pre = str(min(400, args.p))
    nj = str(args.n_jobs)
    main_cohort = "gse37745"

    def run(label, module, argv, entry="main"):
        t = time.time()
        mod = importlib.import_module(module)
        sys.argv = [f"{module}.py", *argv]
        fn = getattr(mod, entry)
        rc = fn() if module == "run_survey_e2_e3_timing" else fn(argv)
        timings[label] = time.time() - t
        print(f"==> {label}: {timings[label]:.1f}s\n", flush=True)
        return rc

    run("Bản chia 5×5 (5 bộ)", "build_e1_repeated_splits", [])
    run("Bản chia few-shot (4 cohort)", "build_fewshot_splits", [])
    run("E0 kiểm tra tín hiệu (2 cohort, 20 hoán vị)", "run_survey_e0_signal_check",
        ["--datasets", main_cohort, "gse10245", "--n-perm", "20", "--n-random", "10", "--prefilter", pre, "--n-jobs", nj])
    run("E1 pha A (chọn gen, 2 bộ, 1 lần lặp)", "run_e1_feature_selection",
        ["--datasets", main_cohort, C.SOURCE_DATASET, "--methods", *methods, "ihc_2", "ihc_4", "--k", "1", "2", "3", "4", "5", "10", "25",
         "--repeats", "0", "--prefilter", pre])
    run("E1 pha A wrapper (GA+BPSO, quần thể 8 × 3 thế hệ)", "run_e1_wrappers_all_cohorts",
        ["--datasets", main_cohort, "--repeats", "0", "--k", "5", "10", "--prefilter", pre, "--pop", "8", "--generations", "3"], entry="run")
    run("E1 pha B (LR-L2 + phụ, B0, random-k)", "run_e1_classification",
        ["--datasets", main_cohort, C.SOURCE_DATASET, "--repeats", "0", "--secondary", "--random-draws", "5", "--prefilter", pre])
    run("E1b đường cong học (1 cohort)", "run_survey_e1b_learning_curve",
        ["--datasets", main_cohort, "--methods", *methods, "ihc_2", "ihc_4", "--k", "2", "4", "5", "10", "--prefilter", pre,
         "--random-draws", "5", "--n-jobs", nj])
    run("E7 chuyển giao đơn giản (1 cohort)", "run_survey_e7_transfer_naive",
        ["--datasets", main_cohort, "--k", "2", "5", "10", "--n-t", "5", "10", "--prefilter", pre, "--n-jobs", nj])
    run("E2+E3 (thời gian)", "run_survey_e2_e3_timing",
        ["--methods", *methods, "--repeats", "1", "--n-grid", "40", "80", "all", "--p-grid", "200", "400", "all", "--k", "10"])
    run("E4+E5 (ổn định, M = 6)", "run_survey_e4_e5_stability",
        ["--datasets", main_cohort, "--methods", *methods, "--k", "2", "5", "10", "--m", "6", "--prefilter", pre, "--wrapper-k", "10",
         "--pop", "8", "--generations", "3", "--n-jobs", nj])
    run("Bảng + hình", "make_survey_tables", ["--e6-fraction", "0.7"])
    rc = run("Sanity Y10", "run_sanity_auroc", ["--repeats", "0", "--prefilter", pre])
    print("\n=== Thời gian từng bước ===")
    for k, v in timings.items():
        print(f"{v:8.1f}s  {k}")
    print(f"\nTổng {sum(timings.values()) / 60:.1f} phút. Kết quả thử ở {root / 'results' / 'survey_v2'} (dữ liệu giả, không dùng cho bài).")
    print("Sanity:", "ĐẠT" if rc == 0 else "KHÔNG ĐẠT (bình thường với dữ liệu giả nhỏ nếu AUROC < 0,99)")


if __name__ == "__main__":
    main()
