"""Chạy thử toàn bộ chuỗi survey v2 trên DỮ LIỆU TỔNG HỢP (≈ 3–10 phút) để kiểm tra môi trường trước khi chạy thật.

Tạo KLTN_DATA_ROOT tạm (hoặc --root), sinh 3 bộ giai đoạn + bộ sanity theo đúng định dạng experiment-data-perm/,
rồi chạy lần lượt: run_survey_e1 → run_survey_e1_wrappers → run_survey_e2_e3_timing → run_survey_e4_e5_stability →
make_survey_tables → run_p0_perm_v3 → aggregate_p0_decision → run_sanity_auroc. In thời gian từng bước.
Kết quả chạy thử KHÔNG được dùng cho bài (dữ liệu giả); xóa thư mục tạm sau khi xem.

    python scripts/smoke_test_survey_v2.py                 # thư mục tạm, cấu hình nhỏ
    python scripts/smoke_test_survey_v2.py --root /tmp/kltn_smoke --n 150 --p 1500 --full-methods
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
    ap.add_argument("--full-methods", action="store_true", help="chạy đủ 8 phương pháp filter/embedded (chậm hơn)")
    ap.add_argument("--n-jobs", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="kltn_smoke_"))
    os.environ["KLTN_DATA_ROOT"] = str(root)
    os.environ.setdefault("KLTN_ALLOW_UNPUSHED", "1")      # chạy thử: không yêu cầu đã push
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import config as C
    from survey_v2 import synthetic

    print(f"KLTN_DATA_ROOT (tạm) = {root}")
    t0 = time.time()
    synthetic.make_all(root, n=args.n, p=args.p, effect_stage=0.7)
    timings = {"synthetic data": time.time() - t0}
    methods = list(C.FILTER_METHODS + C.EMBEDDED_METHODS) if args.full_methods else ["f_test", "mrmr", "l1_lr", "rf_importance"]
    pre = str(min(500, args.p))
    nj = str(args.n_jobs)

    def run(label, module, argv):
        t = time.time()
        mod = importlib.import_module(module)
        sys.argv = [f"{module}.py", *argv]
        rc = mod.main(argv) if module not in ("run_survey_e1_wrappers", "run_survey_e2_e3_timing") else mod.main()
        timings[label] = time.time() - t
        print(f"==> {label}: {timings[label]:.1f}s\n", flush=True)
        return rc

    run("E1 (filter+embedded, 3 bộ, 2 lần lặp)", "run_survey_e1",
        ["--methods", *methods, "--k", "10", "50", "100", "--repeats", "0", "1", "--prefilter", pre, "--random-draws", "5",
         "--secondary", "--n-jobs", nj])
    run("E1 wrapper (GA+BPSO, 1 lần lặp, quần thể 8 × 3 thế hệ)", "run_survey_e1_wrappers",
        ["--repeats", "0", "--k", "10", "50", "--prefilter", pre, "--pop", "8", "--generations", "3", "--n-jobs", nj])
    run("E2+E3 (thời gian)", "run_survey_e2_e3_timing",
        ["--methods", *methods, "--repeats", "1", "--n-grid", "40", "60", "all", "--p-grid", "200", "400", "all", "--k", "10"])
    run("E4+E5 (ổn định, M = 6)", "run_survey_e4_e5_stability",
        ["--methods", *methods, "--k", "10", "50", "--m", "6", "--prefilter", pre, "--wrapper-k", "10", "--pop", "8",
         "--generations", "3", "--n-jobs", nj])
    C.OVERLAP_K = 50
    run("Bảng + hình", "make_survey_tables", ["--e6-fraction", "0.7"])
    run("P0 perm v3 (20 hoán vị, 20 random-k)", "run_p0_perm_v3",
        ["--datasets", "gse68465_stage_early_late", "--n-perm", "20", "--n-random", "20", "--n-jobs", nj])
    run("P0 quyết định hợp nhất", "aggregate_p0_decision",
        ["--summaries", str(root / "results" / "survey_v2" / "P0" / "perm_test_v3" / "gse68465_stage_early_late" / "summary.csv")])
    rc = run("Sanity Y10", "run_sanity_auroc", ["--repeats", "0", "--prefilter", pre])
    print("\n=== Thời gian từng bước ===")
    for k, v in timings.items():
        print(f"{v:8.1f}s  {k}")
    print(f"\nTổng {sum(timings.values()) / 60:.1f} phút. Kết quả thử ở {root / 'results' / 'survey_v2'} (dữ liệu giả, không dùng cho bài).")
    print("Sanity:", "ĐẠT" if rc == 0 else "KHÔNG ĐẠT (bình thường với dữ liệu giả nhỏ nếu AUROC < 0,99)")


if __name__ == "__main__":
    main()
