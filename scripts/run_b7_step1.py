"""B7 bước 1 (QĐ188, Review 13/11 Phụ lục A): chạy đủ chuỗi theo đúng thứ tự, dừng ở bước lỗi đầu tiên.

  1. build_b7_egfr_datasets.py tcga          nhãn EGFR/TP53 từ MAF mở của GDC (kiểm 58/435/12/10 và 243/262/10), biểu hiện TCGA-LUAD
  2. build_b7_egfr_datasets.py gse31210      series matrix GSE31210, nhãn QĐ185 (kiểm 127/99/20), thang A.2, gom probe
  3. check_sex_consistency.py                QC giới tính GSE31210 (như bốn cohort GEO)
  4. geo_sample_qc.py apply-sex-qc           ghi mẫu cách ly vào excluded_samples_gse31210_egfr.csv
  5. geo_sample_qc.py rz-screen              sàng trùng trong GSE31210 và với ba cohort GPL570 (chỉ báo; không tự bỏ mẫu)
  6. geo_sample_qc.py rebuild                bỏ mẫu cách ly khỏi gse31210_egfr
  7. build_e1_repeated_splits.py             bản chia 5 × 5 theo bệnh nhân, phân tầng theo nhãn, seed 42–46
  8. build_b7_egfr_datasets.py card          data card hai bộ
  9. run_survey_e0_signal_check.py           E0 (F-test, k = 10, 200 hoán vị, 100 random-k) → E0_b7/
 10. run_e1_classification.py --k 10         B0 và random-k (100 lần rút mỗi fold), không panel, có điểm từng mẫu → E1_classification_b7/
 11. b7_gate.py                              bảng cổng G1, G2 → B7_gate/
 12. build_b7_egfr_datasets.py pack          chép hai bộ vào results/survey_v2/B7_data/ để bàn giao

Script cổng (b7_gate.py) đã commit và push trước khi chạy (A.4). Cần Internet (GDC, GEO) và Kaggle Input experiment-data-v3.

    python scripts/run_b7_step1.py               # chạy đủ
    python scripts/run_b7_step1.py --list        # chỉ in các lệnh
    python scripts/run_b7_step1.py --from 9      # tiếp tục từ bước 9 (dữ liệu đã dựng)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402

TCGA_B7, GEO_B7 = C.B7_DATASETS


def steps(qc_dir: str, n_jobs: int) -> list[list[str]]:
    b7 = list(C.B7_DATASETS)
    return [
        ["scripts/build_b7_egfr_datasets.py", "tcga"],
        ["scripts/build_b7_egfr_datasets.py", "gse31210"],
        ["scripts/check_sex_consistency.py", "--datasets", GEO_B7, "--out", qc_dir],
        ["scripts/geo_sample_qc.py", "apply-sex-qc", "--cohorts", GEO_B7, "--qc-dir", qc_dir],
        ["scripts/geo_sample_qc.py", "rz-screen", "--cohorts", GEO_B7, *C.GPL570_COHORTS, "--out", qc_dir],
        ["scripts/geo_sample_qc.py", "rebuild", "--cohorts", GEO_B7],
        ["scripts/build_e1_repeated_splits.py", "--datasets", *b7],
        ["scripts/build_b7_egfr_datasets.py", "card"],
        ["scripts/run_survey_e0_signal_check.py", "--datasets", *b7, "--n-jobs", str(n_jobs), "--out", "E0_b7"],
        ["scripts/run_e1_classification.py", "--datasets", *b7, "--k", str(C.B7_GATE_K), "--save-scores", "--out", "E1_classification_b7"],
        ["scripts/b7_gate.py", "--e0-dir", "E0_b7", "--e1-dir", "E1_classification_b7"],
        ["scripts/build_b7_egfr_datasets.py", "pack"],
    ]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=1, help="bắt đầu từ bước này (1-based)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    from survey_v2.runlog import DATA_ROOT
    args = parse_args(argv)
    qc_dir = str(DATA_ROOT.joinpath(*C.RESULT_DIR, "B7_data", "qc"))
    plan = steps(qc_dir, args.n_jobs)
    for i, s in enumerate(plan, 1):
        if args.list or i < args.start:
            print(f"{'(bỏ qua) ' if i < args.start and not args.list else ''}{i:2d}. python {' '.join(s)}", flush=True)
            continue
        t0 = time.time()
        print(f"=== B7 bước 1 · {i}/{len(plan)}: python {' '.join(s)}", flush=True)
        code = subprocess.run([sys.executable, *s], cwd=REPO).returncode
        print(f"=== {i}/{len(plan)} xong sau {time.time() - t0:.0f} s, mã {code}", flush=True)
        if code:
            print(f"DỪNG ở bước {i}: sửa nguyên nhân rồi chạy lại với --from {i}", flush=True)
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
