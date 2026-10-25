"""E1 (phần wrapper) — GA và BPSO với mã hóa đúng k trên GSE68465, CV 3×5 (3 lần lặp đầu; QĐ6, M10).

Fitness = AUROC CV 3 fold của LR-L2 (C = 1) bên trong phần train; quần thể 30, 50 thế hệ (duyệt có
chỉnh sửa, Review 24/10 Bảng 8 mục 4). Kết quả ghi chung vào results/survey_v2/E1/<bộ>/ để
make_survey_tables.py ghép cặp với các phương pháp khác trên cùng 3 lần lặp.

    python scripts/run_survey_e1_wrappers.py                       # GA + BPSO, repeats 0 1 2, GSE68465
    python scripts/run_survey_e1_wrappers.py --methods ga --repeats 0 --k 10 50 --n-jobs 5
    python scripts/run_survey_e1_wrappers.py --pop 10 --generations 20   # ngân sách nhỏ (E3 / chạy thử)

Ước lượng: mỗi (fold, k) ≈ 1.400–1.500 lần đánh giá fitness (30 + 50×28 với GA); vài phút đến vài chục
phút tùy k; toàn bộ 15 fold × 5 k × 2 wrapper ≈ 10–40 giờ CPU (chạy song song theo fold bằng --n-jobs).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from survey_v2 import config as C  # noqa: E402
import run_survey_e1  # noqa: E402  (cùng thư mục scripts/)


def main() -> None:
    argv = sys.argv[1:]
    defaults = []
    if "--datasets" not in argv:
        defaults += ["--datasets", C.RANKING_DATASET]
    if "--methods" not in argv:
        defaults += ["--methods", *C.WRAPPER_METHODS]
    if "--repeats" not in argv:
        defaults += ["--repeats", *map(str, range(C.WRAPPER_REPEATS))]
    for flag in ("--no-b0", "--no-random"):
        if flag not in argv:
            defaults.append(flag)
    run_survey_e1.main(defaults + argv, script=__file__)


if __name__ == "__main__":
    main()
