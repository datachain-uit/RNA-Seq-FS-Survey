"""[ĐÃ THAY THẾ — survey v3] Bộ chạy E1 gộp của bản v2 (bài toán giai đoạn) không còn dùng.

E1 của v3 chạy theo hai pha trên bản chia tính trước:
    python scripts/build_e1_repeated_splits.py                      # bản chia 5×5 cho 5 bộ
    python scripts/run_e1_feature_selection.py --n-jobs ...         # pha A: chọn gen (11 phương pháp + panel IHC)
    python scripts/run_e1_classification.py --secondary             # pha B: LR-L2 (+ RF, SVM ở k ∈ {5, 25}), B0, random-k
Wrapper: scripts/run_e1_wrappers_all_cohorts.py (3 cohort GPL570, 3×5, k ∈ WRAPPER_K).
"""

from __future__ import annotations

import sys


def main(argv=None, script=__file__) -> None:
    print(__doc__, file=sys.stderr)
    raise SystemExit("run_survey_e1.py đã được thay bằng run_e1_feature_selection.py + run_e1_classification.py (survey v3).")


if __name__ == "__main__":
    main()
