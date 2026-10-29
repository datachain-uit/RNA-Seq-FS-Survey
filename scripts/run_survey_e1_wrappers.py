"""[ĐÃ THAY THẾ — survey v3] Dùng scripts/run_e1_wrappers_all_cohorts.py (pha A, GA + BPSO, 3 cohort GPL570, 3×5, k ∈ WRAPPER_K)
rồi scripts/run_e1_classification.py (pha B tự gộp panel từ E1_wrapper_feature_selection)."""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__, file=sys.stderr)
    raise SystemExit("run_survey_e1_wrappers.py đã được thay bằng run_e1_wrappers_all_cohorts.py (survey v3).")


if __name__ == "__main__":
    main()
