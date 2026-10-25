"""Run E1 wrapper selection (GA and BPSO) separately, on all three cohorts.

This is intentionally a Phase-A-only entry point.  Run Phase B afterwards with
``run_e1_classification.py --methods ga bpso`` so classifiers reuse these exact
saved panels rather than re-running a stochastic selector.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_e1_feature_selection import main


if __name__ == "__main__":
    main(["--methods", "ga", "bpso", "--out", "E1_wrapper_feature_selection"])
