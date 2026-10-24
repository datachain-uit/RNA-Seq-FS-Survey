"""Tests for the validated Phase 0 permutation-v3 aggregator."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_phase0_v3.py"
SPEC = importlib.util.spec_from_file_location("aggregate_phase0_v3", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_payload_contains_all_v3_branches_and_sanity_rows() -> None:
    data_root = Path(os.environ.get("KLTN_DATA_ROOT", r"D:\KLTN"))
    root = data_root / "results" / "phase 1" / "perm" / "perm_test_v3_permutation"
    if not root.exists():
        pytest.skip(f"Không có kết quả perm v3 tại {root} (đặt KLTN_DATA_ROOT)")

    payload = MODULE.build_payload(root)

    assert len(payload["decision_rows"]) == 10
    assert len(payload["main_permutation_rows"]) == 66
    assert len(payload["sanity_rows"]) == 3
    assert {row["analysis"] for row in payload["main_permutation_rows"]} == {"main"}
    assert {row["dataset"] for row in payload["sanity_rows"]} == {"tcga_normal_vs_tumor_sanity"}
    assert all(row["complete"] is True for row in payload["decision_rows"])
