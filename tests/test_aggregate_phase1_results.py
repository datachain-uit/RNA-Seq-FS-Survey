"""Tests for the Phase 1 aggregation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_phase1_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_phase1_results", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mean_by_fold_averages_numeric_metrics_and_counts_all_folds() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["gse30219_luad_lusc", "gse30219_luad_lusc"],
            "n_features": [10, 10],
            "fs_group": ["filter", "filter"],
            "fs_method": ["f_classif", "f_classif"],
            "model": ["linear_svm", "linear_svm"],
            "fold": [1, 2],
            "test_balanced_accuracy": [0.70, 0.90],
            "n_train": [87, 88],
            "class_labels": ["[0, 1]", "[0, 1]"],
        }
    )

    result = MODULE.mean_by_fold(
        frame,
        ["dataset", "n_features", "fs_group", "fs_method", "model"],
        descriptive_columns=["class_labels"],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_folds_observed"] == 2
    assert row["test_balanced_accuracy"] == 0.8
    assert row["n_train"] == 87.5
    assert row["class_labels"] == "[0, 1]"
    assert "fold" not in result.columns


def test_mean_by_fold_requires_all_key_columns() -> None:
    frame = pd.DataFrame({"dataset": ["gse30219_luad_lusc"], "fold": [1]})

    try:
        MODULE.mean_by_fold(frame, ["dataset", "n_features"])
    except ValueError as error:
        assert "n_features" in str(error)
    else:
        raise AssertionError("Expected missing key columns to raise ValueError")


def test_records_for_json_converts_nan_to_null() -> None:
    frame = pd.DataFrame({"method": ["pca"], "explained_variance": [float("nan")]})

    records = MODULE.records_for_json(frame)

    assert records == [{"method": "pca", "explained_variance": None}]
    json.dumps(records, allow_nan=False)
