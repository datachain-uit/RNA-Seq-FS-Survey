"""Aggregate validated Phase 1 GSE30219 results without reading archived runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import require
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import require


DATASET = "gse30219_luad_lusc"
FEATURE_GROUPS = ("embedded", "filter", "wrapper")
CLASSIFICATION_GROUPS = ("embedded", "extraction", "filter", "wrapper")


def mean_by_fold(
    frame: pd.DataFrame,
    key_columns: list[str],
    *,
    descriptive_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Average numeric fold-level values and retain stable descriptive values."""
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {', '.join(missing)}")

    descriptive_columns = descriptive_columns or []
    grouped = frame.groupby(key_columns, dropna=False, sort=True)
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {*key_columns, "fold"}
    ]
    result = grouped[numeric_columns].mean()
    result.insert(0, "n_folds_observed", grouped.size())
    result = result.reset_index()

    for column in descriptive_columns:
        if column in frame.columns:
            result[column] = grouped[column].first().to_numpy()
    return result


def _read_feature_metrics(
    metrics_path: Path,
    group: str,
) -> pd.DataFrame:
    config_path = metrics_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for {metrics_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    table = pd.read_csv(metrics_path)
    if set(table["dataset"]) != {DATASET}:
        raise ValueError(f"Unexpected dataset in {metrics_path}")
    table.insert(0, "n_features", int(config["n_features"]))
    table.insert(1, "fs_group", group)
    table.insert(2, "fs_method", str(config["method"]))
    return table


def read_feature_selection(results_root: Path) -> pd.DataFrame:
    """Read only live FS/FE metrics belonging to the corrected GSE30219 cohort."""
    records: list[pd.DataFrame] = []
    fs_root = results_root / "fs" / DATASET
    for group in FEATURE_GROUPS:
        for metrics_path in sorted(fs_root.glob(f"{group}/*/k_*/metrics_per_fold.csv")):
            records.append(_read_feature_metrics(metrics_path, group))

    fe_root = results_root / "fe"
    for metrics_path in sorted(fe_root.glob(f"fe_*_{DATASET}/k_*/metrics_per_fold.csv")):
        records.append(_read_feature_metrics(metrics_path, "extraction"))

    if not records:
        raise FileNotFoundError("No live Phase 1 feature-selection metrics found")
    return pd.concat(records, ignore_index=True)


def aggregate_feature_selection(results_root: Path) -> pd.DataFrame:
    combined = read_feature_selection(results_root)
    keys = ["dataset", "n_features", "fs_group", "fs_method"]
    return mean_by_fold(combined, keys).sort_values(keys).reset_index(drop=True)


def read_classification(results_root: Path) -> pd.DataFrame:
    """Read live classification results, explicitly excluding all archived paths."""
    root = results_root / "clf" / DATASET
    records: list[pd.DataFrame] = []
    for group in CLASSIFICATION_GROUPS:
        path = root / group / f"results_{group}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing live classification result: {path}")
        table = pd.read_csv(path)
        if set(table["dataset"]) != {DATASET}:
            raise ValueError(f"Unexpected dataset in {path}")
        if not (table["n_train"] + table["n_val"] + table["n_test"] == 146).all():
            raise ValueError(f"Non-146 split found in {path}")
        table = table.rename(columns={"k": "n_features"})
        records.append(table)
    return pd.concat(records, ignore_index=True)


def aggregate_classification(results_root: Path) -> pd.DataFrame:
    combined = read_classification(results_root)
    keys = ["dataset", "n_features", "fs_group", "fs_method", "model"]
    return (
        mean_by_fold(combined, keys, descriptive_columns=["class_labels"])
        .sort_values(keys)
        .reset_index(drop=True)
    )


def records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a dataframe to JSON records while representing missing values as null."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def build_payload(results_root: Path) -> dict[str, object]:
    """Build a JSON-safe payload for the aggregate workbook builder."""
    feature_selection = aggregate_feature_selection(results_root)
    classification = aggregate_classification(results_root)
    return {
        "dataset": DATASET,
        "cohort_samples": 146,
        "class_counts": {"LUAD": 85, "LUSC": 61},
        "feature_selection_columns": feature_selection.columns.tolist(),
        "feature_selection_rows": records_for_json(feature_selection),
        "classification_columns": classification.columns.tolist(),
        "classification_rows": records_for_json(classification),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    prov = require(__file__)

    payload = build_payload(args.results_root)
    payload["provenance"] = prov
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Wrote "
        f"{len(payload['feature_selection_rows'])} feature-selection aggregates and "
        f"{len(payload['classification_rows'])} classification aggregates"
    )


if __name__ == "__main__":
    main()
