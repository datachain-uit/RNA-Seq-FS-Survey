"""Create consolidated five-fold mean tables for FS and classification results."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import zipfile

import pandas as pd


FS_ROOT = Path(__file__).resolve().parent
CLF_ROOT = Path(r"D:\KLTN\clf-experiment-result")
OUTPUT_DIR = FS_ROOT / "aggregated_results"


def numeric_mean_table(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Average every numeric measurement while retaining one descriptive value."""
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {*key_columns, "fold"}
    ]
    grouped = frame.groupby(key_columns, dropna=False, sort=True)
    means = grouped[numeric_columns].mean()
    means.insert(0, "n_folds_observed", grouped.size())
    return means.reset_index()


def aggregate_feature_selection() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for metrics_file in FS_ROOT.glob("*/*/*/metrics_per_fold.csv"):
        run_dir = metrics_file.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        parts = run_dir.relative_to(FS_ROOT).parts
        table = pd.read_csv(metrics_file)
        table.insert(0, "n_features", config["n_features"])
        table.insert(1, "fs_group", parts[1])
        table.insert(2, "fs_method", config["method"])
        records.append(table)

    combined = pd.concat(records, ignore_index=True)
    keys = ["dataset", "n_features", "fs_group", "fs_method"]
    result = numeric_mean_table(combined, keys)
    return result.sort_values(keys).reset_index(drop=True)


def read_classification_results() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for archive_path in sorted((CLF_ROOT / "_zips").glob("classification_*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            result_members = [
                name for name in archive.namelist() if Path(name).name.startswith("results_")
            ]
            if len(result_members) != 1:
                raise ValueError(f"Expected one results CSV in {archive_path}, found {result_members}")
            records.append(pd.read_csv(io.BytesIO(archive.read(result_members[0]))))
    return pd.concat(records, ignore_index=True)


def aggregate_classification() -> pd.DataFrame:
    combined = read_classification_results()
    keys = ["dataset", "n_features", "fs_group", "fs_method", "model"]
    result = numeric_mean_table(combined, keys)

    # These values are descriptive rather than quantities that should be averaged.
    grouped = combined.groupby(keys, dropna=False, sort=True)
    result["class_labels"] = grouped["class_labels"].first().to_numpy()
    return result.sort_values(keys).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fs_table = aggregate_feature_selection()
    clf_table = aggregate_classification()

    fs_path = OUTPUT_DIR / "feature_selection_5fold_means.csv"
    clf_path = OUTPUT_DIR / "classification_5fold_means.csv"
    fs_table.to_csv(fs_path, index=False, encoding="utf-8-sig")
    clf_table.to_csv(clf_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(fs_table)} feature-selection rows: {fs_path}")
    print(f"Wrote {len(clf_table)} classification rows: {clf_path}")
    incomplete = clf_table.loc[clf_table["n_folds_observed"] != 5]
    if not incomplete.empty:
        print("Classification rows with fewer than five observed folds:")
        print(incomplete[["n_features", "fs_group", "fs_method", "dataset", "model", "n_folds_observed"]].to_string(index=False))


if __name__ == "__main__":
    main()
