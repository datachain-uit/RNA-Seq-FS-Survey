"""Split saved test confusion matrices into Normal-vs-tumour and stage-only metrics.

Review 2026-10-04, R1 / Section 8.6.1: no model is retrained; every number is
recomputed from the ``confusion_matrix_test`` column of the per-fold
``results_*.csv`` files (rows = true label, columns = prediction, class 0 = Normal).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLF_ROOT = ROOT / "results" / "clf"
OUTPUT_DIR = ROOT / "results" / "stage_split"

ROW_KEYS = ["dataset", "n_features", "fs_group", "fs_method", "model", "fold"]
CONFIG_KEYS = ["dataset", "n_features", "fs_group", "fs_method", "model"]
CHANCE_MARGIN = 0.05


def read_all_results() -> pd.DataFrame:
    """Load every per-fold results_*.csv, both unpacked and inside zip archives."""
    frames: list[pd.DataFrame] = []
    for csv_path in CLF_ROOT.rglob("results_*.csv"):
        frames.append(pd.read_csv(csv_path).assign(source=str(csv_path.relative_to(ROOT))))
    for zip_path in CLF_ROOT.rglob("*.zip"):
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if Path(name).name.startswith("results_") and name.endswith(".csv"):
                    table = pd.read_csv(io.BytesIO(archive.read(name)))
                    frames.append(table.assign(source=f"{zip_path.relative_to(ROOT)}::{name}"))

    combined = pd.concat(frames, ignore_index=True)
    combined["fs_group"] = combined["fs_group"].fillna("extraction")
    # The same run can exist both unpacked and zipped; duplicates must be identical.
    conflicts = combined.groupby(ROW_KEYS)["confusion_matrix_test"].nunique()
    assert (conflicts == 1).all(), f"Conflicting duplicates:\n{conflicts[conflicts > 1]}"
    return combined.drop_duplicates(ROW_KEYS).reset_index(drop=True)


def mcc_from_cm(cm: np.ndarray) -> float:
    """Multiclass Matthews correlation (Gorodkin) from a confusion matrix."""
    t, p = cm.sum(1), cm.sum(0)
    c, s = np.trace(cm), cm.sum()
    denom = np.sqrt((s**2 - (p**2).sum()) * (s**2 - (t**2).sum()))
    return float((c * s - (t * p).sum()) / denom) if denom > 0 else 0.0


def qwk_from_cm(cm: np.ndarray) -> float:
    """Quadratic weighted kappa for ordinal labels from a confusion matrix."""
    n = cm.shape[0]
    i, j = np.indices((n, n))
    w = (i - j) ** 2 / (n - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom > 0 else 0.0


def split_metrics(cm_json: str) -> pd.Series:
    cm = np.array(json.loads(cm_json), dtype=float)
    rec = np.diag(cm) / cm.sum(1).clip(min=1)
    n_stage = len(rec) - 1

    # (a) Normal vs tumour, binary
    tn, fp = cm[0, 0], cm[0, 1:].sum()
    fn, tp = cm[1:, 0].sum(), cm[1:, 1:].sum()

    # (b) stage, tumour rows only; tumour predicted as Normal counts as an error
    tumour = cm[1:, :]
    early_n, late_n = tumour[0].sum(), tumour[1:].sum()
    early_recall = tumour[0, 1] / max(early_n, 1)
    late_recall = tumour[1:, 2:].sum() / max(late_n, 1)

    # Chance-corrected stage scores on the tumour x tumour block
    stage_block = cm[1:, 1:]
    return pd.Series({
        "normal_recall": rec[0],
        "tumour_recall": tp / max(tp + fn, 1),
        "bacc_normal_vs_tumour": 0.5 * (tn / max(tn + fp, 1) + tp / max(tp + fn, 1)),
        "stage_macro_recall": rec[1:].mean(),
        "stage_chance": 1 / n_stage,
        "stage_recall_minus_chance": rec[1:].mean() - 1 / n_stage,
        "stage_mcc": mcc_from_cm(stage_block),
        "stage_qwk": qwk_from_cm(stage_block),
        "early_recall": early_recall,
        "late_recall": late_recall,
        "bacc_early_vs_late": 0.5 * (early_recall + late_recall),
        "n_tumour_pred_normal": cm[1:, 0].sum(),
        "n_test_normal": cm[0].sum(),
        "normal_share_of_bacc": rec[0] / len(rec) / max(rec.mean(), 1e-12),
        **{f"recall_class_{k}": r for k, r in enumerate(rec)},
        "bacc_check": rec.mean(),
    })


def summarise(configs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    near = configs["stage_recall_minus_chance"] <= CHANCE_MARGIN
    g = configs.assign(stage_near_chance=near,
                       early_late_near_chance=configs["bacc_early_vs_late"] <= 0.5 + CHANCE_MARGIN
                       ).groupby(by)
    out = pd.DataFrame({
        "n_configs": g.size(),
        "test_bacc_mean": g["test_balanced_accuracy"].mean(),
        "test_bacc_min": g["test_balanced_accuracy"].min(),
        "test_bacc_max": g["test_balanced_accuracy"].max(),
        "test_mcc_mean": g["test_mcc"].mean(),
        "normal_recall_mean": g["normal_recall"].mean(),
        "bacc_normal_vs_tumour_mean": g["bacc_normal_vs_tumour"].mean(),
        "normal_share_of_bacc_mean": g["normal_share_of_bacc"].mean(),
        "stage_chance": g["stage_chance"].first(),
        "stage_recall_mean": g["stage_macro_recall"].mean(),
        "stage_recall_max": g["stage_macro_recall"].max(),
        "pct_stage_le_chance_plus_0.05": 100 * g["stage_near_chance"].mean(),
        "stage_mcc_mean": g["stage_mcc"].mean(),
        "stage_qwk_mean": g["stage_qwk"].mean(),
        "bacc_early_vs_late_mean": g["bacc_early_vs_late"].mean(),
        "bacc_early_vs_late_max": g["bacc_early_vs_late"].max(),
        "pct_early_late_le_0.55": 100 * g["early_late_near_chance"].mean(),
    })
    return out.reset_index()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = read_all_results()
    split = results["confusion_matrix_test"].apply(split_metrics)
    per_fold = pd.concat([results[ROW_KEYS + ["test_balanced_accuracy", "test_mcc", "source"]],
                          split], axis=1)

    # Sanity check from the review: recomputed BAcc must equal the saved column
    gap = (per_fold["bacc_check"] - per_fold["test_balanced_accuracy"]).abs().max()
    assert gap < 1e-9, f"BAcc mismatch {gap}"
    per_fold.to_csv(OUTPUT_DIR / "split_metrics_per_fold.csv", index=False)

    numeric = per_fold.drop(columns=["fold", "source"]).select_dtypes("number").columns
    configs = per_fold.groupby(CONFIG_KEYS)[list(numeric)].mean().reset_index()
    configs.insert(len(CONFIG_KEYS), "n_folds", per_fold.groupby(CONFIG_KEYS).size().values)
    configs.to_csv(OUTPUT_DIR / "split_metrics_5fold_means.csv", index=False)

    fs_only = configs[~configs["fs_group"].isin(["extraction", "no_fs"])]
    tables = {
        "by_dataset": summarise(fs_only, ["dataset"]),
        "by_dataset_k": summarise(fs_only, ["dataset", "n_features"]),
        "by_dataset_group": summarise(fs_only, ["dataset", "fs_group"]),
        "by_dataset_method": summarise(fs_only, ["dataset", "fs_method"]),
        "by_dataset_model": summarise(fs_only, ["dataset", "model"]),
        "extraction_by_dataset": summarise(configs[configs["fs_group"] == "extraction"],
                                           ["dataset", "fs_method"]),
        # Baseline B0: every gene, no selection (Review R2)
        "no_fs_by_dataset_model": summarise(configs[configs["fs_group"] == "no_fs"],
                                            ["dataset", "model"]),
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"summary_{name}.csv", index=False)

    coverage = per_fold.groupby(["n_features", "fs_group"]).agg(
        rows=("fold", "size"), datasets=("dataset", "nunique"),
        methods=("fs_method", "nunique"), models=("model", "nunique"))
    print(f"BAcc check max gap: {gap:.2e}; per-fold rows: {len(per_fold)}; configs: {len(configs)}")
    print(coverage.to_string())
    pd.set_option("display.width", 250)
    print(tables["by_dataset"].round(3).T.to_string())


if __name__ == "__main__":
    main()
