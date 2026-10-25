"""Precomputed, patient-grouped 5x5 outer splits for the separated E1 workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


SPLIT_COLUMNS = ("repeat", "seed", "fold", "sample_id", "patient_id", "label", "role")


def build_repeated_split_manifest(sample_ids, patient_ids, labels, *, seeds, n_folds: int = 5) -> pd.DataFrame:
    """Build deterministic train/test assignments for every repeat and outer fold.

    The manifest deliberately contains only outer train/test roles.  A classifier
    may tune its hyperparameters only inside its assigned train rows; it must
    never use the outer test rows before final evaluation.
    """
    sample_ids = np.asarray(sample_ids, dtype=str)
    patient_ids = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(labels, dtype=int)
    if not (len(sample_ids) == len(patient_ids) == len(labels)):
        raise ValueError("sample_ids, patient_ids and labels must have the same length")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id must be unique")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must be binary {0, 1}")

    one_sample_per_patient = len(np.unique(patient_ids)) == len(patient_ids)
    rows = []
    for repeat, seed in enumerate(seeds):
        if one_sample_per_patient:
            split_iter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed)).split(sample_ids, labels)
        else:
            split_iter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=int(seed)).split(
                sample_ids, labels, patient_ids)
        for fold, (train_idx, test_idx) in enumerate(split_iter):
            roles = np.full(len(labels), "train", dtype=object)
            roles[test_idx] = "test"
            rows.append(pd.DataFrame({
                "repeat": repeat, "seed": int(seed), "fold": fold,
                "sample_id": sample_ids, "patient_id": patient_ids, "label": labels, "role": roles,
            }))
    manifest = pd.concat(rows, ignore_index=True)
    errors = validate_repeated_split_manifest(manifest, n_folds=n_folds)
    if errors:
        raise ValueError("invalid generated split manifest: " + "; ".join(errors))
    return manifest.loc[:, SPLIT_COLUMNS]


def validate_repeated_split_manifest(manifest: pd.DataFrame, *, n_folds: int = 5) -> list[str]:
    """Return human-readable validation errors; an empty list means safe splits."""
    errors: list[str] = []
    missing = set(SPLIT_COLUMNS) - set(manifest.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    frame = manifest.loc[:, SPLIT_COLUMNS].copy()
    if frame[list(SPLIT_COLUMNS[:-1])].isna().any().any() or frame["role"].isna().any():
        errors.append("missing split values")
    if not set(frame["role"].unique()) <= {"train", "test"}:
        errors.append("roles must be train/test")
    for (repeat, fold), group in frame.groupby(["repeat", "fold"], sort=False):
        if set(group["role"]) != {"train", "test"}:
            errors.append(f"repeat={repeat}, fold={fold}: missing train or test role")
            continue
        if group["sample_id"].duplicated().any():
            errors.append(f"repeat={repeat}, fold={fold}: duplicate sample_id")
        train_patients = set(group.loc[group.role == "train", "patient_id"])
        test_patients = set(group.loc[group.role == "test", "patient_id"])
        if train_patients & test_patients:
            errors.append(f"repeat={repeat}, fold={fold}: patient overlap")
        for role in ("train", "test"):
            if set(group.loc[group.role == role, "label"]) != {0, 1}:
                errors.append(f"repeat={repeat}, fold={fold}: {role} lacks a class")
    for repeat, group in frame.groupby("repeat", sort=False):
        folds = sorted(group["fold"].unique())
        if len(folds) != n_folds:
            errors.append(f"repeat={repeat}: expected {n_folds} folds, got {len(folds)}")
            continue
        counts = group[group.role == "test"].groupby("sample_id").size()
        if counts.empty or not counts.eq(1).all():
            errors.append(f"repeat={repeat}: each sample must be test exactly once")
    return errors
