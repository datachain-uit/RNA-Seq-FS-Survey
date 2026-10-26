"""Build and validate fixed v3 repeated-CV and few-shot split manifests.

This utility is intentionally data-only. It creates the manifests documented
for the LUAD/LUSC v3 protocol without running feature selection or classifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survey_v2.e1_splits import build_repeated_split_manifest, validate_repeated_split_manifest


REPEAT_SEEDS = (42, 43, 44, 45, 46)
PRIMARY_DATASETS = ("tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127")
GEO_DATASETS = ("gse10245", "gse30219", "gse37745", "gse42127")
FEWSHOT_N = {"gse10245": (5, 10), "gse30219": (5, 10, 20, 40), "gse37745": (5, 10, 20, 40), "gse42127": (5, 10, 20)}
FEWSHOT_COLUMNS = ("split", "seed", "n_target", "sample_id", "patient_id", "label", "role")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_input(sample_ids, patient_ids, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_ids = np.asarray(sample_ids, dtype=str)
    patient_ids = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(labels, dtype=int)
    if not (len(sample_ids) == len(patient_ids) == len(labels)):
        raise ValueError("sample_ids, patient_ids and labels must have the same length")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("sample_id must be unique")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must be binary {0, 1}")
    return sample_ids, patient_ids, labels


def _fewshot_test_pool_indices(sample_ids, patient_ids, labels, *, seed: int, test_fraction: float):
    if len(np.unique(patient_ids)) == len(patient_ids):
        train, test = next(StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(sample_ids, labels))
        return train, test
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(splitter.split(sample_ids, labels, patient_ids))
    train, test = folds[seed % len(folds)]
    return train, test


def build_fewshot_split_manifest(sample_ids, patient_ids, labels, *, seeds=REPEAT_SEEDS, n_per_class=(5, 10, 20, 40), test_fraction=1 / 3) -> pd.DataFrame:
    """Build deterministic fixed-test, nested target-training subsets.

    Each split has one fixed test set. For a numeric ``n_target``, train rows
    contain exactly that many samples of each class, nested by increasing n.
    The additional ``pool`` level trains on all remaining non-test samples.
    """
    sample_ids, patient_ids, labels = _validate_input(sample_ids, patient_ids, labels)
    n_per_class = tuple(sorted(set(int(value) for value in n_per_class)))
    rows = []
    for split, seed in enumerate(seeds):
        pool_idx, test_idx = _fewshot_test_pool_indices(sample_ids, patient_ids, labels, seed=int(seed), test_fraction=test_fraction)
        pool_by_label = {label: pool_idx[labels[pool_idx] == label] for label in (0, 1)}
        if any(len(indices) < max(n_per_class) for indices in pool_by_label.values()):
            counts = {label: len(indices) for label, indices in pool_by_label.items()}
            raise ValueError(f"seed={seed}: target pool too small for {max(n_per_class)} per class: {counts}")
        ordered = {
            label: indices[np.random.default_rng(np.random.SeedSequence([int(seed), label])).permutation(len(indices))]
            for label, indices in pool_by_label.items()
        }
        for n_target in n_per_class:
            train_idx = np.concatenate([ordered[0][:n_target], ordered[1][:n_target]])
            train_set = set(train_idx.tolist())
            roles = np.full(len(sample_ids), "pool", dtype=object)
            roles[test_idx] = "test"
            roles[list(train_set)] = "train"
            rows.append(pd.DataFrame({
                "split": split, "seed": int(seed), "n_target": str(n_target),
                "sample_id": sample_ids, "patient_id": patient_ids, "label": labels, "role": roles,
            }))
        roles = np.full(len(sample_ids), "train", dtype=object)
        roles[test_idx] = "test"
        rows.append(pd.DataFrame({
            "split": split, "seed": int(seed), "n_target": "pool",
            "sample_id": sample_ids, "patient_id": patient_ids, "label": labels, "role": roles,
        }))
    manifest = pd.concat(rows, ignore_index=True)
    errors = validate_fewshot_split_manifest(manifest, n_per_class=n_per_class)
    if errors:
        raise ValueError("invalid generated few-shot manifest: " + "; ".join(errors))
    return manifest.loc[:, FEWSHOT_COLUMNS]


def validate_fewshot_split_manifest(manifest: pd.DataFrame, *, n_per_class) -> list[str]:
    """Return validation errors for a v3 few-shot split manifest."""
    errors: list[str] = []
    missing = set(FEWSHOT_COLUMNS) - set(manifest.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    frame = manifest.loc[:, FEWSHOT_COLUMNS].copy()
    numeric_levels = {str(int(value)) for value in n_per_class}
    for (split, level), group in frame.groupby(["split", "n_target"], sort=False):
        if group.sample_id.duplicated().any():
            errors.append(f"split={split}, n_target={level}: duplicate sample_id")
        if not set(group.role).issubset({"test", "train", "pool"}) or "test" not in set(group.role):
            errors.append(f"split={split}, n_target={level}: invalid roles")
            continue
        test_patients = set(group.loc[group.role == "test", "patient_id"])
        train_patients = set(group.loc[group.role == "train", "patient_id"])
        if test_patients & train_patients:
            errors.append(f"split={split}, n_target={level}: patient overlap")
        if set(group.loc[group.role == "test", "label"]) != {0, 1}:
            errors.append(f"split={split}, n_target={level}: test lacks a class")
        if level in numeric_levels:
            expected = int(level)
            counts = group.loc[group.role == "train", "label"].value_counts().to_dict()
            if counts != {0: expected, 1: expected}:
                errors.append(f"split={split}, n_target={level}: train counts {counts}, expected {expected} each")
    for split, group in frame.groupby("split", sort=False):
        test_sets = [tuple(sorted(level_group.loc[level_group.role == "test", "sample_id"])) for _, level_group in group.groupby("n_target")]
        if len(set(test_sets)) != 1:
            errors.append(f"split={split}: test set changes across n_target")
        previous: set[str] = set()
        for level in sorted(numeric_levels, key=int):
            current = set(group.loc[(group.n_target == level) & (group.role == "train"), "sample_id"])
            if previous and not previous <= current:
                errors.append(f"split={split}: training subsets are not nested")
            previous = current
    return errors


def _dataset_dir(data_root: Path, name: str) -> Path:
    parent = "experiment-data-perm" if name == "tcga_luad_lusc_primary_tumor" else "experiment-data-geo-target"
    return data_root / parent / name


def _load_metadata(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = pd.read_csv(data_dir / "metadata.csv", dtype={"sample_id": str, "patient_id": str, "label": int})
    return metadata.sample_id.to_numpy(), metadata.patient_id.to_numpy(), metadata.label.to_numpy(dtype=int)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def build_all(data_root: Path, *, force: bool = False) -> dict:
    repo = Path(__file__).resolve().parents[1]
    repeated_files = []
    fewshot_files = []
    for name in PRIMARY_DATASETS:
        data_dir = _dataset_dir(data_root, name)
        sample_ids, patient_ids, labels = _load_metadata(data_dir)
        path = data_dir / "repeated_splits.csv"
        if path.exists() and not force:
            manifest = pd.read_csv(path)
        else:
            manifest = build_repeated_split_manifest(sample_ids, patient_ids, labels, seeds=REPEAT_SEEDS)
            manifest.to_csv(path, index=False)
        errors = validate_repeated_split_manifest(manifest)
        if errors:
            raise ValueError(f"{name}: invalid repeated split: {errors}")
        repeated_files.append({"dataset": name, "path": str(path.relative_to(data_root)), "rows": len(manifest), "sha256": sha256(path)})
    for name in GEO_DATASETS:
        data_dir = _dataset_dir(data_root, name)
        sample_ids, patient_ids, labels = _load_metadata(data_dir)
        path = data_dir / "fewshot_splits.csv"
        if path.exists() and not force:
            manifest = pd.read_csv(path, dtype={"n_target": str})
        else:
            manifest = build_fewshot_split_manifest(sample_ids, patient_ids, labels, n_per_class=FEWSHOT_N[name])
            manifest.to_csv(path, index=False)
        errors = validate_fewshot_split_manifest(manifest, n_per_class=FEWSHOT_N[name])
        if errors:
            raise ValueError(f"{name}: invalid few-shot split: {errors}")
        fewshot_files.append({"dataset": name, "path": str(path.relative_to(data_root)), "rows": len(manifest), "sha256": sha256(path), "n_per_class": list(FEWSHOT_N[name]) + ["pool"]})
    out = data_root / "split_manifests"
    out.mkdir(exist_ok=True)
    common = {"generator": "scripts/build_v3_splits.py", "generator_commit": _git_commit(repo), "repeat_seeds": list(REPEAT_SEEDS)}
    (out / "e1_repeated_splits_manifest.json").write_text(json.dumps({**common, "protocol": "5 repeats x 5 patient-grouped outer folds", "files": repeated_files}, indent=2) + "\n", encoding="utf-8")
    (out / "fewshot_splits_manifest.json").write_text(json.dumps({**common, "protocol": "fixed test split plus nested balanced target subsets", "test_fraction": 1 / 3, "files": fewshot_files}, indent=2) + "\n", encoding="utf-8")
    return {"repeated": repeated_files, "fewshot": fewshot_files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.data_root is None:
        root = os.environ.get("KLTN_DATA_ROOT")
        if not root:
            parser.error("set KLTN_DATA_ROOT or pass --data-root")
        args.data_root = Path(root)
    report = build_all(args.data_root, force=args.force)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
