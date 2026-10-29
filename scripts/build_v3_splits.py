"""Build and validate the fixed v3 split files (5x5 repeated CV + few-shot) in one data-only pass.

v3.1 (Review 29/10/2026, QĐ19): the few-shot part now writes the KBTN v2.1 Section 6 protocol
(``survey_v2.fewshot``): 5 test_T/pool_T splits (1/3 : 2/3, stratified, seeds 42-46) x FEWSHOT_SUBSETS = 4 nested
subsets per n_T level per class (n_T in {5, 10, 20, 40, pool}; GSE10245 only {5, 10}; levels above the smaller
class of pool_T are dropped automatically, e.g. GSE42127 has no 40). The single-subset format of the first version
(commit ad75caa, column ``n_target``) is kept below as *legacy* helpers for the unit tests only; it is NOT written
any more because one subset per n_T cannot estimate the between-subset variance required by E1b/E7 (Review 29/10, O23).

Outputs (next to each cohort, below $KLTN_DATA_ROOT or --data-root):
  <cohort>/repeated_splits.csv            5 repeats x 5 patient-grouped outer folds (survey_v2.e1_splits)
  <cohort>/fewshot_splits.csv             GEO cohorts only (survey_v2.fewshot, columns incl. ``subset``)
  split_manifests/e1_repeated_splits_manifest.json, split_manifests/fewshot_splits_manifest.json (SHA-256, commit)

Existing files are validated and kept unless --force. Equivalent single-purpose scripts: build_e1_repeated_splits.py,
build_fewshot_splits.py (same generators, same seeds, same output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from survey_v2 import config as C  # noqa: E402
from survey_v2.data import load_dataset, resolve_dataset_dir  # noqa: E402
from survey_v2.e1_splits import build_repeated_split_manifest, validate_repeated_split_manifest  # noqa: E402
from survey_v2.fewshot import build_fewshot_manifest, validate_fewshot_manifest  # noqa: E402


REPEAT_SEEDS = C.REPEAT_SEEDS
PRIMARY_DATASETS = tuple(C.DATASETS)
GEO_DATASETS = tuple(C.FEWSHOT_COHORTS)
# n_T levels per cohort (documentation + legacy helpers). survey_v2.fewshot derives the same levels from the data:
# a level is kept only when the smaller class of pool_T has at least n_T samples; GSE10245 is capped by FEWSHOT_N_T_LIMIT.
FEWSHOT_N = {"gse10245": (5, 10), "gse30219": (5, 10, 20, 40), "gse37745": (5, 10, 20, 40), "gse42127": (5, 10, 20)}
FEWSHOT_COLUMNS = ("split", "seed", "n_target", "sample_id", "patient_id", "label", "role")   # legacy format only


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- legacy single-subset few-shot format (commit ad75caa) — kept for tests, not used by build_all ---------------------
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
    """LEGACY (one subset per n_T). Use survey_v2.fewshot.build_fewshot_manifest for experiments.

    Each split has one fixed test set. For a numeric ``n_target``, train rows contain exactly that many samples of
    each class, nested by increasing n. The additional ``pool`` level trains on all remaining non-test samples.
    """
    warnings.warn("build_fewshot_split_manifest is the legacy single-subset format; experiments use "
                  "survey_v2.fewshot.build_fewshot_manifest (4 nested subsets per n_T)", DeprecationWarning, stacklevel=2)
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
    """Return validation errors for a LEGACY single-subset few-shot manifest."""
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


# --- v3.1 builder ----------------------------------------------------------------------------------------------------
def _dataset_dir(data_root: Path, name: str) -> Path:
    return resolve_dataset_dir(data_root, name)


def _load_ids(data_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """sample_id, patient_id, label via the shared loader (handles both perm/ and geo-target/ metadata formats)."""
    ds = load_dataset(data_root, name, gene_space=None)
    return ds.sample_ids.astype(str), ds.groups.astype(str), ds.y.astype(int), ds.notes


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build_all(data_root: Path, *, force: bool = False) -> dict:
    data_root = Path(data_root)
    repo = Path(__file__).resolve().parents[1]
    repeated_files = []
    fewshot_files = []
    for name in PRIMARY_DATASETS:
        data_dir = _dataset_dir(data_root, name)
        sample_ids, patient_ids, labels, notes = _load_ids(data_root, name)
        path = data_dir / "repeated_splits.csv"
        if path.exists() and not force:
            manifest = pd.read_csv(path)
        else:
            manifest = build_repeated_split_manifest(sample_ids, patient_ids, labels, seeds=REPEAT_SEEDS, n_folds=C.N_FOLDS)
            manifest.to_csv(path, index=False)
        errors = validate_repeated_split_manifest(manifest, n_folds=C.N_FOLDS)
        if errors:
            raise ValueError(f"{name}: invalid repeated split: {errors}")
        repeated_files.append({"dataset": name, "path": str(path.relative_to(data_root)), "rows": int(len(manifest)),
                               "sha256": sha256(path), "n_per_class": notes.get("n_per_class"),
                               "patient_id_assumed_from_sample_id": notes.get("patient_id_assumed_from_sample_id")})
    for name in GEO_DATASETS:
        data_dir = _dataset_dir(data_root, name)
        sample_ids, patient_ids, labels, notes = _load_ids(data_root, name)
        path = data_dir / "fewshot_splits.csv"
        if path.exists() and not force:
            manifest = pd.read_csv(path, dtype={"n_t": str})
            if "subset" not in manifest.columns:
                raise ValueError(f"{name}: {path} is the legacy single-subset format (no 'subset' column); "
                                 f"rebuild with --force (Review 29/10, O23)")
        else:
            manifest = build_fewshot_manifest(name, sample_ids, patient_ids, labels)
            manifest.to_csv(path, index=False, encoding="utf-8")
        errors = validate_fewshot_manifest(manifest)
        if errors:
            raise ValueError(f"{name}: invalid few-shot split: {errors}")
        levels = sorted({str(v) for v in manifest.n_t.unique()}, key=lambda t: 10**9 if t == "pool" else int(t))
        fewshot_files.append({"dataset": name, "path": str(path.relative_to(data_root)), "rows": int(len(manifest)),
                              "sha256": sha256(path), "n_t_levels": levels, "n_splits": int(manifest.split.nunique()),
                              "n_subsets": int(manifest.subset.nunique()), "n_per_class": notes.get("n_per_class")})
    out = data_root / "split_manifests"
    out.mkdir(exist_ok=True)
    common = {"generator": "scripts/build_v3_splits.py", "generator_commit": _git_commit(repo), "repeat_seeds": list(REPEAT_SEEDS)}
    (out / "e1_repeated_splits_manifest.json").write_text(json.dumps({
        **common, "protocol": f"{len(REPEAT_SEEDS)} repeats x {C.N_FOLDS} patient-grouped outer folds", "files": repeated_files},
        indent=2) + "\n", encoding="utf-8")
    (out / "fewshot_splits_manifest.json").write_text(json.dumps({
        **common, "protocol": f"{C.FEWSHOT_SPLITS} splits test_T/pool_T (1/3:2/3, stratified) x {C.FEWSHOT_SUBSETS} nested subsets per n_T per class",
        "test_fraction": C.FEWSHOT_TEST_FRACTION, "n_t_grid": [v if v is not None else "pool" for v in C.N_T_GRID],
        "n_t_limit": {k: list(v) for k, v in C.FEWSHOT_N_T_LIMIT.items()}, "files": fewshot_files}, indent=2) + "\n", encoding="utf-8")
    return {"repeated": repeated_files, "fewshot": fewshot_files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="rebuild existing split files (after validation of the new ones)")
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
