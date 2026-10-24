"""Rebuild the GSE30219 LUAD-vs-LUSC cohort after correcting histology labels.

Only exact GEO histology values ``ADC`` and ``SQC`` are retained.  In
particular, ``SCC`` means small-cell carcinoma in this series and is excluded.
The output uses five patient-disjoint partitions: fold r uses partition r for
test, partition r+1 for validation, and the other three for training.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:  # import khi chạy test (scripts.<x>) hoặc chạy trực tiếp (python scripts/<x>.py)
    from scripts.provenance import DATA_ROOT, require, stamp
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from provenance import DATA_ROOT, require, stamp


REPO = DATA_ROOT  # thư mục dữ liệu (KLTN_DATA_ROOT); code nằm trong kho Git
DEFAULT_SOURCE = REPO / "experiment-data-geo-target" / "gse30219"
DEFAULT_OUTPUT = REPO / "experiment-data-geo-target" / "gse30219_r5_fixed"
LABEL_MAP = {"adc": ("LUAD", 0), "sqc": ("LUSC", 1)}
N_SPLITS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def histology_column(metadata: pd.DataFrame) -> pd.Series:
    if "histology" in metadata:
        return metadata["histology"].astype("string")
    characteristic_columns = [c for c in metadata if c.startswith("Sample_characteristics_ch1")]
    for column in characteristic_columns:
        values = metadata[column].astype("string")
        if values.str.match(r"^histology:\s*", case=False, na=False).any():
            return values.str.replace(r"^histology:\s*", "", regex=True, case=False)
    raise ValueError("GEO metadata has no histology characteristic")


def assign_labels(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = metadata.copy()
    result["histology_raw"] = histology_column(result).str.strip()
    histology_key = result["histology_raw"].str.lower()
    result["class"] = histology_key.map({key: value[0] for key, value in LABEL_MAP.items()})
    result["label"] = histology_key.map({key: value[1] for key, value in LABEL_MAP.items()})
    excluded = result[result["class"].isna()].copy()
    kept = result[result["class"].notna()].copy()
    kept["label"] = kept["label"].astype(int)
    return kept, excluded


def build_fold_assignments(
    metadata: pd.DataFrame, n_splits: int = N_SPLITS, seed_range=range(42, 142)
) -> tuple[pd.DataFrame, dict]:
    labels = metadata["label"].to_numpy()
    groups = metadata["patient_id"].astype(str).to_numpy()
    ideal = pd.Series(labels).value_counts().sort_index() / n_splits
    best = None
    for seed in seed_range:
        parts = np.empty(len(metadata), dtype=int)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for part, (_, test_index) in enumerate(splitter.split(np.zeros(len(metadata)), labels, groups), start=1):
            parts[test_index] = part
        counts = pd.crosstab(parts, labels).reindex(columns=ideal.index, fill_value=0)
        score = float(np.abs(counts - ideal.to_numpy()).to_numpy().sum())
        if best is None or score < best[0]:
            best = (score, seed, parts, counts)

    _, seed, parts, counts = best
    assignment = metadata[["patient_id", "class", "label"]].copy()
    assignment["part"] = parts
    for fold in range(1, n_splits + 1):
        validation_part = fold % n_splits + 1
        assignment[f"fold_{fold}"] = np.select(
            [parts == fold, parts == validation_part], ["test", "val"], default="train"
        )
    return assignment, {"selected_seed": seed, "split_imbalance_score": best[0], "part_class_counts": counts}


def verify_assignments(assignments: pd.DataFrame) -> None:
    fold_columns = [c for c in assignments if c.startswith("fold_")]
    assert assignments["patient_id"].is_unique, "each GSE30219 sample must represent one patient"
    assert ((assignments[fold_columns] == "test").sum(axis=1) == 1).all()
    assert ((assignments[fold_columns] == "val").sum(axis=1) == 1).all()
    for column in fold_columns:
        for split in ("train", "val", "test"):
            subset = assignments.loc[assignments[column] == split]
            assert subset["label"].nunique() == 2, f"{column}/{split} lacks a class"


def write_dataset(source: Path, output: Path, prov: dict | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    expression = pd.read_hdf(source / "expression.h5", key="expression")
    metadata = pd.read_csv(source / "metadata.csv", index_col="sample_id").loc[expression.index]
    labelled, excluded = assign_labels(metadata)
    expected = {"LUAD": 85, "LUSC": 61}
    actual = labelled["class"].value_counts().to_dict()
    assert actual == expected, f"expected {expected}, got {actual}"
    assert not labelled["Sample_title"].str.contains("small cell", case=False, na=False).any()

    expression = expression.loc[labelled.index]
    assignments, split_info = build_fold_assignments(labelled)
    verify_assignments(assignments)
    old_split_columns = [c for c in labelled if c == "part" or c.startswith("fold_")]
    labelled = labelled.drop(columns=old_split_columns).join(
        assignments[["part", *[f"fold_{fold}" for fold in range(1, N_SPLITS + 1)]]]
    )
    output.mkdir(parents=True)
    expression.astype(np.float32).to_hdf(output / "expression.h5", key="expression", mode="w", complevel=9, complib="blosc")
    labelled.to_csv(output / "metadata.csv", index_label="sample_id")
    excluded.to_csv(output / "excluded_samples.csv", index_label="sample_id")
    assignments.to_csv(output / "fold_assignment.csv", index_label="sample_id")
    (output / "label_mapping.json").write_text(
        json.dumps({"LUAD": 0, "LUSC": 1}, indent=2), encoding="utf-8"
    )

    stats = []
    for fold in range(1, N_SPLITS + 1):
        fold_dir = output / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        roles = assignments[f"fold_{fold}"]
        for split in ("train", "val", "test"):
            ids = assignments.index[roles == split]
            expression.loc[ids].assign(label=assignments.loc[ids, "label"]).to_hdf(
                fold_dir / f"{split}.h5", key="data", mode="w", complevel=9, complib="blosc"
            )
            for label, name in ((0, "LUAD"), (1, "LUSC")):
                stats.append({"fold": fold, "split": split, "class": name,
                              "n_samples": int((assignments.loc[ids, "label"] == label).sum())})
    pd.DataFrame(stats).to_csv(output / "fold_stats.csv", index=False)

    config = {
        "dataset": "gse30219_r5_fixed",
        "source_dataset": "GSE30219",
        "source_expression": str(source.relative_to(REPO) / "expression.h5"),
        "source_metadata": str(source.relative_to(REPO) / "metadata.csv"),
        "geo_accession": "GSE30219",
        "geo_series_matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30219/matrix/GSE30219_series_matrix.txt.gz",
        "label_mapping": {"ADC": "LUAD", "SQC": "LUSC"},
        "exclusion_policy": "Exclude every histology other than exact ADC or SQC; SCC is small-cell carcinoma.",
        "n_samples": len(labelled), "n_genes": expression.shape[1],
        "class_counts": actual, "excluded_counts": excluded["histology_raw"].value_counts().to_dict(),
        "split_method": "StratifiedGroupKFold over patient_id; fold r: test=P_r, val=P_(r+1), train=remaining partitions.",
        "selected_seed": split_info["selected_seed"],
        "split_imbalance_score": split_info["split_imbalance_score"],
        "files_sha256": {"expression.h5": sha256(output / "expression.h5"),
                         "metadata.csv": sha256(output / "metadata.csv"),
                         "fold_assignment.csv": sha256(output / "fold_assignment.csv")},
    }
    if prov is not None:
        stamp(config, prov)
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prov = require(__file__, data_files=[args.source / "expression.h5", args.source / "metadata.csv"])
    write_dataset(args.source, args.output, prov)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
