import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path

from scripts.build_v3_splits import FEWSHOT_N, build_fewshot_split_manifest, validate_fewshot_split_manifest


def test_split_builder_cli_loads_from_the_repository_root():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_v3_splits.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_gse42127_fewshot_omits_infeasible_40_per_class_level():
    assert FEWSHOT_N["gse42127"] == (5, 10, 20)


def test_fewshot_manifest_has_nested_balanced_training_sets_and_fixed_test_per_split():
    sample_ids = np.array([f"s{i}" for i in range(80)])
    patient_ids = np.array([f"p{i}" for i in range(80)])
    labels = np.array([0] * 40 + [1] * 40)

    manifest = build_fewshot_split_manifest(
        sample_ids,
        patient_ids,
        labels,
        seeds=(42, 43, 44, 45, 46),
        n_per_class=(5, 10, 20),
        test_fraction=1 / 3,
    )

    assert validate_fewshot_split_manifest(manifest, n_per_class=(5, 10, 20)) == []
    for split, group in manifest.groupby("split"):
        test_ids = set(group.loc[group.role == "test", "sample_id"])
        assert test_ids
        for n_target in (5, 10, 20):
            train = group[(group.n_target == str(n_target)) & (group.role == "train")]
            assert train.groupby("label").size().to_dict() == {0: n_target, 1: n_target}
            assert not (set(train.sample_id) & test_ids)
        small = set(group[(group.n_target == "5") & (group.role == "train")].sample_id)
        large = set(group[(group.n_target == "10") & (group.role == "train")].sample_id)
        assert small <= large


def test_fewshot_validator_rejects_test_train_overlap():
    manifest = pd.DataFrame({
        "split": [0, 0, 0, 0],
        "seed": [42] * 4,
        "n_target": [5] * 4,
        "sample_id": ["s1", "s1", "s2", "s3"],
        "patient_id": ["p1", "p1", "p2", "p3"],
        "label": [0, 0, 1, 1],
        "role": ["test", "train", "train", "pool"],
    })

    assert any("overlap" in error for error in validate_fewshot_split_manifest(manifest, n_per_class=(5,)))
