"""Contract tests for the precomputed patient-grouped E1 5x5 split manifest."""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from survey_v2.e1_splits import build_repeated_split_manifest, validate_repeated_split_manifest
from scripts.run_e1_feature_selection import split_indices


def test_precomputed_manifest_has_25_leak_free_outer_splits():
    sample_ids = np.array([f"s{i}" for i in range(40)])
    patient_ids = np.array([f"p{i}" for i in range(40)])
    labels = np.array([0, 1] * 20)

    manifest = build_repeated_split_manifest(sample_ids, patient_ids, labels, seeds=(42, 43, 44, 45, 46))

    assert set(manifest.columns) == {"repeat", "seed", "fold", "sample_id", "patient_id", "label", "role"}
    assert len(manifest) == 40 * 5 * 5
    assert validate_repeated_split_manifest(manifest) == []
    counts = manifest.groupby(["repeat", "fold", "role"]).size()
    assert (counts.xs("train", level="role") == 32).all()
    assert (counts.xs("test", level="role") == 8).all()


def test_manifest_validator_reports_patient_leakage():
    manifest = pd.DataFrame({
        "repeat": [0, 0, 0, 0], "seed": [42] * 4, "fold": [0] * 4,
        "sample_id": ["s1", "s2", "s3", "s4"], "patient_id": ["p1", "p1", "p2", "p3"],
        "label": [0, 0, 1, 1], "role": ["train", "test", "train", "test"],
    })

    errors = validate_repeated_split_manifest(manifest)

    assert any("patient overlap" in error for error in errors)


def test_split_indices_rejects_manifest_label_mismatch():
    ds = SimpleNamespace(sample_ids=np.array(["s1", "s2", "s3", "s4"]),
                         groups=np.array(["p1", "p2", "p3", "p4"]), y=np.array([0, 0, 1, 1]), name="tiny")
    manifest = pd.DataFrame({
        "repeat": [0] * 4, "seed": [42] * 4, "fold": [0] * 4,
        "sample_id": ["s1", "s2", "s3", "s4"], "patient_id": ["p1", "p2", "p3", "p4"],
        "label": [1, 0, 1, 1], "role": ["train", "train", "test", "test"],
    })

    with pytest.raises(ValueError, match="label does not match"):
        split_indices(ds, manifest, 0, 0)
