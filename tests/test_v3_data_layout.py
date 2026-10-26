from pathlib import Path

from survey_v2.data import dataset_path
from survey_v2 import config as C


def test_v3_dataset_paths_use_the_kaggle_mounted_data_root_directly(tmp_path):
    """The Kaggle dataset mount contains the v3 folders, not an extra v3 parent."""
    root = tmp_path / "survey-v3-data"

    assert dataset_path(root, "tcga_luad_lusc_primary_tumor") == (
        root / "experiment-data-perm" / "tcga_luad_lusc_primary_tumor"
    )
    assert dataset_path(root, "gse10245") == root / "experiment-data-geo-target" / "gse10245"


def test_v3_cohorts_have_deterministic_seed_ids_outside_the_legacy_registry():
    assert C.dataset_seed_id("tcga_luad_lusc_primary_tumor") == 100
    assert C.dataset_seed_id("gse42127") == 104
