from pathlib import Path

from survey_v2.data import dataset_path


def test_v3_dataset_paths_use_the_kaggle_mounted_data_root_directly(tmp_path):
    """The Kaggle dataset mount contains the v3 folders, not an extra v3 parent."""
    root = tmp_path / "survey-v3-data"

    assert dataset_path(root, "tcga_luad_lusc_primary_tumor") == (
        root / "experiment-data-perm" / "tcga_luad_lusc_primary_tumor"
    )
    assert dataset_path(root, "gse10245") == root / "experiment-data-geo-target" / "gse10245"
