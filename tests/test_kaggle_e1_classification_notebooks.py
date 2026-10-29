import json
from pathlib import Path


NOTEBOOKS = {
    "kaggle_e1_classification_v3_runner.ipynb": (
        "CLASSIFICATION_GROUP",
        "evaluate_panel",
        "selection_per_fold.csv",
        "checkpoint",
    ),
    "kaggle_e1_reference_v3_runner.ipynb": (
        "RUN_B0",
        "N_RANDOM_DRAWS",
        "random_k",
        "checkpoint",
    ),
    "kaggle_e1b_fewshot_v3_runner.ipynb": (
        "fewshot_splits.csv",
        "validate_fewshot_split_manifest",
        "n_target",
        "target_only",
        "evaluate_panel",
        "skrebate==0.8.4",
    ),
    "kaggle_e2_e3_v3_runner.ipynb": (
        "E2_K_GRID",
        "E3_N_GRID",
        "E3_P_GRID",
        "machine_config.json",
        "measure",
        "skrebate==0.8.4",
    ),
    "kaggle_e0_v3_runner.ipynb": (
        "N_PERMUTATIONS = 200",
        "N_RANDOM_DRAWS = 100",
        "K = 10",
        "permutation",
        "random_k",
        "repeated_splits.csv",
        "checkpoint",
        "all_cohorts.zip",
    ),
    "kaggle_e4_stability_v3_runner.ipynb": (
        "RESUME_E4_ARCHIVE",
        "zipfile.ZipFile",
        "resume checkpoint",
        "E4_stability",
        "all_cohorts.zip",
    ),
    "kaggle_e2_e3_v3_sharded_runner.ipynb": (
        "SESSION_ID",
        "SESSION_METHODS",
        "e2_cost_point.csv",
        "checkpoint",
        "skrebate==0.8.4",
        "SHARD_ARCHIVES",
        "e2_cost_point.csv",
        "e2_e3_merged",
    ),
}


def test_kaggle_notebooks_expose_resumable_e1_shards():
    # Notebook 26–28/10 đã lưu trữ (M18/M19, Review 29/10): chỉ kiểm tra cấu trúc.
    root = Path(__file__).resolve().parents[1] / "notebooks" / "_archive" / "kaggle_v3_m19"
    for filename, required_markers in NOTEBOOKS.items():
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for marker in required_markers:
            assert marker in source
        assert "RUNTIME_IMPORTS" in source
        assert "requirements.txt" in source
    classification_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads((root / "kaggle_e1_classification_v3_runner.ipynb").read_text(encoding="utf-8"))["cells"]
    )
    assert "COHORTS =" in classification_source
    assert "for COHORT in COHORTS:" in classification_source
    assert "all_cohorts.zip" in classification_source
