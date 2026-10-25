"""Contract checks for the Kaggle Y10 + P0 runner notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_y10_p0_runner.ipynb"


def test_notebook_is_a_kaggle_runner_for_the_uploaded_dataset_layout() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert "/kaggle/input/experiment-data-perm" in source
    assert "gse68465_stage_early_late" in source
    assert "tcga_luad_stage_early_late" in source
    assert "tcga_lusc_stage_early_late" in source
    assert "tcga_normal_vs_tumor_sanity" in source
    assert "run_sanity_auroc.py" in source
    assert "run_p0_perm_v3.py" in source
    assert "aggregate_p0_decision.py" in source
    assert "KLTN_DATA_ROOT" in source
    assert "--n-perm" in source and "--n-random" in source
    assert "config.json" in source
    assert "traceable" in source
