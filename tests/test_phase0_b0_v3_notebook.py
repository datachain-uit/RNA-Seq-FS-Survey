"""Contract checks for the Phase 0 v3 full-gene baseline notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "phase_0"
    / "04_analysis"
    / "b0_full_gene_baseline_v3.ipynb"
)


def test_b0_notebook_declares_v3_datasets_and_protocol() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert "gse10245_luad_lusc" in source
    assert "gse30219_luad_lusc" in source
    assert "gse37745_luad_lusc" in source
    assert "gse42127_luad_lusc" in source
    assert "gse68465_stage_tumor_only" in source
    assert "tcga_luad_lusc_primary_tumor" in source
    assert "tcga_normal_vs_tumor_sanity" in source
    assert "B0_all_genes" in source
    assert "/kaggle/input/experiment-dataset-v2/h5" in source
    assert "/kaggle/input/datasets/lfreedom2750/experiment-dataset-v2" in source
    assert "fold_1" in source
    assert "folds/fold_<r>" in source
    assert "frame.pop('label')" in source
    assert "kaggle_input.rglob('train.h5')" in source
    assert "len(np.unique(ytr)) < 2" in source
    assert "n_classes" in source
    assert "shutil.make_archive" in source
    assert "RESULT_ROOT" in source

