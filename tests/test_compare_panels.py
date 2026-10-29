import pandas as pd

from scripts.compare_panels import compare, summarise


def _panels(rows):
    return pd.DataFrame(rows, columns=["cohort", "repeat", "fold", "method", "k", "rank", "gene"])


def test_equal_sets_reuse_and_order_matters_only_for_ranking_methods():
    old = _panels([
        ("tcga", 0, 0, "f_test", 2, 1, "A"), ("tcga", 0, 0, "f_test", 2, 2, "B"),
        ("tcga", 0, 0, "l1_lr", 2, 1, "A"), ("tcga", 0, 0, "l1_lr", 2, 2, "B"),
    ])
    new = _panels([
        ("tcga", 0, 0, "f_test", 2, 1, "B"), ("tcga", 0, 0, "f_test", 2, 2, "A"),   # same set, different order
        ("tcga", 0, 0, "l1_lr", 2, 1, "B"), ("tcga", 0, 0, "l1_lr", 2, 2, "A"),
    ])
    cells = compare(old, new, repeats=[0])
    s = summarise(cells).set_index("method")
    assert s.loc["f_test", "verdict"] == "RERUN"      # ranking method: order must match
    assert s.loc["l1_lr", "verdict"] == "REUSE"       # per-k method: set equality suffices


def test_missing_cell_or_different_gene_forces_rerun():
    old = _panels([("tcga", 0, 0, "f_test", 1, 1, "A"), ("tcga", 0, 1, "f_test", 1, 1, "A")])
    new = _panels([("tcga", 0, 0, "f_test", 1, 1, "C")])
    cells = compare(old, new, repeats=[0])
    assert set(cells.present_in) == {"both", "old_only"}
    s = summarise(cells)
    assert s.verdict.tolist() == ["RERUN"] and s.n_missing.tolist() == [1]
