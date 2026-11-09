"""Kiểm tra đơn vị cho scripts/run_e7_k2_evidence.py (Y62 phần 1): chỉ dữ liệu giả, không cần dữ liệu thật."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_e7_k2_evidence import (PER_RUN_COLUMNS, direction_row, fit_source_model, gene_direction_rows,
                                        zero_shot_rows, zero_shot_summary)
from survey_v2.transfer import panel_transform


def _toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(5, 1, size=(n, 4))
    X[:, 0] += 0.5 * y          # gen A tăng nhẹ ở LUSC
    X[:, 1] += 2.0 * y          # gen B tăng mạnh ở LUSC -> φ = A − B giảm ở LUSC
    return X, y, np.array(["A", "B", "C", "D"])


def test_direction_row_signs():
    X, y, _ = _toy()
    r = direction_row("toy", "B", "log2", X[:, 1], y)
    assert r["n_luad"] == r["n_lusc"] == 60
    assert r["shift"] > 0 and r["auroc_as_lusc_score"] > 0.5
    r2 = direction_row("toy", "negB", "log2", -X[:, 1], y)
    assert np.isclose(r2["auroc_as_lusc_score"], 1 - r["auroc_as_lusc_score"])


def test_gene_direction_rows_phi_is_first_minus_second():
    X, y, genes = _toy()
    rows = gene_direction_rows("toy", "log2", X, genes, y, ["A", "B"], extra_genes=("C", "ZZZ"))
    names = [r["gene"] for r in rows]
    assert names == ["A", "B", "C", "phi_A_minus_B"]        # gen không có trong cohort bị bỏ qua
    phi = rows[-1]
    assert np.isclose(phi["shift"], rows[0]["shift"] - rows[1]["shift"])
    assert phi["auroc_as_lusc_score"] < 0.5


def test_zero_shot_rows_and_summary():
    per_run = pd.DataFrame({"cohort": ["c1"] * 4 + ["c2"] * 2, "split": [0, 1, 0, 1, 0, 1], "seed": 42, "subset": 0,
                            "n_t": ["5", "5", "10", "10", "5", "5"], "n_train": 10, "n_test": 30,
                            "strategy": ["source_only", "source_only", "source_only", "pooled", "source_only", "source_only"],
                            "k": [2, 2, 2, 2, 3, 2], "auroc": [0.6, 0.4, 0.7, 0.9, 0.1, 0.3]})
    rows = zero_shot_rows(per_run)
    assert list(rows.columns) == PER_RUN_COLUMNS and len(rows) == 4
    s = zero_shot_summary(rows).set_index("cohort")
    assert s.loc["c1", "n_runs"] == 3 and s.loc["c1", "n_runs_below_0_5"] == 1
    assert np.isclose(s.loc["c1", "auroc_mean"], (0.6 + 0.4 + 0.7) / 3)
    assert s.loc["c2", "n_runs"] == 1 and s.loc["c2", "n_runs_below_0_5"] == 1


def test_fit_source_model_reproduces_evaluate_panel_and_sign():
    Xs, ys, _ = _toy(1, 200)
    Xt, yt, _ = _toy(2, 80)
    fit = fit_source_model(panel_transform(Xs[:, :2]), ys, panel_transform(Xt[:, :2]), yt, seed=7)
    assert fit["max_abs_diff_scores_vs_evaluate_panel"] < 1e-12
    assert fit["coef_phi"] < 0                                 # φ giảm ở LUSC -> hệ số âm
    assert np.isclose(fit["coef_phi_raw_scale"] * fit["scaler_sd_phi"], fit["coef_phi"])
