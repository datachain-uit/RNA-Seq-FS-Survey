"""Kiểm tra thống kê (Nadeau–Bengio, Holm, Bayesian ROPE), chỉ số ổn định và bộ phân loại của survey_v2."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st

from survey_v2.classify import evaluate_panel
from survey_v2.stability import jaccard, kuncheva_index, mean_jaccard, nogueira, overlap_matrix, selections_to_matrix
from survey_v2.stats import bayesian_correlated_ttest, holm, mean_ci, nadeau_bengio_ttest


def test_holm_matches_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = holm(p)
    # bước xuống: 0.01*4=0.04; 0.03*3=0.09; 0.04*2=0.08->max(0.09)=0.09; 0.20*1=0.20
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.20])
    adj_nan = holm([0.01, np.nan, 0.5])
    assert np.isnan(adj_nan[1]) and np.allclose(adj_nan[[0, 2]], [0.02, 0.5])


def test_nadeau_bengio_is_more_conservative_than_plain_t():
    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.03, size=25)
    nb = nadeau_bengio_ttest(d, ratio=0.25, alternative="greater")
    t_plain, p_plain = st.ttest_1samp(d, 0.0, alternative="greater")
    assert nb["t"] < t_plain and nb["p"] > p_plain
    assert nb["ci_low"] < nb["mean"] < nb["ci_high"]
    assert np.isnan(nadeau_bengio_ttest([0.1])["p"])  # NaN khi J < 2


def test_bayesian_rope_probabilities_sum_to_one():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 0.01, size=25)
    b = bayesian_correlated_ttest(d, rope=0.01)
    assert abs(b["p_left"] + b["p_rope"] + b["p_right"] - 1) < 1e-9
    big = bayesian_correlated_ttest(np.full(25, 0.05) + rng.normal(0, 0.001, 25), rope=0.01)
    assert big["p_right"] > 0.99
    assert mean_ci(d)["n"] == 25


def test_nogueira_extremes_and_ci():
    d = 100
    same = [list(range(10))] * 20
    Z = selections_to_matrix(same, list(range(d)))
    r = nogueira(Z)
    assert r["stability"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rand = [sorted(rng.choice(d, 10, replace=False).tolist()) for _ in range(200)]
    r2 = nogueira(selections_to_matrix(rand, list(range(d))))
    assert abs(r2["stability"]) < 0.08
    assert r2["ci_low"] < r2["stability"] < r2["ci_high"]
    assert kuncheva_index(same, d) == pytest.approx(1.0)
    assert abs(kuncheva_index(rand, d)) < 0.08
    assert mean_jaccard(same) == 1.0 and jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_overlap_matrix_symmetric():
    names, M = overlap_matrix({"a": [{1, 2, 3}, {1, 2, 4}], "b": [{1, 2, 3}, {5, 6, 7}]})
    assert names == ["a", "b"] and M[0, 0] == 1.0 and M[0, 1] == M[1, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("clf", ["lr_l2", "rf", "svm_linear"])
def test_evaluate_panel_detects_strong_signal(clf):
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, 8))
    X[:, 0] += 3 * y
    r = evaluate_panel(X[:80], y[:80], X[80:], y[80:], classifier=clf, seed=0)
    assert r.auroc > 0.95 and r.bacc > 0.85 and -1 <= r.mcc <= 1
    assert len(r.scores_test) == 40
    if clf != "rf":
        assert r.hyperparam in (0.01, 0.1, 1.0, 10.0)
