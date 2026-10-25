"""Kiểm tra các phương pháp chọn gen, wrapper, mRMR và lọc sơ bộ của survey_v2 (dữ liệu tổng hợp nhỏ)."""

from __future__ import annotations

import numpy as np
import pytest

from survey_v2 import config as C
from survey_v2.data import add_correlated_copies, stratified_subsample, variance_prefilter
from survey_v2.methods import prefix_panels, select
from survey_v2.mrmr import discretize, mrmr_ranking, relevance
from survey_v2.wrappers import InnerCVFitness, binary_pso, genetic_algorithm


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, p = 90, 120
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(0, 1, size=(n, p))
    X[:, :6] += 1.5 * y[:, None]           # 6 gen tín hiệu
    X[:, 6] = X[:, 0] * 0.95 + 0.3 * rng.normal(size=n)   # gen dư thừa với gen 0
    return X, y


def test_variance_prefilter_is_label_free_and_sorted(toy):
    X, _ = toy
    keep = variance_prefilter(X, 50)
    assert len(keep) == 50 and np.all(np.diff(keep) > 0)
    var = X.var(axis=0, ddof=1)
    assert set(keep) == set(np.argsort(-var)[:50])
    assert len(variance_prefilter(X, 10_000)) == X.shape[1]


def test_prefix_panels_are_nested():
    order = np.arange(100)[::-1]
    panels = prefix_panels(order, [5, 10, 20])
    assert set(panels[5]) <= set(panels[10]) <= set(panels[20])
    with pytest.raises(ValueError):
        prefix_panels(order, [200])


@pytest.mark.parametrize("method", ["f_test", "mutual_info", "mrmr", "relieff", "rf_importance", "svm_rfe", "l1_lr", "elastic_net"])
def test_select_returns_exact_k_and_nested_for_rankers(toy, method):
    X, y = toy
    ks = [5, 10, 20]
    sel = select(method, X, y, ks, seed=1)
    for k in ks:
        assert len(sel.panels[k]) == k
        assert len(set(sel.panels[k].tolist())) == k
        assert sel.panels[k].min() >= 0 and sel.panels[k].max() < X.shape[1]
    if method in C.RANKING_METHODS:
        assert set(sel.panels[5]) <= set(sel.panels[10]) <= set(sel.panels[20])
    if method in ("f_test", "mrmr", "svm_rfe", "l1_lr"):
        # phần lớn gen tín hiệu phải nằm trong top 10
        assert len(set(sel.panels[10]) & set(range(7))) >= 4


def test_mrmr_penalises_redundant_gene(toy):
    X, y = toy
    D = discretize(X)
    assert set(np.unique(D)) <= {0, 1, 2}
    rel = relevance(D, y)
    assert rel[:6].mean() > rel[20:].mean()
    order = mrmr_ranking(X, y, 10, criterion="MID")
    assert len(order) == 10 and len(set(order.tolist())) == 10
    # gen 6 gần như trùng gen 0: không được đứng liền ngay sau gen 0 ở vị trí thứ hai trong mRMR
    pos0, pos6 = list(order).index(0) if 0 in order else -1, list(order).index(6) if 6 in order else 99
    assert not (pos0 == 0 and pos6 == 1)
    order_q = mrmr_ranking(X, y, 10, criterion="MIQ")
    assert len(order_q) == 10


def test_sparse_exact_k_reports_fill_when_saturated():
    rng = np.random.default_rng(3)
    n, p = 30, 60
    y = np.array([0, 1] * 15)
    X = rng.normal(size=(n, p))
    X[:, 0] += 2 * y
    sel = select("l1_lr", X, y, [10, 50], seed=0)
    assert len(sel.panels[50]) == 50
    # L1 với n = 30 không thể có 50 hệ số khác 0 -> phải bổ sung bằng F-test và ghi n_filled > 0
    assert sel.info["n_filled"][50] > 0
    assert sel.info["exact"][50] is False


def test_wrappers_return_exact_k_and_count_evaluations(toy):
    X, y = toy
    fit = InnerCVFitness(X, y, seed=0)
    rng = np.random.default_rng(0)
    best, f, hist = genetic_algorithm(fit, X.shape[1], 5, rng, pop=6, generations=3)
    assert len(best) == 5 and len(set(best.tolist())) == 5 and 0 <= f <= 1 and len(hist) == 4
    assert fit.n_eval > 0 and fit.n_eval <= 6 + 6 * 3
    fit2 = InnerCVFitness(X, y, seed=0)
    best2, f2, hist2 = binary_pso(fit2, X.shape[1], 5, np.random.default_rng(0), pop=6, iterations=3)
    assert len(best2) == 5 and len(set(best2.tolist())) == 5
    assert all(b >= a for a, b in zip(hist2, hist2[1:]))  # gbest không giảm
    sel = select("ga", X, y, [5], seed=0, wrapper_kwargs={"pop": 6, "generations": 2})
    assert len(sel.panels[5]) == 5 and sel.info["n_eval"][5] > 0


def test_fitness_uses_only_training_rows(toy):
    """Fitness được dựng chỉ từ (X_tr, y_tr): thay đổi dữ liệu ngoài không ảnh hưởng."""
    X, y = toy
    Xtr, ytr = X[:60], y[:60]
    f1 = InnerCVFitness(Xtr, ytr, seed=0)([0, 1, 2])
    X_mod = X.copy()
    X_mod[60:] += 100.0
    f2 = InnerCVFitness(X_mod[:60], ytr, seed=0)([0, 1, 2])
    assert f1 == f2


def test_subsample_and_copies(toy):
    X, y = toy
    rng = np.random.default_rng(0)
    idx = stratified_subsample(y, 0.5, rng)
    assert len(idx) == pytest.approx(len(y) * 0.5, abs=2)
    assert abs(y[idx].mean() - y.mean()) < 0.1
    genes = np.array([f"g{j}" for j in range(X.shape[1])])
    X2, genes2, src, copies = add_correlated_copies(X, genes, 0.9, 5, rng)
    assert X2.shape == (X.shape[0], X.shape[1] + 5) and len(genes2) == X.shape[1] + 5
    for s, c in zip(src, copies):
        r = np.corrcoef(X2[:, s], X2[:, c])[0, 1]
        assert r > 0.75
        assert genes2[c].startswith(genes[s] + "__copy_rho")
