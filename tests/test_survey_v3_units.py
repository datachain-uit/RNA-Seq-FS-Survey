"""Kiểm tra đơn vị cho phần thêm ở survey v3: abess, stability selection, k-TSP, panel cố định, bản chia few-shot,
k* gộp (đoạn code G), chuẩn hoá mức panel và chiến lược chuyển giao, log-loss/Brier và chế độ ít mẫu."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survey_v2 import config as C
from survey_v2.classify import evaluate_panel, probability_metrics
from survey_v2.fewshot import build_fewshot_manifest, fewshot_runs, validate_fewshot_manifest
from survey_v2.kstar import k_star, kstar_table, per_cohort, verdict_h1
from survey_v2.methods import ktsp_pair_scores, rank_ktsp, rank_stability_selection, select
from survey_v2.panels import all_fixed_panels, panel_indices
from survey_v2.transfer import panel_transform


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(3)
    n, p = 100, 150
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(0, 1, size=(n, p))
    X[:, 0] += 2.0 * y
    X[:, 1] -= 2.0 * y
    X[:, 2] += 1.2 * y
    X[:, 3] -= 1.2 * y
    return X, y


# ---------------------------------------------------------------------------------------------------
# Phương pháp mới
# ---------------------------------------------------------------------------------------------------
def test_stability_selection_ranks_signal_genes_first(toy):
    X, y = toy
    order, info = rank_stability_selection(X, y, k_max=10, seed=0, n_subsamples=20)
    assert len(set(order.tolist())) == X.shape[1]
    assert set(order[:4].tolist()) >= {0, 1}
    assert 0 < info["C"] and info["pi"].max() <= 1.0


def test_ktsp_pairs_are_disjoint_and_even_only(toy):
    X, y = toy
    order, info = rank_ktsp(X, y, k_max=6)
    genes = [g for pair in info["pairs"] for g in pair[:2]]
    assert len(genes) == len(set(genes)) == 6
    assert {0, 1} <= set(order[:2].tolist()) or {0, 1} <= set(order[:4].tolist())
    sel = select("ktsp", X, y, [2, 3, 4, 10])
    assert sel.panels[3] is None and len(sel.panels[2]) == 2 and len(sel.panels[4]) == 4
    assert set(sel.panels[2]) <= set(sel.panels[4]) <= set(sel.panels[10])
    d, g, (ii, jj) = ktsp_pair_scores(X[:, :20], y)
    assert len(d) == 20 * 19 // 2 and d.max() <= 1.0 + 1e-6


def test_ktsp_refuses_too_many_genes(toy):
    X, y = toy
    with pytest.raises(ValueError, match="KTSP_MAX_P"):
        ktsp_pair_scores(X, y, max_p=10)


def test_abess_exact_support(toy):
    pytest.importorskip("abess")
    X, y = toy
    sel = select("abess", X, y, [2, 5], seed=0)
    assert len(sel.panels[2]) == 2 and len(sel.panels[5]) == 5
    assert {0, 1} <= set(sel.panels[5].tolist())


def test_svm_rfe_ranks_down_to_k1(toy):
    X, y = toy
    sel = select("svm_rfe", X, y, [1, 2, 5])
    assert len(sel.panels[1]) == 1 and set(sel.panels[1]) <= set(sel.panels[2]) <= set(sel.panels[5])


# ---------------------------------------------------------------------------------------------------
# Panel cố định
# ---------------------------------------------------------------------------------------------------
def test_fixed_panels_and_missing_genes():
    panels = all_fixed_panels()
    assert panels["ihc_2"] == ["NKX2-1", "TP63"] and panels["ihc_4"][:2] == ["NKX2-1", "TP63"]
    cols, missing = panel_indices(np.array(["X", "TP63", "NKX2-1"]), panels["ihc_4"])
    assert cols.tolist() == [2, 1] and missing == ["NAPSA", "KRT5"]


# ---------------------------------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------------------------------
def test_fewshot_manifest_nested_and_leak_free():
    n = 150
    y = np.array([0] * 90 + [1] * 60)
    ids = np.array([f"s{i}" for i in range(n)])
    man = build_fewshot_manifest("gse37745", ids, ids, y)
    assert validate_fewshot_manifest(man) == []
    assert set(man.n_t.unique()) == {"5", "10", "20", "40", "pool"}   # pool có đúng 40 LUSC -> mức 40 vẫn hợp lệ (n_T <= lớp nhỏ)
    runs = list(fewshot_runs(man))
    assert len(runs) == 5 * 4 * 5
    # tập con lồng nhau và mỗi lớp đúng n_T
    g = man[(man.split == 0) & (man.subset == 0) & (man.role == "train")]
    s5, s10 = set(g[g.n_t == "5"].sample_id), set(g[g.n_t == "10"].sample_id)
    assert s5 <= s10 and len(s5) == 10 and len(s10) == 20
    # GSE10245: chỉ n_T 5, 10
    man2 = build_fewshot_manifest("gse10245", ids[:58], ids[:58], np.array([0] * 40 + [1] * 18))
    assert set(man2.n_t.unique()) == {"5", "10", "pool"}


def test_fewshot_validator_catches_overlap():
    man = pd.DataFrame({"cohort": "x", "split": 0, "seed": 42, "subset": 0, "n_t": "5",
                        "sample_id": ["a", "b", "c", "a"], "patient_id": ["a", "b", "c", "a"], "label": [0, 1, 0, 1],
                        "role": ["train", "train", "test", "test"]})
    assert any("trùng bệnh nhân" in e for e in validate_fewshot_manifest(man))


# ---------------------------------------------------------------------------------------------------
# k* gộp (đoạn code G)
# ---------------------------------------------------------------------------------------------------
def _fake_long(rng, cohorts=("gse10245", "gse30219", "gse37745"), ks=(1, 2, 3, 4, 5, 10, 25, 50), drop_small_k=False, noisy_cohort=None):
    rows = []
    for c in cohorts:
        sd = 0.06 if c == "gse10245" else 0.015
        for r in range(5):
            for f in range(5):
                b0 = 0.95 + rng.normal(0, sd)
                rows.append({"cohort": c, "repeat": r, "fold": f, "method": "b0", "k": 0, "auroc": b0})
                for k in ks:
                    penalty = 0.0
                    if drop_small_k and k <= 3:
                        penalty = 0.08
                    if noisy_cohort == c:
                        penalty += 0.05
                    rows.append({"cohort": c, "repeat": r, "fold": f, "method": "m", "k": k, "auroc": b0 - penalty + rng.normal(0, sd / 3)})
    return pd.DataFrame(rows)


def test_kstar_like_b0_supports_h1():
    df = _fake_long(np.random.default_rng(0))
    tab, k_ivw, k_both = k_star(df, "m", ["gse10245", "gse30219", "gse37745"])
    assert k_both == 1 and k_ivw == 1
    assert verdict_h1(k_ivw, k_both, tab["k"].tolist()).startswith("ủng hộ (")
    assert set(tab.columns) >= {"diff_ivw", "lb_ivw", "lb_eq", "I2", "pass_ivw", "pass_eq"}


def test_kstar_stops_at_first_failure():
    df = _fake_long(np.random.default_rng(1), drop_small_k=True)
    _, k_ivw, k_both = k_star(df, "m", ["gse10245", "gse30219", "gse37745"])
    assert k_both == 4 and k_ivw == 4


def test_kstar_partial_when_only_ivw_passes():
    df = _fake_long(np.random.default_rng(2), noisy_cohort="gse10245")
    tab, k_ivw, k_both = k_star(df, "m", ["gse10245", "gse30219", "gse37745"])
    assert k_ivw is not None
    # cohort nhỏ bị phạt 0,05 nhưng trọng số IVW của nó thấp -> IVW đạt, trọng số bằng nhau có thể không
    assert tab.loc[tab.k == 50, "w_gse10245"].iloc[0] < 0.2
    summ = kstar_table(df, ["m"], cohorts=["gse10245", "gse30219", "gse37745"])[1]
    assert summ.k_sequence_used.iloc[0].startswith("50>25>10")


def test_per_cohort_uses_nadeau_bengio_variance():
    df = _fake_long(np.random.default_rng(3))
    pc = per_cohort(df, "m", 5)
    assert len(pc) == 3 and (pc.J == 25).all()
    row = pc.iloc[0]
    assert np.isclose(row["var"], (1 / 25 + 0.25) * row["sd"] ** 2)


# ---------------------------------------------------------------------------------------------------
# Chuẩn hoá mức panel, chỉ số xác suất, chế độ ít mẫu
# ---------------------------------------------------------------------------------------------------
def test_panel_transform_is_per_sample():
    X = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    Z = panel_transform(X)
    assert np.allclose(Z[0], Z[1])                      # cùng hình dạng trong mẫu -> cùng φ
    assert np.allclose(Z.mean(axis=1), 0)
    D = panel_transform(X[:, :2])
    assert D.shape == (2, 1) and np.allclose(D[:, 0], X[:, 0] - X[:, 1])
    assert panel_transform(X[:, :1]).shape == (2, 1)


def test_probability_metrics_and_small_n_mode():
    y = np.array([0, 0, 1, 1])
    ll, br = probability_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), True)
    assert 0 < ll < 0.3 and 0 < br < 0.05
    assert np.isnan(probability_metrics(y, np.array([-1.0, -0.5, 0.5, 1.0]), False)[0])
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(12, 3)); ytr = np.array([0, 1] * 6); Xtr[:, 0] += 2 * ytr
    Xte = rng.normal(size=(30, 3)); yte = np.array([0, 1] * 15); Xte[:, 0] += 2 * yte
    r = evaluate_panel(Xtr, ytr, Xte, yte)                 # 6 mẫu/lớp <= SMALL_N_T_MAX -> C cố định, ngưỡng 0,5
    assert r.hyperparam == C.SMALL_N_FIXED_C and r.threshold == 0.5 and r.auroc > 0.7 and not np.isnan(r.brier)
    r2 = evaluate_panel(Xtr, ytr, Xte, yte, small_n=False)  # ép CV nội bộ (n đủ cho 3 fold)
    assert r2.hyperparam in C.C_GRID


def test_config_consistency():
    assert set(C.FEWSHOT_K_GRID) <= set(C.K_GRID) and set(C.WRAPPER_K) <= set(C.K_GRID)
    assert set(C.POOLED_KSTAR_COHORTS) <= set(C.DATASETS) and C.SOURCE_DATASET in C.DATASETS
    assert all(k in C.K_GRID for k in C.KSTAR_SEQUENCE) and C.OVERLAP_K in C.K_GRID
    assert set(C.STANDARDIZE_BEFORE) <= set(C.ALL_SELECTORS)
    assert len(C.METHOD_ID) == len(C.ALL_SELECTORS) == 13


def test_ktsp_cap_covers_e5a_copies():
    """QĐ24: E5a thêm REDUNDANCY_N_COPIES bản sao sau lọc sơ bộ P0; k-TSP phải chạy được ở p = P0 + bản sao."""
    from survey_v2 import config as C
    assert C.KTSP_MAX_P >= C.PREFILTER_P0 + C.REDUNDANCY_N_COPIES
    assert C.KTSP_MAX_P < 10000          # điểm p = 10.000 của E3 vẫn bị bỏ cho k-TSP
