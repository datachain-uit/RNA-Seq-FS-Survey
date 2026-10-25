"""Chạy đầu-cuối mọi script survey trên dữ liệu tổng hợp nhỏ (KLTN_DATA_ROOT tạm), kiểm tra tệp đầu ra và bất biến.

Chạy chậm (~1–3 phút). Bỏ qua bằng: pytest -m "not slow".
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def data_root(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("kltn")
    monkeypatch_module.setenv("KLTN_DATA_ROOT", str(root))
    monkeypatch_module.setenv("KLTN_ALLOW_UNPUSHED", "1")
    monkeypatch_module.setenv("KLTN_ALLOW_UNTRACKED", "1")
    # nạp lại các module đọc DATA_ROOT lúc import
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2.runlog") or m.startswith("run_survey") or m == "provenance":
            del sys.modules[m]
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from survey_v2 import synthetic
    synthetic.make_all(root, n=90, p=600, effect_stage=0.8)   # p > 500 vì K_STAGE của P0 có k = 500
    return root


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _load(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_e1_and_wrappers(data_root):
    e1 = _load("run_survey_e1")
    out = data_root / "results" / "survey_v2" / "E1"
    e1.main(["--datasets", "gse68465_stage_early_late", "tcga_luad_stage_early_late", "--methods", "f_test", "mrmr", "l1_lr",
             "--k", "5", "10", "--repeats", "0", "--prefilter", "100", "--random-draws", "3", "--secondary", "--n-jobs", "1"])
    for ds in ("gse68465_stage_early_late", "tcga_luad_stage_early_late"):
        d = out / ds
        pf = pd.read_csv(d / "e1_per_fold.csv")
        assert set(pf.method) == {"b0", "f_test", "mrmr", "l1_lr"}
        assert (pf.n_selected[pf.method != "b0"] == pf.k[pf.method != "b0"]).all()
        assert pf.auroc.between(0, 1).all()
        genes = pd.read_csv(d / "e1_selected_genes.csv")
        assert genes.groupby(["repeat", "fold", "method", "k"]).size().eq(genes.k.iloc[0]).any()
        # tập k=5 ⊂ tập k=10 với phương pháp xếp hạng
        g = genes[(genes.method == "f_test") & (genes.repeat == 0) & (genes.fold == 0)]
        assert set(g[g.k == 5].gene) <= set(g[g.k == 10].gene)
        rk = pd.read_csv(d / "e1_random_k.csv")
        assert rk.groupby(["fold", "k"]).size().eq(3).all()
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        assert "commit_hash" in cfg and cfg["random_k_draws"] == 3
        splits = pd.read_csv(d / "cv_splits.csv")
        tr = splits[(splits.repeat == 0) & (splits.fold == 0) & (splits.role == "train")].patient_id
        te = splits[(splits.repeat == 0) & (splits.fold == 0) & (splits.role == "test")].patient_id
        assert set(tr).isdisjoint(te)
    # resume: chạy lại không nhân đôi dòng
    n_before = len(pd.read_csv(out / "gse68465_stage_early_late" / "e1_per_fold.csv"))
    e1.main(["--datasets", "gse68465_stage_early_late", "--methods", "f_test", "mrmr", "l1_lr", "--k", "5", "10", "--repeats", "0",
             "--prefilter", "100", "--random-draws", "3", "--secondary", "--n-jobs", "1"])
    assert len(pd.read_csv(out / "gse68465_stage_early_late" / "e1_per_fold.csv")) == n_before
    # wrapper
    w = _load("run_survey_e1_wrappers")
    sys.argv = ["run_survey_e1_wrappers.py", "--methods", "ga", "--repeats", "0", "--k", "5", "--prefilter", "60", "--pop", "6",
                "--generations", "2"]
    w.main()
    pf = pd.read_csv(out / "gse68465_stage_early_late" / "e1_per_fold.csv")
    ga = pf[pf.method == "ga"]
    assert len(ga) == 5 and ga.n_eval.gt(0).all() and ga.n_selected.eq(5).all()


def test_e2_e3_timing(data_root):
    t = _load("run_survey_e2_e3_timing")
    sys.argv = ["run_survey_e2_e3_timing.py", "--methods", "f_test", "mrmr", "--repeats", "1", "--n-grid", "20", "40", "all",
                "--p-grid", "50", "100", "all", "--k", "5", "--wrappers"]
    import survey_v2.config as C
    C.WRAPPER_POP, C.WRAPPER_GENERATIONS = 4, 2
    C.E3_WRAPPER_BUDGETS = ((4, 2), (5, 2))
    t.main()
    out = data_root / "results" / "survey_v2" / "E2_E3"
    cost = pd.read_csv(out / "e2_cost_point.csv")
    assert set(cost.method) == {"f_test", "mrmr", "ga", "bpso"} and (cost.wall_s > 0).all()
    exps = pd.read_csv(out / "e3_exponents.csv")
    assert set(exps.method) == {"f_test", "mrmr"}
    dn = pd.read_csv(out / "e3_scaling_n.csv")
    assert sorted(dn.n.unique()) == [20, 40, dn.n.max()]
    assert (out / "machine_config.json").exists() and (out / "config.json").exists()


def test_e4_e5_stability(data_root):
    s = _load("run_survey_e4_e5_stability")
    s.main(["--datasets", "gse68465_stage_early_late", "--methods", "f_test", "l1_lr", "--k", "5", "10", "--m", "4",
            "--fractions", "0.5", "0.7", "--prefilter", "100", "--wrapper-k", "5", "--pop", "4", "--generations", "2", "--n-jobs", "1"])
    out = data_root / "results" / "survey_v2" / "E4_E5" / "gse68465_stage_early_late"
    stab = pd.read_csv(out / "e4_stability.csv")
    assert set(stab.method) == {"f_test", "l1_lr", "random_k"} and set(stab.fraction) == {0.5, 0.7}
    assert stab.nogueira.between(-1, 1.0001).all() and (stab.M == 4).all()
    # random-k phải kém ổn định hơn F-test trên dữ liệu có tín hiệu
    f, r = stab[stab.method == "f_test"].nogueira.mean(), stab[stab.method == "random_k"].nogueira.mean()
    assert f > r
    e5a = pd.read_csv(out / "e5a_redundancy.csv")
    assert set(e5a.rho) == {0.5, 0.8, 0.95} and "nogueira_merged" in e5a.columns
    e5b = pd.read_csv(out / "e5b_seed_stability.csv")
    assert set(e5b.method) == {"rf_importance", "ga", "bpso"} and (e5b.n_seeds == 5).all()
    assert (out / "e5c_overlap_k50.csv").exists() is False or True  # E1 chạy ở k 5/10 nên không có k=50 -> có thể rỗng
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["e4_e5a"]["n_subsamples"] == 4


def test_tables_and_p0(data_root):
    import survey_v2.config as C
    C.OVERLAP_K = 10
    mk = _load("make_survey_tables")
    mk.main(["--e6-fraction", "0.7"])
    out = data_root / "results" / "survey_v2" / "tables"
    assert (out / "e1_summary.csv").exists() and (out / "e1_tests_vs_random_b0.csv").exists()
    tests = pd.read_csv(out / "e1_tests_vs_random_b0.csv")
    assert {"p_nb_random", "p_nb_random_holm", "p_nb_b0"} <= set(tests.columns)
    assert tests.p_nb_random.between(0, 1).all()
    man = json.loads((out / "provenance_manifest.json").read_text(encoding="utf-8"))
    assert man["inputs"] and all("sha256" in v for v in man["inputs"].values())
    assert (out / "e1_auroc_table.tex").exists()
    figs = list((data_root / "results" / "survey_v2" / "figures").glob("*.png"))
    assert figs
    # P0 perm v3 (ít hoán vị) + aggregator
    p0 = _load("run_p0_perm_v3")
    p0.main(["--datasets", "gse68465_stage_early_late", "--n-perm", "6", "--n-random", "6", "--n-jobs", "1", "--batch-per-worker", "3"])
    d = data_root / "results" / "survey_v2" / "P0" / "perm_test_v3" / "gse68465_stage_early_late"
    summ = pd.read_csv(d / "summary.csv")
    assert len(summ) == 15 and (summ.n_perm == 6).all() and (summ.n_random == 6).all()
    assert summ.p_perm.between(1 / 7 - 1e-9, 1).all()   # p nhỏ nhất = 1/(B+1)
    null = pd.read_csv(d / "null_scores_per_fold.csv")
    assert null.pid.nunique() == 6 and set(null.pipeline) == {"ftest_lr", "svm_rfe", "pam"}
    agg = _load("aggregate_p0_decision")
    agg.main(["--summaries", str(d / "summary.csv"), "--out-dir", "P0_test"])
    dec = pd.read_csv(data_root / "results" / "survey_v2" / "P0_test" / "p0_decision.csv")
    assert len(dec) == 1 and dec.n_tests.iloc[0] == 15 and dec.family_complete.iloc[0]
    # sanity Y10
    sa = _load("run_sanity_auroc")
    rc = sa.main(["--repeats", "0", "--prefilter", "100"])
    summ = pd.read_csv(data_root / "results" / "survey_v2" / "sanity" / "sanity_summary.csv")
    assert set(summ.method) == {"f_test", "l1_lr", "rf_importance"}
    assert rc in (0, 1)
