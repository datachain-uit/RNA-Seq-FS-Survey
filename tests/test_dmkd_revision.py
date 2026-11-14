"""Đợt hiệu chỉnh DMKD (B1–B7, đặc tả 12/11/2026): test đơn vị cho quy tắc và một chuỗi đầu-cuối trên dữ liệu giả nhỏ.

Chuỗi (slow): dựng dữ liệu giả → E1 pha A/B, E1b, E7, E4 (đóng vai "kết quả đã công bố", KLTN_PUBLISHED_ROOT) → B1 (bản chia mở rộng +
cổng), B2, B3, B4, B5, B7 → make_revision_tables.py. Mỗi cổng bằng nhau phải đạt trên dữ liệu giả (cùng mã, cùng seed).
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from survey_v2 import config as C  # noqa: E402
from survey_v2 import revision as R  # noqa: E402


# ------------------------------------------------------------------------------------------------- quy tắc chung
def test_seed_ids_of_the_five_published_datasets_are_unchanged():
    assert list(C.DATASETS)[:5] == ["tcga_luad_lusc_primary_tumor", "gse10245", "gse30219", "gse37745", "gse42127"]
    assert [C.DATASET_SEED_IDS[d] for d in list(C.DATASETS)[:5]] == [100, 101, 102, 103, 104]
    assert list(C.DATASETS)[5:] == list(C.B7_DATASETS) and C.MAIN_DATASETS == tuple(list(C.DATASETS)[:5])


def test_ktsp_direction_vote_and_tie_rule():
    X = np.array([[2, 1, 0, 5], [3, 1, 9, 5], [0, 4, 0, 1], [1, 5, 0, 2]], dtype=float)
    y = np.array([1, 1, 0, 0])
    pairs = [(0, 1), (2, 3)]
    dirs, ties = R.ktsp_directions(X, y, pairs)
    # cặp (0,1): x0 > x1 ở lớp 1 (2/2), không ở lớp 0 -> +1; cặp (2,3): x2 > x3 lớp 1 1/2, lớp 0 0/2 -> +1
    assert dirs.tolist() == [1, 1] and not ties.any()
    score, pred = R.ktsp_vote(X, pairs, dirs)
    assert score.tolist() == [0.5, 1.0, 0.0, 0.0]
    assert pred.tolist() == [1, 1, 0, 0]                    # mẫu 0 hoà 0,5: cặp đầu bỏ phiếu lớp 1
    dirs_rev = np.array([-1, 1])
    s2, p2 = R.ktsp_vote(X, pairs, dirs_rev)
    assert s2.tolist() == [0.0, 0.5, 0.5, 0.5] and p2.tolist() == [0, 0, 1, 1]   # hoà: cặp đầu (đảo) quyết định
    assert R.ktsp_pairs_from_panel([4, 7, 1, 2]) == [(4, 7), (1, 2)]
    with pytest.raises(ValueError):
        R.ktsp_pairs_from_panel([1, 2, 3])


def test_evaluate_vote_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)); y = (rng.random(40) < 0.5).astype(int); y[:2] = [0, 1]
    X[:, 0] += 2 * y
    pairs = [(0, 1), (2, 3)]
    a = R.evaluate_vote(X[:20], y[:20], X[20:], y[20:], pairs)
    b = R.evaluate_vote(np.exp(X[:20]), y[:20], np.exp(X[20:]), y[20:], pairs)
    assert a["auroc"] == b["auroc"] and a["directions"] == b["directions"]


def test_split_label_rule():
    assert R.split_label(0.01, 0.03, 0.01, 0.05) == "positive transfer"
    assert R.split_label(0.01, -0.03, -0.05, -0.01) == "negative transfer"
    assert R.split_label(0.5, 0.001, -0.004, 0.006) == "practically equivalent"
    assert R.split_label(0.5, 0.02, -0.01, 0.05) == "inconclusive"
    assert R.split_label(0.01, 0.005, 0.001, 0.009) == "practically equivalent"   # p < 0,05 nhưng |Δ| ≤ 0,01


def test_split_level_labels_average_subsets_and_holm_within_strategy():
    rows = []
    for split in range(20):
        for subset in range(4):
            for k in (2, 5):
                base = 0.8 + 0.001 * subset
                rows += [{"cohort": "c", "n_t": "5", "k": k, "split": split, "subset": subset, "strategy": "target_only", "auroc": base},
                         {"cohort": "c", "n_t": "5", "k": k, "split": split, "subset": subset, "strategy": "pooled",
                          "auroc": base + 0.05 + 0.001 * split},
                         {"cohort": "c", "n_t": "5", "k": k, "split": split, "subset": subset, "strategy": "source_only",
                          "auroc": base + (0.002 if split % 2 else -0.002)}]
    lab = R.split_level_labels(pd.DataFrame(rows), "strategy", "target_only", ["cohort", "n_t", "k"], seed_tag="t")
    assert set(lab.strategy) == {"pooled", "source_only"} and (lab.n_splits == 20).all()
    assert (lab[lab.strategy == "pooled"].label == "positive transfer").all()
    assert (lab[lab.strategy == "source_only"].label == "practically equivalent").all()
    again = R.split_level_labels(pd.DataFrame(rows), "strategy", "target_only", ["cohort", "n_t", "k"], seed_tag="t")
    pd.testing.assert_frame_equal(lab, again)                       # CI bootstrap tất định


def test_compare_rows_gate():
    old = pd.DataFrame({"a": [1, 2], "auroc": [0.5, np.nan]})
    assert R.compare_rows(old.assign(auroc=[0.5 + 1e-13, np.nan]), old, ["a"])["pass"]
    assert not R.compare_rows(old.assign(auroc=[0.51, np.nan]), old, ["a"])["pass"]
    assert not R.compare_rows(old.iloc[:1], old, ["a"])["pass"]


def test_fewshot_manifest_with_20_seeds_keeps_the_first_five_splits():
    from survey_v2.fewshot import build_fewshot_manifest
    rng = np.random.default_rng(3)
    n = 90
    y = (rng.random(n) < 0.4).astype(int); y[:2] = [0, 1]
    ids = np.array([f"S{i:03d}" for i in range(n)])
    m5 = build_fewshot_manifest("gse37745", ids, ids, y)
    m20 = build_fewshot_manifest("gse37745", ids, ids, y, seeds=R.B1_SEEDS)
    assert m20.split.nunique() == 20 and sorted(m20.seed.unique()) == list(range(42, 62))
    pd.testing.assert_frame_equal(m20[m20.split < 5].reset_index(drop=True), m5.reset_index(drop=True))


def test_zscore_helpers():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    Z = R.within_sample_zscore(X)
    assert np.allclose(Z[0], [-1.224744871, 0, 1.224744871]) and np.allclose(Z[1], 0)
    G = R.per_gene_zscore(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.allclose(G[:, 0], [-1, 1]) and np.allclose(G[:, 1], 0)


def test_b5_top_f_genes_keeps_signature_genes_only():
    b5 = importlib.import_module("run_b5_signatures10")
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 20)
    X = rng.normal(size=(40, 30)); X[:, 7] += 3 * y; X[:, 3] += 2 * y
    cols = np.array([3, 5, 7, 9, 11])
    top = b5.top_f_genes(X, y, cols, k=2)
    assert set(top) == {3, 7}
    assert b5.top_f_genes(X, y, cols[:2], k=10).tolist() == [3, 5]
    assert b5.method_name("panel_girard2016") == "sig10_girard2016"


def test_kaggle_jobs_registry_has_revision_jobs_with_prepare_steps():
    kj = importlib.import_module("kaggle_jobs")
    for name in ("b2_b3_b5", "b4", "b1", "b7_e1_phaseA", "b7_e1_phaseB", "b7_e4"):
        assert name in kj.JOBS
    b1 = kj.JOBS["b1"]
    assert b1.prepare and "--gate-against" in b1.prepare[0] and "62" not in b1.prepare[0] and "61" in b1.prepare[0]
    assert all("fewshot_splits_20.csv" in s for s in b1.shards)
    assert kj.JOBS["b7_e1_phaseA"].prepare == [["scripts/b7_restore_data.py"]]
    assert not any(m in kj.JOBS["b7_e1_phaseA"].shards[0] for m in ("ihc_2", "ihc_4", "panel_girard2016", "panel_wilkerson2013"))


def test_e7_config_note_moved_ptlasso_to_survey():
    src = (REPO / "scripts" / "run_survey_e7_transfer_naive.py").read_text(encoding="utf-8")
    assert "ptLasso and Trans-GLM are reported in the survey (B6, QĐ176); MVP belongs to the method paper" in src
    assert "no proposed method here" not in src


# ------------------------------------------------------------------------------------------------- chuỗi đầu-cuối (slow)
COHORT, SMALL = "gse37745", "gse10245"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("kltn_rev")
    pub = root / "results" / "survey_v2"
    mp.setenv("KLTN_DATA_ROOT", str(root))
    mp.setenv("KLTN_PUBLISHED_ROOT", str(pub))
    mp.setenv("KLTN_ALLOW_UNPUSHED", "1")
    mp.setenv("KLTN_ALLOW_UNTRACKED", "1")
    mp.setenv("KLTN_STRICT_N", "0")
    for m in list(sys.modules):
        if m.startswith("scripts.") or m == "scripts" or m.startswith("survey_v2") or m.startswith("run_") or m.startswith("make_") \
                or m.startswith("build_") or m in ("provenance", "kaggle_jobs", "b7_restore_data"):
            del sys.modules[m]
    from survey_v2 import config as C2, synthetic
    synthetic.make_all(root, n=120, p=300, effect=0.9, n_target=80)
    # hai bộ B7 giả: chép layout perm/ rồi chuyển vào experiment-data-b7/
    for name, seed in zip(C2.B7_DATASETS, (31, 32)):
        d = synthetic.make_dataset(root, name, n=90, p=300, effect=0.8, seed=seed, with_folds=False, prevalence=0.4,
                                   genes=np.array([f"G{j:05d}" for j in range(300)]))
        dst = root / C2.B7_DATA_DIR / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dst))
    yield root
    mp.undo()


def _m(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.slow
def test_revision_pipeline_end_to_end(pipeline):
    root = pipeline
    res = root / "results" / "survey_v2"
    small = ["--prefilter", "120"]
    cohorts = [COHORT, SMALL]
    # ---- "kết quả đã công bố" giả
    _m("build_e1_repeated_splits").main(["--datasets", C.SOURCE_DATASET, *cohorts])
    _m("build_fewshot_splits").main(["--datasets", *cohorts])
    fs = _m("run_e1_feature_selection")
    fs.main(["--datasets", C.SOURCE_DATASET, *cohorts, "--methods", "f_test", "l1_lr", "ktsp", "ihc_4", "--k", "2", "4", "5", "10",
             "--repeats", "0", *small])
    _m("run_e1_classification").main(["--datasets", C.SOURCE_DATASET, *cohorts, "--repeats", "0", "--random-draws", "2", *small])
    _m("run_survey_e1b_learning_curve").main(["--datasets", *cohorts, "--methods", "f_test", "l1_lr", "ktsp", "ihc_4", "--k", "2", "4", "10",
                                             "--random-draws", "2", "--n-jobs", "1", *small])
    _m("run_survey_e7_transfer_naive").main(["--datasets", *cohorts, "--k", "2", "4", "10", "--n-t", "5", "--n-jobs", "1", *small])
    # ---- B1: bản chia 7 lần (seed 42–48), cổng 5 lần chia, E7 và E1b trên bản mới
    b1_root = res / "B1_splits"
    _m("build_fewshot_splits").main(["--datasets", *cohorts, "--seeds", *[str(s) for s in range(42, 49)], "--name", "fewshot_splits_20.csv",
                                     "--out-root", str(b1_root), "--manifest-name", "m20.json", "--gate-against", "fewshot_splits.csv"])
    man = pd.read_csv(b1_root / COHORT / "fewshot_splits_20.csv", dtype={"n_t": str})
    assert man.split.nunique() == 7
    b1 = ["--splits-root", str(b1_root), "--splits-file", "fewshot_splits_20.csv"]
    _m("run_survey_e7_transfer_naive").main(["--datasets", *cohorts, "--k", "2", "4", "10", "--n-t", "5", "--n-jobs", "1", *small, *b1,
                                             "--out", "B1_E7_20"])
    _m("run_survey_e1b_learning_curve").main(["--datasets", *cohorts, "--methods", "f_test", "l1_lr", "ktsp", "ihc_4", "--k", "2", "4", "10",
                                             "--random-draws", "2", "--n-jobs", "1", *small, *b1, "--out", "B1_E1b_20"])
    cfg = json.loads((res / "B1_E7_20" / COHORT / "config.json").read_text(encoding="utf-8"))
    assert cfg["fewshot_splits_file"].endswith("fewshot_splits_20.csv") and "B6, QĐ176" in cfg["note"]
    # ---- B2, B3, B4, B5
    _m("run_b2_ktsp_vote").main(["--datasets", C.SOURCE_DATASET, *cohorts, "--n-t", "5", *small])
    b2 = pd.read_csv(res / "B2_ktsp_vote" / COHORT / "b2_e7_per_run.csv", dtype={"n_t": str})
    assert set(b2.strategy) == {"source_only", "select_on_source"} and b2.auroc_vote.between(0, 1).all()
    e1 = pd.read_csv(res / "B2_ktsp_vote" / COHORT / "b2_e1_per_fold.csv")
    assert set(e1.k) == {2, 4, 10} and e1.auroc_lr.notna().all() and (e1.n_pairs == e1.k // 2).all()
    _m("run_b3_zero_shot").main(["--datasets", *cohorts, "--k", "2", "4", "10", *small])
    b3cfg = json.loads((res / "B3_zero_shot" / COHORT / "config.json").read_text(encoding="utf-8"))
    assert b3cfg["source_panels_match_published_e7"] is True
    _m("run_b4_b0_variants").main(["--datasets", C.SOURCE_DATASET, *cohorts, "--repeats", "0", *small])
    b4 = pd.read_csv(res / "B4_b0_variants" / COHORT / "b4_per_fold.csv")
    assert set(b4.method) == {"b0", "b0_full_wide", "b0_p5000", "b0_p5000_wide"} and (b4[b4.method == "b0_p5000"].n_features == 120).all()
    panels_csv = root / "sigs.csv"
    genes = [f"G{j:05d}" for j in range(0, 300, 7)]
    pd.DataFrame({"panel": ["girard2016"] * 20 + ["wilkerson2013"] * 23, "gene": genes[:20] + genes[20:43], "source": "test"}).to_csv(panels_csv, index=False)
    _m("run_b5_signatures10").main(["--datasets", C.SOURCE_DATASET, *cohorts, "--panels-csv", str(panels_csv)])
    b5 = pd.read_csv(res / "B5_signatures10" / COHORT / "b5_e1_per_fold.csv")
    assert set(b5.method) == {"sig10_girard2016", "sig10_wilkerson2013"} and (b5.n_selected == 10).all()
    # ---- B7: E1 pha A/B, E4 (nhỏ)
    b7 = list(C.B7_DATASETS)
    _m("build_e1_repeated_splits").main(["--datasets", *b7])
    _m("run_e1_feature_selection").main(["--datasets", *b7, "--methods", "f_test", "l1_lr", "ktsp", "--k", "2", "4", "10", "--repeats", "0",
                                         "--out", "E1_feature_selection_b7", *small])
    _m("run_e1_classification").main(["--datasets", *b7, "--repeats", "0", "--random-draws", "2", "--save-scores", "--out", "E1_classification_b7",
                                      "--selection-roots", str(res / "E1_feature_selection_b7"), *small])
    _m("run_survey_e4_e5_stability").main(["--datasets", b7[0], "--methods", "f_test", "l1_lr", "--k", "2", "10", "--m", "3", "--fractions", "0.7",
                                           "--skip-e5a", "--skip-e5b", "--out", "E4_E5_b7", "--n-jobs", "1", *small])
    _m("run_survey_e4_e5_stability").main(["--datasets", C.SOURCE_DATASET, "--methods", "f_test", "l1_lr", "--k", "2", "10", "--m", "3",
                                           "--fractions", "0.7", "--skip-e5a", "--skip-e5b", "--n-jobs", "1", *small])
    # ---- bảng (cohort giả: chỉ hai cohort GEO có dữ liệu -> thu hẹp danh sách cohort của script bảng)
    _m("make_revision_tables").main(["b1", "b2", "b3", "b4", "b5", "b7", "--fewshot-cohorts", *cohorts,
                                     "--main-datasets", C.SOURCE_DATASET, *cohorts, "--kstar-cohorts", *cohorts])
    gates = pd.read_csv(res / "B1_tables" / "b1_gates.csv")
    assert gates["pass"].all(), gates
    assert pd.read_csv(res / "B3_tables" / "b3_gate_e7_source_only.csv")["pass"].all()
    assert pd.read_csv(res / "B4_tables" / "b4_gate_b0.csv")["pass"].all()
    lab = pd.read_csv(res / "B1_tables" / "b1_e7_split_labels.csv")
    assert (lab.n_splits == 7).all() and set(lab.label) <= {"positive transfer", "negative transfer", "practically equivalent", "inconclusive"}
    assert (res / "B2_tables" / "b2_source_only_reversal.csv").exists()
    assert (res / "B5_tables" / "b5_noninferiority_b0.csv").exists()
    r1 = pd.read_csv(res / "B7_tables" / "b7_R1.csv")
    assert set(r1.cohort) == set(b7)
    assert "auprc" in pd.read_csv(res / "B7_tables" / "b7_e1_summary.csv").columns
    assert len(pd.read_csv(res / "B7_tables" / "b7_R4_kendall_stability.csv")) == 1
    for d in ("B1_tables", "B2_tables", "B3_tables", "B4_tables", "B5_tables", "B7_tables"):
        c = json.loads((res / d / "config.json").read_text(encoding="utf-8"))
        assert c["published_root"] == str(res) and c["inputs_sha256"]


@pytest.mark.slow
def test_b1_gate_stops_when_published_splits_differ(pipeline):
    root = pipeline
    d = root / C.GEO_TARGET_DIR / SMALL
    ref = pd.read_csv(d / "fewshot_splits.csv", dtype={"n_t": str})
    ref.loc[0, "sample_id"] = "changed"
    ref.to_csv(d / "fewshot_splits_bad.csv", index=False)
    with pytest.raises(SystemExit):
        _m("build_fewshot_splits").main(["--datasets", SMALL, "--seeds", "42", "43", "--name", "x.csv", "--out-root", str(root / "tmp_b1"),
                                         "--gate-against", "fewshot_splits_bad.csv"])


def test_b7_restore_data_checks_sha(tmp_path):
    rd = importlib.import_module("b7_restore_data")
    from survey_v2.data import sha256
    src = tmp_path / "results" / "survey_v2" / "B7_data"
    files = {}
    for name in C.B7_DATASETS:
        f = src / C.B7_DATA_DIR / name / "expression.h5"
        f.parent.mkdir(parents=True)
        f.write_bytes(name.encode())
        files[f"{C.B7_DATA_DIR}/{name}/expression.h5"] = sha256(f)
    (src / "config.json").write_text(json.dumps({"traceable": True, "commit_hash": "abc", "files_sha256": files}), encoding="utf-8")
    rep = rd.restore(tmp_path)
    assert rep["files"] == 2 and (tmp_path / C.B7_DATA_DIR / C.B7_DATASETS[0] / "expression.h5").exists()
    assert rd.restore(tmp_path)["files"] == 2                       # chạy lại: chỉ kiểm
    (tmp_path / C.B7_DATA_DIR / C.B7_DATASETS[0] / "expression.h5").write_bytes(b"x")
    with pytest.raises(SystemExit):
        rd.restore(tmp_path)
