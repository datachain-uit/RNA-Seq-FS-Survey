import numpy as np

from scripts.check_sex_consistency import classify, normalise_sex, split_axis, verdict


def test_axis_split_finds_two_modes_and_rejects_flat_axis():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(6, 0.4, 60), rng.normal(12, 0.4, 40)])
    thr, lo, hi = split_axis(bimodal)
    assert thr is not None and 8 < thr < 10 and lo < 7 and hi > 11
    flat = rng.normal(6, 0.3, 100)
    assert split_axis(flat)[0] is None


def test_categories_and_verdicts_follow_qd22():
    assert classify(True, False) == "F-like" and classify(False, True) == "M-like"
    assert classify(True, True) == "both-high" and classify(False, False) == "both-low"
    assert classify(None, True) == "uninformative"
    assert verdict("F-like", "female") == "match" and verdict("F-like", "male") == "opposite"
    assert verdict("M-like", "female") == "opposite"
    assert verdict("both-high", "male") == "inconsistent" and verdict("both-high", None) == "inconsistent"
    assert verdict("both-low", "male") == "inconclusive" and verdict("uninformative", "female") == "inconclusive"
    assert verdict("F-like", None) == "no-metadata"
    assert normalise_sex("Female") == "female" and normalise_sex("M") == "male" and normalise_sex("") is None


def test_single_axis_mode_when_xist_is_absent(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.check_sex_consistency import sex_table

    class DS:  # protein-coding only: no XIST
        pass
    rng = np.random.default_rng(3)
    n = 40
    genes = np.array(["RPS4Y1", "DDX3Y", "G1", "G2"])
    X = rng.normal(5, 0.3, (n, 4))
    X[:20, 0:2] = rng.normal(0.05, 0.05, (20, 2))      # females
    X[20:, 0:2] = rng.normal(6.5, 0.3, (20, 2))        # males
    X[0, 0:2] = 6.6                                     # annotated female, male-level Y -> opposite
    X[1, 0:2] = 3.1                                     # annotated female, between the clusters -> inconclusive, keep
    ds = DS(); ds.genes = genes; ds.X = X; ds.y = np.zeros(n, dtype=int); ds.sample_ids = np.array([f"T{i}" for i in range(n)])
    meta = pd.DataFrame({"sex": ["female"] * 20 + ["male"] * 20}, index=ds.sample_ids)
    table, info = sex_table(ds, meta)
    assert info["mode"].startswith("single-axis")
    assert table.set_index("sample_id").loc["T0", "verdict"] == "opposite"
    t1 = table.set_index("sample_id").loc["T1"]
    assert t1["category"] == "intermediate(Y)" and t1["verdict"] == "inconclusive" and t1["action"] == "keep"
    assert (table.verdict == "opposite").sum() == 1
    assert sex_table(ds, meta, male_axis="RPS4Y1")[1]["male_axis"] == "RPS4Y1"
