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


def test_single_axis_vote_mode_when_xist_is_absent():
    import pandas as pd
    from scripts.check_sex_consistency import sex_table

    class DS:  # protein-coding only: no XIST
        pass
    rng = np.random.default_rng(3)
    n = 80
    genes = np.array(["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "G1"])
    X = rng.normal(5, 0.3, (n, 6))
    X[:40, 0:5] = rng.normal(0.1, 0.1, (40, 5)).clip(0)        # annotated females: Y genes ~0
    X[40:, 0:5] = rng.normal(5.0, 0.8, (40, 5))                # annotated males
    X[70:, 0:5] = rng.normal(1.5, 0.3, (10, 5))                # ... including a tail with reduced Y expression (tumour LOY)
    X[0, 0:5] = [6.6, 3.0, 1.6, 1.0, 2.4]                      # annotated female, all five genes male-level -> opposite
    X[1, 0:5] = [3.2, 0.2, 0.1, 0.05, 0.2]                     # annotated female, only RPS4Y1 up -> F-like, match
    X[2, 0:5] = [2.5, 2.2, 1.0, 1.0, 2.1]                      # annotated female, several genes up -> never silently 'match'
    X[40, 0:5] = [3.3, 0.9, 0.7, 0.2, 0.3]                     # annotated male with low Y (LOY) -> must NOT be opposite
    ds = DS(); ds.genes = genes; ds.X = X; ds.y = np.zeros(n, dtype=int); ds.sample_ids = np.array([f"T{i}" for i in range(n)])
    meta = pd.DataFrame({"sex": ["female"] * 40 + ["male"] * 40}, index=ds.sample_ids)
    table, info = sex_table(ds, meta)
    t = table.set_index("sample_id")
    assert info["mode"].startswith("single-axis") and info["threshold_rule"].startswith("midpoint")
    assert t.loc["T0", "verdict"] == "opposite" and t.loc["T0", "action"] == "quarantine"
    assert t.loc["T1", "verdict"] == "match" and t.loc["T1", "votes_male"] <= 1
    assert t.loc["T2", "verdict"] in ("inconclusive", "opposite")      # multi-gene elevation is never silently 'match'
    assert t.loc["T40", "verdict"] != "opposite"                       # LOY male keeps
    assert (table.verdict == "opposite").sum() <= 2 and (table[table.meta_sex == "male"].verdict == "opposite").sum() == 0
