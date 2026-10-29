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
