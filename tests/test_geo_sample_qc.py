"""scripts/geo_sample_qc.py (QĐ21): rebuild bỏ mẫu cách ly + thêm cột sex, không đổi phần còn lại; sex-check; rz-screen."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("geo_sample_qc", REPO / "scripts" / "geo_sample_qc.py")
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


def _cohort(root: Path, name: str = "gse37745", n: int = 12):
    rng = np.random.default_rng(0)
    d = root / "experiment-data-geo-target" / name
    d.mkdir(parents=True)
    ids = [f"GSM{i:03d}" for i in range(n)]
    sex = ["male" if i % 2 else "female" for i in range(n)]
    genes = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY"] + [f"G{j}" for j in range(30)]
    X = rng.normal(5, 1, size=(n, len(genes)))
    for i, s in enumerate(sex):
        X[i, 0] = 12 if s == "female" else 4
        X[i, 1:6] = 4 if s == "female" else 11
    pd.DataFrame(X, index=pd.Index(ids, name="sample_id"), columns=genes).to_hdf(d / "expression.h5", key="expression", mode="w")
    meta = pd.DataFrame({"sample_id": ids, "Sample_characteristics_ch1_7": [f"gender: {s}" for s in sex],
                         "patient_id": ids, "class": ["LUAD", "LUSC"] * (n // 2), "label": [0, 1] * (n // 2)})
    meta.to_csv(d / "metadata.csv", index=False)
    pd.DataFrame({"sample_id": ["GSM001", "GSM002"], "reason": ["test", "test"]}).to_csv(d / f"excluded_samples_{name}.csv", index=False)
    return d, X, ids


def test_rebuild_drops_excluded_and_adds_sex_without_changing_the_rest(tmp_path):
    d, X, ids = _cohort(tmp_path)
    log = qc.rebuild(tmp_path, "gse37745", {"commit_hash": None})
    assert log["n_before"] == 12 and log["n_after"] == 10 and log["excluded_now"] == ["GSM001", "GSM002"]
    new_X = pd.read_hdf(d / "expression.h5", key="expression")
    meta = pd.read_csv(d / "metadata.csv")
    keep = [i for i, s in enumerate(ids) if s not in {"GSM001", "GSM002"}]
    assert list(new_X.index) == [ids[i] for i in keep] == meta.sample_id.tolist()
    assert np.array_equal(new_X.to_numpy(), X[keep])
    assert set(meta["sex"]) == {"male", "female"}
    assert (Path(log["backup_dir"]) / "expression.h5").exists()
    again = qc.rebuild(tmp_path, "gse37745", {"commit_hash": None})      # idempotent
    assert not again["changed"] and again["sha256_after"] == log["sha256_after"]


def test_sex_check_and_rz_screen(tmp_path):
    _cohort(tmp_path)
    qc.rebuild(tmp_path, "gse37745", {"commit_hash": None})
    t = qc.sex_check(tmp_path, "gse37745", unclear=2.0)
    assert (t.status == "match").all()
    pairs, summary = qc.rz_screen(tmp_path, ["gse37745"], threshold=0.9, raw_threshold=0.99)
    assert summary.loc[0, "scope"] == "within" and summary.loc[0, "n_pairs"] == 10 * 9 // 2


def test_sex_only_keeps_samples_and_apply_sex_qc_appends_quarantine(tmp_path):
    d, X, ids = _cohort(tmp_path)
    log = qc.rebuild(tmp_path, "gse37745", {"commit_hash": None}, sex_only=True)
    assert log["n_after"] == 12 and log["excluded_now"] == []
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    pd.DataFrame({"sample_id": ["GSM003", "GSM004", "GSM005"], "verdict": ["opposite", "inconsistent", "inconclusive"],
                  "category": ["F-like", "both-high", "both-low"], "meta_sex": ["male", "female", "male"],
                  "action": ["quarantine", "quarantine", "keep"]}).to_csv(qc_dir / "sex_check_gse37745.csv", index=False)
    res = qc.apply_sex_qc(tmp_path, qc_dir, ["gse37745"])
    assert res[0]["added"] == ["GSM003", "GSM004"] and res[0]["inconclusive"] == ["GSM005"]
    excl = pd.read_csv(d / "excluded_samples_gse37745.csv")
    assert excl.sample_id.tolist() == ["GSM001", "GSM002", "GSM003", "GSM004"]
    assert "không xác minh được danh tính mẫu" in excl.loc[3, "reason"]
    assert qc.apply_sex_qc(tmp_path, qc_dir, ["gse37745"])[0]["added"] == []      # chạy lại không thêm trùng
    final = qc.rebuild(tmp_path, "gse37745", {"commit_hash": None})
    assert final["n_after"] == 8
