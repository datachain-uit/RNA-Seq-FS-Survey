"""scripts/kaggle_jobs.py: mọi lệnh trong danh mục trỏ tới script có thật và được parser của script chấp nhận; gộp E2/E3."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass cần module trong sys.modules
    spec.loader.exec_module(mod)
    return mod


kj = _load(REPO / "scripts" / "kaggle_jobs.py")


def _all_commands():
    for job in kj.JOBS.values():
        for argv in job.shards + job.finalize:
            yield job.name, argv


@pytest.mark.parametrize("job,argv", list(_all_commands()), ids=lambda v: v if isinstance(v, str) else " ".join(v[:2]))
def test_every_command_parses(job, argv, monkeypatch, tmp_path):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    script = REPO / argv[0]
    assert script.is_file(), f"{job}: {argv[0]} không tồn tại"
    if script.name == "kaggle_jobs.py":
        assert argv[1] in ("merge-e2e3",)
        return
    args = kj.render(argv)[1:]
    mod = _load(script)
    import inspect
    if inspect.signature(mod.parse_args).parameters:
        mod.parse_args(args)
    else:
        monkeypatch.setattr(sys, "argv", [script.name, *args])
        mod.parse_args()


def test_catalog_shape():
    names = set(kj.JOBS)
    for job in kj.JOBS.values():
        assert job.shards, job.name
        for dep in job.restore_from:
            assert dep in names or dep.startswith("("), (job.name, dep)
        if job.exclusive:
            assert job.parallel == 1
    timing = [j for j in kj.JOBS.values() if any("run_survey_e2_e3_timing.py" in a[0] for a in j.shards)]
    assert timing and all(j.exclusive for j in timing)


def test_run_parallel_reports_exit_codes(tmp_path):
    ok = [sys.executable, "-c", "print('hi')"]
    bad = [sys.executable, "-c", "import sys; sys.exit(4)"]
    assert kj.run_parallel([ok, bad, ok], 2, tmp_path, "t") == [0, 4, 0]
    assert (tmp_path / "t_0.log").read_text().strip() == "hi"


def test_merge_e2e3(tmp_path, monkeypatch):
    monkeypatch.setenv("KLTN_DATA_ROOT", str(tmp_path))
    root = tmp_path.joinpath("results", "survey_v2")
    for p, m in [("E2_E3_p1", "f_test"), ("E2_E3_p2", "l1_lr"), ("E2_E3_p3", "relieff")]:
        d = root / p
        d.mkdir(parents=True)
        pd.DataFrame({"method": [m], "wall_s": [1.0]}).to_csv(d / "e2_cost_point.csv", index=False)
        (d / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": True, "commit_hash": "abc", "methods": [m]}))
    out = kj.merge_e2e3()
    merged = pd.read_csv(out / "e2_cost_point.csv")
    assert merged.method.tolist() == ["f_test", "l1_lr", "relieff"] and set(merged.session) == {"E2_E3_p1", "E2_E3_p2", "E2_E3_p3"}
    assert json.loads((out / "config.json").read_text())["commit_hash"] == "abc"
    (root / "E2_E3_p2" / "config.json").write_text(json.dumps({"traceable": True, "calibration_ok": False, "commit_hash": "abc"}))
    with pytest.raises(SystemExit):
        kj.merge_e2e3()


def test_run_parallel_stops_at_deadline(tmp_path):
    import time
    slow = [sys.executable, "-c", "import time; time.sleep(60)"]
    t0 = time.time()
    codes = kj.run_parallel([slow, slow, slow], 2, tmp_path, "slow", deadline=time.time() + 3)
    assert codes == [kj.TIMEOUT_CODE] * 3 and time.time() - t0 < 40


def test_runner_notebook_only_dispatches_to_kaggle_jobs():
    nb = json.loads((REPO / "notebooks" / "kaggle_survey_v3_runner.ipynb").read_text(encoding="utf-8"))
    code = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "scripts/kaggle_jobs.py" in code and "JOB = " in code
    assert "scripts/run_" not in code and "make_survey_tables" not in code      # M18: không có lệnh thí nghiệm trong notebook
    assert "KLTN_STRICT_N" in code and "'traceable'" not in code               # traceable do script kiểm, không gán tay


mk = _load(REPO / "scripts" / "make_kaggle_notebooks.py")


def test_per_experiment_notebooks_cover_every_job_once_and_are_up_to_date():
    covered = [n for e in mk.EXPERIMENTS for n in e.jobs]
    assert sorted(covered) == sorted(kj.JOBS) and len(covered) == len(set(covered))
    assert mk.main(["--check"]) == 0          # notebooks/kaggle/*.ipynb khớp runner + danh mục (sinh lại nếu lệch)


@pytest.mark.parametrize("exp", mk.EXPERIMENTS, ids=lambda e: e.file)
def test_per_experiment_notebook_only_dispatches(exp):
    nb = json.loads((REPO / "notebooks" / "kaggle" / exp.file).read_text(encoding="utf-8"))
    code = "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert f"JOB = {exp.jobs[0]!r}" in code and "assert JOB in JOBS_IN_NOTEBOOK" in code
    assert "scripts/kaggle_jobs.py" in code and "scripts/run_" not in code and "make_survey_tables" not in code
    assert "KLTN_STRICT_N" in code and "'traceable'" not in code
