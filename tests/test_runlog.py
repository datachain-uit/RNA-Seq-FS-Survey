"""Regression tests for survey_v2.runlog traceability checks."""

from pathlib import Path

from survey_v2 import runlog


def test_is_tracked_passes_repo_relative_path_to_git(monkeypatch, tmp_path):
    """Absolute script paths must work on Linux runners such as Kaggle."""
    repo = tmp_path / "repo"
    script = repo / "scripts" / "run_sanity_auroc.py"
    seen = []

    def fake_git(*args, cwd):
        seen.append((args, cwd))
        return "scripts/run_sanity_auroc.py"

    monkeypatch.setattr(runlog, "_git", fake_git)

    assert runlog.is_tracked(script, repo=repo)
    assert seen == [(("ls-files", "--error-unmatch", "scripts/run_sanity_auroc.py"), repo.resolve())]


def test_is_tracked_rejects_path_outside_repository(tmp_path):
    repo = tmp_path / "repo"
    outside = tmp_path / "other" / "run.py"

    assert not runlog.is_tracked(outside, repo=repo)
