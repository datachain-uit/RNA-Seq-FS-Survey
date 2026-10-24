"""Test cho scripts/provenance.py (sửa O3, Y8 — Review 20/10/2026, §7.7)."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.provenance import (ENV_ALLOW_UNTRACKED, ProvenanceError, provenance, require,
                                save_provenance, stamp)

HAS_GIT = shutil.which("git") is not None


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class TestProvenanceNoGit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    def test_stops_when_not_a_git_clone(self):
        with mock.patch.dict(os.environ, {ENV_ALLOW_UNTRACKED: ""}):
            with self.assertRaises(ProvenanceError):
                provenance(repo_dir=self.tmp)

    def test_require_exits_with_code_2(self):
        with mock.patch.dict(os.environ, {ENV_ALLOW_UNTRACKED: ""}), \
                mock.patch("scripts.provenance.REPO", self.tmp):
            with self.assertRaises(SystemExit) as ctx:
                require(self.tmp / "x.py", repo_dir=self.tmp)
        self.assertEqual(2, ctx.exception.code)

    def test_env_flag_allows_untracked_run_and_marks_it(self):
        with mock.patch.dict(os.environ, {ENV_ALLOW_UNTRACKED: "1"}):
            with self.assertWarns(UserWarning):
                prov = provenance(repo_dir=self.tmp, data_files=[self.tmp / "data.csv"])
        self.assertIsNone(prov["commit_hash"])
        self.assertFalse(prov["tracked"])
        self.assertIn(ENV_ALLOW_UNTRACKED, prov["untracked_reason"])
        self.assertEqual(["data.csv"], list(prov["data_sha256"]))
        self.assertEqual(64, len(prov["data_sha256"]["data.csv"]))

    def test_records_environment(self):
        prov = provenance(repo_dir=self.tmp, allow_untracked=True)
        for key in ("python", "packages", "machine", "started_at", "argv", "schema"):
            self.assertIn(key, prov)
        self.assertIn("pandas", prov["packages"])

    def test_stamp_and_save(self):
        prov = provenance(repo_dir=self.tmp, allow_untracked=True)
        cfg = stamp({"a": 1}, prov)
        self.assertIsNone(cfg["commit_hash"])
        self.assertIs(cfg["provenance"], prov)
        path = save_provenance(self.tmp / "out", prov, outputs=[self.tmp / "data.csv"])
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("finished_at", saved)
        self.assertEqual(1, len(saved["outputs_sha256"]))


@unittest.skipUnless(HAS_GIT, "cần lệnh git")
class TestProvenanceGit(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        git("init", "-q", cwd=self.repo)
        git("config", "user.email", "t@example.com", cwd=self.repo)
        git("config", "user.name", "t", cwd=self.repo)
        (self.repo / "run.py").write_text("print(1)\n", encoding="utf-8")
        git("add", "run.py", cwd=self.repo)
        git("commit", "-q", "-m", "init", cwd=self.repo)
        self.head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, check=True,
                                   capture_output=True, text=True).stdout.strip()

    def test_clean_repo_records_real_commit(self):
        prov = provenance(script=self.repo / "run.py", repo_dir=self.repo)
        self.assertEqual(self.head, prov["commit_hash"])
        self.assertTrue(prov["tracked"])
        self.assertFalse(prov["dirty"])
        self.assertEqual("run.py", prov["script"])

    def test_untracked_new_files_do_not_block(self):
        (self.repo / "result.csv").write_text("x\n", encoding="utf-8")
        prov = provenance(repo_dir=self.repo)
        self.assertEqual(self.head, prov["commit_hash"])

    def test_uncommitted_change_stops(self):
        (self.repo / "run.py").write_text("print(2)\n", encoding="utf-8")
        with self.assertRaises(ProvenanceError):
            provenance(repo_dir=self.repo)

    def test_allow_dirty_records_dirty_files(self):
        (self.repo / "run.py").write_text("print(2)\n", encoding="utf-8")
        prov = provenance(repo_dir=self.repo, allow_dirty=True)
        self.assertTrue(prov["dirty"])
        self.assertTrue(any("run.py" in f for f in prov["dirty_files"]))

    def test_env_flag_does_not_bypass_dirty_check(self):
        (self.repo / "run.py").write_text("print(2)\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {ENV_ALLOW_UNTRACKED: "1"}):
            with self.assertRaises(ProvenanceError):
                provenance(repo_dir=self.repo)


if __name__ == "__main__":
    unittest.main()
