"""Regression tests for the Phase 0 hand-off manifest builder."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_review_completion_artifacts import collect_phase0_manifest
from scripts.run_rq0_prime import cohort_data_dir, sample_balanced_training_indices


class Phase0ManifestTests(unittest.TestCase):
    def test_collects_required_evidence_and_reports_missing_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "results" / "phase 0" / "permu_test" / "perm_test_v2_gse68465"
            run.mkdir(parents=True)
            for name in ("summary.csv", "real_scores_per_fold.csv", "real_confusion.csv", "config.json"):
                (run / name).write_text("ok", encoding="utf-8")
            (run / "config.json").write_text(json.dumps({"n_permutations": 1000}), encoding="utf-8")
            stage = root / "results" / "phase 0" / "stage_split"
            stage.mkdir(parents=True)
            (stage / "summary_no_fs_by_dataset_model.csv").write_text("dataset\n", encoding="utf-8")
            (stage / "split_metrics_per_fold.csv").write_text("dataset\n", encoding="utf-8")

            rows = collect_phase0_manifest(root)

            self.assertEqual(len(rows), 3)
            row = next(item for item in rows if item["dataset"] == "GSE68465")
            self.assertEqual(row["summary_csv"], "Có")
            self.assertEqual(row["real_scores_per_fold_csv"], "Có")
            self.assertEqual(row["real_confusion_csv"], "Có")
            self.assertEqual(row["stage_split_no_fs_summary"], "Có")
            self.assertEqual(row["stage_split_metrics_per_fold"], "Có")
            self.assertEqual(row["commit_hash"], "Không có (thư mục không phải Git)")
            self.assertEqual(row["handoff_status"], "Cần bổ sung commit hash")


class Rq0PrimeTests(unittest.TestCase):
    def test_gse30219_uses_the_r5_fixed_dataset_directory(self):
        root = Path("D:/temporary-rq0-root")

        self.assertEqual(
            root / "experiment-data-geo-target" / "gse30219_r5_fixed",
            cohort_data_dir(root, "gse30219"),
        )
        self.assertEqual(
            root / "experiment-data-geo-target" / "gse37745",
            cohort_data_dir(root, "gse37745"),
        )

    def test_balanced_training_sample_uses_exactly_five_per_class(self):
        indices = sample_balanced_training_indices([0] * 9 + [1] * 7, 5, seed=42)
        labels = ([0] * 9 + [1] * 7)
        self.assertEqual(len(indices), 10)
        self.assertEqual(sum(labels[i] == 0 for i in indices), 5)
        self.assertEqual(sum(labels[i] == 1 for i in indices), 5)


if __name__ == "__main__":
    unittest.main()
