import unittest

import pandas as pd

from scripts.rebuild_gse30219_r5 import assign_labels, build_fold_assignments


class TestGse30219R5Rebuild(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame(
            {
                "patient_id": [f"P{i:02d}" for i in range(22)],
                "Sample_title": ["Lung cancer"] * 22,
                "histology": ["ADC"] * 10 + ["SQC"] * 10 + ["SCC"] * 2,
            },
            index=[f"GSM{i:03d}" for i in range(22)],
        )

    def test_keeps_only_explicit_luad_and_lusc_labels(self):
        labelled, excluded = assign_labels(self.metadata)

        self.assertEqual({"LUAD": 10, "LUSC": 10}, labelled["class"].value_counts().to_dict())
        self.assertEqual({"SCC": 2}, excluded["histology_raw"].value_counts().to_dict())
        self.assertEqual({0, 1}, set(labelled["label"]))

    def test_each_fold_has_disjoint_stratified_train_val_and_test_sets(self):
        labelled, _ = assign_labels(self.metadata)
        assignments, _ = build_fold_assignments(labelled, n_splits=5, seed_range=range(42, 43))

        for fold in range(1, 6):
            roles = assignments[f"fold_{fold}"]
            self.assertEqual({"train", "val", "test"}, set(roles))
            for split in ("train", "val", "test"):
                self.assertEqual(2, assignments.loc[roles == split, "label"].nunique())

        self.assertTrue(((assignments.filter(like="fold_") == "test").sum(axis=1) == 1).all())
        self.assertTrue(((assignments.filter(like="fold_") == "val").sum(axis=1) == 1).all())


if __name__ == "__main__":
    unittest.main()
