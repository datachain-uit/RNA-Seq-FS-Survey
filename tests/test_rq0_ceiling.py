"""Regression test cho quy tắc trần RQ0′ ở mức cohort (sửa O14, Review 20/10/2026)."""

import unittest

import pandas as pd

from scripts.run_rq0_prime import CEILING_SET, ceiling_by_cohort, summarize


def summary_rows(rows):
    return pd.DataFrame(rows, columns=["cohort", "train_per_class", "method", "auroc_mean"])


def full_cohort(cohort, single, ihc2, **others):
    """Một cohort với đủ phương pháp ở n_T = 5 và train đủ."""
    base = {"best_single_gene": single, "IHC_2": ihc2, "IHC_4": 0.99,
            "B0_all_genes": 0.99, "random_k_1": 0.55, "random_k_2": 0.60}
    base.update(others)
    rows = [(cohort, "5", m, v) for m, v in base.items()]
    rows += [(cohort, "all", m, 0.999) for m in base]
    return rows


class TestCeilingByCohort(unittest.TestCase):
    def test_ceiling_set_only_contains_k_le_2_selection_methods(self):
        self.assertEqual(("best_single_gene", "IHC_2"), CEILING_SET)

    def test_both_methods_reach_threshold_means_ceiling(self):
        out = ceiling_by_cohort(summary_rows(full_cohort("A", 0.985, 0.99)))
        self.assertEqual([True], out["ceiling_hit"].tolist())

    def test_one_method_below_threshold_means_no_ceiling(self):
        out = ceiling_by_cohort(summary_rows(full_cohort("A", 0.85, 0.99)))
        self.assertEqual([False], out["ceiling_hit"].tolist())
        self.assertEqual("best_single_gene", out.loc[0, "min_method"])
        self.assertAlmostEqual(0.85, out.loc[0, "min_auroc"])

    def test_threshold_is_inclusive(self):
        out = ceiling_by_cohort(summary_rows(full_cohort("A", 0.98, 0.98)))
        self.assertEqual([True], out["ceiling_hit"].tolist())

    def test_random_k_b0_and_ihc4_do_not_affect_decision(self):
        # Lỗi cũ: random-k (mức sàn) làm điều kiện "mọi phương pháp" không bao giờ thỏa.
        rows = full_cohort("A", 0.99, 0.99, random_k_1=0.50, random_k_2=0.52,
                           B0_all_genes=0.70, IHC_4=0.60)
        out = ceiling_by_cohort(summary_rows(rows))
        self.assertEqual([True], out["ceiling_hit"].tolist())

    def test_full_training_results_are_ignored(self):
        rows = [r for r in full_cohort("A", 0.80, 0.80)]
        out = ceiling_by_cohort(summary_rows(rows))  # train 'all' đều 0.999 nhưng không được tính
        self.assertEqual([False], out["ceiling_hit"].tolist())

    def test_one_row_per_cohort(self):
        rows = full_cohort("A", 0.99, 0.99) + full_cohort("B", 0.70, 0.92)
        out = ceiling_by_cohort(summary_rows(rows))
        self.assertEqual({"A": True, "B": False}, dict(zip(out["cohort"], out["ceiling_hit"])))

    def test_missing_method_raises(self):
        rows = [r for r in full_cohort("A", 0.99, 0.99) if r[2] != "IHC_2"]
        with self.assertRaises(ValueError):
            ceiling_by_cohort(summary_rows(rows))

    def test_works_on_summarize_output_with_integer_train_size(self):
        # Trong main(), train_per_class là 5 (int) hoặc "all"; khi đọc lại từ CSV là "5".
        metrics = pd.DataFrame([
            {"cohort": "A", "train_per_class": tp, "method": m, "auroc": a,
             "balanced_accuracy": 0.9, "status": "OK"}
            for tp in (5, "all") for m in ("best_single_gene", "IHC_2", "random_k_1")
            for a in (0.99, 0.985)
        ])
        out = ceiling_by_cohort(summarize(metrics))
        self.assertEqual([True], out["ceiling_hit"].tolist())


if __name__ == "__main__":
    unittest.main()
