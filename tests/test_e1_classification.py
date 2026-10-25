"""Tests for the isolated E1 classification phase."""

import numpy as np

from survey_v2.e1_classification import METRIC_NAMES, compute_metrics, tune_and_evaluate


def test_compute_metrics_contains_phase1_metrics_and_binary_rates():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = compute_metrics(y, pred, scores)

    assert set(METRIC_NAMES) <= set(metrics)
    assert metrics["accuracy"] == 0.75
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["roc_auc"] == 1.0


def test_tuning_uses_only_outer_train_and_evaluates_outer_test():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(40, 8))
    y_train = np.array([0, 1] * 20)
    X_train[:, 0] += 2 * y_train
    X_test = rng.normal(size=(20, 8))
    y_test = np.array([0, 1] * 10)
    X_test[:, 0] += 2 * y_test

    result = tune_and_evaluate("logistic_regression", X_train, y_train, X_test, y_test, seed=2, quick=True)

    assert result["model"] == "logistic_regression"
    assert result["n_tuning_configs"] >= 1
    assert 0 <= result["test_roc_auc"] <= 1
    assert result["test_roc_auc"] > 0.8
    assert "val_f1_macro" in result and "confusion_matrix_test" in result
