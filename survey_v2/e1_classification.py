"""Models, inner-train tuning, and metrics for the separated E1 classifier phase.

This module never receives an outer-test row during tuning.  Feature panels are
provided by the independent feature-selection phase and are treated as fixed
within the outer split.
"""

from __future__ import annotations

import json
import time
from itertools import product

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


METRIC_NAMES = (
    "accuracy", "balanced_accuracy", "precision_macro", "precision_weighted", "recall_macro",
    "recall_weighted", "f1_macro", "f1_weighted", "mcc", "cohen_kappa", "roc_auc",
    "sensitivity", "specificity", "ppv", "npv",
)

CLASSIFICATION_MODELS = (
    "knn", "gaussian_nb", "logistic_regression", "decision_tree", "random_forest",
    "linear_svm", "rbf_svm", "mlp", "xgboost", "lightgbm", "stacking",
)


def _scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def compute_metrics(y_true, y_pred, scores) -> dict:
    """Phase-1 classification metrics plus binary clinical rates and confusion matrix."""
    y_true, y_pred, scores = np.asarray(y_true, int), np.asarray(y_pred, int), np.asarray(scores, float)
    kw = {"zero_division": 0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", **kw)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", **kw)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "confusion_matrix": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
    }


def _optional_estimator(model: str, params: dict, seed: int):
    if model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for model=xgboost; install requirements.txt") from exc
        return XGBClassifier(n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0, **params)
    if model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is required for model=lightgbm; install requirements.txt") from exc
        return LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, **params)
    raise ValueError(model)


def build_estimator(model: str, params: dict, seed: int):
    """Construct one deterministic model; scaling is fit only inside each fit."""
    p = dict(params)
    if model == "knn":
        return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(**p))])
    if model == "gaussian_nb":
        return GaussianNB(**p)
    if model == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000, random_state=seed, **p))])
    if model == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if model == "random_forest":
        return RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=seed, **p)
    if model == "linear_svm":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="linear", probability=True, random_state=seed, **p))])
    if model == "rbf_svm":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=seed, **p))])
    if model == "mlp":
        return Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(max_iter=500, random_state=seed, **p))])
    if model in {"xgboost", "lightgbm"}:
        return _optional_estimator(model, p, seed)
    if model == "stacking":
        bases = [
            ("lr", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000, random_state=seed))])),
            ("rf", RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed, class_weight="balanced")),
            ("svm", Pipeline([("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=seed))])),
        ]
        return StackingClassifier(estimators=bases, final_estimator=LogisticRegression(max_iter=5000, random_state=seed, **p),
                                  stack_method="predict_proba", cv=3, n_jobs=1)
    raise ValueError(f"unknown classifier: {model}")


def parameter_grid(model: str, *, quick: bool = False) -> list[dict]:
    """Predeclared grids adapted from the phase_1 classification notebooks."""
    grids = {
        "knn": {"n_neighbors": [5, 11] if quick else [3, 5, 7, 11, 15, 21], "weights": ["distance"] if quick else ["uniform", "distance"]},
        "gaussian_nb": {"var_smoothing": [1e-9, 1e-3] if quick else [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]},
        "logistic_regression": {"C": [0.01, 1.0] if quick else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "class_weight": ["balanced"] if quick else [None, "balanced"]},
        "decision_tree": {"max_depth": [None, 5] if quick else [None, 3, 5, 10, 20], "min_samples_leaf": [5] if quick else [1, 5, 10], "class_weight": ["balanced"] if quick else [None, "balanced"]},
        "random_forest": {"max_features": ["sqrt"], "min_samples_leaf": [1, 5] if quick else [1, 5, 10], "class_weight": ["balanced"]},
        "linear_svm": {"C": [1e-3, 1e-1] if quick else [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], "class_weight": ["balanced"] if quick else [None, "balanced"]},
        "rbf_svm": {"C": [1.0, 10.0] if quick else [0.1, 1.0, 10.0, 100.0], "gamma": ["scale"] if quick else ["scale", 1e-4, 1e-3, 1e-2], "class_weight": ["balanced"] if quick else [None, "balanced"]},
        "mlp": {"hidden_layer_sizes": [(100,)] if quick else [(64,), (256,), (256, 64)], "alpha": [1e-4, 1.0] if quick else [1e-4, 1e-2, 1.0, 10.0]},
        "xgboost": {"n_estimators": [200] if quick else [200, 500], "max_depth": [3] if quick else [3, 6], "learning_rate": [0.1] if quick else [0.05, 0.1], "subsample": [0.8], "colsample_bytree": [0.5] if quick else [0.5, 1.0]},
        "lightgbm": {"n_estimators": [200] if quick else [200, 500], "num_leaves": [15] if quick else [7, 15], "learning_rate": [0.1] if quick else [0.05, 0.1], "min_child_samples": [10] if quick else [5, 20], "class_weight": ["balanced"] if quick else [None, "balanced"]},
        "stacking": {"C": [1.0] if quick else [0.01, 0.1, 1.0, 10.0], "class_weight": ["balanced"]},
    }
    try:
        grid = grids[model]
    except KeyError as exc:
        raise ValueError(f"unknown classifier: {model}") from exc
    return [dict(zip(grid, values)) for values in product(*grid.values())]


def tune_and_evaluate(model: str, X_train, y_train, X_test, y_test, *, seed: int, quick: bool = False,
                      inner_folds: int = 3) -> dict:
    """Choose a configuration by inner-CV macro-F1 and evaluate exactly once on outer test."""
    X_train, y_train, X_test, y_test = map(np.asarray, (X_train, y_train, X_test, y_test))
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y_train)), y_train))
    best_params, best_score, tune_s = None, -np.inf, 0.0
    for params in parameter_grid(model, quick=quick):
        oof = np.empty(len(y_train), dtype=int)
        t0 = time.perf_counter()
        for tr, va in splits:
            est = build_estimator(model, params, seed)
            est.fit(X_train[tr], y_train[tr])
            oof[va] = est.predict(X_train[va])
        tune_s += time.perf_counter() - t0
        score = f1_score(y_train, oof, average="macro", zero_division=0)
        if score > best_score:
            best_params, best_score = params, float(score)
    t0 = time.perf_counter()
    best = build_estimator(model, best_params, seed)
    best.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = best.predict(X_test)
    scores = _scores(best, X_test)
    infer_s = time.perf_counter() - t0
    metrics = compute_metrics(y_test, pred, scores)
    test_metrics = {f"test_{key}": value for key, value in metrics.items() if key != "confusion_matrix"}
    return {
        "model": model, "best_params": json.dumps(best_params, sort_keys=True, default=str),
        "val_f1_macro": best_score, "n_tuning_configs": len(parameter_grid(model, quick=quick)),
        "tuning_time_s": tune_s, "train_time_s": train_s, "infer_time_test_s": infer_s,
        "infer_time_per_sample_ms": infer_s * 1000.0 / len(y_test),
        "confusion_matrix_test": metrics["confusion_matrix"],
        **test_metrics,
    }
