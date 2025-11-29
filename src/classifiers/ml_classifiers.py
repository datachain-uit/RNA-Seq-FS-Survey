import os
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.metrics.classification_metrics import compute_classification_metrics


def run_all_classifiers(X_train, X_test, y_train, y_test, fs_info):
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=200, solver='lbfgs', random_state=42),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        ),
        "SVM": SVC(C=0.1, kernel="linear", probability=True, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")

        clf = Pipeline([("clf", model)])
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        cls_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

        cls_metrics.update(fs_info)        # add FS info
        cls_metrics["Classifier"] = name
        cls_metrics["NumFeatures"] = X_train.shape[1]
        cls_metrics["Timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        results.append(cls_metrics)

        print(f"{name}: Acc={cls_metrics['Accuracy']:.4f}, MacroF1={cls_metrics['MacroF1']:.4f}")

    return pd.DataFrame(results)
