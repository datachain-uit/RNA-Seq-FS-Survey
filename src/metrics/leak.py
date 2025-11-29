import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def calc_sleak(missing_flags, labels):
    if missing_flags is None:
        return np.nan

    aucs = []
    skf = StratifiedKFold(3, shuffle=True, random_state=42)

    for tr, te in skf.split(missing_flags, labels):
        clf = LogisticRegression(max_iter=200)
        clf.fit(missing_flags[tr], labels[tr])
        y_prob = clf.predict_proba(missing_flags[te])
        aucs.append(roc_auc_score(labels[te], y_prob, multi_class="ovr"))

    return float(1 - np.mean(aucs))
