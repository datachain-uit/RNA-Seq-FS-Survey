import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
    classification_report, confusion_matrix
)

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    acc     = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    mcc      = matthews_corrcoef(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)
    sperf = (macro_f1 * bal_acc * ((mcc+1)/2) * ((kappa+1)/2)) ** 0.25

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    wide_cols = {}
    classes = np.unique(y_true)

    for i, cls in enumerate(classes):
        wide_cols[f"precision_class_{cls}"] = float(per_precision[i])
        wide_cols[f"recall_class_{cls}"] = float(per_recall[i])
        wide_cols[f"f1_class_{cls}"] = float(per_f1[i])

    if y_prob is not None and y_prob.ndim == 2:
        for i, cls in enumerate(classes):
            try:
                y_true_bin = (y_true == cls).astype(int)
                auc_cls = roc_auc_score(y_true_bin, y_prob[:, i])
            except Exception:
                auc_cls = np.nan
            wide_cols[f"auc_class_{cls}"] = float(auc_cls)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)

    return {
        "Accuracy": acc,
        "MacroF1": macro_f1,
        "BalancedAcc": bal_acc,
        "MCC": mcc,
        "Kappa": kappa,
        "Sperf": sperf,
        "ReportDF": report_df,
        "ConfusionMatrix": cm_df,
        **wide_cols
    }
