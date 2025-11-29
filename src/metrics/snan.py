import numpy as np

def calc_snan(pred_probs, eps=1e-3):
    if pred_probs is None:
        return np.nan

    if not np.all(np.isfinite(pred_probs)):
        return 0.0

    row_sums = pred_probs.sum(axis=1)
    return float(np.mean(np.abs(row_sums - 1) <= eps))
