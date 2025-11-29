import numpy as np

def calc_sent(pred_probs):
    if pred_probs is None or pred_probs.ndim != 2:
        return np.nan

    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-12), axis=1)
    entropy /= np.log(pred_probs.shape[1])

    m = np.median(entropy)
    return float(1 - min(abs(m - 0.5) / 0.5, 1))
