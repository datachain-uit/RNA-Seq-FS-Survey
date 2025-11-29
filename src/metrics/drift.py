from scipy.spatial.distance import jensenshannon
import numpy as np

def calc_sdrift(pred_probs, true_labels):
    if pred_probs is None:
        return np.nan

    K = pred_probs.shape[1]
    P_hat = pred_probs.mean(axis=0)
    P_true = np.bincount(true_labels, minlength=K) / len(true_labels)

    jsd = jensenshannon(P_hat, P_true, base=2) ** 2
    return float(1 - jsd / np.log(2))
