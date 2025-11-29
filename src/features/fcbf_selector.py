import numpy as np
import pandas as pd
from skfeature.function.information_theoretical_based import FCBF


class FCBFSelector:

    def __init__(self, k=100):
        self.k = k
        self.selected_indices = None
        self.mask = None
        self.num_evals = None

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for FCBF.")

        selected_idx = FCBF.fcbf(X.values, y)

        selected_idx = selected_idx[: self.k]
        self.selected_indices = selected_idx

        mask = np.zeros(X.shape[1], dtype=bool)
        mask[selected_idx] = True
        self.mask = mask

        self.num_evals = X.shape[1]

        X_sel = X.iloc[:, selected_idx]

        return X_sel, mask, self.num_evals
