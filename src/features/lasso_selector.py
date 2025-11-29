import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


class LassoSelector:
    def __init__(self, alpha=0.001, max_iter=5000, k=100, random_state=42):
        self.alpha = alpha
        self.max_iter = max_iter
        self.k = k
        self.random_state = random_state

        self.mask = None
        self.num_evals = 0

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for LASSO FS.")

        model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        model.fit(X, y)

        coef = model.coef_
        abs_coef = np.abs(coef)

        top_k_idx = np.argsort(abs_coef)[-self.k:]
        mask = np.zeros_like(coef, dtype=bool)
        mask[top_k_idx] = True

        self.mask = mask
        self.num_evals = 1  

        X_sel = X.iloc[:, top_k_idx]

        return X_sel, mask, self.num_evals
