import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


class SVMRFESelector:

    def __init__(self, k=100, C=0.1, step=0.1, random_state=42):
        self.k = k
        self.C = C
        self.step = step
        self.random_state = random_state

        self.mask = None
        self.num_evals = 0

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for SVM-RFE.")

        estimator = SVC(
            kernel="linear",
            C=self.C,
            random_state=self.random_state
        )

        selector = RFE(
            estimator=estimator,
            n_features_to_select=self.k,
            step=self.step
        )

        selector.fit(X, y)

        mask = selector.support_
        self.mask = mask
        self.num_evals = X.shape[1]  

        X_sel = X.loc[:, mask]

        return X_sel, mask, self.num_evals
