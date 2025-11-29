import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

class ChiSquareSelector:
    def __init__(self, k=100):
        self.k = k
        self.selector = None
        self.mask = None
        self.num_evals = None

    def fit_transform(self, X, y):

        k = min(self.k, X.shape[1])

        self.selector = SelectKBest(score_func=chi2, k=k)
        X_sel = self.selector.fit_transform(X, y)

        self.mask = self.selector.get_support()
        self.num_evals = X.shape[1]  

        return X_sel, self.mask, self.num_evals
