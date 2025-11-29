import numpy as np
import pandas as pd
from mrmr import mrmr_classif


class MRMRSelector:

    def __init__(self, k=100):
        self.k = k
        self.selected_features = None
        self.mask = None
        self.num_evals = None

    def fit_transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for mRMR.")

        selected = mrmr_classif(X=X, y=y, K=self.k)
        self.selected_features = selected

        self.mask = X.columns.isin(selected)

        self.num_e_
