import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TenureBucket(BaseEstimator, TransformerMixin):
    """Bins tenure (months) into 4 ordinal buckets: 0-12, 13-24, 25-48, 49+."""

    _BINS = [-1, 12, 24, 48, float("inf")]
    _LABELS = ["0-12", "13-24", "25-48", "49+"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = (
            X.iloc[:, 0]
            if isinstance(X, pd.DataFrame)
            else pd.Series(np.asarray(X).ravel())
        )
        bucketed = pd.cut(col, bins=self._BINS, labels=self._LABELS, right=True)
        return bucketed.astype(str).values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["tenure_bucket"])
