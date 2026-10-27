"""Combining models trained on different inputs."""

import numpy as np
import pandas as pd
from scipy.special import expit, logit


class LogitAverage:
    """Average the log-odds of several models, each reading its own columns of X.

    Averaging log-odds instead of probabilities is fairer when the models are
    calibrated differently (one may say 0.99 where another says 0.8 for the
    same evidence). `parts` maps a name to (make_model, columns), where
    columns is a column name (e.g. "URL" for a text pipeline) or a list of
    feature columns.
    """

    def __init__(self, parts: dict):
        self.parts = parts

    def _input(self, X: pd.DataFrame, columns):
        return X[columns]

    def fit(self, X: pd.DataFrame, y):
        self.models_ = {name: make().fit(self._input(X, cols), y)
                        for name, (make, cols) in self.parts.items()}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_odds = [logit(np.clip(model.predict_proba(self._input(X, self.parts[name][1]))[:, 1], 1e-6, 1 - 1e-6))
                    for name, model in self.models_.items()]
        p = expit(np.mean(log_odds, axis=0))
        return np.column_stack([1 - p, p])
