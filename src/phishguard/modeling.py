"""The train → tune on validation → test once routine used by every model."""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from phishguard.evaluate import best_threshold, evaluate


def param_grid(**options) -> list[dict]:
    """param_grid(C=[1, 10], max_iter=[500]) -> [{'C': 1, 'max_iter': 500}, {'C': 10, ...}]"""
    keys = list(options)
    return [dict(zip(keys, values)) for values in product(*options.values())]


def tune(make_model, grid: list[dict], X, y, splits) -> tuple[pd.DataFrame, dict, object]:
    """Fit one model per setting on train and score it on validation.

    `X` is anything indexable with .loc (a feature table, or a Series of URLs
    for text pipelines). The best setting has the lowest validation log-loss:
    unlike accuracy, log-loss changes smoothly, so one lucky validation URL
    can't decide the winner. Returns the scores table, the best setting and
    the model fitted with it.
    """
    train_idx, val_idx, _ = splits
    rows, best_model, best_loss = [], None, np.inf
    for params in grid:
        model = make_model(**params).fit(X.loc[train_idx], y.loc[train_idx])
        prob = model.predict_proba(X.loc[val_idx])[:, 1]
        scores = evaluate(y.loc[val_idx], prob, threshold=best_threshold(y.loc[val_idx], prob))
        rows.append({**params, "val_log_loss": scores["log_loss"],
                     "val_accuracy": scores["accuracy"], "val_roc_auc": scores["roc_auc"]})
        if scores["log_loss"] < best_loss:
            best_model, best_loss = model, scores["log_loss"]

    table = pd.DataFrame(rows).sort_values("val_log_loss")
    return table.reset_index(drop=True), grid[table.index[0]], best_model


@dataclass
class Fit:
    """A model fitted on train, with its validation-chosen threshold and scores."""
    model: object
    threshold: float
    val_metrics: dict
    test_metrics: dict
    val_prob: np.ndarray
    test_prob: np.ndarray


def fit_and_test(model, X, y, splits, already_fitted: bool = False) -> Fit:
    """Fit on train, choose the threshold on validation, score test exactly once."""
    train_idx, val_idx, test_idx = splits
    if not already_fitted:
        model.fit(X.loc[train_idx], y.loc[train_idx])
    val_prob = model.predict_proba(X.loc[val_idx])[:, 1]
    threshold = best_threshold(y.loc[val_idx], val_prob)
    test_prob = model.predict_proba(X.loc[test_idx])[:, 1]
    return Fit(
        model=model,
        threshold=threshold,
        val_metrics=evaluate(y.loc[val_idx], val_prob, threshold),
        test_metrics=evaluate(y.loc[test_idx], test_prob, threshold),
        val_prob=val_prob,
        test_prob=test_prob,
    )
