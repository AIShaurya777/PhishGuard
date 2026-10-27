"""The train → tune on validation → test once routine used by every model."""

from itertools import product

import pandas as pd

from phishguard.evaluate import best_threshold, evaluate


def param_grid(**options) -> list[dict]:
    """param_grid(C=[1, 10], max_iter=[500]) -> [{'C': 1, 'max_iter': 500}, {'C': 10, ...}]"""
    keys = list(options)
    return [dict(zip(keys, values)) for values in product(*options.values())]


def tune(make_model, grid: list[dict], X, y, splits) -> tuple[pd.DataFrame, dict]:
    """Fit one model per setting on train and score it on validation.

    `X` is anything indexable with .loc (a feature table, or a Series of URLs
    for text pipelines). Returns the scores table and the best setting.
    """
    train_idx, val_idx, _ = splits
    rows = []
    for params in grid:
        model = make_model(**params).fit(X.loc[train_idx], y.loc[train_idx])
        prob = model.predict_proba(X.loc[val_idx])[:, 1]
        scores = evaluate(y.loc[val_idx], prob, threshold=best_threshold(y.loc[val_idx], prob))
        rows.append({**params, "val_accuracy": scores["accuracy"], "val_roc_auc": scores["roc_auc"]})

    table = pd.DataFrame(rows).sort_values(["val_accuracy", "val_roc_auc"], ascending=False)
    return table.reset_index(drop=True), grid[table.index[0]]


def fit_and_test(model, X, y, splits) -> tuple[object, float, dict]:
    """Fit on train, choose the threshold on validation, score test exactly once."""
    train_idx, val_idx, test_idx = splits
    model.fit(X.loc[train_idx], y.loc[train_idx])
    threshold = best_threshold(y.loc[val_idx], model.predict_proba(X.loc[val_idx])[:, 1])
    metrics = evaluate(y.loc[test_idx], model.predict_proba(X.loc[test_idx])[:, 1], threshold)
    return model, threshold, metrics
