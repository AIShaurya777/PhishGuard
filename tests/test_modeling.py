import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from phishguard.modeling import fit_and_test, param_grid, tune


def toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = (X["signal"] > 0).astype(int)
    idx = X.index.to_numpy()
    return X, y, (idx[:200], idx[200:250], idx[250:])


def test_param_grid_is_cartesian_product():
    assert param_grid(a=[1, 2], b=[None]) == [{"a": 1, "b": None}, {"a": 2, "b": None}]


def test_tune_returns_a_setting_from_the_grid():
    X, y, splits = toy_data()
    grid = param_grid(C=[0.001, 10.0])
    table, best = tune(lambda C: LogisticRegression(C=C), grid, X, y, splits)
    assert best in grid
    assert len(table) == 2


def test_fit_and_test_scores_test_split():
    X, y, splits = toy_data()
    _, threshold, metrics = fit_and_test(LogisticRegression(), X, y, splits)
    assert 0 < threshold < 1
    assert metrics["tp"] + metrics["fn"] + metrics["tn"] + metrics["fp"] == len(splits[2])
    assert metrics["accuracy"] > 0.9
