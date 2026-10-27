import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from phishguard.ensemble import LogitAverage
from phishguard.modeling import fit_and_test, param_grid, tune


def toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = (X["signal"] > 0).astype(int)
    idx = X.index.to_numpy()
    return X, y, (idx[:200], idx[200:250], idx[250:])


def test_param_grid_is_cartesian_product():
    assert param_grid(a=[1, 2], b=[None]) == [{"a": 1, "b": None}, {"a": 2, "b": None}]


def test_tune_picks_the_setting_with_lowest_validation_log_loss():
    X, y, splits = toy_data()
    grid = param_grid(C=[0.0001, 10.0])  # C=0.0001 barely learns anything
    table, best, model = tune(lambda C: LogisticRegression(C=C), grid, X, y, splits)
    assert best == {"C": 10.0}
    assert table.iloc[0]["val_log_loss"] <= table.iloc[1]["val_log_loss"]
    assert model.C == 10.0


def test_fit_and_test_scores_test_split():
    X, y, splits = toy_data()
    fit = fit_and_test(LogisticRegression(), X, y, splits)
    m = fit.test_metrics
    assert 0 < fit.threshold < 1
    assert m["tp"] + m["fn"] + m["tn"] + m["fp"] == len(splits[2])
    assert len(fit.test_prob) == len(splits[2]) and len(fit.val_prob) == len(splits[1])
    assert m["accuracy"] > 0.9


class RecordingVectorizer(TfidfVectorizer):
    """A TF-IDF vectorizer that remembers which documents it was fitted on."""
    fitted_on = []

    def fit_transform(self, raw_documents, y=None):
        RecordingVectorizer.fitted_on.append(list(raw_documents))
        return super().fit_transform(raw_documents, y)


def test_text_pipeline_is_fitted_on_train_only():
    urls = pd.Series([f"http://site{i}.com/{'login' if i % 2 else 'home'}" for i in range(60)])
    y = pd.Series([i % 2 for i in range(60)])
    splits = (urls.index[:40].to_numpy(), urls.index[40:50].to_numpy(), urls.index[50:].to_numpy())
    RecordingVectorizer.fitted_on = []
    model = make_pipeline(RecordingVectorizer(analyzer="char"), LogisticRegression())
    fit_and_test(model, urls, y, splits)
    assert RecordingVectorizer.fitted_on == [urls.iloc[:40].tolist()]


def test_logit_average_combines_models_on_their_own_columns():
    X, y, splits = toy_data()
    model = LogitAverage({
        "signal": (LogisticRegression, ["signal"]),
        "noise": (LogisticRegression, ["noise"]),
    })
    fit = fit_and_test(model, X, y, splits)
    prob = fit.test_prob
    assert prob.shape == (len(splits[2]),) and ((0 < prob) & (prob < 1)).all()
    assert fit.test_metrics["accuracy"] > 0.8  # the useful model still dominates
