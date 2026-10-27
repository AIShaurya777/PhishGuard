import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from phishguard.deep import CharURLClassifier
from phishguard.explain import char_occlusion, explain_url, feature_occlusion, highlight, tree_shap


@pytest.fixture(scope="module")
def toy_hybrid():
    # Phishing ⇔ the domain ends in .top; the "signal" feature also carries the label.
    rng = np.random.default_rng(0)
    n = 600
    y = pd.Series(np.arange(n) % 2)
    X = pd.DataFrame({
        "url": [f"{''.join(rng.choice(list('abcdefgh'), 8))}.{'top' if i % 2 else 'org'}" for i in range(n)],
        "signal": y * 2.0 + rng.random(n), "noise": rng.random(n),
    })
    return CharURLClassifier(arch="hybrid", max_epochs=6, batch_size=32, device="cpu").fit(X, y), X


def test_char_occlusion_gives_one_score_per_character_and_finds_the_tld(toy_hybrid):
    model, X = toy_hybrid
    phishing = X[X["url"].str.endswith(".top")].head(20)
    scores = char_occlusion(model, phishing)
    assert [len(s) for s in scores] == phishing["url"].str.len().tolist()
    # On average the ".top" characters push towards phishing more than the random name does.
    tld = np.mean([s[-4:].mean() for s in scores])
    name = np.mean([s[:8].mean() for s in scores])
    assert tld > name


def test_feature_occlusion_ranks_the_informative_feature_first(toy_hybrid):
    model, X = toy_hybrid
    attributions = feature_occlusion(model, X.head(100))
    assert list(attributions.columns) == ["signal", "noise"]
    assert attributions["signal"].abs().mean() > attributions["noise"].abs().mean()


def test_tree_shap_values_add_up_to_the_model_log_odds():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 4)), columns=list("abcd"))
    y = (X["a"] + 0.3 * X["b"] > 0.7).astype(int)
    model = XGBClassifier(n_estimators=30, max_depth=3).fit(X, y)
    explanation = tree_shap(model, X)
    log_odds = model.predict(X, output_margin=True)
    assert np.allclose(explanation.base_values + explanation.values.sum(axis=1), log_odds, atol=1e-4)
    assert np.abs(explanation.values).mean(axis=0).argmax() == 0     # "a" matters most


def test_highlight_escapes_html_and_colours_each_character():
    out = highlight("<a>", np.array([1.0, -0.5, 0.0]))
    assert "&lt;" in out and "<a>" not in out
    assert out.count("<span style=\"background") == 3
    assert "rgba(214, 39, 40, 1.00)" in out and "rgba(31, 119, 180, 0.50)" in out


def test_explain_url_reduces_to_the_view_and_scores_each_character(toy_hybrid):
    _, X = toy_hybrid
    cnn = CharURLClassifier(arch="cnn", max_epochs=3, batch_size=32, device="cpu").fit(
        X["url"], X["url"].str.endswith(".top").astype(int))
    result = explain_url(cnn, "https://www.login.abcdefgh.top/verify?id=1")
    assert result["input"] == "login.abcdefgh.top"
    assert len(result["char_scores"]) == len(result["input"])
    assert 0 <= result["probability"] <= 1
    assert "feature_scores" not in result          # a CNN has no features branch


def test_explanations_do_not_vanish_for_very_confident_urls(toy_hybrid):
    # Confident predictions round to P = 1.0; scores must come from the log-odds, not the probability.
    model, X = toy_hybrid
    ids, lengths, features = model.encode(X.head(50))
    logits = model.predict_encoded(ids, lengths, features, logits=True)
    assert np.allclose(1 / (1 + np.exp(-logits)), model.predict_encoded(ids, lengths, features), atol=1e-6)
    scores = feature_occlusion(model, X.head(50).assign(signal=X["signal"].head(50) * 20))
    assert (scores["signal"].abs() > 0).all()      # even when the probability is saturated
