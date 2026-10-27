"""Explanations (Milestone 6): which characters and which features made a model say "phishing".

Three methods, all measured in **log-odds** (log(p / (1 − p))), where the
contributions of different inputs add up:

- `char_occlusion`: hide one character at a time and measure how much the
  CNN/hybrid prediction changes (characters branch).
- `feature_occlusion`: set one feature at a time to its training average and
  measure the change (the hybrid's features branch).
- `tree_shap`: exact SHAP values for XGBoost on the 40 features.

Positive = pushes towards phishing, negative = towards legitimate.
"""

import html

import numpy as np
import pandas as pd
import shap

from phishguard.deep import PAD, CharURLClassifier
from phishguard.features import extract_features
from phishguard.urls import URL_VIEWS


def char_occlusion(model: CharURLClassifier, X) -> list[np.ndarray]:
    """For each URL: one score per character = log-odds(original) − log-odds(character hidden).

    A hidden character is replaced by PAD, an all-zero input, so the network
    sees "nothing" there. For the hybrid, the features stay those of the
    original URL, so only the characters branch is being explained.
    """
    ids, lengths, features = model.encode(X)
    rows, positions = [], []
    for i, length in enumerate(lengths):
        rows += [i] * (length + 1)                           # the original, then one copy per hidden character
        positions += [-1] + list(range(length))
    rows, positions = np.array(rows), np.array(positions)
    variant_ids = ids[rows].copy()
    hidden = positions >= 0
    variant_ids[np.flatnonzero(hidden), positions[hidden]] = PAD
    scores = model.predict_encoded(variant_ids, lengths[rows], None if features is None else features[rows], logits=True)

    result, start = [], 0
    for length in lengths:
        original, variants = scores[start], scores[start + 1:start + 1 + length]
        result.append(original - variants)
        start += length + 1
    return result


def feature_occlusion(model: CharURLClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """For each URL and feature: log-odds(original) − log-odds(feature set to its training average).

    After standardisation the training average is 0, so "occluding" a
    feature means setting its scaled value to 0.
    """
    ids, lengths, features = model.encode(X)
    n, k = features.shape
    variant_features = np.repeat(features, k + 1, axis=0)
    for j in range(k):
        variant_features[np.arange(n) * (k + 1) + j + 1, j] = 0.0
    scores = model.predict_encoded(np.repeat(ids, k + 1, axis=0), np.repeat(lengths, k + 1),
                                   variant_features, logits=True).reshape(n, k + 1)
    columns = [c for c in X.columns if c != "url"]
    return pd.DataFrame(scores[:, :1] - scores[:, 1:], index=X.index, columns=columns)


def tree_shap(xgb_model, features: pd.DataFrame) -> shap.Explanation:
    """Exact SHAP values (TreeSHAP) of an XGBoost classifier, in log-odds.

    For every URL: base value + sum of the SHAP values = the model's log-odds.
    """
    return shap.TreeExplainer(xgb_model)(features)


def highlight(text: str, scores: np.ndarray, scale: float | None = None) -> str:
    """HTML: each character on a red (→ phishing) or blue (→ legitimate) background.

    The colour strength is |score| / scale; by default scale is the largest |score|.
    """
    scale = scale or max(float(np.abs(scores).max()), 1e-9)
    spans = []
    for char, score in zip(text, scores):
        alpha = min(abs(float(score)) / scale, 1.0)
        colour = f"rgba(214, 39, 40, {alpha:.2f})" if score > 0 else f"rgba(31, 119, 180, {alpha:.2f})"
        spans.append(f'<span style="background:{colour}" title="{score:+.2f}">{html.escape(char)}</span>')
    return f'<span style="font-family:monospace;font-size:15px">{"".join(spans)}</span>'


def explain_url(model: CharURLClassifier, url: str, view: str = "domain only") -> dict:
    """Everything the app shows for one URL typed by a user.

    The URL is reduced to the view the model was trained on (D10), then we
    return that text, P(phishing), one score per character and, for the
    hybrid, one score per feature (largest effect first).
    """
    text = URL_VIEWS[view](pd.Series([url]))
    X = extract_features(text).assign(url=text) if hasattr(model, "feature_mean_") else text
    result = {
        "input": text.iloc[0],
        "probability": float(model.predict_proba(X)[0, 1]),
        "char_scores": char_occlusion(model, X)[0],
    }
    if hasattr(model, "feature_mean_"):
        scores = feature_occlusion(model, X).iloc[0]
        result["feature_scores"] = scores.reindex(scores.abs().sort_values(ascending=False).index)
    return result
