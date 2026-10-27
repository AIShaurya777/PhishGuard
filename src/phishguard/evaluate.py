"""One way of scoring and recording every model, so results are directly comparable."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from phishguard.data import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Metrics with phishing (1) as the positive class.

    `y_prob` is the predicted probability of phishing. ROC-AUC uses it
    directly; the other metrics use predictions at `threshold`.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, y_prob) -> float:
    """Threshold with the fewest errors on the given (validation) data.

    Choose it on validation, then apply that same value to test. Choosing
    it on test would make the test score optimistic.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidates = np.round(np.arange(0.01, 1.0, 0.01), 2)
    errors = [((y_prob >= t) != y_true).sum() for t in candidates]
    return float(candidates[int(np.argmin(errors))])


def save_result(model: str, features: str, threshold: float, metrics: dict) -> None:
    """Record a model's test metrics in results/<model>.json for the comparison table."""
    RESULTS_DIR.mkdir(exist_ok=True)
    record = {"model": model, "features": features, "threshold": threshold, **metrics}
    path = RESULTS_DIR / f"{model.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps(record, indent=2))


def load_results() -> pd.DataFrame:
    """All recorded test results, best accuracy first."""
    records = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    return pd.DataFrame(records).sort_values("accuracy", ascending=False, ignore_index=True)
