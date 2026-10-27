"""One way of scoring and recording every model, so results are directly comparable."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from phishguard.data import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def tpr_at_fpr(y_true, y_prob, max_fpr: float) -> float:
    """Share of phishing caught while flagging at most `max_fpr` of legitimate URLs."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Metrics with phishing (1) as the positive class.

    `y_prob` is the predicted probability of phishing. The threshold-free
    metrics (ROC-AUC, PR-AUC, TPR at a fixed FPR, Brier, log-loss) use it
    directly; the rest use predictions at `threshold`.
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, y_prob, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, y_prob, 0.001),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, y_prob) -> float:
    """Threshold with the fewest errors on the given (validation) data.

    Choose it on validation, then apply that same value to test. Choosing
    it on test would make the test score optimistic. When several thresholds
    tie, the middle one is taken, so a flat error curve doesn't push the
    threshold to an extreme like 0.02.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidates = np.round(np.arange(0.01, 1.0, 0.01), 2)
    errors = np.array([((y_prob >= t) != y_true).sum() for t in candidates])
    tied = candidates[errors == errors.min()]
    return float(tied[len(tied) // 2])


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    """Precision if only `base_rate` of URLs were phishing, e.g. 0.01 in real traffic.

    precision = recall·p / (recall·p + fpr·(1 − p))
    """
    caught = recall * base_rate
    false_alarms = fpr * (1 - base_rate)
    return caught / (caught + false_alarms) if caught + false_alarms else 0.0


def save_result(model: str, features: str, threshold: float, metrics: dict, group: str = "m2") -> None:
    """Record a single-split result in results/<group>/<model>.json.

    Multi-seed experiments use phishguard.experiment.run(), which saves more.
    """
    folder = RESULTS_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    record = {"model": model, "features": features, "threshold": threshold, **metrics}
    (folder / f"{slugify(model)}.json").write_text(json.dumps(record, indent=2))


def slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def load_results(group: str) -> pd.DataFrame:
    """One row per result in results/<group>/, best mean accuracy first."""
    rows = []
    for path in sorted((RESULTS_DIR / group).glob("*.json")):
        record = json.loads(path.read_text())
        if "summary" in record:  # multi-seed result from experiment.run()
            row = {"model": record["model"], "features": record["features"],
                   "seeds": len(record["seeds"])}
            for metric, stats in record["summary"].items():
                row[metric] = stats["mean"]
                row[f"{metric}_sd"] = stats["sd"]
            ref = record["reference"]
            row["ref_accuracy"] = ref["test"]["accuracy"]
            row["ref_ci_low"], row["ref_ci_high"] = ref["accuracy_ci"]
            row["ref_errors"] = ref["test"]["fp"] + ref["test"]["fn"]
            rows.append(row)
        else:
            rows.append(record)
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False, ignore_index=True)
