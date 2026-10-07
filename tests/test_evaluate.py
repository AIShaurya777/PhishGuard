import pytest

from phishguard.evaluate import evaluate


def test_metrics_treat_phishing_as_positive():
    # 3 phishing, 2 legitimate; one phishing URL is missed.
    y_true = [1, 1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.2, 0.1, 0.3]
    m = evaluate(y_true, y_prob)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 0, 2)
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 5)
    # 6 (phishing, legitimate) pairs; only (0.2, 0.3) is ranked the wrong way.
    assert m["roc_auc"] == pytest.approx(5 / 6)


def test_threshold_changes_predictions_not_auc():
    y_true = [1, 0, 1, 0]
    y_prob = [0.6, 0.4, 0.45, 0.1]
    low, high = evaluate(y_true, y_prob, threshold=0.3), evaluate(y_true, y_prob, threshold=0.5)
    assert low["recall"] == 1.0 and high["recall"] == 0.5
    assert low["roc_auc"] == high["roc_auc"]
