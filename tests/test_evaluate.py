import pytest

from phishguard.evaluate import best_threshold, evaluate, precision_at_base_rate, tpr_at_fpr


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


def test_extra_metrics_are_present_and_sane():
    m = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["mcc"] == 1.0 and m["pr_auc"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["tpr_at_fpr_1pct"] == 1.0
    assert 0 < m["brier"] < 0.05 and m["log_loss"] > 0


def test_tpr_at_fpr():
    # Catching the 0.7 phishing URL would also flag the 0.75 legitimate one (FPR 0.5).
    assert tpr_at_fpr([0, 0, 1, 1], [0.1, 0.75, 0.7, 0.9], max_fpr=0.0) == 0.5


def test_best_threshold_minimises_errors():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.25, 0.9]  # 0.5 would miss the 0.25 phishing URL
    t = best_threshold(y_true, y_prob)
    assert 0.2 < t <= 0.25
    assert evaluate(y_true, y_prob, threshold=t)["accuracy"] == 1.0


def test_best_threshold_takes_the_middle_of_ties():
    # Every threshold from 0.11 to 0.90 makes zero errors; the middle is ~0.5, not 0.11.
    assert 0.49 <= best_threshold([0, 1], [0.1, 0.9]) <= 0.51


def test_threshold_changes_predictions_not_auc():
    y_true = [1, 0, 1, 0]
    y_prob = [0.6, 0.4, 0.45, 0.1]
    low, high = evaluate(y_true, y_prob, threshold=0.3), evaluate(y_true, y_prob, threshold=0.5)
    assert low["recall"] == 1.0 and high["recall"] == 0.5
    assert low["roc_auc"] == high["roc_auc"]


def test_precision_at_base_rate():
    # 78% recall, 3.6% false alarms, 1% phishing: most alarms are false.
    assert precision_at_base_rate(0.78, 0.036, 0.01) == pytest.approx(0.18, abs=0.005)
