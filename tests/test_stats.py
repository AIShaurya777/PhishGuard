import numpy as np

from phishguard.stats import mcnemar, owner_bootstrap_ci


def test_owner_bootstrap_ci_contains_the_accuracy_and_widens_with_clustering():
    rng = np.random.default_rng(0)
    correct = rng.random(2000) < 0.9
    independent = owner_bootstrap_ci(correct, np.arange(2000))
    clustered = owner_bootstrap_ci(correct, np.arange(2000) // 100)  # 20 owners of 100 URLs
    assert independent[0] < correct.mean() < independent[1]
    assert (clustered[1] - clustered[0]) > (independent[1] - independent[0])


def test_mcnemar_counts_disagreements_only():
    a = np.array([1, 1, 1, 0, 1], bool)
    b = np.array([1, 0, 0, 1, 1], bool)
    result = mcnemar(a, b)
    assert (result["only_a_right"], result["only_b_right"]) == (2, 1)
    assert result["p_value"] == 1.0  # 2 vs 1 is no evidence of a difference


def test_mcnemar_detects_a_clear_difference():
    a = np.ones(100, bool)
    b = np.r_[np.zeros(30, bool), np.ones(70, bool)]
    assert mcnemar(a, b)["p_value"] < 1e-6
