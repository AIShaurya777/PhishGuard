"""Uncertainty: how much would a result move with different data?"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def owner_bootstrap_ci(correct, owners, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% confidence interval for accuracy, resampling whole site owners.

    URLs from the same owner aren't independent (one owner can have 1,000+
    URLs), so we resample owners, not single URLs. That gives a wider and
    more honest interval.
    """
    per_owner = pd.DataFrame({"correct": np.asarray(correct, dtype=float), "owner": np.asarray(owners)})
    per_owner = per_owner.groupby("owner")["correct"].agg(["sum", "count"])
    sums, counts = per_owner["sum"].to_numpy(), per_owner["count"].to_numpy()

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_owner), size=(n_boot, len(per_owner)))
    accuracies = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(accuracies, [2.5, 97.5])
    return float(low), float(high)


def mcnemar(correct_a, correct_b) -> dict:
    """Are two models' accuracies on the same test URLs really different?

    Only URLs where the models disagree matter: b = only A right, c = only B
    right. If the models were equally good, b and c would be a fair coin split.
    """
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": float(p)}
