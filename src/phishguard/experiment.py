"""Run a model over several split seeds and record everything (decision D11).

One split can be lucky: the domain-only n-gram model scores 88.6% on the
seed-42 split but ~85–86% on others. So every model is trained and tested on
5 different splits, and we report the mean ± standard deviation. Seed 42 is
the *reference* split, used for examples, plots, confidence intervals and
model-vs-model significance tests.
"""

import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from phishguard.evaluate import RESULTS_DIR, slugify
from phishguard.modeling import Fit, fit_and_test, tune
from phishguard.split import get_splits, site_owner
from phishguard.stats import owner_bootstrap_ci

SEEDS = (42, 0, 1, 2, 3)
REFERENCE_SEED = 42
SUMMARY_METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc",
                   "tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct", "brier"]


@dataclass
class Experiment:
    name: str
    per_seed: pd.DataFrame      # one row of test metrics per seed
    summary: pd.DataFrame       # mean and sd per metric
    reference: Fit              # the seed-42 fit (model, threshold, probabilities)
    reference_splits: tuple     # (train_idx, val_idx, test_idx) for seed 42
    accuracy_ci: tuple          # owner-bootstrap 95% CI of seed-42 test accuracy


def run(name: str, make_model, grid: list[dict], X, y: pd.Series, df: pd.DataFrame,
        features: str, group: str, seeds=SEEDS) -> Experiment:
    """Tune and test `make_model` on every seed's split, save results/<group>/<name>.json.

    The seed-42 validation and test probabilities are saved to a matching .npz,
    so later notebooks can compare or combine models without retraining.
    """
    owners = site_owner(df["URL"])
    seed_records, reference = [], None
    for seed in seeds:
        splits = get_splits(df, seed=seed)
        start = time.perf_counter()
        _, params, model = tune(make_model, grid, X, y, splits)
        fit = fit_and_test(model, X, y, splits, already_fitted=True)
        seed_records.append({
            "seed": seed, "params": params, "threshold": fit.threshold,
            "val": fit.val_metrics, "test": fit.test_metrics,
            "runtime_s": round(time.perf_counter() - start, 1),
        })
        if seed == REFERENCE_SEED:
            reference, reference_splits = fit, splits

    test_idx = reference_splits[2]
    correct = (reference.test_prob >= reference.threshold) == y.loc[test_idx].to_numpy()
    ci = owner_bootstrap_ci(correct, owners.loc[test_idx])

    per_seed = pd.DataFrame([{"seed": r["seed"], "params": r["params"], "threshold": r["threshold"], **r["test"]}
                             for r in seed_records])
    summary = per_seed[SUMMARY_METRICS].agg(["mean", "std"]).T.rename(columns={"std": "sd"})

    ref_record = next(r for r in seed_records if r["seed"] == REFERENCE_SEED)
    record = {
        "model": name,
        "features": features,
        "summary": {m: {"mean": summary.loc[m, "mean"], "sd": summary.loc[m, "sd"]} for m in SUMMARY_METRICS},
        "reference": {**ref_record, "accuracy_ci": list(ci)},
        "seeds": seed_records,
        "environment": _environment(),
    }
    folder = RESULTS_DIR / group
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slugify(name)}.json").write_text(json.dumps(record, indent=2, default=_jsonable))
    np.savez_compressed(
        folder / f"{slugify(name)}.npz",
        val_idx=reference_splits[1], val_prob=reference.val_prob.astype(np.float32),
        test_idx=test_idx, test_prob=reference.test_prob.astype(np.float32),
        threshold=reference.threshold,
    )
    return Experiment(name, per_seed, summary, reference, reference_splits, ci)


_PERCENT = {"accuracy", "precision", "recall", "f1", "tpr_at_fpr_1pct", "tpr_at_fpr_0_1pct"}


def summary_table(experiments, metrics=("accuracy", "f1", "roc_auc", "pr_auc", "mcc", "tpr_at_fpr_1pct")) -> pd.DataFrame:
    """One row per experiment: 'mean ± sd' over the seeds, plus the reference split's 95% CI.

    Rates (accuracy, F1, TPR …) are shown in %, scores (ROC-AUC, MCC …) as 0–1.
    """
    rows = []
    for exp in experiments:
        row = {"model": exp.name}
        for metric in metrics:
            mean, sd = exp.summary.loc[metric, "mean"], exp.summary.loc[metric, "sd"]
            row[metric] = f"{mean:.2%} ± {sd:.2%}".replace("%", "") if metric in _PERCENT else f"{mean:.4f} ± {sd:.4f}"
        low, high = exp.accuracy_ci
        row["accuracy, seed 42 [95% CI]"] = (f"{exp.reference.test_metrics['accuracy']:.2%} "
                                             f"[{low:.2%}, {high:.2%}]")
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table.columns = [f"{c} (%)" if c in _PERCENT else c for c in table.columns]
    return table


def load_record(group: str, name: str) -> dict:
    """The full JSON record saved by run(): settings, per-seed metrics, environment."""
    return json.loads((RESULTS_DIR / group / f"{slugify(name)}.json").read_text())


def load_predictions(group: str, name: str) -> dict:
    """Seed-42 validation/test probabilities saved by run()."""
    with np.load(RESULTS_DIR / group / f"{slugify(name)}.npz") as npz:
        return {key: npz[key] for key in npz.files}


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _environment() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "tldextract", "torch"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=RESULTS_DIR.parent).stdout.strip()
    except OSError:
        commit = ""
    return {"python": platform.python_version(), "packages": packages, "git_commit": commit,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
