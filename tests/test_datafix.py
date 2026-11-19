import numpy as np
import pandas as pd

from phishguard.datafix import TRAIN_PREFIXES, _add_prefix, _augment, _in_half, _owner_split, _random_edit, summarise
from phishguard.robustness import LEGIT_SUBDOMAINS
from phishguard.split import site_owner


def test_train_prefixes_never_overlap_the_tested_ones():
    # If we trained on "mail." we'd be testing on what we taught.
    assert not set(TRAIN_PREFIXES) & set(LEGIT_SUBDOMAINS)


def test_tranco_halves_are_fixed_and_balanced():
    owners = [f"site{i}.com" for i in range(2000)]
    first = [_in_half(o) for o in owners]
    assert first == [_in_half(o) for o in owners]          # same answer every time, no seed
    assert 0.45 < np.mean(first) < 0.55


def test_owner_split_keeps_owners_whole():
    domains = pd.Series([f"{sub}.owner{i}.com" for i in range(200) for sub in ("a", "b", "c")])
    fit, val = _owner_split(domains, seed=0)
    assert len(fit) + len(val) == len(domains)
    assert not set(site_owner(fit)) & set(site_owner(val))
    assert 0.10 < len(val) / len(domains) < 0.20


def test_augmentations():
    rng = np.random.default_rng(0)
    prefix, rest = _add_prefix("example.org", rng).split(".", 1)
    assert prefix in TRAIN_PREFIXES and rest == "example.org"
    assert _random_edit("example.com", rng) is None or _random_edit("example.com", rng) != "example.com"

    domains = pd.Series([f"site{i}.org" for i in range(400)] * 3)    # repeats count once
    extra = _augment(domains, _add_prefix, seed=0)
    assert 90 <= len(extra) <= 110                                  # about 25% of 400 distinct domains
    assert extra.str.endswith(".org").all() and not extra.isin(domains).any()


def test_summarise_mean_and_sd():
    records = [{"phiusiil_test": {"accuracy": a, "roc_auc": 0.9}, "crux_bare": {"flagged": f, "label": 0}}
               for a, f in [(0.8, 0.1), (0.9, 0.3)]]
    s = summarise(records)
    assert round(s["phiusiil_accuracy"]["mean"], 6) == 0.85 and round(s["crux_bare"]["mean"], 6) == 0.2
    assert round(s["crux_bare"]["sd"], 6) == round(np.std([0.1, 0.3], ddof=1), 6)
