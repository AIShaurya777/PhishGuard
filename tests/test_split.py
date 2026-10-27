import hashlib

import numpy as np
import pandas as pd
import pytest

from phishguard import data
from phishguard.split import get_splits, site_owner


def test_site_owner_groups_subdomains_but_not_platform_customers():
    urls = pd.Series([
        "http://a.evil.com/login",
        "https://b.evil.com",
        "http://Login.PayPal.com.evil.co.uk/x",
        "https://foo.web.app",
        "https://bar.web.app",
        "http://192.168.0.1:8080/admin",
        "http://www.paypal.com:911@evil.top/fix-account",   # really goes to evil.top
        "www.no-scheme.com/x",
    ])
    assert site_owner(urls).tolist() == [
        "evil.com", "evil.com", "evil.co.uk", "foo.web.app", "bar.web.app", "192.168.0.1",
        "evil.top", "no-scheme.com",
    ]


@pytest.fixture(scope="module")
def real():
    if not data.RAW_PATH.exists():
        pytest.skip("raw dataset not present")
    df = data.load_clean()
    return df, get_splits(df)


def test_splits_cover_every_row_exactly_once(real):
    df, splits = real
    combined = np.concatenate(splits)
    assert len(combined) == len(df)
    assert len(set(combined)) == len(df)


def test_no_owner_in_two_splits(real):
    df, splits = real
    owners = [set(site_owner(df.loc[idx, "URL"])) for idx in splits]
    assert not owners[0] & owners[1]
    assert not owners[0] & owners[2]
    assert not owners[1] & owners[2]


def test_split_sizes_and_class_ratio(real):
    df, (train, val, test) = real
    overall = df[data.TARGET].mean()
    for idx, share in [(train, 0.70), (val, 0.15), (test, 0.15)]:
        assert abs(len(idx) / len(df) - share) < 0.01
        assert abs(df.loc[idx, data.TARGET].mean() - overall) < 0.01


def test_same_seed_gives_same_split(real):
    df, splits = real
    again = get_splits(df)
    assert all(np.array_equal(a, b) for a, b in zip(splits, again))
    other = get_splits(df, seed=7)
    assert not np.array_equal(splits[2], other[2])


# Fingerprints of the seed-42 reference split (decision D11). If this test
# fails, the split changed: every saved result is then out of date. Change
# the split only on purpose, log it in docs/DECISIONS.md and re-run notebooks.
REFERENCE_FINGERPRINTS = {"train": "af8220940c56b351", "val": "69838e1a331058a6", "test": "10b36a5065ec1869"}


def test_reference_split_is_unchanged(real):
    df, splits = real
    fingerprints = {
        name: hashlib.sha256(",".join(sorted(df.loc[idx, "URL"])).encode()).hexdigest()[:16]
        for name, idx in zip(["train", "val", "test"], splits)
    }
    assert fingerprints == REFERENCE_FINGERPRINTS
