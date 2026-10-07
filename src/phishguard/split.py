"""Train/validation/test split that keeps every site owner in one split.

If two URLs from the same owner (e.g. a.evil.com and b.evil.com) landed in
train and test, the model could score well by recognising the owner rather
than by detecting phishing. See decision D5 in docs/DECISIONS.md.
"""

import numpy as np
import pandas as pd
import tldextract

from phishguard.data import TARGET

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Uses the public suffix list bundled with tldextract (no network access), so
# everyone gets the same groups. The private section of the list makes
# customers of hosting platforms separate owners: x.web.app != y.web.app.
_extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def site_owner(urls: pd.Series) -> pd.Series:
    """Map each URL to its registered domain, e.g. http://a.evil.co.uk/x -> evil.co.uk."""
    hosts = urls.str.extract(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/:?#]+)", expand=False).str.lower()
    # Hosts with no registrable domain (IP addresses, "localhost") are their own owner.
    return hosts.map(lambda h: _extract(h).top_domain_under_public_suffix or h)


def get_splits(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) as index labels of `df`, roughly 70/15/15.

    Owners are shuffled separately within each class (by the owner's majority
    label) and then assigned to splits until each split holds its share of
    that class's rows. This keeps owners whole and the class ratio equal
    across splits.
    """
    owners = site_owner(df["URL"])
    groups = df.groupby(owners)[TARGET].agg(rows="size", phishing_share="mean")
    groups["cls"] = (groups["phishing_share"] >= 0.5).astype(int)

    rng = np.random.default_rng(seed)
    split_of_owner = {}
    for _, cls_groups in groups.sort_index().groupby("cls"):
        shuffled = cls_groups.iloc[rng.permutation(len(cls_groups))]
        # Fraction of this class's rows covered once each owner is added.
        filled = shuffled["rows"].cumsum() / shuffled["rows"].sum()
        split = np.where(filled <= TEST_SIZE, "test",
                         np.where(filled <= TEST_SIZE + VAL_SIZE, "val", "train"))
        split_of_owner.update(zip(shuffled.index, split))

    assigned = owners.map(split_of_owner)
    return tuple(df.index[assigned == name].to_numpy() for name in ("train", "val", "test"))
