"""A look-alike rule, kept apart from the models (Milestone 8, D17).

Our models never learned what a brand is: PhiUSIIL has almost no look-alike
phishing, so they pass goog1e.com and paypa1.com as legitimate. This rule
compares a domain with the most popular sites (Tranco) and names the one it
imitates. It is a hand-written rule, not machine learning, and the website
says so.
"""

import re

import pandas as pd

from phishguard.robustness import EXTERNAL_DIR, LOOK_ALIKE
from phishguard.urls import TLD_EXTRACT

TOP_SITES = 1000
MIN_NAME = 5  # shorter names (bit, ok, x) match other domains by chance


def one_edit_apart(a: str, b: str) -> bool:
    """True if one inserted, deleted or replaced letter turns a into b (gooogle ↔ google)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:] if len(a) < len(b) else a[i + 1:] == b[i + 1:]


class LookAlikeRule:
    """match(domain) -> (imitated site, how) or None."""

    def __init__(self, sites: list[str]):
        self.sites = set(sites)
        self.brands = {}  # site name -> most popular site with that name: google -> google.com
        for site in sites:
            name = TLD_EXTRACT(site).domain
            if len(name) >= MIN_NAME:
                self.brands.setdefault(name, site)

    @classmethod
    def from_tranco(cls, top: int = TOP_SITES) -> "LookAlikeRule":
        tranco = pd.read_csv(EXTERNAL_DIR / "tranco_GQ3QK_top20k.csv").sort_values("rank")
        return cls(tranco["domain"].head(top).tolist())

    def match(self, domain: str) -> tuple[str, str] | None:
        ext = TLD_EXTRACT(domain)
        owner, name = ext.top_domain_under_public_suffix, ext.domain
        if not owner or owner in self.sites or len(name) < MIN_NAME - 1:
            return None
        swapped = name.translate(LOOK_ALIKE)
        if swapped != name and swapped in self.brands:
            changes = sorted({f"{a} for {b}" for a, b in zip(name, swapped) if a != b})
            return self.brands[swapped], "digits that look like letters: " + ", ".join(changes)
        for brand, site in self.brands.items():
            if len(brand) >= MIN_NAME + 1 and one_edit_apart(name, brand):
                return site, "one letter added, removed or changed"
        for token in re.split(r"[.-]", f"{ext.subdomain}.{name}"):
            if token in self.brands:
                return self.brands[token], f"contains the name “{token}”"
        return None
