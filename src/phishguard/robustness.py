"""Realistic URL edits for the robustness test (Milestone 7).

Each edit takes one domain (the "domain only" view) and returns the edited
domain, or None when the edit doesn't apply (e.g. "TLD → .com" on a domain
that is already .com). The models are never retrained: we only measure how
their predictions change.

Phishing edits ask "could an attacker get past the model with a cheap
change?". The legitimate edit asks "does a legitimate site get flagged just
because it has a subdomain?".
"""

import re
from urllib.parse import urlsplit

import pandas as pd

from phishguard.data import REPO_ROOT
from phishguard.split import site_owner
from phishguard.urls import TLD_EXTRACT, URL_VIEWS, split_url

_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_LOOK_ALIKE = str.maketrans("0134578", "oleastb")


def _parts(domain: str):
    """(subdomain, name, suffix, on a hosting platform?) or None for IP addresses / unparseable hosts."""
    ext = TLD_EXTRACT(domain)
    if _IP.match(domain) or not ext.suffix or not ext.domain:
        return None
    return ext.subdomain, ext.domain, ext.suffix, ext.is_private


def _join(*labels: str) -> str:
    return ".".join(label for label in labels if label)


def tld_to_com(domain: str) -> str | None:
    """login.paypal-check.cf → login.paypal-check.com (own domains only: a hosting platform's suffix can't be changed)."""
    parts = _parts(domain)
    if parts is None or parts[3] or parts[2] == "com":
        return None
    sub, name, _, _ = parts
    return _join(sub, name, "com")


def drop_subdomain(domain: str) -> str | None:
    """login.paypal-check.cf → paypal-check.cf"""
    parts = _parts(domain)
    if parts is None or not parts[0]:
        return None
    _, name, suffix, _ = parts
    return _join(name, suffix)


def remove_digits_hyphens(domain: str) -> str | None:
    """pay-pal-2fa.top → paypalfa.top (the suffix is kept; every label must stay non-empty)."""
    parts = _parts(domain)
    if parts is None:
        return None
    sub, name, suffix, _ = parts
    cleaned = [re.sub(r"[\d-]", "", label) for label in sub.split(".") if label] + [re.sub(r"[\d-]", "", name)]
    if any(not label for label in cleaned):
        return None
    edited = _join(*cleaned, suffix)
    return edited if edited != domain else None


def look_alike_letters(domain: str) -> str | None:
    """paypa1-l0gin.top → paypal-login.top: digits that look like letters become the letters."""
    parts = _parts(domain)
    if parts is None:
        return None
    sub, name, suffix, _ = parts
    edited = _join(sub.translate(_LOOK_ALIKE), name.translate(_LOOK_ALIKE), suffix)
    return edited if edited != domain else None


def leave_hosting_platform(domain: str) -> str | None:
    """my-wallet.web.app → my-wallet.com: register an own .com instead of using a free hosting platform."""
    parts = _parts(domain)
    if parts is None or not parts[3]:
        return None
    return _join(parts[1], "com")


def all_edits(domain: str) -> str | None:
    """Every applicable edit above, in order: own .com domain, no subdomain, no digits/hyphens."""
    edited = domain
    for edit in (leave_hosting_platform, tld_to_com, drop_subdomain, look_alike_letters, remove_digits_hyphens):
        edited = edit(edited) or edited
    return edited if edited != domain else None


PHISHING_EDITS = {
    "TLD → .com": tld_to_com,
    "drop subdomain": drop_subdomain,
    "remove digits and hyphens": remove_digits_hyphens,
    "look-alike digits → letters": look_alike_letters,
    "leave hosting platform (own .com)": leave_hosting_platform,
    "all of the above": all_edits,
}

LEGIT_SUBDOMAINS = ["www2", "mail", "blog", "shop", "login", "accounts"]


def add_subdomain(domain: str, prefix: str) -> str | None:
    """example.org → mail.example.org (legitimate sites commonly have such subdomains)."""
    if _parts(domain) is None:
        return None
    return f"{prefix}.{domain}"


def https_www_root(url: str) -> str | None:
    """Raw-URL attack from the independent audit (ollivianguy): make a URL look like PhiUSIIL's legitimate ones.

    http://login.evil.top/x?id=1 → https://www.login.evil.top . Applies only to
    URLs the attacker fully controls (not on a hosting platform), as in that audit.
    """
    host = split_url(url).hostname or ""
    parts = _parts(host.removeprefix("www."))
    if parts is None or parts[3]:
        return None
    edited = f"https://www.{host.removeprefix('www.')}"
    return edited if edited != url else None


def switch_to_https(url: str) -> str | None:
    """http://… → https://… (free certificates make this trivial)."""
    return "https://" + url[len("http://"):] if urlsplit(url).scheme.lower() == "http" else None


EXTERNAL_DIR = REPO_ROOT / "DataSets" / "external"


def site_kind(host: str) -> str:
    """'bare domain', 'with a subdomain' or 'on a hosting platform'."""
    ext = TLD_EXTRACT(host)
    if ext.is_private:
        return "on a hosting platform"
    return "with a subdomain" if ext.subdomain else "bare domain"


def external_sites(phiusiil_owners: set[str]) -> dict[str, pd.Series]:
    """Popular real websites (domain-only view) whose owner is nowhere in PhiUSIIL, as in notebook 11.

    CrUX origins are grouped by site_kind(); Tranco gives registered domains.
    """
    crux = pd.read_csv(EXTERNAL_DIR / "crux_top10k_2026-10.csv")
    tranco = pd.read_csv(EXTERNAL_DIR / "tranco_GQ3QK_top20k.csv")
    sets = {}
    crux_hosts = URL_VIEWS["domain only"](crux["origin"]).drop_duplicates()
    crux_hosts = crux_hosts[~site_owner(crux_hosts).isin(phiusiil_owners)]
    for group, hosts in crux_hosts.groupby(crux_hosts.map(site_kind)):
        sets[f"CrUX top 10k: {group}"] = hosts.reset_index(drop=True)
    tranco_domains = tranco["domain"].drop_duplicates()
    sets["Tranco top 20k: registered domains"] = (
        tranco_domains[~site_owner(tranco_domains).isin(phiusiil_owners)].reset_index(drop=True))
    return sets
