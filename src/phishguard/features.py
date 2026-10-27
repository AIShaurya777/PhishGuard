"""Handcrafted features computed from the URL string alone.

Nothing here downloads the page, looks up WHOIS or uses the dataset's own
columns, so the same features can be computed for any URL typed into the app.
See decision D7 in docs/DECISIONS.md.
"""

import ipaddress
import math
import re
from collections import Counter

import pandas as pd

from phishguard.urls import TLD_EXTRACT, split_url, strip_prefix

SUSPICIOUS_WORDS = {
    "login", "signin", "logon", "verify", "account", "secure", "update", "password",
    "bank", "confirm", "paypal", "wallet", "webscr", "billing", "support", "unlock",
    "suspend", "recover", "invoice", "free", "bonus",
}
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly",
}
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _is_ip(host: str) -> bool:
    """IPv4/IPv6 in any form a browser accepts: 1.2.3.4, 3232235521, 0x7f.0.0.1, ::1."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return False
    try:
        values = [int(p, 0) if p.lower().startswith("0x") else int(p) for p in parts]
    except ValueError:
        return False
    return all(v >= 0 for v in values) and values[-1] < 2 ** (8 * (5 - len(parts)))


def _has_port(parts) -> int:
    try:
        return int(parts.port is not None)
    except ValueError:  # a port that isn't a valid number is still a port
        return 1


def _entropy(text: str) -> float:
    """Shannon entropy in bits per character: random-looking strings score high."""
    n = len(text)
    if n == 0:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _url_features(url: str, body: str) -> dict:
    parts = split_url(url)
    host = (parts.hostname or "").lower()
    ext = TLD_EXTRACT(host)
    path_segments = [s for s in parts.path.split("/") if s]
    tokens = _TOKEN.findall(body.lower())

    return {
        "host_length": len(host),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "host_dots": host.count("."),
        "host_hyphens": host.count("-"),
        "host_digits": sum(ch.isdigit() for ch in host),
        "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "domain_length": len(ext.domain),
        "tld_length": len(ext.suffix),
        "is_https": int(parts.scheme.lower() == "https"),
        "has_www": int(host.startswith("www.")),
        "has_ip": int(_is_ip(host)),
        "has_port": _has_port(parts),
        "has_punycode": int("xn--" in host),
        "on_hosting_platform": int(ext.is_private),
        "is_shortener": int(ext.top_domain_under_public_suffix in SHORTENERS),
        "path_depth": len(path_segments),
        "num_query_params": len([q for q in parts.query.split("&") if q]),
        "has_fragment": int(bool(parts.fragment)),
        "double_slash_in_path": int("//" in parts.path),
        "suspicious_words": sum(t in SUSPICIOUS_WORDS for t in tokens),
        "longest_token": max((len(t) for t in tokens), default=0),
        "url_entropy": _entropy(body),
        "host_entropy": _entropy(host),
    }


def extract_features(urls: pd.Series) -> pd.DataFrame:
    """One row of numeric features per URL, with the same index as `urls`.

    Counts, ratios and entropy use the URL without its scheme and "www.", so
    only `is_https` and `has_www` carry the prefix (see D9).
    """
    urls = urls.astype(str)
    body = strip_prefix(urls)
    length = body.str.len()
    letters = body.str.count(r"[A-Za-z]")
    digits = body.str.count(r"\d")

    counts = pd.DataFrame({
        "url_length": length,
        "num_letters": letters,
        "num_digits": digits,
        "letter_ratio": letters / length.clip(lower=1),
        "digit_ratio": digits / length.clip(lower=1),
        "special_ratio": (length - letters - digits) / length.clip(lower=1),
        "num_dots": body.str.count(r"\."),
        "num_hyphens": body.str.count("-"),
        "num_underscores": body.str.count("_"),
        "num_slashes": body.str.count("/"),
        "num_at": body.str.count("@"),
        "num_question": body.str.count(r"\?"),
        "num_equals": body.str.count("="),
        "num_ampersand": body.str.count("&"),
        "num_percent": body.str.count("%"),
        "num_tilde": body.str.count("~"),
    }, index=urls.index)

    parsed = pd.DataFrame([_url_features(u, b) for u, b in zip(urls, body)], index=urls.index)
    return pd.concat([counts, parsed], axis=1)
