"""Handcrafted features computed from the URL string alone.

Nothing here downloads the page, looks up WHOIS or uses the dataset's own
columns, so the same features can be computed for any URL typed into the app.
See decision D7 in docs/DECISIONS.md.
"""

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

from phishguard.urls import TLD_EXTRACT

SUSPICIOUS_WORDS = [
    "login", "signin", "logon", "verify", "account", "secure", "update", "password",
    "bank", "confirm", "paypal", "wallet", "webscr", "billing", "support", "unlock",
    "suspend", "recover", "invoice", "free", "bonus",
]
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly",
}
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _split(url: str):
    """urlsplit that also accepts URLs without a scheme and malformed IPv6 hosts."""
    if "://" not in url:
        url = "//" + url
    try:
        return urlsplit(url)
    except ValueError:
        return urlsplit(url.replace("[", "").replace("]", ""))


def _entropy(text: str) -> float:
    """Shannon entropy in bits per character: random-looking strings score high."""
    n = len(text)
    if n == 0:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _url_features(url: str) -> dict:
    parts = _split(url)
    host = (parts.hostname or "").lower()
    ext = TLD_EXTRACT(host)
    path_segments = [s for s in parts.path.split("/") if s]
    lower = url.lower()

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
        "is_https": int(parts.scheme == "https"),
        "has_www": int(host.startswith("www.")),
        "has_ip": int(bool(_IPV4.match(host))),
        "has_port": int(":" in parts.netloc.rsplit("@", 1)[-1]),
        "on_hosting_platform": int(ext.is_private),
        "is_shortener": int(ext.top_domain_under_public_suffix in SHORTENERS),
        "path_depth": len(path_segments),
        "num_query_params": len([q for q in parts.query.split("&") if q]),
        "has_fragment": int(bool(parts.fragment)),
        "double_slash_in_path": int("//" in parts.path),
        "suspicious_words": sum(word in lower for word in SUSPICIOUS_WORDS),
        "longest_token": max((len(t) for t in _TOKEN.findall(url)), default=0),
        "url_entropy": _entropy(url),
        "host_entropy": _entropy(host),
    }


def extract_features(urls: pd.Series) -> pd.DataFrame:
    """One row of numeric features per URL, with the same index as `urls`."""
    urls = urls.astype(str)
    length = urls.str.len()
    letters = urls.str.count(r"[A-Za-z]")
    digits = urls.str.count(r"\d")

    counts = pd.DataFrame({
        "url_length": length,
        "num_letters": letters,
        "num_digits": digits,
        "letter_ratio": letters / length.clip(lower=1),
        "digit_ratio": digits / length.clip(lower=1),
        "special_ratio": (length - letters - digits) / length.clip(lower=1),
        "num_dots": urls.str.count(r"\."),
        "num_hyphens": urls.str.count("-"),
        "num_underscores": urls.str.count("_"),
        "num_slashes": urls.str.count("/"),
        "num_at": urls.str.count("@"),
        "num_question": urls.str.count(r"\?"),
        "num_equals": urls.str.count("="),
        "num_ampersand": urls.str.count("&"),
        "num_percent": urls.str.count("%"),
        "num_tilde": urls.str.count("~"),
    }, index=urls.index)

    parsed = pd.DataFrame([_url_features(u) for u in urls], index=urls.index)
    return pd.concat([counts, parsed], axis=1)
