"""URL parsing helpers shared by the split, the feature extractor and the models."""

from urllib.parse import SplitResult, urlsplit

import pandas as pd
import tldextract

# Uses the public suffix list bundled with tldextract (no network access), so
# everyone gets the same results. The private section of the list treats
# customers of hosting platforms as separate sites: x.web.app != y.web.app.
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
# ICANN-only variant: hosting platforms count as one domain (foo.web.app -> web.app).
_ICANN_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)

_PREFIX = r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:www\.)?"


def split_url(url: str) -> SplitResult:
    """urlsplit that also accepts URLs without a scheme and malformed IPv6 hosts."""
    if "://" not in url:
        url = "//" + url
    try:
        return urlsplit(url)
    except ValueError:
        return urlsplit(url.replace("[", "").replace("]", ""))


def parse_host(url: str) -> str:
    """Lower-cased host: drops scheme, user@, port, path and query.

    http://paypal.com@evil.top:8080/x -> evil.top  (the browser goes to evil.top)
    """
    return split_url(url).hostname or ""


def hostnames(urls: pd.Series) -> pd.Series:
    """Host of each URL, e.g. http://A.evil.com:8080/x -> a.evil.com."""
    return urls.astype(str).map(parse_host)


def strip_prefix(urls: pd.Series) -> pd.Series:
    """Drop the scheme and a leading www., e.g. https://www.a.com/x -> a.com/x."""
    return urls.str.replace(_PREFIX, "", regex=True, case=False)


def host_only(urls: pd.Series) -> pd.Series:
    """Host without www., e.g. https://www.a.b.com/x?q=1 -> a.b.com."""
    return hostnames(urls).str.replace(r"^www\.", "", regex=True)


def registered_domain(urls: pd.Series) -> pd.Series:
    """Registered domain only (subdomains removed), e.g. https://login.a.b.com/x -> b.com.

    Uses the ICANN part of the public suffix list, so customers of hosting
    platforms collapse to the platform: foo.web.app -> web.app.
    """
    return hostnames(urls).map(lambda h: _ICANN_EXTRACT(h).top_domain_under_public_suffix or h)


# The inputs every URL model is evaluated on (decisions D10, D11).
# In PhiUSIIL every legitimate URL is a bare "https://www.<domain>", so the
# prefix and the presence of a path give the label away; each later view
# removes more of that collection shortcut.
URL_VIEWS = {
    "raw URL": lambda urls: urls,
    "no scheme/www": strip_prefix,
    "domain only": host_only,
    "registered domain": registered_domain,
}
