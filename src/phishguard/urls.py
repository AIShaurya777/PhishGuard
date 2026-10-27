"""URL parsing helpers shared by the split, the feature extractor and the models."""

import pandas as pd
import tldextract

# Uses the public suffix list bundled with tldextract (no network access), so
# everyone gets the same results. The private section of the list treats
# customers of hosting platforms as separate sites: x.web.app != y.web.app.
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

_HOST = r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/:?#]+)"
_PREFIX = r"^[a-zA-Z][a-zA-Z0-9+.-]*://(?:www\.)?"


def hostnames(urls: pd.Series) -> pd.Series:
    """Lower-cased host of each URL, e.g. http://A.evil.com:8080/x -> a.evil.com."""
    return urls.str.extract(_HOST, expand=False).str.lower()


def strip_prefix(urls: pd.Series) -> pd.Series:
    """Drop the scheme and a leading www., e.g. https://www.a.com/x -> a.com/x."""
    return urls.str.replace(_PREFIX, "", regex=True, case=False)


def host_only(urls: pd.Series) -> pd.Series:
    """Host without www., port, path or query, e.g. https://www.a.com/x?q=1 -> a.com."""
    return hostnames(urls).str.replace(r"^www\.", "", regex=True)


# The three inputs every URL model is evaluated on (decision D10).
# In PhiUSIIL every legitimate URL is a bare "https://www.<domain>", so the
# prefix and the presence of a path give the label away; the later views remove them.
URL_VIEWS = {
    "raw URL": lambda urls: urls,
    "no scheme/www": strip_prefix,
    "domain only": host_only,
}
