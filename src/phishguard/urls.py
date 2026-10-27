"""URL parsing helpers shared by the split and the feature extractor."""

import pandas as pd
import tldextract

# Uses the public suffix list bundled with tldextract (no network access), so
# everyone gets the same results. The private section of the list treats
# customers of hosting platforms as separate sites: x.web.app != y.web.app.
TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

_HOST = r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/:?#]+)"


def hostnames(urls: pd.Series) -> pd.Series:
    """Lower-cased host of each URL, e.g. http://A.evil.com:8080/x -> a.evil.com."""
    return urls.str.extract(_HOST, expand=False).str.lower()
