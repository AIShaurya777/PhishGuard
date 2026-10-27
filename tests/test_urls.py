import pandas as pd
import pytest

from phishguard import data
from phishguard.columns import COLUMN_GROUPS
from phishguard.urls import URL_VIEWS, host_only, hostnames, parse_host, registered_domain, strip_prefix

URLS = pd.Series([
    "https://www.example.com",
    "http://login.evil.com/verify?id=1",
    "HTTPS://WWW.Shop.co.uk:8443/cart",
    "http://192.168.0.1/admin",
])


@pytest.mark.parametrize("url, host", [
    ("http://paypal.com@evil.top/login", "evil.top"),          # the browser goes to evil.top
    ("http://www.paypal.com:911@xn--80aaa.xn--p1ai/fix", "xn--80aaa.xn--p1ai"),
    ("http://[2001:db8::1]/login", "2001:db8::1"),
    ("www.example.com:8080/a", "www.example.com"),             # no scheme
    ("HTTPS://WWW.EXAMPLE.COM", "www.example.com"),
    ("http://[bad", "bad"),                                     # malformed IPv6 doesn't crash
    ("", ""),
])
def test_parse_host_edge_cases(url, host):
    assert parse_host(url) == host


def test_strip_prefix_removes_scheme_and_www_only():
    assert strip_prefix(URLS).tolist() == [
        "example.com", "login.evil.com/verify?id=1", "Shop.co.uk:8443/cart", "192.168.0.1/admin",
    ]


def test_host_only_drops_everything_but_the_host():
    assert host_only(URLS).tolist() == ["example.com", "login.evil.com", "shop.co.uk", "192.168.0.1"]


def test_registered_domain_drops_subdomains_and_platform_customers():
    urls = pd.Series(["http://login.evil.com/x", "https://foo.web.app", "https://a.b.co.uk"])
    assert registered_domain(urls).tolist() == ["evil.com", "web.app", "b.co.uk"]


def test_hostnames_never_returns_missing_values():
    assert not hostnames(pd.Series(["no-scheme.com/x", "", "http://[bad"])).isna().any()


def test_views_keep_the_index():
    urls = URLS.set_axis([5, 6, 7, 8])
    for view in URL_VIEWS.values():
        assert view(urls).index.tolist() == [5, 6, 7, 8]


def test_column_groups_cover_every_numeric_column_once():
    if not data.RAW_PATH.exists():
        pytest.skip("raw dataset not present")
    numeric = set(data.load_raw().drop(columns="label").select_dtypes("number").columns)
    grouped = [c for cols in COLUMN_GROUPS.values() for c in cols]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == numeric
