import pandas as pd

from phishguard import data
from phishguard.columns import COLUMN_GROUPS
from phishguard.urls import URL_VIEWS, host_only, strip_prefix

URLS = pd.Series([
    "https://www.example.com",
    "http://login.evil.com/verify?id=1",
    "HTTPS://WWW.Shop.co.uk:8443/cart",
    "http://192.168.0.1/admin",
])


def test_strip_prefix_removes_scheme_and_www_only():
    assert strip_prefix(URLS).tolist() == [
        "example.com", "login.evil.com/verify?id=1", "Shop.co.uk:8443/cart", "192.168.0.1/admin",
    ]


def test_host_only_drops_everything_but_the_host():
    assert host_only(URLS).tolist() == ["example.com", "login.evil.com", "shop.co.uk", "192.168.0.1"]


def test_views_keep_the_index():
    urls = URLS.set_axis([5, 6, 7, 8])
    for view in URL_VIEWS.values():
        assert view(urls).index.tolist() == [5, 6, 7, 8]


def test_column_groups_cover_every_numeric_column_once():
    if not data.RAW_PATH.exists():
        return
    numeric = set(data.load_raw().drop(columns="label").select_dtypes("number").columns)
    grouped = [c for cols in COLUMN_GROUPS.values() for c in cols]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == numeric
