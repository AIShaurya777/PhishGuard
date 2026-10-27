import pandas as pd
import pytest

from phishguard import data
from phishguard.features import extract_features


def features_of(url):
    return extract_features(pd.Series([url])).iloc[0]


def test_ip_host_with_login_path():
    f = features_of("http://192.168.0.1:8080/secure/login.php?id=1&u=2")
    assert f["has_ip"] == 1
    assert f["has_port"] == 1
    assert f["is_https"] == 0
    assert f["path_depth"] == 2
    assert f["num_query_params"] == 2
    assert f["suspicious_words"] == 2  # "secure", "login"


def test_bare_homepage():
    f = features_of("https://www.example.com")
    assert (f["is_https"], f["has_www"], f["has_ip"]) == (1, 1, 0)
    assert f["num_subdomains"] == 1  # "www"
    assert f["path_depth"] == 0
    assert f["domain_length"] == len("example")
    assert f["tld_length"] == len("com")


def test_brand_in_subdomain_and_hosting_platform():
    f = features_of("https://paypal.com.account-verify.web.app/x")
    assert f["on_hosting_platform"] == 1
    assert f["host_hyphens"] == 1
    assert f["suspicious_words"] == 3  # "paypal", "account", "verify"


def test_url_without_scheme_still_parses_host():
    f = features_of("bit.ly/abc")
    assert f["is_shortener"] == 1
    assert f["is_https"] == 0
    assert f["host_length"] == len("bit.ly")


@pytest.mark.parametrize("url", [
    "http://3232235521/login",          # decimal form of 192.168.0.1
    "http://0x7f.0x0.0x0.0x1/x",        # hex form of 127.0.0.1
    "http://[2001:db8::1]/login",       # IPv6
])
def test_has_ip_recognises_other_ip_forms(url):
    assert features_of(url)["has_ip"] == 1


def test_ipv6_brackets_are_not_a_port():
    assert features_of("http://[2001:db8::1]/login")["has_port"] == 0


def test_suspicious_words_match_whole_words_only():
    assert features_of("https://www.freedomhouse.org")["suspicious_words"] == 0
    assert features_of("http://free-bonus.top/login")["suspicious_words"] == 3


def test_punycode_host():
    assert features_of("http://xn--pypal-4ve.com")["has_punycode"] == 1


def test_counts_ignore_the_scheme_and_www_prefix():
    a, b = extract_features(pd.Series(["https://www.example.com", "http://example.com"])).to_dict("records")
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"is_https", "has_www", "host_length", "host_dots", "num_subdomains", "host_entropy"}


def test_entropy_is_higher_for_random_strings():
    plain, random_ = extract_features(pd.Series(["http://aaaa.com", "http://x7q9kz.com"]))["url_entropy"]
    assert random_ > plain


def test_output_shape_index_and_no_missing_values():
    urls = pd.Series(["https://a.com", "http://[bad", "", "ftp://x.org/a/b"], index=[10, 11, 12, 13])
    out = extract_features(urls)
    assert list(out.index) == [10, 11, 12, 13]
    assert not out.isna().any().any()


@pytest.mark.skipif(not data.RAW_PATH.exists(), reason="raw dataset not present")
def test_real_dataset_has_no_missing_values():
    df = data.load_clean()
    out = extract_features(df["URL"].head(5000))
    assert not out.isna().any().any()
