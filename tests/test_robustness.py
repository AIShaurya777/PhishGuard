import pytest

from phishguard.robustness import (
    add_subdomain, all_edits, drop_subdomain, https_www_root, leave_hosting_platform, look_alike_letters,
    remove_digits_hyphens, switch_to_https, tld_to_com,
)


@pytest.mark.parametrize("edit, domain, expected", [
    (tld_to_com, "login.paypal-check.cf", "login.paypal-check.com"),
    (tld_to_com, "paypal.com", None),                       # already .com
    (tld_to_com, "my-wallet.web.app", None),                # platform suffix can't be changed
    (tld_to_com, "192.168.0.1", None),                      # IP address
    (drop_subdomain, "login.paypal-check.cf", "paypal-check.cf"),
    (drop_subdomain, "paypal-check.cf", None),
    (drop_subdomain, "a.b.my-site.co.uk", "my-site.co.uk"),
    (remove_digits_hyphens, "pay-pal-2fa.top", "paypalfa.top"),
    (remove_digits_hyphens, "123.example.org", None),       # a label would become empty
    (remove_digits_hyphens, "example.org", None),
    (look_alike_letters, "paypa1-l0gin.top", "paypal-login.top"),
    (look_alike_letters, "example.org", None),
    (leave_hosting_platform, "my-wallet.web.app", "my-wallet.com"),
    (leave_hosting_platform, "my-wallet.top", None),
    (all_edits, "secure-1.paypa1-check.cf", "paypalcheck.com"),
    (all_edits, "example.com", None),
])
def test_domain_edits(edit, domain, expected):
    assert edit(domain) == expected


def test_add_subdomain():
    assert add_subdomain("example.org", "mail") == "mail.example.org"
    assert add_subdomain("10.0.0.1", "mail") is None


def test_raw_url_attacks():
    assert https_www_root("http://login.evil.top/x?id=1") == "https://www.login.evil.top"
    assert https_www_root("https://www.evil.top") is None              # already looks legitimate
    assert https_www_root("https://site.web.app/login") is None        # hosting platform: not attacker-controlled
    assert switch_to_https("http://evil.top/a") == "https://evil.top/a"
    assert switch_to_https("https://evil.top/a") is None
