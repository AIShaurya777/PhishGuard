import pytest

from phishguard.lookalike import LookAlikeRule, one_edit_apart

RULE = LookAlikeRule(["google.com", "paypal.com", "yahoo.com", "github.com"])


@pytest.mark.parametrize("domain, site", [
    ("g0ogle.com", "google.com"),                  # digit for a look-alike letter
    ("paypa1.top", "paypal.com"),
    ("gooogle.com", "google.com"),                 # one letter added
    ("paypal-login.top", "paypal.com"),            # brand name inside another name
    ("accounts.google.com.evil.top", "google.com"),
    ("google.fr", "google.com"),                   # same name, other suffix: could be the company's own site
])
def test_finds_look_alikes(domain, site):
    assert RULE.match(domain)[0] == site


@pytest.mark.parametrize("domain", [
    "google.com", "mail.google.com", "mail.yahoo.com",   # the real sites
    "example.org", "bbc.co.uk", "10.0.0.1",
    "yaho.org",                                           # one-letter edits only count for names of 6+ letters
])
def test_leaves_other_domains_alone(domain):
    assert RULE.match(domain) is None


def test_one_edit_apart():
    assert one_edit_apart("google", "gooogle") and one_edit_apart("google", "googie") and one_edit_apart("google", "goole")
    assert not one_edit_apart("google", "google") and not one_edit_apart("google", "elgoog")
