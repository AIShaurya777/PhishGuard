import pandas as pd
import pytest

from phishguard.serving import NGRAM_PATH, Checker, display_parts

NUMBERS = {
    "real_sites": {"CrUX top 10k: with a subdomain": {"hybrid": 0.81}},
    "suffix_priors": {"ru": {"train_rows": 900, "phishing_share": 0.79},
                      "org": {"train_rows": 5000, "phishing_share": 0.05},
                      "museum": {"train_rows": 3, "phishing_share": 1.0}},
}
OWNERS = pd.DataFrame({"rows": [782], "phishing_share": [1.0]}, index=pd.Index(["google.com"], name="owner"))


def warnings_for(domain, hybrid=True, ngram=True):
    return Checker(hybrid=None, ngram=None, numbers=NUMBERS, owners=OWNERS).warnings(domain, hybrid, ngram)


@pytest.mark.parametrize("url, domain, expected", [
    ("https://www.Example.com/login", "example.com",
     [("https://www.", False), ("Example.com", True), ("/login", False)]),
    ("example.com", "example.com", [("example.com", True)]),
    ("http://paypal.com@evil.top/x", "evil.top",          # the part before @ is not the host
     [("http://paypal.com@", False), ("evil.top", True), ("/x", False)]),
])
def test_display_parts(url, domain, expected):
    assert [(p["text"], p["read"]) for p in display_parts(url, domain)] == expected


def test_warnings_explain_known_weak_spots():
    assert any("in our training data: 782 links, 100%" in w for w in warnings_for("accounts.google.com"))
    assert any("subdomain (mail.)" in w and "81%" in w for w in warnings_for("mail.example.org"))
    assert any("79% of .ru" in w for w in warnings_for("yandex.ru"))
    assert any("benefit of the doubt" in w for w in warnings_for("example.org"))
    assert any("disagree" in w for w in warnings_for("example.net", hybrid=True, ngram=False))


def test_no_warnings_without_evidence():
    assert warnings_for("example.museum") == []         # too few training rows to call it a prior
    assert warnings_for("my-site.web.app") == []        # platform customers are their own site, not a subdomain


@pytest.mark.skipif(not NGRAM_PATH.exists(), reason="website models not built: python -m phishguard.serving")
def test_website_api():
    from fastapi.testclient import TestClient
    from phishguard.web import app

    with TestClient(app) as client:
        result = client.post("/api/check", json={"url": "https://mail.yahoo.com/inbox"}).json()
        assert result["domain"] == "mail.yahoo.com"
        assert len(result["char_scores"]) == len("mail.yahoo.com")
        assert set(result["models"]) == {"hybrid", "ngram"}
        assert result["warnings"] and result["lookalike"] is None
        assert client.post("/api/check", json={"url": "http://paypa1.com/x"}).json()["lookalike"]["site"] == "paypal.com"

        for bad in ["", "   ", "not a link", "x" * 3000]:
            response = client.post("/api/check", json={"url": bad})
            assert response.status_code == 422 and isinstance(response.json()["detail"], str)

        numbers = client.get("/api/numbers").json()
        assert len(numbers["accuracy"]) == 9 and "suffix_priors" not in numbers
        assert [r["domain"] for r in numbers["brand_examples"]][:2] == ["google.com", "g0ogle.com"]
        assert client.get("/").status_code == 200
