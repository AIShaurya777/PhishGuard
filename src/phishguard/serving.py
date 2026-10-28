"""What the project website runs (Milestone 8): the saved models, plus the numbers and warnings shown next to them.

Build once (a few minutes; trains the two n-gram models, nothing else):
    .venv/bin/python -m phishguard.serving
That writes models/ngram_domain_only.joblib, models/train_owners.csv.gz and
models/site_numbers.json. The website (phishguard.web) only loads these files
and the hybrid from notebook 10: it never trains.

Every number the website quotes about robustness is recomputed here from the
same saved models it serves, with the method of notebook 11 (D15).
"""

import json

import joblib
import numpy as np
import pandas as pd

from phishguard.data import REPO_ROOT, TARGET, load_clean
from phishguard.deep import CharURLClassifier
from phishguard.evaluate import best_threshold, evaluate
from phishguard.experiment import load_record
from phishguard.explain import explain_url
from phishguard.features import extract_features
from phishguard.lookalike import LookAlikeRule
from phishguard.modeling import char_ngram_model
from phishguard.robustness import LEGIT_SUBDOMAINS, add_subdomain, all_edits, external_sites, https_www_root, switch_to_https
from phishguard.split import get_splits, site_owner
from phishguard.urls import TLD_EXTRACT, URL_VIEWS

MODELS_DIR = REPO_ROOT / "models"
HYBRID_PATH = MODELS_DIR / "hybrid_domain_only.pt"
NGRAM_PATH = MODELS_DIR / "ngram_domain_only.joblib"
OWNERS_PATH = MODELS_DIR / "train_owners.csv.gz"
NUMBERS_PATH = MODELS_DIR / "site_numbers.json"

# Real and look-alike brand domains, scored by both models (shown on the website).
BRAND_EXAMPLES = ["google.com", "g0ogle.com", "goog1e.com", "paypal.com", "paypa1.com"]

MAX_URL_LENGTH = 2000
# A suffix "leans" one way when it has enough training rows and a lopsided label share.
PRIOR_MIN_ROWS = 30
PRIOR_PHISHING, PRIOR_LEGIT = 0.60, 0.15

FEATURE_LABELS = {
    "num_subdomains": "number of subdomains",
    "on_hosting_platform": "hosted on a platform (e.g. web.app)",
    "host_length": "length of the domain",
    "domain_length": "length of the site name",
    "tld_length": "length of the suffix",
    "host_dots": "dots in the domain",
    "host_hyphens": "hyphens in the domain",
    "host_digits": "digits in the domain",
    "has_ip": "is an IP address",
    "has_punycode": "uses punycode (xn--)",
    "is_shortener": "is a link shortener",
    "suspicious_words": "words like login, verify, secure",
    "longest_token": "longest word",
    "url_entropy": "randomness of the characters",
    "host_entropy": "randomness of the domain",
    "url_length": "length",
    "num_letters": "letters",
    "num_digits": "digits",
    "letter_ratio": "share of letters",
    "digit_ratio": "share of digits",
    "special_ratio": "share of symbols",
    "num_dots": "dots",
    "num_hyphens": "hyphens",
}


def build(seed: int = 42) -> dict:
    """Train the n-gram models on the seed-42 split and save everything the website needs."""
    df = load_clean()
    y = df[TARGET]
    train_idx, val_idx, test_idx = get_splits(df, seed=seed)
    domain = URL_VIEWS["domain only"](df["URL"])

    hybrid = CharURLClassifier.load(HYBRID_PATH, device="cpu")
    hybrid_card = json.loads(HYBRID_PATH.with_suffix(".json").read_text())
    ngram_C = load_record("m3_views", "Char n-gram LogReg | domain only")["reference"]["params"]["C"]
    ngram = char_ngram_model(ngram_C).fit(domain.loc[train_idx], y.loc[train_idx])
    ngram_threshold = best_threshold(y.loc[val_idx], ngram.predict_proba(domain.loc[val_idx])[:, 1])

    predict = {
        "hybrid": lambda d: _hybrid_proba(hybrid, d),
        "ngram": lambda d: ngram.predict_proba(d)[:, 1],
    }
    threshold = {"hybrid": hybrid_card["threshold"], "ngram": ngram_threshold}

    def flagged(name, domains):
        return float((predict[name](domains) >= threshold[name]).mean())

    numbers = {
        "split": f"seed {seed}",
        "models": {
            "hybrid": {"threshold": threshold["hybrid"], "test_accuracy": hybrid_card["test_accuracy"]},
            "ngram": {"C": ngram_C, "threshold": ngram_threshold,
                      "test_accuracy": evaluate(y.loc[test_idx], predict["ngram"](domain.loc[test_idx]),
                                                ngram_threshold)["accuracy"]},
        },
    }

    # Popular real websites outside PhiUSIIL (notebook 11, part D).
    sites = external_sites(set(site_owner(df["URL"])))
    numbers["real_sites"] = {name: {"sites": len(hosts), **{m: flagged(m, hosts) for m in predict}}
                             for name, hosts in sites.items()}

    # Legitimate test domains, before and after adding a subdomain (part C).
    yt = y.loc[test_idx]
    legit = pd.Series(domain.loc[test_idx][yt == 0].unique())
    numbers["legit_with_subdomain"] = {"domains": len(legit), "original": {m: flagged(m, legit) for m in predict}}
    for prefix in LEGIT_SUBDOMAINS:
        numbers["legit_with_subdomain"][prefix] = {m: flagged(m, legit.map(lambda d: add_subdomain(d, prefix)).dropna())
                                                   for m in predict}

    # Phishing test domains, before and after every applicable edit (part B).
    phish = pd.Series(domain.loc[test_idx][yt == 1].unique())
    edited = phish.map(all_edits)
    applies = edited.notna()
    numbers["all_edits"] = {"domains": int(applies.sum()),
                            **{m: {"before": flagged(m, phish[applies]), "after": flagged(m, edited[applies])}
                               for m in predict}}

    # The raw-URL model and the HTTPS attack (part A): why the website reads only the domain.
    raw_C = load_record("m3_views", "Char n-gram LogReg | raw URL")["reference"]["params"]["C"]
    raw = char_ngram_model(raw_C).fit(df.loc[train_idx, "URL"], y.loc[train_idx])
    raw_threshold = best_threshold(y.loc[val_idx], raw.predict_proba(df.loc[val_idx, "URL"])[:, 1])
    phish_urls = df.loc[test_idx, "URL"][yt == 1]
    numbers["raw_url_attacks"] = {}
    for name, edit in {"switch to HTTPS": switch_to_https, "https://www. + no path": https_www_root}.items():
        after = phish_urls.map(edit).dropna()
        before = phish_urls.loc[after.index]
        numbers["raw_url_attacks"][name] = {
            "urls": len(after),
            "before": float((raw.predict_proba(before)[:, 1] >= raw_threshold).mean()),
            "after": float((raw.predict_proba(after)[:, 1] >= raw_threshold).mean()),
        }

    # Real brands vs look-alikes: the models never learned what a brand is.
    examples = pd.Series(BRAND_EXAMPLES)
    numbers["brand_examples"] = [{"domain": d, **{m: float(p) for m, p in zip(predict, ps)}}
                                 for d, *ps in zip(examples, *(predict[m](examples) for m in predict))]

    # The look-alike rule (not a model): how often it fires on phishing and on popular real sites.
    rule = LookAlikeRule.from_tranco()

    def fires(domains):
        return float(pd.Series(domains).map(rule.match).notna().mean())

    numbers["lookalike_rule"] = {
        "PhiUSIIL test phishing": fires(phish),
        "PhiUSIIL test legitimate": fires(legit),
        **{name: fires(hosts) for name, hosts in sites.items() if name.startswith("CrUX") and len(hosts) > 100},
    }

    # Per-suffix phishing share in training: the "country prior" behind many false alarms.
    train_suffix = domain.loc[train_idx].map(lambda h: TLD_EXTRACT(h).suffix)
    by_suffix = y.loc[train_idx].groupby(train_suffix.to_numpy()).agg(["size", "mean"])
    numbers["suffix_priors"] = {s: {"train_rows": int(r["size"]), "phishing_share": round(float(r["mean"]), 4)}
                                for s, r in by_suffix.iterrows() if s}

    # Site owners seen in training: a check on those sites says nothing about new sites.
    owners = y.loc[train_idx].groupby(site_owner(df.loc[train_idx, "URL"]).to_numpy()).agg(["size", "mean"])
    owners.rename_axis("owner").rename(columns={"size": "rows", "mean": "phishing_share"}).round(4).to_csv(
        OWNERS_PATH, compression={"method": "gzip", "mtime": 0})  # mtime 0: same data, same file

    joblib.dump(ngram, NGRAM_PATH)
    NUMBERS_PATH.write_text(json.dumps(numbers, indent=2) + "\n")
    return numbers


def _hybrid_proba(hybrid: CharURLClassifier, domains: pd.Series) -> np.ndarray:
    domains = pd.Series(domains).reset_index(drop=True)
    return hybrid.predict_proba(extract_features(domains).assign(url=domains.to_numpy()))[:, 1]


class Checker:
    """Checks one link the way the website shows it. Load once, then call check(url)."""

    def __init__(self, hybrid, ngram, numbers: dict, owners: pd.DataFrame, lookalike: LookAlikeRule | None = None):
        self.hybrid, self.ngram, self.numbers, self.owners = hybrid, ngram, numbers, owners
        self.lookalike = lookalike

    @classmethod
    def load(cls, models_dir=MODELS_DIR) -> "Checker":
        missing = [p.name for p in (NGRAM_PATH, OWNERS_PATH, NUMBERS_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{', '.join(missing)} not found: run  .venv/bin/python -m phishguard.serving")
        return cls(
            hybrid=CharURLClassifier.load(models_dir / HYBRID_PATH.name, device="cpu"),
            ngram=joblib.load(models_dir / NGRAM_PATH.name),
            numbers=json.loads((models_dir / NUMBERS_PATH.name).read_text()),
            owners=pd.read_csv(models_dir / OWNERS_PATH.name, index_col="owner", keep_default_na=False),
            lookalike=LookAlikeRule.from_tranco(),
        )

    def check(self, url: str) -> dict:
        url = url.strip()
        if not url:
            raise ValueError("Paste a link to check.")
        if len(url) > MAX_URL_LENGTH:
            raise ValueError(f"That link is longer than {MAX_URL_LENGTH:,} characters. Paste a shorter one.")
        domain = URL_VIEWS["domain only"](pd.Series([url])).iloc[0]
        ext = TLD_EXTRACT(domain)
        if not domain or not (ext.suffix or ext.ipv4):
            raise ValueError("No domain name found in that link. A link looks like example.com or https://example.com/page.")

        explained = explain_url(self.hybrid, url)
        models = self.numbers["models"]
        hybrid_p = explained["probability"]
        ngram_p = float(self.ngram.predict_proba(pd.Series([domain]))[0, 1])
        values = extract_features(pd.Series([domain])).iloc[0]
        top = explained["feature_scores"].head(6)
        return {
            "url": url,
            "domain": domain,
            "parts": display_parts(url, domain),
            "char_scores": [round(float(s), 4) for s in explained["char_scores"]],
            "models": {
                "hybrid": {"probability": hybrid_p, "threshold": models["hybrid"]["threshold"],
                           "phishing": hybrid_p >= models["hybrid"]["threshold"]},
                "ngram": {"probability": ngram_p, "threshold": models["ngram"]["threshold"],
                          "phishing": ngram_p >= models["ngram"]["threshold"]},
            },
            "features": [{"name": FEATURE_LABELS.get(name, name.replace("_", " ")),
                          "value": round(float(values[name]), 2), "score": round(float(score), 3)}
                         for name, score in top.items()],
            "lookalike": self.lookalike_note(domain),
            "warnings": self.warnings(domain, hybrid_p >= models["hybrid"]["threshold"],
                                      ngram_p >= models["ngram"]["threshold"]),
        }

    def warnings(self, domain: str, hybrid_says_phishing: bool, ngram_says_phishing: bool) -> list[str]:
        """Reasons to distrust this particular result, each backed by a number we measured."""
        ext = TLD_EXTRACT(domain)
        real = self.numbers["real_sites"]
        notes = []

        owner = ext.top_domain_under_public_suffix or domain
        if owner in self.owners.index:
            row = self.owners.loc[owner]
            notes.append(f"{owner} is in our training data: {int(row['rows']):,} links, {row['phishing_share']:.0%} of "
                         "them phishing. The model may simply remember that, so this result says nothing about new sites.")

        if ext.subdomain and not ext.is_private:
            share = real["CrUX top 10k: with a subdomain"]["hybrid"]
            notes.append(f"This link has a subdomain ({ext.subdomain}.). Legitimate sites in our data almost never do, "
                         f"so the hybrid model flags {share:.0%} of popular real sites with one, like mail.yahoo.com.")

        prior = self.numbers["suffix_priors"].get(ext.suffix)
        if prior and prior["train_rows"] >= PRIOR_MIN_ROWS:
            share = prior["phishing_share"]
            if share >= PRIOR_PHISHING:
                notes.append(f"In our training data, {share:.0%} of .{ext.suffix} links were phishing. The model learned "
                             f"to distrust .{ext.suffix} itself, so real .{ext.suffix} sites get flagged more often.")
            elif share <= PRIOR_LEGIT:
                notes.append(f"In our training data, only {share:.0%} of .{ext.suffix} links were phishing, so the model "
                             f"gives .{ext.suffix} sites the benefit of the doubt. A phishing site here could slip through.")

        if hybrid_says_phishing != ngram_says_phishing:
            notes.append("The two models disagree. Treat this result as uncertain.")
        return notes


    def lookalike_note(self, domain: str) -> dict | None:
        """The look-alike rule's finding (a rule, not a model), with how often the rule fires."""
        found = self.lookalike.match(domain) if self.lookalike else None
        if found is None:
            return None
        site, how = found
        rates = self.numbers["lookalike_rule"]
        return {"site": site, "how": how,
                "caught_phishing": rates["PhiUSIIL test phishing"],
                "popular_sites_flagged": max(v for k, v in rates.items() if k.startswith("CrUX"))}


def display_parts(url: str, domain: str) -> list[dict]:
    """Split the typed link into the part the model reads and the parts it ignores.

    https://www.Example.com/login -> [https://www. (ignored), Example.com (read), /login (ignored)]
    """
    start = url.find("://") + 3 if "://" in url else 0
    at = url.lower().find(domain, start)
    if at < 0:  # e.g. an internationalised domain the parser rewrote
        return [{"text": domain, "read": True}]
    parts = [{"text": url[:at], "read": False}, {"text": url[at:at + len(domain)], "read": True},
             {"text": url[at + len(domain):], "read": False}]
    return [p for p in parts if p["text"]]


if __name__ == "__main__":
    built = build()
    print(json.dumps({k: v for k, v in built.items() if k != "suffix_priors"}, indent=2))
