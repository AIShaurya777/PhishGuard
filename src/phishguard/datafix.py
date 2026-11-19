"""Milestone 7b: does better training data fix the real-world failures of Milestone 7?

Two training sets, same models, same protocol, same tests:
- "PhiUSIIL only": the domain view of PhiUSIIL's train split (as in Milestones 3–5);
- "PhiUSIIL + fixes": the same rows plus four additions aimed at the failures we found:
  1. real legitimate sites: a fixed half of the Tranco top 20k (split by site owner);
  2. legitimate subdomains: copies of legitimate training domains with a prefix
     from TRAIN_PREFIXES, which shares nothing with the prefixes we test with;
  3. more, newer phishing: one domain per owner from Phishing.Database;
  4. edited phishing: copies of phishing training domains with one realistic edit.

Tests never share a site owner with anything a model trains on (pools below):
- PhiUSIIL's own test split (accuracy: how much does the fix cost in-dataset?);
- CrUX top sites and the other half of Tranco (false alarms on real sites);
- PhishTank phishing submitted from 2025 on (phishing caught that's new to every model);
- the Milestone 7 attacks: every edit at once, and "mail." on legitimate domains.

Run:  python -m phishguard.datafix --model ngram  (CPU, minutes)
      python -m phishguard.datafix --model hybrid (GPU; use the lab server)
Results: results/m7b/<model>__<variant>.json
"""

import argparse
import gzip
import hashlib
import json
import time

import numpy as np
import pandas as pd

from phishguard.data import TARGET, load_clean
from phishguard.evaluate import RESULTS_DIR, best_threshold, evaluate
from phishguard.features import extract_features
from phishguard.modeling import char_ngram_model
from phishguard.robustness import EXTERNAL_DIR, PHISHING_EDITS, add_subdomain, all_edits, site_kind
from phishguard.split import get_splits, site_owner
from phishguard.urls import URL_VIEWS

SEEDS = [42, 0, 1, 2, 3]
# Prefixes added to legitimate training domains. Disjoint from robustness.LEGIT_SUBDOMAINS
# (www2, mail, blog, shop, login, accounts), which the tests use.
TRAIN_PREFIXES = ["app", "docs", "support", "news", "m", "en", "help", "store", "portal", "status"]
AUGMENT_SHARE = 0.25            # share of training domains that get a subdomain copy / an edited copy
PDB_SAMPLE = 25_000             # Phishing.Database domains added (one per owner)
PHISHTANK_FROM = "2025-01-01"   # "new" phishing: submitted after PhiUSIIL was published
VAL_SHARE = 0.15                # share of each addition's owners kept for validation
OUT_DIR = RESULTS_DIR / "m7b"
VARIANTS = ["PhiUSIIL only", "PhiUSIIL + fixes"]

domain_view = URL_VIEWS["domain only"]


def _in_half(owner: str, salt: str = "m7b") -> bool:
    """A fixed, seed-independent coin flip per owner (hash), so the Tranco halves never change."""
    return int(hashlib.sha256(f"{salt}:{owner}".encode()).hexdigest(), 16) % 2 == 0


def build_pools(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Fixed sets of distinct domains. Training pools and test pools never share a site owner."""
    phiusiil = set(site_owner(df["URL"]))

    tranco = pd.read_csv(EXTERNAL_DIR / "tranco_GQ3QK_top20k.csv")["domain"].drop_duplicates()
    tranco = tranco[~site_owner(tranco).isin(phiusiil)].reset_index(drop=True)
    tranco_owner = site_owner(tranco)
    tranco_train = tranco[tranco_owner.map(_in_half)].reset_index(drop=True)
    tranco_test = tranco[~tranco_owner.map(_in_half)].reset_index(drop=True)

    crux = domain_view(pd.read_csv(EXTERNAL_DIR / "crux_top10k_2026-10.csv")["origin"]).drop_duplicates()
    phishtank = pd.read_csv(EXTERNAL_DIR / "phishtank_online-valid_2026-11-19.csv.gz")
    phishtank = phishtank[pd.to_datetime(phishtank["submission_time"]) >= PHISHTANK_FROM]
    phishtank = domain_view(phishtank["url"]).drop_duplicates()

    # Owners nobody may train on: everything in a test pool.
    test_owners = set(site_owner(tranco_test)) | set(site_owner(crux)) | set(site_owner(phishtank))

    with gzip.open(EXTERNAL_DIR / "phishing-database_domains-active_2026-11-19.txt.gz", "rt") as f:
        pdb = pd.Series([line.strip() for line in f if line.strip() and not line.startswith("#")])
    pdb = pdb.str.lower().drop_duplicates()
    pdb_owner = site_owner(pdb)
    keep = ~pdb_owner.isin(phiusiil | test_owners | set(site_owner(tranco_train)))
    # One domain per owner, so huge hosting owners (weeblysite.com: 9,432 domains) count once.
    pdb = pdb[keep].groupby(pdb_owner[keep].to_numpy()).first()
    pdb = pdb.sample(n=min(PDB_SAMPLE, len(pdb)), random_state=0).reset_index(drop=True)

    train_owners = phiusiil | set(site_owner(tranco_train)) | set(site_owner(pdb))
    fresh = lambda s: s[~site_owner(s).isin(train_owners)].reset_index(drop=True)
    crux, phishtank = fresh(crux), fresh(phishtank)
    kind = crux.map(site_kind)
    return {
        "tranco_train": tranco_train,
        "pdb_train": pdb,
        "tranco_test": fresh(tranco_test),
        "crux_bare": crux[kind == "bare domain"].reset_index(drop=True),
        "crux_subdomain": crux[kind == "with a subdomain"].reset_index(drop=True),
        "phishtank_new": phishtank,
    }


def _owner_split(domains: pd.Series, seed: int) -> tuple[pd.Series, pd.Series]:
    """Train / validation parts of an addition, whole owners only."""
    owners = site_owner(domains)
    unique = owners.drop_duplicates().sample(frac=1, random_state=seed)
    val_owners = set(unique.iloc[: int(len(unique) * VAL_SHARE)])
    is_val = owners.isin(val_owners)
    return domains[~is_val], domains[is_val]


def _augment(domains: pd.Series, edit, seed: int) -> pd.Series:
    """Apply `edit(domain, rng)` to a random AUGMENT_SHARE of distinct domains; drop where it doesn't apply."""
    rng = np.random.default_rng(seed)
    distinct = pd.Series(domains.unique())
    chosen = distinct.sample(frac=AUGMENT_SHARE, random_state=seed)
    return chosen.map(lambda d: edit(d, rng)).dropna().drop_duplicates()


def _add_prefix(domain: str, rng) -> str | None:
    return add_subdomain(domain, TRAIN_PREFIXES[rng.integers(len(TRAIN_PREFIXES))])


def _random_edit(domain: str, rng) -> str | None:
    edits = list(PHISHING_EDITS.values())
    return edits[rng.integers(len(edits))](domain)


def make_sets(df: pd.DataFrame, pools: dict, seed: int, variant: str) -> dict:
    """(domains, labels) for train and validation, and the PhiUSIIL test rows, for one split."""
    y = df[TARGET]
    train_idx, val_idx, test_idx = get_splits(df, seed=seed)
    domain = domain_view(df["URL"])
    train = [domain.loc[train_idx]], [y.loc[train_idx]]
    val = [domain.loc[val_idx]], [y.loc[val_idx]]

    if variant == "PhiUSIIL + fixes":
        for pool, label in [("tranco_train", 0), ("pdb_train", 1)]:
            fit_part, val_part = _owner_split(pools[pool], seed)
            train[0].append(fit_part), train[1].append(pd.Series(label, index=fit_part.index))
            val[0].append(val_part), val[1].append(pd.Series(label, index=val_part.index))
        legit = pd.concat([d[l == 0] for d, l in zip(*train)])
        phish = pd.concat([d[l == 1] for d, l in zip(*train)])
        for extra, label in [(_augment(legit, _add_prefix, seed), 0), (_augment(phish, _random_edit, seed), 1)]:
            train[0].append(extra), train[1].append(pd.Series(label, index=extra.index))

    join = lambda parts: pd.concat(parts, ignore_index=True)
    return {"train": (join(train[0]), join(train[1])), "val": (join(val[0]), join(val[1])),
            "test": (domain.loc[test_idx].reset_index(drop=True), y.loc[test_idx].reset_index(drop=True))}


def external_tests(df: pd.DataFrame, pools: dict, seed: int) -> dict[str, tuple[pd.Series, int]]:
    """Test sets beyond PhiUSIIL: (distinct domains, true label)."""
    _, _, test_idx = get_splits(df, seed=seed)
    test = domain_view(df.loc[test_idx, "URL"])
    yt = df.loc[test_idx, TARGET]
    legit, phish = pd.Series(test[yt == 0].unique()), pd.Series(test[yt == 1].unique())
    return {
        "crux_bare": (pools["crux_bare"], 0),
        "crux_subdomain": (pools["crux_subdomain"], 0),
        "tranco_test": (pools["tranco_test"], 0),
        "phishtank_new": (pools["phishtank_new"], 1),
        "phiusiil_legit_plus_mail": (legit.map(lambda d: add_subdomain(d, "mail")).dropna(), 0),
        "phiusiil_phish_all_edits": (phish.map(all_edits).dropna(), 1),
    }


# --- models: each is (make(setting) -> model, grid, inputs(domains) -> X) ---

def _features(domains: pd.Series) -> pd.DataFrame:
    domains = pd.Series(domains).reset_index(drop=True)
    return extract_features(domains).assign(url=domains.to_numpy())


def _models():
    from phishguard.deep import CharURLClassifier
    return {
        "ngram": (lambda C, seed: char_ngram_model(C), [{"C": C} for C in (1.0, 3.0, 10.0)],
                  lambda d: pd.Series(d).reset_index(drop=True)),
        "hybrid": (lambda dropout, seed: CharURLClassifier(arch="hybrid", dropout=dropout, seed=seed),
                   [{"dropout": p} for p in (0.3, 0.5)], _features),
    }


def run_one(model_name: str, variant: str, seed: int, df: pd.DataFrame, pools: dict) -> dict:
    """Tune on validation log-loss, threshold on validation, then score every test set once."""
    make, grid, inputs = _models()[model_name]
    sets = make_sets(df, pools, seed, variant)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = sets["train"], sets["val"], sets["test"]
    best = None
    for params in grid:
        start = time.time()
        model = make(seed=seed, **params).fit(inputs(Xtr), ytr)
        prob = model.predict_proba(inputs(Xva))[:, 1]
        loss = evaluate(yva, prob, threshold=best_threshold(yva, prob))["log_loss"]
        print(f"  {model_name} | {variant} | seed {seed} | {params}: val log-loss {loss:.4f} ({time.time() - start:.0f}s)")
        if best is None or loss < best[0]:
            best = (loss, params, model, prob)
    _, params, model, val_prob = best
    threshold = best_threshold(yva, val_prob)
    record = {"seed": seed, "params": params, "threshold": threshold, "train_rows": len(Xtr),
              "train_phishing_share": float(ytr.mean()),
              "phiusiil_test": evaluate(yte, model.predict_proba(inputs(Xte))[:, 1], threshold)}
    for name, (domains, label) in external_tests(df, pools, seed).items():
        flagged = model.predict_proba(inputs(domains))[:, 1] >= threshold
        # legitimate sets: share flagged = false alarms; phishing sets: share flagged = caught
        record[name] = {"domains": len(domains), "label": label, "flagged": float(flagged.mean())}
    return record


def summarise(records: list[dict]) -> dict:
    """Mean and sd over seeds of PhiUSIIL accuracy and every external rate."""
    out = {"phiusiil_accuracy": [r["phiusiil_test"]["accuracy"] for r in records],
           "phiusiil_roc_auc": [r["phiusiil_test"]["roc_auc"] for r in records]}
    for name in records[0]:
        if isinstance(records[0][name], dict) and "flagged" in records[0][name]:
            out[name] = [r[name]["flagged"] for r in records]
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["ngram", "hybrid"], required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    df = load_clean()
    pools = build_pools(df)
    print({k: len(v) for k, v in pools.items()})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        records = [run_one(args.model, variant, seed, df, pools) for seed in args.seeds]
        result = {"model": args.model, "variant": variant, "seeds": args.seeds,
                  "pools": {k: len(v) for k, v in pools.items()},
                  "settings": {"train_prefixes": TRAIN_PREFIXES, "augment_share": AUGMENT_SHARE,
                               "pdb_sample": PDB_SAMPLE, "phishtank_from": PHISHTANK_FROM, "val_share": VAL_SHARE},
                  "summary": summarise(records), "per_seed": records}
        slug = variant.lower().replace(" + ", "_plus_").replace(" ", "_")
        path = OUT_DIR / f"{args.model}__{slug}.json"
        path.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"saved {path}")
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
