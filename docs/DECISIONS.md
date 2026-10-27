# Decision log

One entry per decision that affects the data, the models or the evaluation.
Add yours in the same format and link the evidence. This file becomes the
"Methodology" section of the report.

---

## D1 · Load the full raw file and check its shape
- **Context:** The first cleaning notebook read only 196,403 of 235,795 rows (a truncated Colab upload) and nobody noticed.
- **Decision:** `load_raw()` reads the zip in the repo directly and raises an error unless the shape is exactly (235,795, 55) with no missing values.
- **Evidence:** `notebooks/01_data_loading.ipynb` §1, `tests/test_data.py::test_load_raw_rejects_truncated_file`.

## D2 · Phishing is the positive class (`is_phishing = 1`)
- **Context:** PhiUSIIL uses `label = 1` for legitimate and `label = 0` for phishing. Precision, recall and F1 are reported for class 1.
- **Decision:** Replace `label` with `is_phishing = 1 - label` and drop `label`.
- **Evidence:** notebook §2.

## D3 · Strip trailing `/` from URLs
- **Context:** 36.5% of phishing URLs end in `/` and no legitimate URL does. The difference comes from how the data was collected, so a model would learn the label from it.
- **Decision:** Strip surrounding whitespace and trailing `/` before anything else.
- **Consequence:** The dataset's precomputed columns were computed on the *original* URL, so the slash shortcut survives inside them. `URLLength − len(cleaned URL)` is −1 for every legitimate URL but often 0 or +1 for phishing, so "≠ −1 → phishing" has 100% precision (notebook 06 §4b). That's why our own features use the cleaned URL, and why the hybrid model (Milestone 5) will never mix in the dataset's columns.
- **Evidence:** notebook §3a.

## D4 · Drop label conflicts, then duplicates
- **Context:** After normalisation, 1,590 URLs are duplicates. One URL (`https://www.education.gouv.fr`) appears with both labels.
- **Decision:** Drop every row of a URL that has conflicting labels (2 rows), then keep the first copy of each remaining URL (1,589 rows dropped).
- **Result:** 234,204 rows: 134,849 legitimate (57.6%) and 99,355 phishing (42.4%).
- **Evidence:** notebook §3b and §4.

## D5 · Split 70/15/15 by site owner
- **Context:** With a plain random split, 39% of phishing test URLs have a site owner (e.g. `evil.com` for `a.evil.com`) that also appears in training, so test scores would partly measure recognising the owner rather than detecting phishing.
- **Options compared:**
  - A: hostname (220k groups). Leaves the same-owner leak in place.
  - B: registered domain (176k groups). Puts whole hosting platforms like `web.app` in one split, which tests a harder, different question.
  - C: registered domain where customers of platforms on the public suffix list count separately (198k groups).
- **Decision:** C. `get_splits()` shuffles owners within each class (seed 42) and fills test, then val, then train to 15/15/70% of that class's rows. Each owner joins the class of its majority label. No owner is shared between splits, and each split is 42.4% phishing. `tldextract` is pinned, because its bundled suffix list defines the groups. Split sizes after the parsing fix are in D11.
- **Mixed-label owners:** About 105 owners (≈560 rows) have both phishing and legitimate URLs, e.g. `pinata.cloud` (mostly phishing), `washington.edu` (mostly legitimate), `adobe.com`, `office.com`. These are usually abused or compromised parts of real sites. Their minority-label URLs set a small error floor that no model can avoid.
- **Concentration:** A few owners are large, e.g. `ipfs.io` with about 1,400 phishing URLs. A whole owner always lands in one split, so a single split can be easier or harder than average. That's one reason we report several splits (D11).
- **Rule:** tune on validation only; use test once, for the final table. Every model reports through `evaluate()` (phishing = positive class).
- **Limitation:** Platforms missing from the suffix list (e.g. `weeblysite.com`, 3,079 URLs) still act as a single owner.
- **Evidence:** `notebooks/01b_train_test_split.ipynb`, `tests/test_split.py`.

## D6 · One protocol for every model: tune on validation, threshold on validation, test once
- **Context:** Published results on PhiUSIIL are hard to compare, because papers use different splits and some choose settings using the test set.
- **Decision:**
  - Every model goes through `phishguard.modeling`. `tune()` fits each setting on train and keeps the one with the lowest **validation log-loss**. Log-loss changes smoothly, while accuracy would let a single validation URL decide the winner.
  - `fit_and_test()` picks the decision threshold that minimises validation errors (`best_threshold()`; ties go to the middle of the tied range) and scores test exactly once.
  - `experiment.run()` repeats all of this on 5 splits (D11) and saves the results to `results/<milestone>/`. Summary notebooks only read them.
- **Why choose the threshold too:** The default 0.5 is arbitrary, and a validation-chosen threshold costs nothing in honesty. In practice it changes only a handful of test predictions, so the gain is small. For the app, a stable threshold matters more than one extra correct validation URL.
- **Evidence:** `src/phishguard/modeling.py`, `tests/test_modeling.py`, notebooks 02–05.

## D7 · URL features are computed from the URL string only
- **Context:** The app receives only a URL. It can't supply page-content columns (`LineOfCode`, `NoOfImage`, …) or `URLSimilarityIndex`, which compares the URL against a list of known legitimate URLs.
- **Decision:**
  - `extract_features()` computes 39 numeric features from the URL string: sizes, character counts and ratios, host structure (subdomains, IP, port, `www`), scheme, hosting platform or shortener, path and query structure, suspicious words, and entropy.
  - No network calls and no dataset columns, so the same code runs in the app.
- **Deliberately included:** the `is_https` and `has_www` features. Our first target is the highest accuracy we can get (docs/ROADMAP.md, Milestone 2). Whether these are shortcuts is tested in Milestone 3.
- **Evidence:** `src/phishguard/features.py`, `tests/test_features.py`, `notebooks/03_url_feature_models.ipynb`.

## D8 · Milestone 2 results (mean ± sd over 5 splits, D11)
- **All provided columns:** Random Forest and XGBoost 100.00 ± 0.00%, LogReg 99.99 ± 0.00% (0–3 errors out of 35,130 per split). This reproduces the published 99.79% (Prasad & Chandra 2024) and 99.99% (Bao et al. 2026). With so few errors the three models are indistinguishable from each other and from perfect, so we don't rank them.
- **URL only:**
  - Char n-gram LogReg: 99.69 ± 0.05%.
  - Log-odds average of n-gram + XGBoost: 99.70 ± 0.03%.
  - XGBoost / Random Forest on our 40 URL features: 99.63–99.64 ± 0.03–0.04%.
  - Published URL-only results on a similar host-disjoint split: 99.62–99.84% (Ahamed et al., Frontiers 2026; see `docs/references.md`).
- **Significance (McNemar, seed-42 test set):** the n-gram model is significantly better than the feature models (p = 0.004 vs XGBoost). The averaged model is **not** better than the n-gram model alone (17 vs 17 disagreements, p = 1.0).
- **Remaining errors:** nearly all are phishing sites shaped like a bare `https://www.<domain>` homepage, the shape of every legitimate URL in the dataset.
- **Evidence:** `notebooks/05_m2_results.ipynb`.

## D9 · The provided columns are dominated by shortcuts
- **Context:** In Milestone 2 even Logistic Regression reached 99.99% on the provided columns.
- **Findings (notebook 06; single-column AUCs on the training split, rules and models scored on test):**
  - `URLSimilarityIndex < 100 → phishing` alone: 99.66% accuracy (seed-42 test split), precision 1.0.
  - "Doesn't start with `https://www.` → phishing" alone: 99.08%. Every legitimate URL starts with it; only 2.4% of phishing URLs do.
  - XGBoost on page columns only: 99.71 ± 0.08% (5 splits). They describe what the crawler downloaded.
  - XGBoost on URL columns without `IsHTTPS`: still 99.61 ± 0.06%. `URLLength` minus the counted characters is 12 (`https://www.`) for all but 4 of 134,849 legitimate URLs, and `URLLength` also keeps the trailing slash we removed (D3).
- **Decision:** Column groups are defined in `src/phishguard/columns.py` (19 URL, 3 statistic, 28 page). The all-column results are reported as a *reproduction* of published work, not as evidence of phishing detection. The app and the hybrid model use only our own URL features, never the provided columns.
- **Protocol note:** The rules were first noticed while exploring the full dataset (notebook 01), which is stated openly. Scoring them on test is a check, not a tuned result.
- **Evidence:** `notebooks/06_shortcut_audit.ipynb`.

## D10 · Report every URL model on the raw URL and on shortcut-reduced views
- **Context:** URL-only models reach ~99.7% mostly from the prefix and from whether there's a path.
- **Findings (notebook 07, test accuracy, mean ± sd over 5 splits):**

  | View | Char n-gram LogReg | XGBoost (URL features) |
  |---|---|---|
  | raw URL | 99.70 ± 0.03% | 99.64 ± 0.03% |
  | no scheme/www | 89.18 ± 0.59% | 85.95 ± 0.95% |
  | domain only | 86.27 ± 0.89% | 81.43 ± 1.15% |
  | registered domain | 82.49 ± 1.93% | 72.82 ± 0.70% |

- **Decision:** `phishguard.urls.URL_VIEWS` defines the four inputs. Every later URL model (CNN, BiLSTM, hybrid) is reported on **raw URL** (for comparison with published work) and on **domain only** and **registered domain**. We call the latter two "shortcut-reduced", not "honest".
- **Why "shortcut-reduced":** Even the domain carries a collection difference. Only 6.2% of legitimate domains have a subdomain, a hosting platform, an IP, a digit or `xn--`, against 56.9% of phishing domains, and a no-learning rule on that gets 78%. On bare registered domains, the n-gram model catches only 53% of phishing (ROC-AUC 0.84).
- **Why n-grams win on the domain:** the signal is in the characters of the name (cheap TLDs such as `.cf`/`.ga`/`.top`, digits, hyphens, free-hosting names), which counts miss. That motivates the character-level deep models.
- **Limitations:**
  - Compromised legitimate sites (`smithtony.co.uk`) can't be recognised from the domain name by any URL-only model.
  - At a 1% phishing rate, the domain-only model's precision would be about 15%.
- **Evidence:** `notebooks/07_url_views.ipynb`.

## D11 · Results are mean ± sd over 5 splits; URL parsing fixed; split re-made once
- **Context (senior review, `docs/review/senior_review_m1_m3.md`):**
  1. The first domain-only result (88.59%) came from a single split, which turned out to be the most favourable of six. Other splits gave ~85–87%.
  2. Differences such as 99.70% vs 99.68% were presented as a ranking, without uncertainty.
  3. `urls.hostnames()` used a regex that kept the `user@` part of URLs like `http://www.paypal.com:911@xn--80aaa….xn--p1ai/fix-account`, assigning them to the owner `paypal.com` (10 phishing URLs), and failed on URLs without `http://`.
- **Decision:**
  - **Parsing:** one `parse_host()` built on Python's `urlsplit`, with tests for `@`, ports, IPv6, missing scheme and uppercase. Because 10 owners changed, the seed-42 split changed. We re-made it **once, on purpose**: 164,045 / 35,029 / 35,130 rows, and `tests/test_split.py` now fingerprints it so it can't change silently again.
  - **Every model runs on 5 splits** (seeds 42, 0, 1, 2, 3) through `phishguard.experiment.run()`, re-tuned on each, and is reported as **mean ± sd**. Seed 42 is the reference split for examples, plots, a 95% confidence interval that resamples whole site owners, and McNemar tests between models.
  - **Metrics added:** PR-AUC, MCC, balanced accuracy, TPR at 1% and 0.1% FPR, Brier score, log-loss, and precision at realistic base rates.
  - **Saved with each result:** settings per seed, validation metrics, run time, library versions and git commit, plus seed-42 probabilities (`.npz`). The environment is pinned in `requirements.lock`.
  - **Features** (`features.py`): IP detection covers decimal/hex/IPv6; ports use the parser; suspicious words match whole words; `has_punycode` was added; counts and entropy ignore the `https://www.` prefix.
- **Effect:** URL features 99.68% → 99.63 ± 0.03%; domain-only n-gram 88.59% → 86.27 ± 0.89%. All-column models are unchanged (≈100%).
- **Evidence:** `src/phishguard/{urls,experiment,stats,evaluate}.py`, `tests/`, all notebooks re-run.

## Reporting protocol from Milestone 4 on (fixed in advance)
To avoid choosing what to report after seeing test results, every new model reports exactly this, and nothing is dropped if it looks bad:
- **raw URL, domain only, registered domain** views (D10)
- each as **mean ± sd over seeds 42, 0, 1, 2, 3** of accuracy, F1, ROC-AUC, PR-AUC, MCC and TPR at 1% FPR
- the seed-42 95% CI, and a McNemar test against the char n-gram model on the same view
