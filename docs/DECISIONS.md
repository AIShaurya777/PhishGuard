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
- **Consequence:** The dataset's precomputed columns (e.g. `URLLength`) were computed on the original URL, so they may be off by one. Features we engineer ourselves use the cleaned URL.
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
- **Decision:** C. `get_splits()` shuffles owners within each class (seed 42) and fills test, then val, then train to 15/15/70% of that class's rows. Result: 163,944 / 35,130 / 35,130 rows, 42.4% phishing in each, no owner shared between splits. `tldextract` is pinned, because its bundled suffix list defines the groups.
- **Rule:** tune on validation only; use test once, for the final table. Every model reports through `evaluate()` (phishing = positive class).
- **Limitation:** Platforms missing from the suffix list (e.g. `weeblysite.com`, 3,079 URLs) still act as a single owner.
- **Evidence:** `notebooks/01b_train_test_split.ipynb`, `tests/test_split.py`.

## D6 · One protocol for every model: tune on validation, threshold on validation, test once
- **Context:** Published results on PhiUSIIL are hard to compare, because papers use different splits and some choose settings using the test set.
- **Decision:**
  - Every model goes through `phishguard.modeling`. `tune()` fits each setting on train and scores it on validation. `fit_and_test()` fits the chosen setting on train, picks the decision threshold that minimises validation errors (`best_threshold()`), and scores test exactly once.
  - Test scores are saved to `results/*.json`. Notebook 05 only reads them.
- **Why choose the threshold too:** The default 0.5 is arbitrary. The char n-gram model, for example, had the fewest validation errors at 0.16. Choosing it on validation and not on test keeps the test score honest.
- **Evidence:** `src/phishguard/modeling.py`, `tests/test_modeling.py`, notebooks 02–05.

## D7 · URL features are computed from the URL string only
- **Context:** The app receives only a URL. It can't supply page-content columns (`LineOfCode`, `NoOfImage`, …) or `URLSimilarityIndex`, which compares the URL against a list of known legitimate URLs.
- **Decision:**
  - `extract_features()` computes 39 numeric features from the URL string: sizes, character counts and ratios, host structure (subdomains, IP, port, `www`), scheme, hosting platform or shortener, path and query structure, suspicious words, and entropy.
  - No network calls and no dataset columns, so the same code runs in the app.
- **Deliberately included:** the `is_https` and `has_www` features. Our first target is the highest accuracy we can get (M2_PLAN). Whether these are shortcuts is tested in Milestone 3.
- **Evidence:** `src/phishguard/features.py`, `tests/test_features.py`, `notebooks/03_url_feature_models.ipynb`.

## D8 · Milestone 2 results
- **All provided columns:** LogReg 99.994%, Random Forest 99.997%, XGBoost 100% test accuracy (0–2 errors out of 35,130). This reproduces the published 99.79% (Prasad & Chandra 2024) and 99.99% (Bao et al. 2026).
- **URL only:**
  - Random Forest and XGBoost on URL features: 99.68%.
  - Char n-gram LogReg: 99.70%.
  - Average of n-gram + XGBoost: 99.71%.
  - Published URL-only results are 99.62–99.84% (Frontiers in Computer Science, 2026).
- **Remaining errors:** nearly all are phishing sites shaped like a bare `https://www.<domain>` homepage, which is the shape of every legitimate URL in the dataset.
- **Evidence:** `notebooks/05_m2_results.ipynb`.

## Open questions (Milestone 3)
- **How strong are the dataset's shortcuts?** Every legitimate URL starts with `https://www.` (2.4% of phishing URLs do), has no path, and has `URLSimilarityIndex = 100`. The char n-gram model's strongest weights are `p://` (phishing) and `www`/`s://` (legitimate).
- **How much accuracy is left without them?** A first check: char n-gram LogReg drops from 99.73% to 90.42% on validation when `https://www.` is removed from every URL.
