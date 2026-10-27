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
- **Deliberately included:** the `is_https` and `has_www` features. Our first target is the highest accuracy we can get (docs/ROADMAP.md, Milestone 2). Whether these are shortcuts is tested in Milestone 3.
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

## D9 · The provided columns are dominated by shortcuts
- **Context:** In Milestone 2 even Logistic Regression reached 99.99% on the provided columns.
- **Findings (notebook 06, test set):**
  - `URLSimilarityIndex < 100 → phishing` alone: 99.72% accuracy.
  - "Doesn't start with `https://www.` → phishing" alone: 99.16%. Every legitimate URL starts with it; only 2.4% of phishing URLs do.
  - Page columns only: 99.64%. They measure what the crawler downloaded.
  - URL columns without `IsHTTPS`: still 99.56%, because `URLLength` minus the counted characters is exactly 12 (`https://www.`) for legitimate URLs.
- **Decision:** Column groups are defined in `src/phishguard/columns.py` (19 URL, 3 statistic, 28 page). The all-column results are reported as a *reproduction* of published work, not as evidence of phishing detection. The app can't use statistic or page columns.
- **Evidence:** `notebooks/06_shortcut_audit.ipynb`, `tests/test_urls.py`.

## D10 · Report every URL model on the raw URL and on the domain only
- **Context:** URL-only models reach 99.7% mostly from the prefix and from whether there's a path.
- **Findings (notebook 07, test accuracy):**

  | View | Char n-gram LogReg | XGBoost (URL features) |
  |---|---|---|
  | raw URL | 99.70% | 99.68% |
  | no scheme/www | 90.89% | 87.97% |
  | domain only | 88.59% | 81.42% |

- **Decision:** `phishguard.urls.URL_VIEWS` defines the three inputs. Every later URL model (CNN, BiLSTM, hybrid) is reported on **raw URL**, for comparison with published work, and **domain only**, the honest number.
- **Why n-grams win on the domain:** the signal lies in the characters of the name (cheap TLDs such as `.cf`/`.ga`/`.top`, digits, hyphens, free-hosting subdomains), which counts miss. That motivates the character-level deep models.
- **Limitation:** compromised legitimate sites (`cama.it`) can't be recognised from the domain name by any URL-only model.
- **Evidence:** `notebooks/07_url_views.ipynb`.
