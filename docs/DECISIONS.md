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

## Open questions (next milestone)
- Which of the 54 provided columns may a URL-only model use? Every legitimate URL has `IsHTTPS = 1` and `URLSimilarityIndex = 100`, and many columns describe page content rather than the URL. (Shaurya, D6)
