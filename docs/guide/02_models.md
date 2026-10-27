# 2 · Models (Milestone 2)

Code: `src/phishguard/features.py`, `modeling.py`, `evaluate.py`.
Notebooks: `02`–`05`.

## The goal
1. **Reproduce the published ≥99.9%** on PhiUSIIL.
2. **Build models that need only the URL**, because a user of our app types a URL. They can't give us the page's HTML statistics.

## The same fair routine for every model
```python
from phishguard.modeling import param_grid, tune, fit_and_test
from phishguard.evaluate import save_result

table, best = tune(make_model, param_grid(max_depth=[6, 8, 10]), X, y, splits)   # pick settings on validation
model, threshold, metrics = fit_and_test(make_model(**best), X, y, splits)        # train → threshold on val → test once
save_result("XGBoost (URL features)", "URL features", threshold, metrics)
```
1. **Tune:** train each setting on *train*, score on *validation*, keep the best.
2. **Threshold:** a model outputs a probability, and the threshold decides "phishing" vs "legitimate". Instead of always using 0.5, we pick the threshold with the fewest errors **on validation**.
3. **Test once:** score the test set a single time and save the result.

## The three groups of models

### A. All provided columns (notebook 02)
We train **Logistic Regression**, **Random Forest** and **XGBoost** on the dataset's 50 numeric columns.

| Model | In one line |
|---|---|
| Logistic Regression | Weighted sum of the columns → sigmoid → probability. Columns are standardised first. |
| Random Forest | 300 decision trees, each trained on random rows and columns; they vote. |
| XGBoost | Trees added one by one, each fixing the previous trees' mistakes (gradient boosting). |

**Result:** 99.994%, 99.997% and **100%**, i.e. 0–2 mistakes out of 35,130 test URLs.

### B. URL features (notebook 03)
We wrote `extract_features()`, which turns each URL into **39 numbers** using only the URL text:
```python
from phishguard.features import extract_features
extract_features(pd.Series(["http://192.168.0.1:8080/secure/login.php?id=1"]))
# has_ip=1, has_port=1, is_https=0, path_depth=2, num_query_params=1, suspicious_words=2, ...
```
| Group | Examples |
|---|---|
| Size | `url_length`, `host_length`, `path_length` |
| Character counts | `num_digits`, `num_hyphens`, `num_at`, `digit_ratio` |
| Host | `num_subdomains`, `has_ip`, `has_www`, `on_hosting_platform`, `is_shortener` |
| Words | `suspicious_words` (login, verify, account, paypal, …) |
| Randomness | `url_entropy`: random-looking names like `x7q9kz` score high |

**Result:** Random Forest 99.68%, XGBoost 99.68%.

### C. Character n-grams (notebook 04)
An **n-gram** is a chunk of *n* characters. `paypa1.top` has the 3-grams `pay`, `ayp`, `ypa`, `pa1`, … The model:
1. Turns each URL into a vector of n-gram scores (**TF-IDF**: n-grams common in this URL but rare overall score high).
2. Learns one weight per n-gram with Logistic Regression.

```python
make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True),
    LogisticRegression(C=10),
)
```
**Result:** 99.70%. Averaging its probability with XGBoost's: **99.71%**.

## Final table (test set, 35,130 URLs)
| Model | Input | Accuracy | Errors |
|---|---|---|---|
| XGBoost | all columns | 100.000% | 0 |
| Random Forest | all columns | 99.997% | 1 |
| Logistic Regression | all columns | 99.994% | 2 |
| Average n-gram + XGBoost | URL only | 99.707% | 103 |
| Char n-gram LogReg | URL only | 99.695% | 107 |
| Random Forest | URL only | 99.681% | 112 |
| XGBoost | URL only | 99.675% | 114 |

Published: original paper 99.79%, Bao et al. RF 99.99% (all columns); URL-only 99.62–99.84% (Frontiers 2026).

## What the errors tell us
Nearly every mistake by the URL-only models is a **phishing site that looks like `https://www.something.com`** with no path. In this dataset, *every* legitimate URL looks exactly like that. For those URLs the only clue is the domain name itself (`ncsbarnk.com`, `lcioud-icu…`, `.top`).

We also saw that the n-gram model's strongest signals are `p://` (= `http://` without `s` → phishing) and `www` (→ legitimate). The models lean heavily on **how the URL starts**. Milestone 3 tests whether that's real phishing knowledge or a dataset shortcut.

## Likely viva questions
**Q: Why do the all-column models score 100%?**
A: Some provided columns almost give the answer away. For example, `URLSimilarityIndex` is 100 for every legitimate URL, and page-content columns like `LineOfCode` differ hugely between classes. Even Logistic Regression gets 99.99%. That's why we also built URL-only models, and why Milestone 3 studies these shortcuts.

**Q: Why not use the all-column model in the app?**
A: The app only has the URL. It can't compute page-content columns or `URLSimilarityIndex` for a new URL.

**Q: Why tune the threshold?**
A: 0.5 isn't special. For the n-gram model, 0.16 gave the fewest validation errors. We choose it on validation, not test, so the test score stays honest.

**Q: Random Forest vs XGBoost?**
A: Random Forest trains trees independently and averages them (reduces variance). XGBoost trains trees one after another, each correcting the previous ones (reduces bias). Here they perform almost the same.

**Q: What is TF-IDF?**
A: Term Frequency × Inverse Document Frequency. An n-gram gets a high score if it appears in this URL but is rare across all URLs, so common chunks like `com` count for less.

**Q: Why did averaging two models help?**
A: They make different mistakes. XGBoost looks at counts and structure; the n-gram model looks at the actual characters. Averaging cancels some errors.

**Q: Your validation score was 99.78% but test 99.71%. Why?**
A: We chose the settings and threshold that were best *on validation*, so validation is slightly optimistic. Test is the honest number, which is exactly why we keep it separate.
