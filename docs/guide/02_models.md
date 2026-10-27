# 2 · Models (Milestone 2)

Code: `src/phishguard/features.py`, `modeling.py`, `experiment.py`, `evaluate.py`, `stats.py`.
Notebooks: `02`–`05`.

## The goal
1. **Reproduce the published ≥99.9%** on PhiUSIIL.
2. **Build models that need only the URL**, because a user of our app types a URL. They can't give us the page's HTML statistics.

## The same fair routine for every model
```python
from phishguard.experiment import run, summary_table
from phishguard.modeling import param_grid

exp = run("XGBoost (URL features)", make_xgb, param_grid(max_depth=[6, 10], min_child_weight=[1, 5]),
          X, y, df, features="URL features", group="m2")
summary_table([exp])        # accuracy etc. as mean ± sd over 5 splits, plus a 95% CI
```
For **each of 5 splits** (seeds 42, 0, 1, 2, 3):
1. **Tune:** train each setting on *train*, keep the one with the lowest **validation log-loss**. Log-loss is smooth, so one lucky validation URL can't decide the winner.
2. **Threshold:** a model outputs a probability; the threshold turns it into "phishing" or "legitimate". We pick the threshold with the fewest **validation** errors (ties → the middle of the tied range).
3. **Test once,** and save everything: settings, scores, versions, and the seed-42 predictions.

Then we report **mean ± standard deviation** over the 5 splits. **Why 5 splits?** One split can be lucky. Our first domain-only result was 88.6%, but over 5 splits it's 86.3 ± 0.9%.

## The three groups of models

### A. All provided columns (notebook 02)
| Model | In one line |
|---|---|
| Logistic Regression | Weighted sum of the standardised columns → sigmoid → probability |
| Random Forest | 300 decision trees on random rows/columns; they vote |
| XGBoost | Trees added one by one, each fixing the previous trees' mistakes (gradient boosting) |

**Result:** 99.99–100.00% on every split (0–3 mistakes out of 35,130). With so few errors, the three are **indistinguishable from perfect**. By the "rule of three", 0 errors in 35,130 still only proves accuracy ≥ ~99.99%.

### B. URL features (notebook 03)
`extract_features()` turns each URL into **40 numbers** using only the URL text:
```python
from phishguard.features import extract_features
extract_features(pd.Series(["http://192.168.0.1:8080/secure/login.php?id=1"]))
# has_ip=1, has_port=1, is_https=0, path_depth=2, num_query_params=1, suspicious_words=2, ...
```
| Group | Examples |
|---|---|
| Size | `url_length`, `host_length`, `path_length` |
| Character counts | `num_digits`, `num_hyphens`, `num_at`, `digit_ratio` |
| Host | `num_subdomains`, `has_ip` (incl. decimal/hex/IPv6), `has_www`, `has_punycode`, `on_hosting_platform`, `is_shortener` |
| Words | `suspicious_words`: whole words only, so `freedomhouse` isn't "free" |
| Randomness | `url_entropy`: random names like `x7q9kz` score high |

Counts are computed **without** the `https://www.` prefix, so only `is_https` and `has_www` carry it.

**Result:** Random Forest 99.63 ± 0.04%, XGBoost 99.64 ± 0.03%.

### C. Character n-grams (notebook 04)
An **n-gram** is a chunk of *n* characters: `paypa1.top` has the 3-grams `pay`, `ayp`, `ypa`, `pa1`, … The model:
1. Turns each URL into a vector of n-gram scores (**TF-IDF**: n-grams common in this URL but rare overall score high).
2. Learns one weight per n-gram with Logistic Regression.

```python
make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(1, max_n), min_df=2, sublinear_tf=True),
    LogisticRegression(C=C, solver="liblinear"),
)   # tuned: max_n ∈ {3, 4}, C ∈ {10, 30, 100}; usually max_n=4, C=30
```
**Result:** 99.69 ± 0.05%. Combined with XGBoost by averaging their **log-odds**: 99.70 ± 0.03%.

## Final table (test accuracy, mean ± sd over 5 splits)
| Model | Input | Accuracy | Seed-42 95% CI |
|---|---|---|---|
| Random Forest | all columns | 100.00 ± 0.00% | [100.00, 100.00] |
| XGBoost | all columns | 100.00 ± 0.00% | [100.00, 100.00] |
| Logistic Regression | all columns | 99.99 ± 0.00% | [99.98, 100.00] |
| Average n-gram + XGBoost | URL only | 99.70 ± 0.03% | [99.69, 99.80] |
| Char n-gram LogReg | URL only | 99.69 ± 0.05% | [99.69, 99.80] |
| XGBoost | URL features | 99.64 ± 0.03% | [99.62, 99.74] |
| Random Forest | URL features | 99.63 ± 0.04% | [99.63, 99.76] |

Published: original paper 99.79%, Bao et al. RF 99.99% (all columns); URL-only 99.62–99.84% (Ahamed et al., Frontiers 2026, on a host-disjoint split like ours).

## Is one model really better? (McNemar test)
On the same test URLs, we only look at the URLs where two models **disagree**. If they were equally good, those would split about 50/50.

| A vs B | only A right | only B right | p-value | Verdict |
|---|---|---|---|---|
| n-gram vs XGBoost (features) | 38 | 16 | 0.004 | n-gram really is better |
| Average vs n-gram alone | 17 | 17 | 1.0 | no difference: averaging didn't help |
| XGBoost vs Random Forest (features) | 7 | 12 | 0.36 | no difference |

## What the numbers mean in real traffic
In PhiUSIIL, 42% of URLs are phishing; in real browsing it's far less. Precision depends on that **base rate**:

precision = recall·p / (recall·p + false-alarm-rate·(1 − p))

| Model | 42% phishing | 1% phishing | 0.1% phishing |
|---|---|---|---|
| Best raw-URL model (recall 99.5%, false alarms 0.05%) | 99.9% | 94.9% | 64.7% |
| Domain-only n-gram (recall 76%, false alarms 4.3%) | ~93% | **~15%** | ~2% |

So the shortcut-reduced model would mostly raise false alarms on its own. That's why we also report PR-AUC and the catch rate at 1% false alarms.

## What the errors tell us
Nearly every mistake by the URL-only models is a **phishing site that looks like `https://www.something.com`** with no path. In this dataset, *every* legitimate URL looks exactly like that. The n-gram model's strongest signals are `p://` (= `http://` without `s` → phishing) and `www` (→ legitimate), so it leans on **how the URL starts**. Milestone 3 tests that.

## Likely viva questions
**Q: Why do the all-column models score 100%?**
A: Some provided columns almost give the answer away: `URLSimilarityIndex` is 100 for every legitimate URL, and page columns like `LineOfCode` differ hugely between classes. Even Logistic Regression gets 99.99%. That's why we built URL-only models and studied these shortcuts in Milestone 3.

**Q: Why not use the all-column model in the app?**
A: The app only has the URL. It can't compute page-content columns or `URLSimilarityIndex` for a new URL.

**Q: Why report mean ± sd over 5 splits?**
A: One split can be lucky or unlucky. Repeating everything on 5 splits shows how stable a number is. 99.69 ± 0.05% means the result moves by about 0.05 points between splits.

**Q: Is the averaged model better than the n-gram model?**
A: No. 99.70 vs 99.69% is within noise, and McNemar's test on the same test URLs gives p = 1.0 (17 vs 17 disagreements). We say they're equivalent.

**Q: Why choose hyperparameters by log-loss instead of accuracy?**
A: Accuracy jumps in steps of one URL, so a single lucky validation URL could pick the winner. Log-loss rewards confident correct probabilities and changes smoothly.

**Q: Why tune the threshold?**
A: 0.5 isn't special. We choose the threshold on validation, never on test, so the test score stays honest. The gain is small (a few URLs), and for the app a stable threshold matters more.

**Q: What is TF-IDF?**
A: Term Frequency × Inverse Document Frequency. An n-gram gets a high score if it appears in this URL but is rare across all URLs, so common chunks like `com` count for less.

**Q: Why average log-odds instead of probabilities?**
A: The two models are calibrated differently (one says 0.99 where the other says 0.8). Log-odds put them on the same scale before averaging.

**Q: Your model is 99.7% accurate, so it's ready for real use?**
A: No. At a realistic 1% phishing rate, precision drops, and much of the 99.7% comes from dataset shortcuts (Milestone 3). Without them, a URL-only model gets ~86%.
