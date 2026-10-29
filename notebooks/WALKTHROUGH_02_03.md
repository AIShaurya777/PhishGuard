# Walkthrough: notebooks 02 and 03

A guide for presenting these two notebooks. The notebooks themselves now explain every step in plain words; this file adds what to say, and answers to likely questions. The step numbers match the headings in the notebooks.

**Don't re-run them live.** They train hundreds of models (several settings × 5 splits) and take a long time. All outputs are already saved in the notebooks. If asked to run something, run the first cell or `.venv/bin/pytest`.

---

## Where we are in the project

Last time we showed **01 and 01b**: loading and cleaning the data, and splitting it by site owner. Today:

| Notebook | Question it answers | Answer |
|---|---|---|
| **02** | Can we reproduce the published result (≥ 99.9%) using all the dataset's columns? | Yes: 99.99–100% |
| **03** | Can we do it with **only the URL**, which is all our app will get? | Almost: 99.63% |

**One sentence for the start:** "First we checked we can match the paper. Then we rebuilt the models to use only the link, because that's all a real user gives us."

---

## Things both notebooks use

### The data (from notebook 01)
```python
from phishguard.data import load_clean, TARGET
df = load_clean()        # 234,204 cleaned rows
y = df[TARGET]           # is_phishing: 1 = phishing, 0 = legitimate
```

### The fair routine: `run(...)`
Every model goes through the same function, `phishguard.experiment.run`, so no model gets an unfair advantage. For **each of 5 splits** (seeds 42, 0, 1, 2, 3):

1. **Tune.** Train the model once for every setting in the grid on the *train* part. Keep the setting with the lowest *validation* log-loss.
2. **Threshold.** The model gives a probability, e.g. 0.73. The threshold turns it into a yes/no. We pick the threshold with the fewest mistakes on *validation*.
3. **Test once.** Score the *test* part one time and save everything to `results/m2/`.

Then it reports **mean ± standard deviation** over the 5 splits.

**Why five splits?** One split can be lucky or unlucky. If a result holds on five different splits, it's real. The "±" shows how much it moves between them.

**Why validation for tuning and test only once?** If we chose settings by looking at the test score, the test score would be too optimistic. It's like seeing the exam paper before the exam.

### `param_grid(...)`: the settings we try
```python
param_grid(C=[0.01, 0.1, 1.0, 10.0])   # → 4 settings: C=0.01, C=0.1, C=1, C=10
```
These settings are called **hyperparameters**: choices we make, not things the model learns.

### The scores you'll see in the results table

Mistakes come in two kinds:
- **fp (false positive):** a legitimate link wrongly called phishing.
- **fn (false negative):** a phishing link the model missed.

| Score | Meaning in plain words |
|---|---|
| **accuracy** | Share of links classified correctly |
| **F1** | Balances "of what we flagged, how much was phishing" (precision) and "of all phishing, how much we caught" (recall) |
| **ROC-AUC** | How well the model *ranks* phishing above legitimate, whatever the threshold. 1.0 = perfect, 0.5 = guessing |
| **PR-AUC** | Like ROC-AUC, but focused on the phishing class |
| **MCC** | One number from −1 to 1 that stays honest even if the classes are unbalanced |
| **TPR at 1% FPR** | How much phishing we catch if we allow only 1 false alarm per 100 legitimate links |
| **95% CI** | The range the true accuracy probably lies in. We get it by resampling whole site owners (a bootstrap) |

---

## Notebook 02: models on all provided columns

**Goal:** reproduce the published results. The original paper reports 99.79%; a 2026 paper reports 99.99% with Random Forest.

### Step 1: load the data and pick the columns
```python
X = df.drop(columns=TARGET).select_dtypes("number")
# → 50 numeric columns (the 4 text columns URL, Domain, TLD, Title are left out)
```
**Say:** "We give the models every numeric column the dataset provides: 50 of them. These include columns about the URL, columns counted from the downloaded web page (like lines of code or number of images), and a similarity score."

### Step 2: Logistic Regression
```python
make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
```
- **What it is:** it multiplies each column by a weight, adds them up, and squashes the total into a probability between 0 and 1 (the *sigmoid*). It's the simplest model we use.
- **`StandardScaler`:** rescales every column to mean 0 and standard deviation 1. Without it, a column measured in thousands (line count) would dominate one measured in 0/1 (is HTTPS).
- **`C`:** the regularisation strength. Smaller `C` forces smaller, simpler weights, which helps avoid overfitting. We try 0.01, 0.1, 1 and 10.
- **Pipeline:** the scaler is learned on the training data only, then applied to validation and test, so nothing leaks.

**Output:** per split, 1–6 mistakes out of ~35,000 test links (99.98–99.99%).

### Step 3: Random Forest
```python
RandomForestClassifier(n_estimators=300, max_depth=..., min_samples_leaf=...)
```
- **What it is:** 300 decision trees. Each tree is a flowchart of yes/no questions ("Is URLLength > 40?"). Each tree sees a random sample of rows and columns, and the forest takes a vote.
- **Settings tried:**
  - `max_depth`: how many questions deep a tree may go (`None` = unlimited, or 20);
  - `min_samples_leaf`: at least 1 or 3 links must end up at each final answer, which stops trees memorising single links.

**Output:** 0 mistakes on every split.

### Step 4: XGBoost
```python
XGBClassifier(max_depth=..., n_estimators=..., learning_rate=0.1)
```
- **What it is:** gradient boosting. Trees are added one at a time, and each new tree focuses on the mistakes the previous trees made. It's usually the strongest model for tables of numbers.
- **Settings tried:**
  - `max_depth`: 4, 6 or 8;
  - `n_estimators`: 200 or 500 trees;
  - `learning_rate` 0.1: how much each new tree may change the answer.

**Output:** 0 mistakes on every split.

### Step 5: the results table
| Model | Accuracy (5 splits) |
|---|---|
| Logistic Regression | 99.99 ± 0.00% |
| Random Forest | 100.00 ± 0.00% |
| XGBoost | 100.00 ± 0.00% |

**How to read it:**
- A handful of errors out of 35,130 is noise. We **don't rank** these three models; we say all three are indistinguishable from perfect.
- **The rule of three:** even 0 errors in 35,130 only proves accuracy is above about 99.99% (3 / 35,130) with 95% confidence.

### Step 6: which columns the models rely on
A bar chart of feature importance from Random Forest and XGBoost.
- **Feature importance:** how much each column helped the trees make their decisions.
- At the top are columns like `URLSimilarityIndex` (similarity to a list of known legitimate sites) and page counts like `LineOfCode`.

### What we learned: the conclusion
**Say:**
- "We reproduced the published 99.9%+, on all 5 splits, even with logistic regression, and with site owners kept out of more than one split."
- "But when even a linear model is nearly perfect, it usually means a few columns give the answer away."
- "Also, our app can't compute those top columns from a link a user types. You'd need to download the page, or have the similarity list. So the next notebook uses only the URL."

The "why is it so perfect?" question is Milestone 3. You can say "we'll investigate this next".

---

## Notebook 03: models on URL features only

**Why:** the app only gets a URL. So we compute our own features from the URL text and nothing else. The code is in `src/phishguard/features.py`, decision D7 in `docs/DECISIONS.md`.

**Published comparison:** a 2026 paper reports XGBoost at 99.74% on URL features. It used a different split, so the comparison is only indicative.

### Step 1: turn every URL into 40 numbers
```python
from phishguard.features import extract_features
X = extract_features(df["URL"])     # → (234204, 40)
```
**Say:** "Feature engineering: we turn each link into 40 numbers a model can use."

Open `src/phishguard/features.py` in VS Code if she wants to see the code.

### Step 1, continued: the 40 features, in groups
| Group | Examples | The idea |
|---|---|---|
| Size | `url_length`, `host_length`, `path_length` | Phishing links are often long, to hide the real domain |
| Character counts | `num_digits`, `num_hyphens`, `num_at`, `digit_ratio` | Random or disguised links have unusual mixes of characters |
| Host structure | `num_subdomains`, `has_ip`, `has_www`, `has_punycode` | `paypal.com.secure-login.xyz` hides a brand in the subdomain; `xn--` hides look-alike letters |
| Scheme | `is_https` | |
| Platform | `on_hosting_platform`, `is_shortener` | Free hosting (`web.app`) and link shorteners are popular with attackers |
| Path and query | `path_depth`, `num_query_params` | |
| Words | `suspicious_words` (login, verify, account …), `longest_token` | Whole words only, so `freedomhouse` doesn't count as "free" |
| Randomness | `url_entropy`, `host_entropy` | Generated names like `x7q9kz.top` score high |

**Entropy:** a measure of how random or unpredictable the characters are. `google` is low; `x7q9kz` is high.

**Detail worth mentioning:** counts and ratios are computed **without** the `https://www.` part, so only `is_https` and `has_www` know about it. This keeps one feature from secretly duplicating another.

**Quick demo you can run live:**
```python
from phishguard.features import extract_features
import pandas as pd
extract_features(pd.Series(["http://192.168.0.1:8080/secure/login.php?id=1"])).T
# has_ip=1, has_port=1, is_https=0, path_depth=2, num_query_params=1, suspicious_words=2 ...
```

### Steps 2–3: how do the two classes differ? (EDA)
Six histograms, plus a table of the average of each feature per class:

| Feature (average) | Legitimate | Phishing |
|---|---|---|
| `url_length` | 15.2 | 36.6 |
| `num_digits` | 0.05 | 4.33 |
| `num_hyphens` | 0.08 | 0.70 |
| `path_length` | 0.0 | 8.2 |
| `num_slashes` | 0.0 | 0.65 |

**Say:** "Legitimate links are short, with no path and almost no digits. Phishing links vary a lot more."

Then point out the odd thing: `path_length` is **exactly 0** for every legitimate link. "Every legitimate link in this dataset is just a homepage. We'll come back to why that matters." That's the hook for Milestone 3.

### Step 4: Random Forest
Same as in notebook 02, now on our 40 features. Settings: `max_depth` unlimited or 30; `min_samples_leaf` 1 or 3.

### Step 5: XGBoost
A slightly wider search than in notebook 02:
- **`subsample=0.8`, `colsample_bytree=0.8`:** each tree sees a random 80% of rows and columns. That adds variety and reduces overfitting.
- **`min_child_weight`:** 1 or 5. It stops trees from splitting on just a handful of links.

### Step 5: the results
| Model | Accuracy (5 splits) | ROC-AUC | Seed 42 [95% CI] |
|---|---|---|---|
| Random Forest | 99.63 ± 0.04% | 0.9972 | 99.70% [99.63, 99.76] |
| XGBoost | 99.63 ± 0.03% | 0.9973 | 99.68% [99.62, 99.74] |

**Say:** "From the link text alone, our own features get 99.6%, close to the 99.74% a 2026 paper reports. It's slightly below the all-columns models, which makes sense: we have less information."

### Step 6: top 15 features of XGBoost
Point out which features the model uses most. You'll see `is_https` and `has_www` near the top. Remember that for next week.

### Step 7: what does it still get wrong? (error analysis)
We look at the mistakes on the **validation** set, never test, because test is only for the final score.
```
missed phishing URLs on validation: 117
share that are a bare https://www.<domain> homepage: 1.0
```
Examples: `https://www.netflix-vn.com`, `https://www.lglsvy.top`, `https://www.k365v.com`.

**Say:** "Every phishing link the model missed looks exactly like a legitimate link in this dataset: `https://www.` plus a domain, no path. For these, the only clue left is the domain name itself, like the fake `netflix-vn.com` or a cheap `.top` domain."

### What we learned: conclusion, and what comes next
"Character-level models can read the domain name directly. That's notebook 04."

---

## If she asks …

**Why did you leave out the text columns (URL, Domain, TLD, Title) in notebook 02?**
They're text, and these three models need numbers. Notebook 04 handles the URL text directly with character n-grams.

**Why log-loss for tuning and not accuracy?**
Log-loss also measures how confident the model is, and it changes smoothly. With accuracy, one lucky validation link could decide which setting wins.

**Isn't 100% suspicious?**
Yes, and we say so. A near-perfect linear model usually means some columns give the label away. That's exactly what we investigate next (Milestone 3).

**Why not use the page columns in the app?**
The app gets only a link. It would have to download the page, which is slow, possibly dangerous, and often the page is already gone.

**How do you make sure the same website isn't in training and test?**
The split from notebook 01b keeps all links of one site owner in the same part, and tests in `tests/` check it.

**Where are the results saved?**
In `results/m2/` as JSON files. The README's results table is generated from them, so nobody types numbers by hand.

**Which is better, Random Forest or XGBoost?**
Neither, on this data: they tie at 99.63% over 5 splits, and the gap is smaller than the ±. We don't claim a winner.

---

## Files to have open in VS Code
- `notebooks/02_all_columns_models.ipynb`
- `notebooks/03_url_feature_models.ipynb`
- `src/phishguard/features.py`: the 40 features
- `src/phishguard/experiment.py`: the `run()` routine
- `docs/DECISIONS.md`: D6 (the routine), D7 (URL features), D8 (results)
