# 3 · Shortcut analysis (Milestone 3)

Notebooks: `06_shortcut_audit.ipynb`, `07_url_views.ipynb`.
Code: `src/phishguard/columns.py`, `src/phishguard/urls.py`.

## The question
In Milestone 2, even Logistic Regression, the simplest model, scored 99.99%. When a simple model is almost perfect, the usual reason is that **some inputs give the answer away**. A model that uses such a **shortcut** looks great on the dataset but fails in the real world.

## Finding 1: one line of code beats most papers
No machine learning, just a rule, scored on the test set:

| Rule | Accuracy |
|---|---|
| `URLSimilarityIndex < 100` → phishing | **99.72%** |
| URL doesn't start with `https://www.` → phishing | **99.16%** |
| `IsHTTPS = 0` → phishing | 77.09% |
| URL has a path → phishing | 72.82% |

**Why?** In PhiUSIIL, **every single legitimate URL** looks like `https://www.something.com`: HTTPS, `www.`, no path. The legitimate side is a list of homepages. Only 2.4% of phishing URLs look like that.

```python
df["URL"].str.startswith("https://www.").groupby(df["is_phishing"]).mean()
# legitimate 1.000, phishing 0.024
```

## Finding 2: where the columns come from
We sorted the 50 numeric columns into three groups (`columns.py`):

| Group | Count | Example | Usable in our app? |
|---|---|---|---|
| URL | 19 | `URLLength`, `NoOfSubDomain`, `IsHTTPS` | Yes |
| statistic | 3 | `URLSimilarityIndex`: similarity to a list of known legitimate URLs | No |
| page | 28 | `LineOfCode`, `NoOfImage`: counted in the downloaded HTML | No |

The **page columns alone give 99.64%**. Legitimate homepages are big pages, while many phishing pages were tiny or already taken down when crawled. That says more about how the data was collected than about phishing.

## Finding 3: the prefix hides inside the counts
Even without `IsHTTPS`, the URL columns still score 99.56%. Why?
```python
URLLength − (letters + digits + special characters + = ? &)
# legitimate: always 12   ← len("https://www.") = 12
# phishing:   usually 7, 8 or 11  ← "http://", "https://", "http://www."
```
The characters that were never counted are the prefix, so trees can read it.

## Finding 4: what's left when we remove the shortcuts
We created three **views** of each URL:

| View | `https://www.paypal-secure.top/login?id=7` becomes |
|---|---|
| raw URL | `https://www.paypal-secure.top/login?id=7` |
| no scheme/www | `paypal-secure.top/login?id=7` |
| domain only | `paypal-secure.top` |

```python
from phishguard.urls import URL_VIEWS
domains = URL_VIEWS["domain only"](df["URL"])
```

Test accuracy:

| View | Char n-gram LogReg | XGBoost (URL features) |
|---|---|---|
| raw URL | 99.70% | 99.68% |
| no scheme/www | 90.89% | 87.97% |
| domain only | **88.59%** | 81.42% |

So **about 10 points** came from the shortcuts. The honest URL-only score is **~88.6%**. That's still well above guessing (57.6%), so the domain name does carry real information.

## What the honest model learned
On domain-only input, the strongest n-gram weights are real phishing signals:
- **→ phishing:** free or cheap TLDs (`.cf`, `.ga`, `.ml`, `.top`), digits and hyphens, random subdomains on free hosting (`repl.co`, IPFS)
- **→ legitimate:** `.gov`, `.edu`, `.ac.` and country TLDs (`.de`, `.jp`, `.it`)
- **Still missed:** normal-looking domains like `cama.it`, probably hacked legitimate websites. No URL-only model can catch those from the name.

## Our decision (D10)
From now on, every URL model is reported **twice**:
- **raw URL:** to compare with published papers
- **domain only:** the honest number

## Likely viva questions
**Q: So is your 100% result fake?**
A: No, it's correct on this dataset, and it reproduces the published results. But we showed *why* it's so high: the dataset's legitimate URLs are all homepages, and some columns nearly contain the answer. We report both the reproduction and the honest number.

**Q: What is a shortcut / spurious correlation?**
A: A pattern that predicts the label in the dataset but has nothing to do with the real task. Here, "starts with `https://www.`" predicts "legitimate" only because of how the data was collected. Real phishing sites can use HTTPS and `www.` too.

**Q: How did you find the shortcuts?**
A: Single-column ROC-AUC for every column, one-line rule baselines, removing column groups and watching the accuracy, and reading the n-gram model's weights (its top weights were `p://` and `www`).

**Q: Why does accuracy drop to 88%? Is that bad?**
A: It's the honest difficulty of the task: judging a website only by its domain name. Some phishing sites are hacked normal websites with normal names. 88.6% from the name alone, with clear real signals (TLDs, digits, free hosting), is a meaningful result.

**Q: Why do n-grams beat XGBoost on the domain only?**
A: The useful information is in the characters themselves (`.cf`, `paypa1`, random strings). Our 39 features are mostly counts, which lose that detail. That's why we try character-level deep learning next.

**Q: What would fix the dataset?**
A: Legitimate URLs with real paths (not only homepages), collected the same way as the phishing URLs, and no columns computed against the label.
