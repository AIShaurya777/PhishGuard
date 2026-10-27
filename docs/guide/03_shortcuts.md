# 3 · Shortcut analysis (Milestone 3)

Notebooks: `06_shortcut_audit.ipynb`, `07_url_views.ipynb`.
Code: `src/phishguard/columns.py`, `src/phishguard/urls.py`.

## The question
In Milestone 2, even Logistic Regression, the simplest model, scored 99.99%. When a simple model is almost perfect, the usual reason is that **some inputs give the answer away**. A model that uses such a **shortcut** looks great on the dataset but fails in the real world.

## Finding 1: one line of code beats most papers
No machine learning, just a rule, scored on the test split:

| Rule | Accuracy |
|---|---|
| `URLSimilarityIndex < 100` → phishing | **99.66%** |
| URL doesn't start with `https://www.` → phishing | **99.08%** |
| `IsHTTPS = 0` → phishing | 79.43% |
| URL has a path → phishing | 72.51% |

**Why?** In PhiUSIIL, **every single legitimate URL** looks like `https://www.something.com`: HTTPS, `www.`, no path. The legitimate side is a list of homepages. Only 2.4% of phishing URLs look like that.

```python
df["URL"].str.startswith("https://www.").groupby(df["is_phishing"]).mean()
# legitimate 1.000, phishing 0.024
```

## Finding 2: where the columns come from
We sorted the 50 numeric columns into three groups (`columns.py`):

| Group | Count | Example | Usable in our app? | XGBoost on this group alone |
|---|---|---|---|---|
| URL | 19 | `URLLength`, `NoOfSubDomain`, `IsHTTPS` | Yes | 99.74 ± 0.02% |
| statistic | 3 | `URLSimilarityIndex`: similarity to a list of known legitimate URLs | No | 99.66 ± 0.03% |
| page | 28 | `LineOfCode`, `NoOfImage`: counted in the downloaded HTML | No | 99.71 ± 0.08% |

(± = standard deviation over 5 splits.) **Every** group alone is near-perfect. The page columns probably reflect how the pages were crawled: legitimate homepages are large, while many phishing pages are small, perhaps already taken down or just a login form. We didn't test that explanation, so we call it likely.

## Finding 3: two leaks hidden in the "URL" columns
Even without `IsHTTPS`, the URL columns still score 99.61%. Why?
```python
URLLength − (letters + digits + special characters + = ? &)
# legitimate: 12 for all but 4 of 134,849 URLs  ← len("https://www.") = 12
# phishing:   usually 7, 8 or 11  ← "http://", "https://", "http://www."
```
The characters that were never counted are the prefix, so trees can read it.

And `URLLength` was computed **before** we removed trailing slashes (D3): `URLLength − len(cleaned URL)` is −1 for every legitimate URL, but often 0 or +1 for phishing. So the slash shortcut we cleaned away is still inside the dataset's own column. That's why our models use only **our own** features.

## Finding 4: what's left when we remove the shortcuts
Four **views** of each URL:

| View | `https://www.login.paypal-secure.top/x?id=7` becomes |
|---|---|
| raw URL | `https://www.login.paypal-secure.top/x?id=7` |
| no scheme/www | `login.paypal-secure.top/x?id=7` |
| domain only | `login.paypal-secure.top` |
| registered domain | `paypal-secure.top` |

```python
from phishguard.urls import URL_VIEWS
domains = URL_VIEWS["domain only"](df["URL"])
```

Test accuracy, mean ± sd over 5 splits:

| View | Char n-gram LogReg | XGBoost (URL features) |
|---|---|---|
| raw URL | 99.70 ± 0.03% | 99.64 ± 0.03% |
| no scheme/www | 89.18 ± 0.59% | 85.95 ± 0.95% |
| domain only | **86.27 ± 0.89%** | 81.43 ± 1.15% |
| registered domain | 82.49 ± 1.93% | 72.82 ± 0.70% |

So **about 13 points** came from the prefix and the path. From the domain name alone, a URL-only model gets **~86%**, still far above guessing (57.6%).

## Finding 5: even the domain isn't completely clean
The legitimate list is popular *registered* domains, so almost none have extra structure:

| | legitimate | phishing |
|---|---|---|
| has a subdomain, hosting platform, IP, digit or `xn--` | 6.2% | 56.9% |

A rule with no learning ("structured → phishing") already gets **78%**. That's why we say **"shortcut-reduced"**, not "honest". On **bare registered domains** (no structure), the n-gram model catches only 53% of phishing.

## What the model learned on domain-only input
- **→ phishing:** free or cheap TLDs (`.cf`, `.ga`, `.ml`, `.top`), digits and hyphens, IPFS and free-hosting names
- **→ legitimate:** `.gov`, `.edu`, `.ac.`, `.mil` and country TLDs (`.de`, `.jp`, `.it`)
- **Still missed:** normal-looking domains like `smithtony.co.uk`, probably hacked legitimate websites. No URL-only model can catch those from the name.

## Our decision (D10)
From now on, every URL model is reported on **raw URL** (to compare with papers) **and** on **domain only** and **registered domain** (shortcut-reduced), each as mean ± sd over 5 splits.

## Likely viva questions
**Q: So is your 100% result fake?**
A: No, it's correct on this dataset and reproduces the published results. But we showed *why* it's so high: the legitimate URLs are all homepages, and some columns nearly contain the answer. We report both the reproduction and the shortcut-reduced numbers.

**Q: What is a shortcut / spurious correlation?**
A: A pattern that predicts the label in the dataset but has nothing to do with the real task. "Starts with `https://www.`" predicts "legitimate" only because of how the data was collected; real phishing sites can use HTTPS and `www.` too.

**Q: How did you find the shortcuts?**
A: Single-column ROC-AUC for every column (on the training split), one-line rule baselines, removing column groups, reading the n-gram model's weights (its top weights were `p://` and `www`), and arithmetic on the dataset's own columns.

**Q: Why does accuracy drop to ~86%? Is that bad?**
A: It's the real difficulty of judging a website by its domain name. Some phishing sites are hacked normal websites with normal names. 86% from the name alone, with clear real signals, is a meaningful result.

**Q: Why not call 86% the "honest" number?**
A: Even the domain still differs in structure between the classes (6% vs 57% have subdomains or similar), and a no-learning rule gets 78%. So it's *shortcut-reduced*, not shortcut-free.

**Q: Why do n-grams beat XGBoost on the domain?**
A: The information is in the characters (`.cf`, `paypa1`, random strings). Our 40 features are mostly counts, which lose that detail. That's why we try character-level deep learning next.

**Q: What would fix the dataset?**
A: Legitimate URLs with real paths, not only homepages, collected the same way as the phishing URLs, and no columns computed against reference lists of legitimate sites.
