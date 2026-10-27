# 6 · Explainability (Milestone 6)

Code: `src/phishguard/explain.py`. Notebook: `10_explainability.ipynb`. Decision: D14.

## Why explain?
A score like 86.7% doesn't say *what* the model learned. It might have learned real phishing signals, or a dataset shortcut. Explanations let us check, and they let the app tell a user **why** a URL looks suspicious.

All explanation scores are in **log-odds**, log(p / (1 − p)), because contributions add up on that scale. **Positive / red = pushes towards phishing; negative / blue = towards legitimate.**

## Method 1: SHAP for XGBoost (features)
**Idea (Shapley values, from game theory):** the prediction is a "payout" shared among the features. A feature's share is its average contribution over all the orders in which the features could be added.

**Key property:** for every URL,

  base value (the average log-odds) + sum of all SHAP values = the model's log-odds.

```python
explanation = tree_shap(xgb_model, features)     # shap.TreeExplainer, exact for trees
shap.plots.beeswarm(explanation)                 # global: which features matter overall
explanation[0]                                   # local: why this one URL got its score
```
For tree models, `TreeExplainer` computes Shapley values **exactly** and fast. We verify the "adds up" property in a test and in the notebook.

## Method 2: Permutation importance (the hybrid's features)
Shuffle one feature across all test URLs. Now it no longer matches its URL. Then measure how much ROC-AUC drops. A big drop means the model needs that feature. It is model-agnostic and needs no retraining.

## Method 3: Occlusion (the hybrid's characters)
```python
scores = char_occlusion(model, X)     # one score per character
highlight(url, scores)                # HTML: red/blue background per character
```
For each character: hide it (replace it with the all-zero `PAD` input), predict again, and record

  score = log-odds(original) − log-odds(with this character hidden).

If hiding a character makes the URL look *less* phishy, that character was pushing towards phishing (positive score, red).

The same idea works for the hybrid's features (`feature_occlusion`): set one feature to its training average and measure the change.

## Are the explanations right? The deletion test
An explanation is **faithful** if the characters it calls important really drive the prediction. Test: on correctly caught phishing domains, hide the **top-k** characters by score, then **k random** characters, then the **bottom-k**. If the explanation is faithful, hiding the top-k should hurt the phishing score far more than random.

## One URL, as the app shows it
```python
from phishguard.deep import CharURLClassifier
from phishguard.explain import explain_url, highlight

model = CharURLClassifier.load("models/hybrid_domain_only.pt")
r = explain_url(model, "http://paypal-account-verify.top/login")
r["input"]            # 'paypal-account-verify.top'  (the domain only view, D10)
r["probability"]      # P(phishing)
r["char_scores"]      # one score per character → highlight(r["input"], r["char_scores"])
r["feature_scores"]   # one score per feature, largest effect first
```

## Results (`notebooks/10_explainability.ipynb`, seed-42 test split)
**1. Raw URL: the explanations reveal the shortcut.**

| XGBoost feature | mean \|SHAP\| |
|---|---|
| `is_https` | **4.18** |
| `has_www` | **2.51** |
| `url_length` | 0.87 |

In the character view, `https://www.` is blue on every legitimate URL. On `http://www.…` phishing URLs, the reddest character is the `:` right after `http`, the place where the `s` is missing.

**2. Domain only: what the hybrid uses.**
- **Features branch:** mainly `num_subdomains` and `on_hosting_platform` (structure).
- **Characters branch:** the suffix.

| → legitimate | → phishing |
|---|---|
| `gov.au`, `gov.uk`, `org.uk`, `ac.uk`, `.gov`, `.edu`, `.mil` | free TLDs `.gq`, `.ml`, `.cf`, `.ga`; `.xyz`, `.top`; hosting platforms `web.app`, `square.site` |

The n-gram model learned the same suffixes (Spearman 0.89).

**3. The explanations are faithful** (deletion test, 2,000 caught phishing domains):

| Characters hidden | k = 1 | k = 5 |
|---|---|---|
| most "phishing" (top-k) | 10% flip to legitimate | 20% |
| random k | 1.4% | 2.8% |
| most "legitimate" (bottom-k) | 0% | 0% |

**4. ⚠️ A limitation we found by explaining.** `https://accounts.google.com/...` → **P = 0.999 phishing**, mostly because of `num_subdomains` (+3.70). Only ~6% of legitimate training domains have a subdomain, so the model learned "subdomain ⇒ phishing". The app must warn about this.

**5. Typical mistakes:**
- **Missed phishing:** hacked sites on trusted suffixes (`nro.gov.sd`, `amurt.org.uk`).
- **False alarms:** legitimate sites with subdomains on less-known hosts (`*.wwwhubs.com`).

## Likely viva questions
**Q: What is a SHAP value?**
A: A feature's fair share of the prediction, from Shapley values in game theory: its average contribution over all orders of adding features. For one URL, the base value plus all SHAP values equals the model's output (in log-odds).

**Q: Why not use SHAP for the neural network too?**
A: Exact SHAP is only cheap for trees. For the network we use occlusion, which asks a direct question, "what if this character weren't there?", and is easy to verify with the deletion test. Approximate SHAP for networks (KernelSHAP) would need thousands of model calls per URL.

**Q: What does occlusion mean?**
A: Hide one input (a character, or a feature set to its average) and see how much the prediction changes. The bigger the change, the more the model relied on that input.

**Q: How do you know your explanations are correct?**
A: Two checks. For SHAP, the values must add up exactly to the model's output (checked in a test and in the notebook). For occlusion, the deletion test: hiding the characters the explanation marks as most important changes the prediction far more than hiding random characters.

**Q: Global vs local explanation?**
A: Global means what matters across all URLs (beeswarm plots, permutation importance). Local means why *this* URL got *this* score (one row of SHAP values, a highlighted URL).

**Q: Your model says accounts.google.com is phishing. Isn't that a failure?**
A: Yes, and we found it *because* of the explanations: `num_subdomains` pushes it to phishing. In PhiUSIIL almost no legitimate domain has a subdomain, so the model learned a dataset artefact. We report it as a limitation and show a warning in the app instead of hiding it.

**Q: The first explanations were all zero for confident URLs. What happened?**
A: We explained log(p / (1 − p)) computed from the probability. For very confident URLs the probability rounds to exactly 1.0 in float32, so hiding a character couldn't change it. We now explain the network's raw output (log-odds) before the sigmoid, and a test checks this.

**Q: Do the explanations show the dataset shortcuts?**
A: Yes. On the raw-URL model, `https://www.` comes out strongly blue (towards legitimate), and `has_www`/`is_https` are top SHAP features, which is exactly the shortcut from Milestone 3. That's why the app uses the domain-only model.
