# 8 · The website (Milestone 8)

Code: `src/phishguard/serving.py` (models, numbers, warnings), `src/phishguard/web.py` (server), `web/` (the page). Decision: D16. Tests: `tests/test_web.py`.

## Run it
```bash
.venv/bin/python -m phishguard.web      # then open http://127.0.0.1:8000
```
A link like `http://127.0.0.1:8000/?url=https://mail.yahoo.com` opens with that link already checked. This is handy in the viva.

## What it shows for one link
Take `https://accounts.google.com/signin`:

1. **What the model reads:** `https://` and `/signin` are crossed out, because the model only sees `accounts.google.com` (D10).
2. **Letter shading:** red letters pushed the hybrid towards phishing, blue towards legitimate. This is character occlusion from Milestone 6: blank one letter, measure how much the score changes.
3. **Two verdicts:** the hybrid (our best model on the test split) and the character n-gram model (more robust to edits, Milestone 7). Each has its score and its threshold, which was chosen on validation.
4. **Warnings,** only when one of our measured weak spots applies:
   - the site is in the training data, so the model may simply remember it;
   - the link has a subdomain, and we know the model flags 81% of popular sites with one;
   - the suffix (`.ru`, `.top`, `.co.uk` …) was lopsided in training;
   - the two models disagree.
5. **What moved the hybrid:** the measured properties (number of subdomains, hyphens …) with the biggest effect, from feature occlusion.

## How it fits together
```
browser (web/index.html + app.js)
   │  POST /api/check {"url": "..."}           GET /api/numbers
   ▼                                           ▼
phishguard.web (FastAPI) ──► serving.Checker.check(url)
                              ├─ URL_VIEWS["domain only"]   → accounts.google.com
                              ├─ explain_url(hybrid, url)   → score + letter + feature effects
                              ├─ ngram.predict_proba(...)   → second opinion
                              └─ warnings(...)              → from models/site_numbers.json
```
```python
from phishguard.serving import Checker
checker = Checker.load()
result = checker.check("https://mail.yahoo.com")
result["models"]["hybrid"]["probability"]   # 0.998
result["warnings"][0]                        # 'This link has a subdomain (mail.). ...'
```

## Where the numbers come from
Nothing on the page is typed by hand:
- **Accuracy chart:** read from `results/`, the same files as the README table.
- **Real-website and attack numbers:** `python -m phishguard.serving` recomputes them from the **same saved models the site uses**, with the method of notebook 11. They match notebook 11 exactly, and they're saved in `models/site_numbers.json`.

## What we found while building it
Every `google.com` link in PhiUSIIL (782 of them) is labelled **phishing**. They are phishing pages hosted on Google Docs, Google Sites and Cloud Storage, and 7 are on `accounts.google.com` itself. So when the model flags `accounts.google.com`, it is partly doing what the labels taught it. A model that sees only the domain can't tell a phishing form on Google Docs from Google's real sign-in page.

## The models don't know brands (D17)
| Domain | Hybrid | n-gram |
|---|---|---|
| `google.com` (real) | 81.5% phishing | 95.3% phishing |
| `g0ogle.com` | 70.4% phishing | 87.8% phishing |
| `goog1e.com` | 46.4%, legitimate | 73.7% phishing |
| `paypal.com` (real) | 33.3%, legitimate | 44.6%, legitimate |
| `paypa1.com` | 38.3%, legitimate | 75.7% phishing |

The real Google scores *higher* than the fake one. So when `g0ogle.com` is flagged, the model isn't detecting a look-alike: it's reacting to the letters "google", which were always phishing in training. In the letter shading, the `0` is even blue.

So the site adds a **look-alike rule** (`src/phishguard/lookalike.py`) in its own box, clearly labelled as a rule, not ML. It compares the name with the 1,000 most popular sites: `goog1e` → google.com, `gooogle` → google.com, `paypal-login.top` → paypal.com. Honest numbers: it catches only 1.9% of PhiUSIIL's phishing (the dataset has few look-alikes) and fires on 1.6–2.8% of popular real sites (often a company's own regional site, like `google.fr`).

## Likely viva questions
**Q: Your model flagged `g0ogle.com`. So it detects look-alikes?**
A: No. The real `google.com` scores even higher (81.5%), because every google.com link in the training data is phishing. `goog1e.com` and `paypa1.com` pass the hybrid as legitimate. The look-alike box on the site is a separate rule, and it says so.

**Q: Why a website and not Streamlit?**
A: We wanted to show the link letter by letter, with the ignored parts crossed out. That needs control over the layout. FastAPI plus a plain HTML page gives that with no build tools. The logic is in Python (`serving.py`) and tested. The page only displays it.

**Q: Is this a real phishing detector?**
A: No, and the page says so. It's trained on one dataset whose legitimate side doesn't look like the real web (Milestone 7). The site's purpose is to show *what* the model learned, including its mistakes.

**Q: Why show two models?**
A: The hybrid is best on the test split. The n-gram model is more robust to edits and subdomains. When they disagree, that is useful information, so we say so.

**Q: How do you know the warnings are right?**
A: Each one quotes a number we measured: subdomain false alarms from the CrUX test, suffix shares from the training split, training membership from the training split's site owners. `tests/test_web.py` checks when each warning appears and when it must not.

**Q: Does the website retrain anything?**
A: No. It loads `models/hybrid_domain_only.pt` (notebook 10) and `models/ngram_domain_only.joblib` (built by `phishguard.serving` on the same seed-42 split).
