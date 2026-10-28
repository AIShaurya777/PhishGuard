# 7 · Robustness (Milestone 7)

Code: `src/phishguard/robustness.py`. Notebook: `11_robustness.ipynb`. Decision: D15. External data: `DataSets/external/`.

## Why test robustness?
The test split contains URLs *collected the same way* as the training URLs. Real attackers and real websites don't follow that recipe. We ask, **without retraining any model**:
1. **Evasion:** if an attacker makes a cheap change to a phishing URL, is it still caught?
2. **False alarms:** is a legitimate site flagged just because it looks different from PhiUSIIL's legitimate URLs?

## The edits
Every edit is a small function with a unit test:
```python
from phishguard.robustness import PHISHING_EDITS, add_subdomain
PHISHING_EDITS["TLD → .com"]("login.paypal-check.cf")      # 'login.paypal-check.com'
PHISHING_EDITS["all of the above"]("secure-1.paypa1-check.cf")  # 'paypalcheck.com'
add_subdomain("example.org", "mail")                       # 'mail.example.org'
```
| Edit (phishing) | Example | Why it's realistic |
|---|---|---|
| TLD → `.com` | `paypal-check.cf` → `paypal-check.com` | a `.com` costs ~$10 |
| drop subdomain | `login.paypal-check.cf` → `paypal-check.cf` | the attacker chooses the host name |
| remove digits and hyphens | `pay-pal-2fa.top` → `paypalfa.top` | pick a cleaner name |
| look-alike digits → letters | `paypa1-l0gin.top` → `paypal-login.top` | use the real letters |
| leave hosting platform | `my-wallet.web.app` → `my-wallet.com` | register an own domain |
| all of the above | `secure-1.paypa1-check.cf` → `paypalcheck.com` | |

An edit only applies where it makes sense. For example, you can't change the TLD of a site hosted on `web.app`.

**Raw-URL edits** (as in an independent audit): switch `http` → `https`, or make the URL look like `https://www.<host>` with no path.

**Legitimate edit:** add a common subdomain (`www2.`, `mail.`, `blog.`, `shop.`, `login.`, `accounts.`).

**Real websites:** the 10,000 most-visited origins in Chrome (CrUX, which includes subdomains), and the Tranco top 20,000 domains, minus every site that appears in PhiUSIIL.

## Results (`notebooks/11_robustness.ipynb`, seed-42 models)
**1. The raw-URL model is trivially evaded.**

| Edit to phishing URLs | Raw-URL n-gram catches |
|---|---|
| none | 100% |
| `http` → `https` | **35.5%** |
| `https://www.<host>`, no path | **17.8%** |

The domain-only models never see the scheme, `www.` or the path, so their input doesn't change at all.

**2. But cheap domain edits evade the domain-only models** (% of phishing domains caught, before → after):

| Edit | Hybrid | n-gram |
|---|---|---|
| drop subdomain | 99 → 54% | 88 → 63% |
| leave hosting platform for an own `.com` | 100 → 72% | 100 → 75% |
| TLD → `.com` | 74 → 51% | 74 → 52% |
| all edits | **86 → 20%** | **83 → 32%** |

**3. Legitimate sites with a subdomain get flagged.** Adding `mail.` to PhiUSIIL's legitimate test domains raises the hybrid's false-alarm rate from 3.3% to 89%.

**4. Real popular websites (not in PhiUSIIL), % flagged as phishing:**

| | Hybrid | n-gram |
|---|---|---|
| CrUX: bare domains (`github.com`) | 15% | 22% |
| CrUX: with a subdomain (`mail.yahoo.com`) | **81%** | 52% |
| Tranco top domains | 19% | 26% |

**5. Why bare domains too? Country priors.** In PhiUSIIL's training data, 79% of `.ru` rows are phishing, so the model flags 37% of *popular* `.ru` sites (`.ir` 34%, `.in` 33%), but 0% of `.org` and `.de`.

**Conclusion:** inside PhiUSIIL the domain-only models reach ~88%. On the real web they would raise many false alarms and miss phishing on cheap `.com` domains. Against edits and subdomains the **n-gram model is the most robust**, though it flags more popular *bare* domains than the hybrid (22% vs 15%). The fix is better legitimate data, not a bigger model.

**Later finding (Milestone 8):** all 782 `google.com` links in PhiUSIIL are labelled phishing (pages hosted on Google Docs, Sites and Cloud Storage, and 7 on `accounts.google.com`). So flagging `accounts.google.com` also follows the training labels, not only the subdomain shortcut. See D16.

## Likely viva questions
**Q: So your best model doesn't work in the real world?**
A: Within PhiUSIIL it reaches ~88% on domains alone, but it doesn't transfer. It flags 15% of popular plain domains and 81% of popular sites with a subdomain, because PhiUSIIL's legitimate list has almost no subdomains and its phishing feed over-represents some countries. We measured this ourselves on 25,000 real websites, and independent audits report the same. Showing *why* a model fails is part of the result.

**Q: Why is the hybrid worse than the n-gram model here, when it was better on the test split?**
A: Its extra features, `num_subdomains` and `on_hosting_platform`, are exactly what our edits and real subdomains change. They help on PhiUSIIL-like data and hurt outside it. The n-gram model spreads its weight over many character chunks, so one change matters less.

**Q: Why not retrain on the edited URLs (adversarial training)?**
A: The task is to measure the deployed model as it is. An independent audit tried adversarial training on PhiUSIIL and found it mostly moved the threshold, so false alarms rose a lot. Retraining would also need legitimate data that looks like the real web, which PhiUSIIL doesn't have.

**Q: Why are the domain-only models immune to the HTTPS attack?**
A: They never see the scheme, `www.` or the path, only the host name. The HTTPS/`www.` edit leaves their input unchanged. We check this in the notebook (100% unchanged).

**Q: Why use CrUX and Tranco?**
A: They are independent lists of popular, real websites. CrUX lists *origins* (including subdomains such as `accounts.google.com`). Tranco lists registered domains. Both are research-grade and documented (`DataSets/external/README.md`). We only use them to test, never to train.
