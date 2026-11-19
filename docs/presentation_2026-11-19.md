# Lab presentation, 19 Nov 2026: read this first (5 minutes)

**The project in one line:** we detect phishing links with machine learning, and we found that the famous 99.9% accuracy on this dataset is a trick of how the data was collected. The honest score is about 86%, and this week we started fixing the data.

**Who says what** (suggested; swap freely):
- **Shaurya:** parts 1–2
- **Abhishek:** parts 3–4
- **Manan:** parts 5–6 and the questions

---

## Part 1 · The task and the data (Shaurya)
- **Task:** given a link, say *phishing* (a fake site that steals passwords) or *legitimate*.
- **Data:** PhiUSIIL, a public dataset. 234,204 links after cleaning: 58% legitimate, 42% phishing.
- **Cleaning:** we removed duplicate links and links labelled both ways, and set phishing = 1.
- **Fair split:** 70% train, 15% validation, 15% test, **grouped by website owner**. All links of one website go into the same part, so the model is never tested on a site it has already seen. With a random split, 39% of phishing test links had a "sibling" from the same site in training.
- **Fair testing:** settings are chosen on validation, the test part is scored once, and everything is repeated on **5 different splits**. We report the average ± how much it varies.

## Part 2 · We matched the published results (Shaurya)
| Model | What it sees | Accuracy |
|---|---|---|
| Logistic Regression, Random Forest, XGBoost | all 50 dataset columns | 99.99–100% |
| XGBoost | 40 numbers we compute from the link (length, digits, hyphens, subdomains…) | 99.6% |
| Character n-grams | every 1–4 letter chunk of the link | 99.7% |

**Say:** "This matches the papers. But when even the simplest model is nearly perfect, something is giving the answer away."

## Part 3 · The shortcut (Abhishek)
- A **single rule with no machine learning**, "link doesn't start with `https://www.` → phishing", is right **99.08%** of the time.
- **Why:** *every* legitimate link in this dataset is a bare homepage like `https://www.example.org`. Only 2.4% of phishing links look like that.
- So the models learned how the data was collected, not what phishing looks like. This is called a **shortcut**.
- **Our fix:** remove the `https://`, the `www.` and the path, and give models **only the domain** (e.g. `login.example.top`). The score drops to **~86%**. That's the honest number.

## Part 4 · Better models on the honest task (Abhishek)
| Model | Domain only, 5 splits |
|---|---|
| XGBoost on 40 features | 81.4% |
| BiLSTM (reads letters in order, both directions) | 85.6% |
| Character n-grams | 86.3% |
| Char-CNN (learns letter patterns like "-login") | 86.6% |
| **Hybrid: CNN + 40 features, trained together** | **86.7%** |

**Say:** "The hybrid is best, on 4 of 5 splits, but only by about half a point. The model matters less than the data."

## Part 5 · Does it work on real websites? (Manan)
We tested on about **25,000 popular real websites** that aren't in the dataset:
- It wrongly flags **15%** of plain sites (`github.com`) and **81%** of sites with a subdomain (`mail.yahoo.com`). The dataset's legitimate links almost never have subdomains.
- Attackers can fool it cheaply. Moving a phishing site to a clean `.com`: caught drops from 86% to 20%. On full links, just switching to `https` drops caught from 100% to 36%.
- We can show *why* with explanations (SHAP, and highlighting which letters push the score).

## Part 6 · This week: fixing the data, on the lab's H100 GPU (Manan)
We added to training:
- real legitimate websites (half of the Tranco top-sites list);
- legitimate links with subdomains;
- fresh phishing (Phishing.Database);
- deliberately edited phishing.

The tests use **other** websites, never seen in training (CrUX and the other half of Tranco), plus **new phishing from 2025–26** (PhishTank).

**Results so far** (character n-gram model, 5 splits; the hybrid is training on the GPU now):
| | Before | After the fix |
|---|---|---|
| False alarms, popular sites with a subdomain | 53% | **39%** |
| False alarms, Tranco sites | 25% | **16%** |
| Edited phishing caught | 31% | **39%** |
| New 2025–26 phishing caught | 82% | 81% |
| Accuracy on the original dataset | 86.3% | 85.7% |

**Say:** "Better data cuts false alarms by a third to a half, for a 0.6-point accuracy cost. It's a partial fix: 39% is still too high."

---

## Trade-offs (have these ready)
- **Dataset accuracy vs. real-world use:** we give up 0.6 points on the dataset to cut false alarms on real sites a lot.
- **Domain only vs. full link:** less information, but it can't be fooled by `https` and can't use the shortcut.
- **Hybrid vs. n-grams:** the hybrid is slightly better but needs a GPU and is harder to explain; n-grams are fast and transparent.
- **Partial fix:** adding `mail.` to a real site still fools the n-gram model, because it learned the exact subdomain words we trained on (`app.`, `docs.`), not the general idea.

## What's left (the remaining ~20%)
1. Finish this week's experiment: the hybrid results, and the write-up.
2. The demo website (built; we'll show it with the final results).
3. The final report and slides.

## Likely questions, short answers
| Question | Answer |
|---|---|
| Why split by website? | Otherwise the model recognises the site instead of detecting phishing (39% of test phishing had a same-site link in training). |
| Why is your score lower than the paper's 99.8%? | The paper's score uses the shortcut. Without it, ~86% is realistic. |
| Why only the domain? | `https`, `www.` and the path give the answer away in this dataset. |
| Is the hybrid really better? | On 4 of 5 splits, by under half a point: small but consistent. |
| Why use a GPU? | The hybrid is a neural network: about 30 s per model on the H100 vs. several minutes on a laptop. |
| How do you know test sites weren't in training? | Every test set is checked to share **zero** website owners with the training data. |
| What would you do next? | More realistic legitimate data with subdomains, and testing on phishing from new sources. Better data, not bigger models. |

**Words:** *false alarm* = a real site called phishing. *Caught* = a phishing link correctly flagged. *Split* = one way of dividing data into train, validation and test. *Owner* = the registered website, e.g. `example.co.uk` for `login.example.co.uk`.
