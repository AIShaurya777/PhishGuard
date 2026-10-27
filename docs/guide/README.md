# PhishGuard: the project explained simply

Read these in order. Each one explains one part of the project in plain words, shows the important code, and ends with questions the professor may ask, along with their answers.

| # | Guide | Covers |
|---|---|---|
| 1 | [Data and split](01_data_and_split.md) | The dataset, cleaning, label direction, train/validation/test split |
| 2 | [Models (Milestone 2)](02_models.md) | The 7 models, how we trained them fairly, and the results |
| 3 | [Shortcuts (Milestone 3)](03_shortcuts.md) | Why the scores are so high, and the shortcut-reduced URL-only number |
| 4 | [Deep learning (Milestone 4)](04_deep_models.md) | Char-CNN and BiLSTM: how they read a URL, how they're trained, results |
| 5 | [Hybrid model (Milestone 5)](05_hybrid.md) | Combining the CNN with handcrafted features: joint network vs averaging |
| 6 | [Explainability (Milestone 6)](06_explainability.md) | SHAP, permutation importance, character highlighting, and checking the explanations are right |

## The project in one minute
- **Problem:** given a URL, say whether it is phishing (a fake site that steals passwords or money) or legitimate.
- **Data:** PhiUSIIL, 235,795 URLs (after cleaning, 234,204): 57.6% legitimate, 42.4% phishing. Each row has the URL plus 54 columns the dataset authors computed, some from the URL and some from the downloaded web page.
- **What we've done so far:**
  1. Cleaned the data and split it so that **no website owner appears in both training and testing**.
  2. Reproduced the published accuracy: **99.99–100%** using all the dataset's columns.
  3. Built models that use **only the URL**, because that's all our app gets from a user: **~99.7%**.
  4. Found out *why* the scores are so high. Every legitimate URL in the dataset is a bare `https://www.` homepage, and some columns nearly contain the answer. With those shortcuts removed, a URL-only model gets **~86%** from the domain name alone.
  5. Made every number trustworthy: each model is trained and tested on **5 different splits**, and we report mean ± standard deviation (after a senior review, `docs/review/`).
  6. Trained deep learning models that read URLs character by character (a CNN and a BiLSTM). They tie with the n-gram model on raw URLs (~99.7%). On domain only, the CNN is at most slightly better (86.6%, fewer false alarms). On the registered domain, the simpler n-gram model wins.
  7. Built a **hybrid** network that reads the characters (CNN) *and* our 40 handcrafted features, trained together. On domain names it's our best URL model (86.7%), slightly but consistently ahead of the n-gram model and the CNN.
  8. Explained the models with **SHAP** and **character highlighting**, and checked the explanations are faithful. They expose the raw-URL shortcut and a real weakness: the domain model flags legitimate subdomains like `accounts.google.com`.
- **What's next:**
  - A robustness test with modified URLs.
  - The Streamlit app.

## Words you'll hear
| Word | Meaning |
|---|---|
| **Positive class** | The class we're trying to detect: phishing = 1 |
| **Train / validation / test** | Train: the model learns from it. Validation: we pick settings with it. Test: we score once at the end. |
| **Accuracy** | Share of URLs classified correctly |
| **Precision** | Of the URLs we *flagged* as phishing, how many really were |
| **Recall** | Of the *real* phishing URLs, how many we caught |
| **F1** | One number balancing precision and recall |
| **ROC-AUC** | How well the model *ranks* phishing above legitimate, regardless of threshold (1.0 = perfect) |
| **Threshold** | The cut-off on the model's probability: above it = phishing |
| **Mean ± sd over 5 splits** | We repeat the whole train/validate/test on 5 different splits; ± is how much the score moves between them |
| **95% confidence interval (CI)** | The range the true accuracy probably lies in, given our test set |
| **McNemar test / p-value** | Checks whether two models really differ, using only the URLs they disagree on. p > 0.05 means "could be chance" |
| **PR-AUC, MCC** | Scores that stay honest when one class is rare |
| **Data leakage** | The model sees information in training that it wouldn't have in real use, which makes scores look better than they are |
| **Shortcut** | A pattern that predicts the label in this dataset but not in the real world |
| **Embedding** | A learned vector of numbers for each character |
| **Epoch** | One pass of training over all training URLs |
| **Early stopping** | Stop training when a held-out set stops improving, and keep the best weights |
| **Dropout** | Randomly switching off neurons during training so the network can't rely on any single one |
| **Log-odds** | log(p / (1 − p)): the probability on a scale where contributions from different inputs add up |
| **SHAP value** | A feature's fair share of one prediction (from game theory); all shares add up to the model's output |

## Where things are
```
src/phishguard/   ← reusable code (every notebook imports from here)
notebooks/        ← experiments, numbered in order, with outputs saved
tests/            ← automatic checks (run: .venv/bin/pytest)
results/          ← test scores of every model (JSON)
docs/DECISIONS.md ← every important decision and why we made it
docs/guide/       ← you are here
```
