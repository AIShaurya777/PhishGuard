# PhishGuard: the project explained simply

Read these in order. Each one explains one part of the project in plain words, shows the important code, and ends with questions the professor may ask, along with their answers.

| # | Guide | Covers |
|---|---|---|
| 1 | [Data and split](01_data_and_split.md) | The dataset, cleaning, label direction, train/validation/test split |
| 2 | [Models (Milestone 2)](02_models.md) | The 7 models, how we trained them fairly, and the results |

## The project in one minute
- **Problem:** given a URL, say whether it is phishing (a fake site that steals passwords or money) or legitimate.
- **Data:** PhiUSIIL, 235,795 URLs (after cleaning, 234,204): 57.6% legitimate, 42.4% phishing. Each row has the URL plus 54 columns the dataset authors computed, some from the URL and some from the downloaded web page.
- **What we've done so far:**
  1. Cleaned the data and split it so that **no website owner appears in both training and testing**.
  2. Reproduced the published accuracy: **99.99–100%** using all the dataset's columns.
  3. Built models that use **only the URL**, because that's all our app gets from a user: **~99.7%**.
- **What's next:**
  - Find out *why* the scores are so high (shortcuts in the dataset).
  - Deep learning models (CNN, BiLSTM) on the raw URL.
  - A hybrid model.
  - Explanations with SHAP.
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
| **Data leakage** | The model sees information in training that it wouldn't have in real use, which makes scores look better than they are |
| **Shortcut** | A pattern that predicts the label in this dataset but not in the real world |

## Where things are
```
src/phishguard/   ← reusable code (every notebook imports from here)
notebooks/        ← experiments, numbered in order, with outputs saved
tests/            ← automatic checks (run: .venv/bin/pytest)
results/          ← test scores of every model (JSON)
docs/DECISIONS.md ← every important decision and why we made it
docs/guide/       ← you are here
```
