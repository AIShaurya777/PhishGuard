# Milestone 2 · Reproduce ≥99.9% and build URL-only models ✅

**Status: done.** Results are in `notebooks/05_m2_results.ipynb` and decision D8.

## What we show the professor
> "We reproduce the published PhiUSIIL results with our own split, which keeps every site owner in one split: 99.99–100% test accuracy on all provided columns. Our app only receives a URL, so we also built URL-only models. They reach ~99.7%, in line with the best published URL-only results."

| Model | Input | Test accuracy | Errors (of 35,130) | Published reference |
|---|---|---|---|---|
| XGBoost | all provided columns | **100.000%** | 0 | 99.79% (Prasad & Chandra 2024) |
| Random Forest | all provided columns | 99.997% | 1 | 99.99% (Bao et al., CMC 2026) |
| Logistic Regression | all provided columns | 99.994% | 2 | |
| Average: n-gram + XGBoost | raw URL + URL features | 99.707% | 103 | |
| Char n-gram TF-IDF + LogReg | raw URL | 99.695% | 107 | LR 99.62% (Frontiers 2026) |
| Random Forest | URL features | 99.681% | 112 | |
| XGBoost | URL features | 99.675% | 114 | XGB 99.74% (Frontiers 2026) |

## Where the work is
| Notebook | What it does |
|---|---|
| `01_data_loading.ipynb` | Loads the data, fixes the label direction, cleans URLs (D1–D4) |
| `01b_train_test_split.ipynb` | Split by site owner (D5) |
| `02_all_columns_models.ipynb` | LogReg, Random Forest and XGBoost on all provided columns |
| `03_url_feature_models.ipynb` | URL features: distributions, Random Forest, XGBoost, error analysis |
| `04_char_ngram_model.ipynb` | Char n-gram model, what it learned, average with XGBoost |
| `05_m2_results.ipynb` | Final table and chart against published results |

| Code (`src/phishguard/`) | What it does |
|---|---|
| `data.py` | `load_clean()`: the one way to load data |
| `split.py`, `urls.py` | `get_splits()`: 70/15/15 by site owner |
| `features.py` | `extract_features()`: 39 features from the URL string |
| `modeling.py` | `tune()` and `fit_and_test()`: the shared training routine (D6) |
| `evaluate.py` | `evaluate()`, `best_threshold()`, `save_result()`, `load_results()` |

## Next: Milestone 3 · Shortcut analysis
The scores are very high. Milestone 3 asks *why*. It measures how much each model depends on `https://www.`, on "has a path" and on `URLSimilarityIndex`, and what accuracy is left when those signals are removed.
