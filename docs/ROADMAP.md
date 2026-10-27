# Roadmap

| # | Milestone | Status | Notebooks | Guide |
|---|---|---|---|---|
| 1 | Data loading, label direction, cleaning, split by site owner | ✅ | 01, 01b | [01](guide/01_data_and_split.md) |
| 2 | Reproduce published ≥99.9% and build URL-only models | ✅ | 02–05 | [02](guide/02_models.md) |
| 3 | Shortcut analysis: why the scores are so high | ✅ | 06, 07 | [03](guide/03_shortcuts.md) |
| 4 | Deep learning on raw URLs: char-CNN, BiLSTM | next | | |
| 5 | Hybrid model: deep URL encoder + handcrafted features | | | |
| 6 | Explainability: SHAP for trees, character attributions for deep models | | | |
| 7 | Robustness: original vs modified phishing URLs, model not retrained | | | |
| 8 | Streamlit app | | | |
| 9 | Report and slides | | | |

## Headline results so far (test set, 35,130 URLs, split by site owner)
| Model | Input | Accuracy |
|---|---|---|
| XGBoost | all provided columns | 100.00% (reproduces the published 99.79–99.99%) |
| `URLSimilarityIndex < 100` (one-line rule) | 1 column | 99.72% |
| Char n-gram LogReg | raw URL | 99.70% |
| Char n-gram LogReg | domain only | **88.59%** (the honest number, D10) |
| XGBoost (URL features) | domain only | 81.42% |

## Rules every new model follows
- Load with `load_clean()`, split with `get_splits()`, train with `modeling.tune()` / `fit_and_test()`. Test once.
- URL models report **raw URL** and **domain only** (`phishguard.urls.URL_VIEWS`).
- Save results with `save_result(..., group="<milestone>")`.
- Add a decision to `docs/DECISIONS.md` and a plain-language page to `docs/guide/`.
