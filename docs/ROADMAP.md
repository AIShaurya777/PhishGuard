# Roadmap

| # | Milestone | Status | Notebooks | Guide |
|---|---|---|---|---|
| 1 | Data loading, label direction, cleaning, split by site owner | ✅ | 01, 01b | [01](guide/01_data_and_split.md) |
| 2 | Reproduce published ≥99.9% and build URL-only models | ✅ | 02–05 | [02](guide/02_models.md) |
| 3 | Shortcut analysis: why the scores are so high | ✅ | 06, 07 | [03](guide/03_shortcuts.md) |
| – | Senior review and fixes: 5-split evaluation, CIs, parsing fix (D11) | ✅ | all re-run | [review](review/senior_review_m1_m3.md) |
| 4 | Deep learning on raw URLs: char-CNN, BiLSTM | next | | |
| 5 | Hybrid model: deep URL encoder + our handcrafted features | | | |
| 6 | Explainability: SHAP for trees, character attributions for deep models | | | |
| 7 | Robustness: original vs modified phishing URLs, model not retrained | | | |
| 8 | Streamlit app | | | |
| 9 | Report and slides | | | |

## Headline results (test accuracy, mean ± sd over 5 site-owner splits)
| Model | Input | Accuracy |
|---|---|---|
| Random Forest / XGBoost | all provided columns | 100.00 ± 0.00% (reproduces the published 99.79–99.99%) |
| `URLSimilarityIndex < 100` (one-line rule, seed 42) | 1 column | 99.66% |
| Char n-gram LogReg | raw URL | 99.69 ± 0.05% |
| Char n-gram LogReg | domain only (shortcut-reduced) | **86.27 ± 0.89%** |
| Char n-gram LogReg | registered domain (shortcut-reduced) | 82.49 ± 1.93% |
| XGBoost (URL features) | domain only | 81.43 ± 1.15% |

The README table is generated from `results/` (`python -m phishguard.report`), and a test keeps it in sync.

## Rules every new model follows
- Load with `load_clean()`. Run with `experiment.run()`, which handles 5 splits, tuning on validation log-loss, the threshold on validation and a single test pass.
- URL models report **raw URL**, **domain only** and **registered domain** (D10), using the fixed reporting protocol at the end of `docs/DECISIONS.md`.
- Compare against the char n-gram model with `stats.mcnemar` before claiming a model is better.
- Never mix the dataset's provided columns into URL models (D3, D9).
- Add a decision to `docs/DECISIONS.md` and a plain-language page to `docs/guide/`.

## Milestone 4 plan (from the senior review)
- **Char-CNN:** a scaled-down URLNet character branch.
  - URLs cut or padded to 200 characters; a vocabulary of about 100 characters, built on train only; embedding size 32.
  - Convolutions of width 3/4/5/6 with 64–128 filters each, then max-pooling, a dense layer and dropout.
- **BiLSTM:** embedding 32 → 1 layer of 64 units per direction → pooling → dense.
- **Training:** PyTorch on Apple MPS, batch size 256, early stopping on validation log-loss, a few epochs per split.
- **Interface:** wrapped as a scikit-learn-style estimator (`fit` / `predict_proba`), so `experiment.run()` works unchanged.
