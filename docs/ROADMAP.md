# Roadmap

| # | Milestone | Status | Notebooks | Guide |
|---|---|---|---|---|
| 1 | Data loading, label direction, cleaning, split by site owner | ✅ | 01, 01b | [01](guide/01_data_and_split.md) |
| 2 | Reproduce published ≥99.9% and build URL-only models | ✅ | 02–05 | [02](guide/02_models.md) |
| 3 | Shortcut analysis: why the scores are so high | ✅ | 06, 07 | [03](guide/03_shortcuts.md) |
| – | Senior review and fixes: 5-split evaluation, CIs, parsing fix (D11) | ✅ | all re-run | [review](review/senior_review_m1_m3.md) |
| 4 | Deep learning on raw URLs: char-CNN, BiLSTM | ✅ | 08 | [04](guide/04_deep_models.md) |
| 5 | Hybrid model: deep URL encoder + our handcrafted features | ✅ | 09 | [05](guide/05_hybrid.md) |
| 6 | Explainability: SHAP for trees, character attributions for deep models | next | | |
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
| Char-CNN | raw URL | 99.68 ± 0.03% |
| Char-CNN | domain only | 86.58 ± 0.92% (≈ n-gram: same ROC-AUC, fewer false alarms) |
| BiLSTM | domain only | 85.55 ± 1.35% |
| Char-CNN / BiLSTM | registered domain | 80.85 ± 1.53% / 80.95 ± 0.62% (n-gram better) |
| **Hybrid CNN + features** | **domain only** | **86.72 ± 0.99%** (best URL model; > n-gram and CNN on 4/5 splits) |
| Hybrid CNN + features | registered domain | 82.01 ± 0.85% (n-gram still better) |

The README table is generated from `results/` (`python -m phishguard.report`), and a test keeps it in sync.

## Rules every new model follows
- Load with `load_clean()`. Run with `experiment.run()`, which handles 5 splits, tuning on validation log-loss, the threshold on validation and a single test pass.
- URL models report **raw URL**, **domain only** and **registered domain** (D10), using the fixed reporting protocol at the end of `docs/DECISIONS.md`.
- Compare against the char n-gram model with `stats.mcnemar` before claiming a model is better.
- Never mix the dataset's provided columns into URL models (D3, D9).
- Add a decision to `docs/DECISIONS.md` and a plain-language page to `docs/guide/`.

## Milestone 4 outcome (D12)
- **Char-CNN** and **BiLSTM** in `src/phishguard/deep.py`, run through `experiment.run()` with early stopping on held-out training owners.
- On raw URLs every URL-only model ties (~99.7%). On domain only, the CNN is at most slightly better than the n-gram model (same ROC-AUC, fewer false alarms). On registered domain, the n-gram model is better.
- The CNN goes forward to the hybrid model: never worse than the BiLSTM, and up to 3× faster.

## Milestone 5 outcome (D13)
- The **hybrid network** (char-CNN + 40 features, trained jointly) is the best domain-only model: 86.72 ± 0.99%. It beats the n-gram model and the CNN on 4/5 splits. The gain is small but consistent.
- Late fusion (log-odds average of CNN + XGBoost) is worse than the CNN alone. Joint training is what makes the combination work.
- On registered domains, the n-gram model is still best.

## Milestone 6 plan
- **Features branch:** SHAP for XGBoost on the 40 features (domain only and raw URL), plus permutation importance of the hybrid's feature inputs.
- **Characters branch:** per-character attributions for the char-CNN/hybrid (gradient × input or occlusion: hide one character, see the probability change), shown as coloured URLs.
- **N-gram model:** its top weights, as a sanity check that the explanations agree.
- **Goal:** explanations the Streamlit app can show for any URL.
