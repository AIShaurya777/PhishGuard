# Roadmap

| # | Milestone | Status | Notebooks | Guide |
|---|---|---|---|---|
| 1 | Data loading, label direction, cleaning, split by site owner | ✅ | 01, 01b | [01](guide/01_data_and_split.md) |
| 2 | Reproduce published ≥99.9% and build URL-only models | ✅ | 02–05 | [02](guide/02_models.md) |
| 3 | Shortcut analysis: why the scores are so high | ✅ | 06, 07 | [03](guide/03_shortcuts.md) |
| – | Senior review and fixes: 5-split evaluation, CIs, parsing fix (D11) | ✅ | all re-run | [review](review/senior_review_m1_m3.md) |
| 4 | Deep learning on raw URLs: char-CNN, BiLSTM | ✅ | 08 | [04](guide/04_deep_models.md) |
| 5 | Hybrid model: deep URL encoder + our handcrafted features | ✅ | 09 | [05](guide/05_hybrid.md) |
| 6 | Explainability: SHAP for trees, character attributions for deep models | ✅ | 10 | [06](guide/06_explainability.md) |
| 7 | Robustness: original vs modified phishing URLs, model not retrained | ✅ | 11 | [07](guide/07_robustness.md) |
| 8 | Project website: link checker with explanations and honest warnings | ✅ | – | [08](guide/08_website.md) |
| 9 | Report and slides | next | | |

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

## Milestone 6 outcome (D14)
- `phishguard.explain`: exact TreeSHAP for XGBoost; character and feature occlusion plus permutation importance for the hybrid; `explain_url()` for the app.
- Checked: SHAP adds up (gap 2·10⁻⁵). Deletion test: top-5 characters flip 20% of caught phishing, random 5 only 2.8%. The hybrid's suffix scores agree with the n-gram weights (ρ = 0.89).
- Found: the raw-URL shortcut is visible in both explanations. The domain-only model flags legitimate subdomains (`accounts.google.com` → 0.999).
- Saved `models/hybrid_domain_only.pt` for the app.

## Milestone 7 outcome (D15)
- The raw-URL model is evaded by switching to HTTPS (100% → 35.5% caught). The domain-only models are immune by construction.
- Cheap domain edits evade the domain-only models (all edits: hybrid 86% → 20%, n-gram 83% → 32%).
- On popular real websites outside PhiUSIIL, the hybrid flags 15% of bare domains and 81% of hosts with a subdomain. Country priors from the phishing feed (`.ru`, `.ir`, `.in`) explain part of it.
- Against edits and subdomains the n-gram model is the most robust (it flags more popular bare domains, 22% vs 15%). The fix is better legitimate data.

## Milestone 8 outcome (D16)
- A website instead of Streamlit: `phishguard.web` (FastAPI) and `web/` (HTML, CSS, JavaScript). Run `.venv/bin/python -m phishguard.web`, open http://127.0.0.1:8000.
- It checks a link with the hybrid and the n-gram model, crosses out what the models ignore, shades each letter by its effect, lists the features that moved the hybrid, and warns when one of our measured weak spots applies.
- "What we found" tells the project's story with numbers read from `results/` and recomputed from the saved models.
- Found: every `google.com` row in PhiUSIIL is labelled phishing (Docs/Sites-hosted phishing). That explains part of the `accounts.google.com` false alarm.
