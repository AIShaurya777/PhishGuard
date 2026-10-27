# Notebooks

Run them in this order. Each one is saved with its outputs, so you can also just read them. All of them import shared code from `src/phishguard/`.

| Notebook | Milestone | What it does |
|---|---|---|
| `01_data_loading` | 1 | Load the full dataset, fix the label direction, clean URLs (D1–D4) |
| `01b_train_test_split` | 1 | Split by site owner, so no website is in both train and test (D5, D11) |
| `02_all_columns_models` | 2 | LogReg, Random Forest, XGBoost on all provided columns: reproduces the published ≥99.9% |
| `03_url_feature_models` | 2 | Our 40 URL features + Random Forest / XGBoost |
| `04_char_ngram_model` | 2 | Character n-gram model, what it learned, McNemar test, log-odds average with XGBoost |
| `05_m2_results` | 2 | All Milestone 2 results with error bars, significance tests, real-traffic precision |
| `06_shortcut_audit` | 3 | Why the scores are so high: one-line rules, column groups, hidden leaks |
| `07_url_views` | 3 | URL-only models with the shortcuts removed (raw → registered domain) |
| `08_deep_models` | 4 | Char-CNN and BiLSTM on raw URL / domain only / registered domain, McNemar vs n-gram (Apple GPU, ~2 h) |
| `09_hybrid` | 5 | Hybrid network (char-CNN + 40 URL features) and log-odds average (CNN + XGBoost), vs n-gram and CNN (~1.5 h) |
| `10_explainability` | 6 | SHAP for XGBoost, permutation importance and character occlusion for the hybrid, deletion test; saves `models/hybrid_domain_only.pt` (~15 min) |

Every model runs on 5 site-owner splits (seeds 42, 0, 1, 2, 3) and reports mean ± sd (D11). Results are saved in `results/<milestone>/`. Running everything takes roughly four to five hours on a laptop; notebooks 08, 09 and 03 are the slowest.
