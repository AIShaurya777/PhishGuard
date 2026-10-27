# Archive

The team's first two notebooks, kept for the history of the project. **Don't use them for results.**

| File | What it was | Replaced by |
|---|---|---|
| `processed_dataset_colab/ML_PhishGuard_proceesed_dataset.ipynb` | First cleaning notebook (Colab) | `notebooks/01_data_loading.ipynb` |
| `ML_EDA.ipynb` | First EDA: URL length, dots, digits, special characters, keywords, IP (Colab) | `notebooks/03_url_feature_models.ipynb`, `notebooks/06_shortcut_audit.ipynb` |

**Why they were replaced.** Both read a Colab upload of the CSV that had been cut off: 196,403 of 235,795 rows (decision D1). They also use `label` with 1 = legitimate (D2), and they upload the file by hand, so nobody else can re-run them. Their findings (longer URLs, more digits and more special characters in phishing) still hold on the full data, and the EDA's hand-made features became the starting point for `src/phishguard/features.py`.
