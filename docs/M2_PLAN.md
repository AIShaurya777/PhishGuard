# Milestone 2 · Reproduce ≥99.9% and build URL-only models

**Lab demo: Thursday 15 Oct 2026**

## What we show the professor
> "We match the published results on PhiUSIIL (≥99.9%) using our own leak-safe split, which keeps every site owner in one split. We also have URL-only models, the kind our Streamlit app needs, because a user can only give us a URL, and they reach ~99.7–99.9%."

The demo ends in one table; the targets come from published work:

| Model | Input | Test accuracy | F1 | ROC-AUC | Published reference |
|---|---|---|---|---|---|
| Logistic Regression | all provided columns | | | | |
| Random Forest | all provided columns | | | | RF 99.99% (Bao et al., CMC 2026) |
| XGBoost | all provided columns | | | | 99.79% (Prasad & Chandra 2024, original paper) |
| Char n-gram TF-IDF + LogReg | raw URL | | | | LR 99.62% (Frontiers Comp. Sci. 2026) |
| Random Forest / XGBoost | our URL features | | | | XGB 99.74% (Frontiers 2026) |

First check, on the validation set: all-columns LogReg/RF made 1 error in 35,130. Char n-gram LogReg on the raw URL got 99.73%. So the all-columns target is easy; the URL-only target of 99.9% is the real challenge.

**Next milestone (M3):** explain *why* the scores are so high (the shortcut audit) and what happens when the shortcuts are removed. Example: removing the `https://www.` prefix drops char n-gram LogReg from 99.73% to 90.42%. That work waits until the 99.9% target is reached.

---

## Tasks

### Manan · Split, evaluation, char n-gram baseline
**`split.py` + `evaluate.py` due Fri 9 Oct (done, in review); baseline by Tue 13 Oct.**
- `get_splits(df)`: 70/15/15, stratified, with no site owner in more than one split (D5).
- `evaluate(y_true, y_prob)`: phishing is the positive class.
- `notebooks/05_char_ngram_baseline.ipynb`: TF-IDF over character n-grams of the raw URL + Logistic Regression, tuned on validation.
- Tue–Wed: put the final table together and run the demo once before the lab.

### Shaurya · Classical ML on all provided columns
**Due Wed 14 Oct.**
- `notebooks/02_all_columns_models.ipynb`: Logistic Regression (with StandardScaler), Random Forest and XGBoost on all numeric provided columns.
- Use `get_splits()` and `evaluate()`. Tune on validation only; touch test once.
- Compare against the published numbers in the table, and show the top features (feature importance).
- Log any modelling choices (tuning ranges, which columns count as "numeric") as **D6**.

### Abhishek · URL feature extraction + models on it
**`features.py` merged Mon 12 Oct; results by Wed 14 Oct.**
- `extract_features(urls: pd.Series) -> pd.DataFrame`: everything computable from the URL string. Include length, counts, entropy, subdomains, path depth, query parameters, IP host, keywords, TLD, scheme and `www`. Use whatever helps reach the target; in M3 we'll test which features are shortcuts.
- Tests with hand-checked examples.
- `notebooks/03_url_features.ipynb`: feature distributions by class, then Random Forest and XGBoost on the features, tuned on validation.
- Log the feature list and reasoning as **D7**.

---

## How we work
- The team repo is **https://github.com/AIShaurya777/PhishGuard**. Manan and Abhishek work in their own forks (`origin` = fork, `upstream` = team repo); Shaurya works on branches in the team repo itself.
- Branch per task: `m2/split`, `m2/all-columns`, `m2/features`. Open a PR into `AIShaurya777/PhishGuard:main`; **someone else reviews and Shaurya merges**.
- Start each task from the latest team code: `git fetch upstream && git checkout -b <branch> upstream/main`.
- Every notebook loads data with `load_clean()`, splits with `get_splits()` and scores with `evaluate()`. No copied loading code.
- Stuck for more than a day? Say so in the group chat, don't sit on it.

| Date | Milestone |
|---|---|
| Fri 9 Oct | `split.py` + `evaluate.py` merged |
| Mon 12 Oct | `features.py` merged |
| Wed 14 Oct | All models trained, table filled in, practice run |
| Thu 15 Oct | Lab demo |
