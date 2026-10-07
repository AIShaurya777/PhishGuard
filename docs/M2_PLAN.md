# Milestone 2 · Leak-free baseline

**Lab demo: Thursday 15 Oct 2026**

## What we show the professor
> "Trained on all 54 provided columns, a simple model scores ~99.9%. We found out why: several columns give the label away. Here is our leak-free setup (split by domain, URL-only features) and the honest baseline score that every later model (CNN, BiLSTM, hybrid) has to beat."

The demo is one notebook, `notebooks/04_baselines.ipynb`, ending in this table:

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | all provided columns (leaky) | | | | | |
| Random Forest | all provided columns (leaky) | | | | | |
| Logistic Regression | our URL features | | | | | |
| Random Forest | our URL features | | | | | |

The table is the result. The process is what earns marks: decision log entries D5–D7 and one reviewed pull request per person.

---

## Tasks

### Manan · Data split + shared evaluation (`split.py`, `evaluate.py`)
**Due: Fri 9 Oct (merged), because the others need it.**
- `get_splits(df) -> (train_idx, val_idx, test_idx)`: 70/15/15, fixed seed, stratified by `is_phishing`, and **no domain in more than one split**.
- Decide what "domain" means: the full hostname (`Domain` column) or the registered domain (`tldextract`, so that `a.hosting-test.net` and `b.hosting-test.net` count as the same site). Show the numbers and log it as **D5**.
- `evaluate(y_true, y_prob) -> dict` with accuracy, precision, recall, F1, ROC-AUC and the confusion matrix, all with phishing as the positive class.
- Tests: no domain overlap between splits, class ratio within ±1% across splits, same seed gives the same split.
- Tue–Wed: put the final table together, update the README, run the demo once before the lab.

### Shaurya · Shortcut audit + baseline notebook
**Audit due Mon 12 Oct; baselines due Wed 14 Oct.**
- `notebooks/02_shortcut_audit.ipynb`: for each of the 54 provided columns, record its type (from the URL / from the page / computed from dataset statistics) and its single-feature ROC-AUC. Flag columns that almost separate the classes on their own.
- Propose `ALLOWED_COLUMNS` / `BANNED_COLUMNS` with a reason for each; log as **D6**.
- `notebooks/04_baselines.ipynb`: Logistic Regression + Random Forest on (a) all columns and (b) Abhishek's URL features, using `get_splits` and `evaluate`. Tune only on validation; touch the test set once.

### Abhishek · URL feature extraction (`features.py`)
**Due: Mon 12 Oct (merged).**
- `extract_features(urls: pd.Series) -> pd.DataFrame`, computed only from the cleaned URL string. Start from the EDA features (length, dots, digits, special characters, slashes, suspicious keywords, IP address) and add: number of subdomains, path depth, query parameters, character entropy, digit ratio, hyphens in the domain, TLD length.
- **No HTTPS or trailing-slash features**: our data shows those just encode how the dataset was collected (see D3).
- Tests with hand-checked examples (e.g. `http://192.168.0.1/login` → `has_ip = 1`).
- `notebooks/03_url_features.ipynb`: the distribution of each feature by class; log the feature list and reasoning as **D7**.

---

## How we work
- The team repo is **https://github.com/AIShaurya777/PhishGuard**. Manan and Abhishek work in their own forks (`origin` = fork, `upstream` = team repo); Shaurya works on branches in the team repo itself.
- Branch per task: `m2/split`, `m2/shortcut-audit`, `m2/features`. Open a PR into `AIShaurya777/PhishGuard:main`; **someone else reviews and Shaurya merges**.
- Start each task from the latest team code: `git fetch upstream && git checkout -b <branch> upstream/main`.
- Every notebook loads data with `load_clean()` and splits with `get_splits()`. No copied loading code.
- Stuck for more than a day? Say so in the group chat, don't sit on it.

| Date | Milestone |
|---|---|
| Fri 9 Oct | `split.py` + `evaluate.py` merged |
| Mon 12 Oct | `features.py` + shortcut audit merged |
| Wed 14 Oct | Baseline notebook done, table filled in, practice run |
| Thu 15 Oct | Lab demo |
