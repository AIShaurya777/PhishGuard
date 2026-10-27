# Decision log

One entry per decision that affects the data, the models or the evaluation.
Add yours in the same format and link the evidence. This file becomes the
"Methodology" section of the report.

---

## D1 · Load the full raw file and check its shape
- **Context:** The first cleaning notebook read only 196,403 of 235,795 rows (a truncated Colab upload) and nobody noticed.
- **Decision:** `load_raw()` reads the zip in the repo directly and raises an error unless the shape is exactly (235,795, 55) with no missing values.
- **Evidence:** `notebooks/01_data_loading.ipynb` §1, `tests/test_data.py::test_load_raw_rejects_truncated_file`.

## D2 · Phishing is the positive class (`is_phishing = 1`)
- **Context:** PhiUSIIL uses `label = 1` for legitimate and `label = 0` for phishing. Precision, recall and F1 are reported for class 1.
- **Decision:** Replace `label` with `is_phishing = 1 - label` and drop `label`.
- **Evidence:** notebook §2.

## D3 · Strip trailing `/` from URLs
- **Context:** 36.5% of phishing URLs end in `/` and no legitimate URL does. The difference comes from how the data was collected, so a model would learn the label from it.
- **Decision:** Strip surrounding whitespace and trailing `/` before anything else.
- **Consequence:** The dataset's precomputed columns were computed on the *original* URL, so the slash shortcut survives inside them. `URLLength − len(cleaned URL)` is −1 for every legitimate URL but often 0 or +1 for phishing, so "≠ −1 → phishing" has 100% precision (notebook 06 §4b). That's why our own features use the cleaned URL, and why the hybrid model (Milestone 5) will never mix in the dataset's columns.
- **Evidence:** notebook §3a.

## D4 · Drop label conflicts, then duplicates
- **Context:** After normalisation, 1,590 URLs are duplicates. One URL (`https://www.education.gouv.fr`) appears with both labels.
- **Decision:** Drop every row of a URL that has conflicting labels (2 rows), then keep the first copy of each remaining URL (1,589 rows dropped).
- **Result:** 234,204 rows: 134,849 legitimate (57.6%) and 99,355 phishing (42.4%).
- **Evidence:** notebook §3b and §4.

## D5 · Split 70/15/15 by site owner
- **Context:** With a plain random split, 39% of phishing test URLs have a site owner (e.g. `evil.com` for `a.evil.com`) that also appears in training, so test scores would partly measure recognising the owner rather than detecting phishing.
- **Options compared:**
  - A: hostname (220k groups). Leaves the same-owner leak in place.
  - B: registered domain (176k groups). Puts whole hosting platforms like `web.app` in one split, which tests a harder, different question.
  - C: registered domain where customers of platforms on the public suffix list count separately (198k groups).
- **Decision:** C. `get_splits()` shuffles owners within each class (seed 42) and fills test, then val, then train to 15/15/70% of that class's rows. Each owner joins the class of its majority label. No owner is shared between splits, and each split is 42.4% phishing. `tldextract` is pinned, because its bundled suffix list defines the groups. Split sizes after the parsing fix are in D11.
- **Mixed-label owners:** About 105 owners (≈560 rows) have both phishing and legitimate URLs, e.g. `pinata.cloud` (mostly phishing), `washington.edu` (mostly legitimate), `adobe.com`, `office.com`. These are usually abused or compromised parts of real sites. Their minority-label URLs set a small error floor that no model can avoid.
- **Concentration:** A few owners are large, e.g. `ipfs.io` with about 1,400 phishing URLs. A whole owner always lands in one split, so a single split can be easier or harder than average. That's one reason we report several splits (D11).
- **Rule:** tune on validation only; use test once, for the final table. Every model reports through `evaluate()` (phishing = positive class).
- **Limitation:** Platforms missing from the suffix list (e.g. `weeblysite.com`, 3,079 URLs) still act as a single owner.
- **Evidence:** `notebooks/01b_train_test_split.ipynb`, `tests/test_split.py`.

## D6 · One protocol for every model: tune on validation, threshold on validation, test once
- **Context:** Published results on PhiUSIIL are hard to compare, because papers use different splits and some choose settings using the test set.
- **Decision:**
  - Every model goes through `phishguard.modeling`. `tune()` fits each setting on train and keeps the one with the lowest **validation log-loss**. Log-loss changes smoothly, while accuracy would let a single validation URL decide the winner.
  - `fit_and_test()` picks the decision threshold that minimises validation errors (`best_threshold()`; ties go to the middle of the tied range) and scores test exactly once.
  - `experiment.run()` repeats all of this on 5 splits (D11) and saves the results to `results/<milestone>/`. Summary notebooks only read them.
- **Why choose the threshold too:** The default 0.5 is arbitrary, and a validation-chosen threshold costs nothing in honesty. In practice it changes only a handful of test predictions, so the gain is small. For the app, a stable threshold matters more than one extra correct validation URL.
- **Evidence:** `src/phishguard/modeling.py`, `tests/test_modeling.py`, notebooks 02–05.

## D7 · URL features are computed from the URL string only
- **Context:** The app receives only a URL. It can't supply page-content columns (`LineOfCode`, `NoOfImage`, …) or `URLSimilarityIndex`, which compares the URL against a list of known legitimate URLs.
- **Decision:**
  - `extract_features()` computes 39 numeric features from the URL string: sizes, character counts and ratios, host structure (subdomains, IP, port, `www`), scheme, hosting platform or shortener, path and query structure, suspicious words, and entropy.
  - No network calls and no dataset columns, so the same code runs in the app.
- **Deliberately included:** the `is_https` and `has_www` features. Our first target is the highest accuracy we can get (docs/ROADMAP.md, Milestone 2). Whether these are shortcuts is tested in Milestone 3.
- **Evidence:** `src/phishguard/features.py`, `tests/test_features.py`, `notebooks/03_url_feature_models.ipynb`.

## D8 · Milestone 2 results (mean ± sd over 5 splits, D11)
- **All provided columns:** Random Forest and XGBoost 100.00 ± 0.00%, LogReg 99.99 ± 0.00% (0–3 errors out of 35,130 per split). This reproduces the published 99.79% (Prasad & Chandra 2024) and 99.99% (Bao et al. 2026). With so few errors the three models are indistinguishable from each other and from perfect, so we don't rank them.
- **URL only:**
  - Char n-gram LogReg: 99.69 ± 0.05%.
  - Log-odds average of n-gram + XGBoost: 99.70 ± 0.03%.
  - XGBoost / Random Forest on our 40 URL features: 99.63–99.64 ± 0.03–0.04%.
  - Published URL-only results on a similar host-disjoint split: 99.62–99.84% (Ahamed et al., Frontiers 2026; see `docs/references.md`).
- **Significance (McNemar, seed-42 test set):** the n-gram model is significantly better than the feature models (p = 0.004 vs XGBoost). The averaged model is **not** better than the n-gram model alone (17 vs 17 disagreements, p = 1.0).
- **Remaining errors:** nearly all are phishing sites shaped like a bare `https://www.<domain>` homepage, the shape of every legitimate URL in the dataset.
- **Evidence:** `notebooks/05_m2_results.ipynb`.

## D9 · The provided columns are dominated by shortcuts
- **Context:** In Milestone 2 even Logistic Regression reached 99.99% on the provided columns.
- **Findings (notebook 06; single-column AUCs on the training split, rules and models scored on test):**
  - `URLSimilarityIndex < 100 → phishing` alone: 99.66% accuracy (seed-42 test split), precision 1.0.
  - "Doesn't start with `https://www.` → phishing" alone: 99.08%. Every legitimate URL starts with it; only 2.4% of phishing URLs do.
  - XGBoost on page columns only: 99.71 ± 0.08% (5 splits). They describe what the crawler downloaded.
  - XGBoost on URL columns without `IsHTTPS`: still 99.61 ± 0.06%. `URLLength` minus the counted characters is 12 (`https://www.`) for all but 4 of 134,849 legitimate URLs, and `URLLength` also keeps the trailing slash we removed (D3).
- **Decision:** Column groups are defined in `src/phishguard/columns.py` (19 URL, 3 statistic, 28 page). The all-column results are reported as a *reproduction* of published work, not as evidence of phishing detection. The app and the hybrid model use only our own URL features, never the provided columns.
- **Protocol note:** The rules were first noticed while exploring the full dataset (notebook 01), which is stated openly. Scoring them on test is a check, not a tuned result.
- **Independent confirmation:** several later-found audits report the same shortcuts: the `URLSimilarityIndex` rule (99.67%), the `https://www.` rule (98.96–99.7%), and the prefix being recoverable from `URLLength` (ollivianguy, Yoyagm, phish-drift; `docs/references.md`, *Independent audits*). They are not peer-reviewed.
- **Evidence:** `notebooks/06_shortcut_audit.ipynb`.

## D10 · Report every URL model on the raw URL and on shortcut-reduced views
- **Context:** URL-only models reach ~99.7% mostly from the prefix and from whether there's a path.
- **Findings (notebook 07, test accuracy, mean ± sd over 5 splits):**

  | View | Char n-gram LogReg | XGBoost (URL features) |
  |---|---|---|
  | raw URL | 99.70 ± 0.03% | 99.64 ± 0.03% |
  | no scheme/www | 89.18 ± 0.59% | 85.95 ± 0.95% |
  | domain only | 86.27 ± 0.89% | 81.43 ± 1.15% |
  | registered domain | 82.49 ± 1.93% | 72.82 ± 0.70% |

- **Decision:** `phishguard.urls.URL_VIEWS` defines the four inputs. Every later URL model (CNN, BiLSTM, hybrid) is reported on **raw URL** (for comparison with published work) and on **domain only** and **registered domain**. We call the latter two "shortcut-reduced", not "honest".
- **Why "shortcut-reduced":** Even the domain carries a collection difference. Only 6.2% of legitimate domains have a subdomain, a hosting platform, an IP, a digit or `xn--`, against 56.9% of phishing domains, and a no-learning rule on that gets 78%. On bare registered domains, the n-gram model catches only 53% of phishing (ROC-AUC 0.84).
- **Why n-grams win on the domain:** the signal is in the characters of the name (cheap TLDs such as `.cf`/`.ga`/`.top`, digits, hyphens, free-hosting names), which counts miss. That motivates the character-level deep models.
- **Limitations:**
  - Compromised legitimate sites (`smithtony.co.uk`) can't be recognised from the domain name by any URL-only model.
  - At a 1% phishing rate, the domain-only model's precision would be about 15%.
- **Comparison:** a host-name-only model in an independent audit (Yoyagm; `docs/references.md`) reports ROC-AUC 0.919 on PhiUSIIL and 0.937 on Tranco + later PhishTank URLs, close to our domain-only ROC-AUC (~0.92). That suggests the shortcut-reduced number is realistic.
- **Evidence:** `notebooks/07_url_views.ipynb`.

## D11 · Results are mean ± sd over 5 splits; URL parsing fixed; split re-made once
- **Context (senior review, `docs/review/senior_review_m1_m3.md`):**
  1. The first domain-only result (88.59%) came from a single split, which turned out to be the most favourable of six. Other splits gave ~85–87%.
  2. Differences such as 99.70% vs 99.68% were presented as a ranking, without uncertainty.
  3. `urls.hostnames()` used a regex that kept the `user@` part of URLs like `http://www.paypal.com:911@xn--80aaa….xn--p1ai/fix-account`, assigning them to the owner `paypal.com` (10 phishing URLs), and failed on URLs without `http://`.
- **Decision:**
  - **Parsing:** one `parse_host()` built on Python's `urlsplit`, with tests for `@`, ports, IPv6, missing scheme and uppercase. Because 10 owners changed, the seed-42 split changed. We re-made it **once, on purpose**: 164,045 / 35,029 / 35,130 rows, and `tests/test_split.py` now fingerprints it so it can't change silently again.
  - **Every model runs on 5 splits** (seeds 42, 0, 1, 2, 3) through `phishguard.experiment.run()`, re-tuned on each, and is reported as **mean ± sd**. Seed 42 is the reference split for examples, plots, a 95% confidence interval that resamples whole site owners, and McNemar tests between models.
  - **Metrics added:** PR-AUC, MCC, balanced accuracy, TPR at 1% and 0.1% FPR, Brier score, log-loss, and precision at realistic base rates.
  - **Saved with each result:** settings per seed, validation metrics, run time, library versions and git commit, plus seed-42 probabilities (`.npz`). The environment is pinned in `requirements.lock`.
  - **Features** (`features.py`): IP detection covers decimal/hex/IPv6; ports use the parser; suspicious words match whole words; `has_punycode` was added; counts and entropy ignore the `https://www.` prefix.
- **Effect:** URL features 99.68% → 99.63 ± 0.03%; domain-only n-gram 88.59% → 86.27 ± 0.89%. All-column models are unchanged (≈100%).
- **Evidence:** `src/phishguard/{urls,experiment,stats,evaluate}.py`, `tests/`, all notebooks re-run.

## D12 · Character-level deep models: char-CNN and BiLSTM (Milestone 4)
- **Context:** On every view, the char n-gram model beat XGBoost on our features (D10), so the signal is in the characters. Character-level networks learn their own character patterns.
- **Decision:** `phishguard.deep.CharURLClassifier` wraps a **char-CNN** (embedding 32 → convolutions of width 3/4/5/6 × 128 filters → max-pool → dense 128) and a **BiLSTM** (embedding 32 → 64 units per direction → max-pool → dense 64) as scikit-learn-style estimators, run through `experiment.run()`.
  - **Vocabulary:** built on training URLs only; lowercased; characters seen fewer than 10 times map to UNK; inputs cut at 200 characters.
  - **Padding can't leak:** the CNN zeroes windows that only cover padding, and the LSTM uses packed sequences. `tests/test_deep.py` checks that padding never changes a prediction.
  - **Training:** Adam, learning rate 0.001, batches of 256 similar-length URLs, binary cross-entropy.
  - **Early stopping:** on 10% of the *training* site owners (grouped, so no owner is on both sides), at most 15 epochs, patience 2. The validation split is used only for tuning (dropout ∈ {0.3, 0.5}) and the threshold, as for every model.
  - **Hardware:** Apple GPU (MPS). It isn't bit-for-bit deterministic, so a re-run can differ in the last decimals.
- **Results** (test accuracy, mean ± sd over 5 splits; the n-gram model from D10 for comparison):

  | View | Char n-gram LogReg | Char-CNN | BiLSTM |
  |---|---|---|---|
  | raw URL | 99.69 ± 0.05% | 99.68 ± 0.03% | 99.68 ± 0.04% |
  | domain only | 86.27 ± 0.89% | **86.58 ± 0.92%** | 85.55 ± 1.35% |
  | registered domain | **82.49 ± 1.93%** | 80.85 ± 1.53% | 80.95 ± 0.62% |

  - **Raw URL:** no difference (McNemar p = 0.15 / 0.47). Every model reads the prefix shortcut.
  - **Domain only:** the CNN's accuracy is higher on 4 of 5 splits (+0.31 points) and significant on seed 42 (1,074 vs 898, p = 0.0001). Its ROC-AUC is equal (0.9165 vs 0.9162, higher on only 3 of 5 splits). The gain comes from fewer false alarms at the threshold (3.3% vs 4.3%) at the same recall (~73%). We call it **a small gain at the threshold, not a better ranking**. The BiLSTM is lower than the n-gram model on all 5 splits.
  - **Registered domain:** the n-gram model is better (CNN lower on all 5 splits; both significant on seed 42). On ~14-character inputs, a regularised linear model over n-grams generalises better than 60–140k learned weights.
- **Consequence:** the **char-CNN** is the deep component for the hybrid model (Milestone 5): it is as good as or better than the BiLSTM everywhere and up to 3× faster. The n-gram model remains the baseline to beat.
- **Evidence:** `notebooks/08_deep_models.ipynb`, `results/m4_deep/`.

## D13 · Hybrid model: char-CNN + handcrafted features, trained jointly (Milestone 5)
- **Context:** The char-CNN reads character patterns. Our 40 features (D7) describe counts and structure. If they make different errors, combining them could help.
- **Decision:** two combinations, both through `experiment.run()` on 3 views × 5 splits:
  1. **Hybrid network** (`CharURLClassifier(arch="hybrid")`, `HybridNet`): the CNN's 512 pooled pattern scores and the 40 features are concatenated inside one network (features → 64 units), then dense 128 → logit, trained end to end. Features are `log1p`-scaled and standardised on training rows only; near-constant features (sd < 1e-6) are left unscaled. Dropout ∈ {0.3, 0.5} is tuned on validation.
  2. **Log-odds average** of a char-CNN and an XGBoost on the features (`LogitAverage`, late fusion). The CNN's dropout ∈ {0.3, 0.5} is tuned on validation; XGBoost uses `max_depth=6`, which it chose on every split and view in notebook 07.
  - Features are always computed **from the same view** the model reads, so the domain views can't see the prefix or path.
- **Results** (test accuracy, mean ± sd over 5 splits):

  | View | n-gram | Char-CNN | **Hybrid** | Average CNN + XGBoost |
  |---|---|---|---|---|
  | raw URL | 99.69 ± 0.05% | 99.68 ± 0.03% | 99.68 ± 0.03% | 99.68 ± 0.03% |
  | domain only | 86.27 ± 0.89% | 86.58 ± 0.92% | **86.72 ± 0.99%** | 85.70 ± 0.99% |
  | registered domain | **82.49 ± 1.93%** | 80.85 ± 1.53% | 82.01 ± 0.85% | 80.97 ± 1.80% |

  - **Domain only:** the hybrid beats the n-gram model on 4/5 splits (+0.45 points; seed 42 p < 0.0001; ROC-AUC higher on 3/5) and the CNN on 4/5 (+0.14; ROC-AUC 4/5; p = 0.027). This is a small but consistent gain, smaller than the split-to-split sd.
  - **Registered domain:** the hybrid beats the CNN on 5/5 splits (+1.16), but the n-gram model remains better (ROC-AUC higher on 5/5).
  - **Raw URL:** no differences (p ≥ 0.31).
  - **Averaging is worse than the CNN alone** on domain only (0/5 splits). An equal vote for the weaker XGBoost (81.4%) hurts, whereas joint training learns how much to trust each input.
  - **Permutation test (seed 42, domain only):** shuffling the features drops accuracy from 88.1% to 77.9%, and shuffling the URL text drops it to 65.4%. Both inputs are used, with the characters carrying more.
- **Consequence:** the **domain-only hybrid** is the model to explain (Milestone 6) and to test for robustness (Milestone 7). The n-gram model stays as the simple baseline and is the best model on registered domains.
- **Evidence:** `notebooks/09_hybrid.ipynb`, `results/m5_hybrid/`, `tests/test_deep.py`.

## D14 · Explanations: TreeSHAP for XGBoost, occlusion for the hybrid (Milestone 6)
- **Decision:** `phishguard.explain` provides three methods, all in log-odds:
  - `tree_shap`: exact SHAP values for XGBoost.
  - `char_occlusion`: hide one character (all-zero PAD input) and measure the change in the hybrid's log-odds.
  - `feature_occlusion`: set one feature to its training mean.
  - Plus permutation importance (ROC-AUC drop) for the hybrid's features, and `explain_url()` for the app.
- **Why occlusion, not SHAP, for the network:** exact SHAP is only cheap for trees. Occlusion asks a direct counterfactual question and can be verified with a deletion test.
- **Bug found and fixed:** the first version explained `logit(probability)`. For confident URLs the float32 sigmoid rounds to exactly 1.0, so every score came out as 0. Explanations now use the network's raw log-odds (`predict_encoded(..., logits=True)`), and a test covers it.
- **Environment:** on macOS, torch and xgboost load different OpenMP libraries, and both running threads in one process segfaults or hangs. `phishguard.deep` sets `OMP_NUM_THREADS=1` on macOS; `tests/conftest.py` imports it first. XGBoost is ~2× slower; GPU training is unaffected.
- **Checks:**
  - SHAP values add up to XGBoost's log-odds (max gap 2·10⁻⁵).
  - Deletion test on 2,000 caught phishing domains: hiding the top-5 characters flips 20% to legitimate, 5 random characters only 2.8%.
  - The hybrid's TLD scores agree with the n-gram weights (Spearman 0.89, 38 suffixes).
- **Findings:**
  - On the raw URL, the explanations show the shortcut: `is_https` and `has_www` dominate SHAP, and `https://www.` is blue character by character.
  - On domains, the hybrid's features branch is used for **structure** (`num_subdomains`, `on_hosting_platform`), and its characters branch for **TLDs and names** (free TLDs `.gq/.ml/.cf/.ga` → phishing; `.gov`, `.ac.uk`, `.edu` → legitimate).
- **Limitation found:** the domain-only model flags legitimate subdomains: `accounts.google.com` gets P = 0.999, mainly from `num_subdomains`, because only ~6% of legitimate training domains have any structure (D10). The app must state this, and Milestone 7 measures it.
- **Known elsewhere:** the same false alarm (`accounts.google.com/signin`) is reported for another URL model (valezion model card), and `login.microsoftonline.com` for a model whose legitimate data held only main domains (NclRsi46). See `docs/references.md`.
- **Artifact:** `models/hybrid_domain_only.pt` + `.json` (seed-42 train split, dropout 0.5, threshold 0.54, test accuracy 87.8%) for the Streamlit app.
- **Evidence:** `notebooks/10_explainability.ipynb`, `tests/test_explain.py`.

## D15 · Robustness: edited URLs and popular real websites, no retraining (Milestone 7)
- **Decision:** test the seed-42 models as they are (threshold from validation) against:
  1. the raw-URL attack from an independent audit (`http`→`https`; `https://www.<host>` with no path);
  2. six attacker edits to phishing **domains** (`phishguard.robustness.PHISHING_EDITS`; each applies only where realistic, e.g. no TLD change on a hosting platform; distinct domains, each counted once);
  3. common subdomains added to legitimate test domains;
  4. popular real websites not in PhiUSIIL: CrUX top-10k origins (with subdomains) and Tranco top-20k registered domains (`DataSets/external/README.md`). Every site whose owner appears in PhiUSIIL is excluded.
- **Results:**
  - **Raw URL:** HTTPS alone drops the raw-URL n-gram from 100% to 35.5% caught; `https://www.` + no path to 17.8%. Domain-only inputs are unchanged by these edits (checked: 100%).
  - **Domain edits** (caught before → after): drop subdomain, hybrid 99% → 54%; leave hosting platform, 100% → 72%; TLD → `.com`, 74% → 51%. All edits together: hybrid 86% → 20%, n-gram 83% → 32%, XGBoost 77% → 4%.
  - **Legitimate + subdomain:** the hybrid's false-alarm rate rises from 3.3% to 74–95%; the n-gram's to 21–73%.
  - **Real websites flagged as phishing:**

    | | Hybrid | n-gram |
    |---|---|---|
    | CrUX bare domains | 15% | 22% |
    | CrUX with a subdomain | 81% | 52% |
    | Tranco top domains | 19% | 26% |

    For comparison, PhiUSIIL's own test domains: 3.3% (hybrid) and 4.3% (n-gram).
  - **Country priors:** popular `.ru`, `.ir` and `.in` sites are flagged 33–37% of the time, matching their phishing share in PhiUSIIL's training split (79%, 78%, 50%). Popular `.org`/`.de` sites: 0%.
- **Interpretation:** the domain-only models resist the prefix shortcut but learned other collection artefacts (no subdomains on the legitimate side, country mix of the phishing feed). Against edits and subdomains the n-gram model is the most robust, though on popular bare domains it raises more false alarms than the hybrid (22% vs 15%). The hybrid's test-split advantage partly rests on the subdomain/hosting features that these edits break. The remedy is legitimate data that looks like the real web, not a larger model.
- **Limitation of the test:** "own domain" is decided with the Public Suffix List. Shared gateways that aren't on it (e.g. `ipfs.io`) are treated as attacker-owned, which makes some edits (`ipfs.io → ipfs.com`) unrealistic. Each such domain counts once among ~11,600.
- **Evidence:** `notebooks/11_robustness.ipynb`, `src/phishguard/robustness.py`, `tests/test_robustness.py`. It agrees with the independent audits in `docs/references.md`.

## Reporting protocol from Milestone 4 on (fixed in advance)
To avoid choosing what to report after seeing test results, every new model reports exactly this, and nothing is dropped if it looks bad:
- **raw URL, domain only, registered domain** views (D10)
- each as **mean ± sd over seeds 42, 0, 1, 2, 3** of accuracy, F1, ROC-AUC, PR-AUC, MCC and TPR at 1% FPR
- the seed-42 95% CI, and a McNemar test against the char n-gram model on the same view
