# Senior review: PhishGuard, Milestones 1–3

**Scope:** everything in the repo up to M3: `README.md`, `docs/`, `src/phishguard/`, `tests/`, notebooks 01, 01b and 02–07, and `results/**/*.json`.
**Method:** I read the code and the saved notebook outputs, then re-ran the parts I doubted. Every number marked *verified* below comes from a check I ran on the current code and data with the repo's `.venv` (Python 3.12.12, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, tldextract 5.4.0). Each check took under a minute. The appendix shows how to repeat the important ones.
**What I changed:** nothing except this file.

> **How to read this.** Section 1 is the summary. Section 2 lists every problem, worst first. Section 3 is your to-do list before M4. Section 4 is viva practice: try to answer each question *before* opening the model answer. Section 5 is advice for the remaining milestones.

### Terms used in this review
| Term | Meaning |
|---|---|
| **Confidence interval (CI)** | A range that probably contains the true value. "88.6% [87.4, 89.8]" means that, given the sampling noise, the real accuracy could plausibly be anywhere from 87.4% to 89.8%. |
| **Wilson interval** | A standard CI for a proportion such as accuracy. It assumes every test URL is independent of the others. |
| **Cluster (owner) bootstrap** | Resample whole site owners with replacement, recompute accuracy, and repeat 2,000 times. URLs from one owner are similar to each other, not independent, so this CI is wider and more honest than Wilson. |
| **Split-seed variance** | Re-make the train/val/test split with a different random seed, retrain and retest. The spread across seeds shows how much the score depends on *which* owners happened to land in test. |
| **McNemar test** | Compares two models on the *same* test URLs. It looks only at the URLs where they disagree (A right and B wrong, or the reverse). If those two counts are similar, the models aren't really different. |
| **PR-AUC** | Area under the precision–recall curve. Unlike ROC-AUC, it reacts to how rare the positive class is. |
| **MCC** | Matthews correlation coefficient: one number from −1 to 1 that uses all four cells of the confusion matrix. It stays honest when the classes are imbalanced. |
| **TPR at fixed FPR** | "What share of phishing do we catch if we allow only 1% (or 0.1%) of legitimate URLs to be wrongly flagged?" This is the number a real deployment cares about. |
| **Calibration** | Whether a predicted 0.8 really means "phishing 80% of the time". Measured with the **Brier score** (mean squared error of the probabilities) and **ECE** (expected calibration error: the average gap between predicted and actual frequency). |
| **Base rate / prevalence** | How common phishing is in the data the model will actually see. It's 42% in PhiUSIIL but far lower in real browsing. |

---

## 1. Verdict

**This is a strong student project.** The engineering discipline is unusually good for a course project, and M3's shortcut analysis is the kind of finding most published PhiUSIIL papers miss. Specifically:

- **One data entry point that fails loudly.** `load_raw()` rejects a truncated file (`data.py:26`). That bug really happened to the team, and the code now catches it.
- **A grouped split, justified with evidence.** Notebook 01b compares three ways of grouping (hostname, registered domain, site owner) and measures the leak a random split would cause (39.4% of phishing test URLs have a same-owner sibling in train). I checked: no owner, and no domain string, appears in two splits.
- **One protocol for every model, with no leakage.** `tune()`/`fit_and_test()` tune on validation, choose the threshold on validation and touch test once. TF-IDF sits inside the pipeline, so it is fitted on train only (`modeling.py:25,37`). For the domain-only model, the threshold chosen on validation (0.56) is exactly the best threshold on test, so threshold choice isn't inflating anything.
- **An honest shortcut audit.** The one-line rules, the "URLLength minus counted characters = 12" trick and the URL-views experiment are excellent, and they reproduce exactly. I re-ran the domain-only n-gram model and got 0.885938, the same as the JSON.
- **Good documentation habits:** a decision log, plain-language guides and tests (27 pass in 8 s).

**Top 3 risks to your marks**

1. **The headline "honest" number (88.59%) comes from one lucky split.** With five other split seeds, the same model scores 85.0–87.1%. Seed 42 was the **best of six**: mean **86.3% ± 1.3** (SD). Every doc quotes 88.59% to two decimals with no error bar. A professor who asks "how stable is that?" will catch it. (Findings C1, M1.)
2. **"Honest" is too strong a word for the domain-only view.** It still contains a data-collection shortcut. Legitimate PhiUSIIL hosts almost never have a subdomain (2.8%) or sit on a hosting platform (0.08%), while 33% and 26.5% of phishing hosts do. A rule with no learned weights gets **79.4%** on the domain-only test set. And because a few phishing domains repeat (`ipfs.io` alone is 1,115 test rows, 7.5% of test phishing), row-weighted recall is inflated. (Findings M2, M3.)
3. **What the professor sees in VS Code contradicts the good work.** `notebooks/ML_EDA.ipynb` and `procssesed_dataset/` still run on the **truncated** 195k/196k-row Colab file that D1 exists to prevent. The README calls the project "PhishResist", marks CNN/BiLSTM/hybrid/SHAP as "✅ Current" and says to run notebooks 01→05. There are also smaller doc inaccuracies a careful examiner could find. (Findings M8, m6.)

There is also an important latent bug for M7 and M8. `urls.hostnames()` mis-parses the `user@host` phishing trick, IPv6 hosts and scheme-less input, and the last of these **crashes** `site_owner()` (Finding M4). Fix it before building the app and the robustness test on top of it.

---

## 2. Findings

Severity scale:
- **Critical:** changes a headline claim.
- **Major:** a viva or lab-check risk, or something that will break a later milestone.
- **Minor:** worth fixing, low risk.
- **Polish:** cosmetic.

Effort: **S** < 1 h, **M** 1–3 h, **L** half a day or more.

### 2a. Verified findings

| # | Severity | Area | Finding | Evidence | Fix | Effort |
|---|---|---|---|---|---|---|
| C1 | **critical** | Statistics / headline | The "honest" domain-only accuracy of 88.59% is a single-split number, and that split happens to be the most favourable of the six I tried. | **Check:** re-split with seeds 42, 0, 1, 2, 3, 7 (same `get_splits`, n-gram LR, C=3, threshold on val). Test accuracy: 88.59, 84.97, 85.64, 85.71, 87.06, 86.00. **Mean 86.33%, SD 1.30, range 85.0–88.6.** ROC-AUC: 0.931 at seed 42 vs 0.906–0.919 at the others. Quoted without error bars in `README.md:34`, `ROADMAP.md:21`, `DECISIONS.md:85`, `guide/03_shortcuts.md:66–68`, `guide/README.md:18`. | Add `evaluate_over_seeds()` (5 split seeds; re-tune C per seed). Report **mean ± SD** as the headline and keep seed 42 as "the reference split". Redo the views chart with error bars. Log as D11. | M |
| M1 | major | Statistics | No uncertainty is reported anywhere, and the docs rank models whose differences are noise: 99.707 vs 99.695 vs 99.681 vs 99.675, and 100% vs 99.997% vs 99.994%. | **Raw URL, n-gram vs XGBoost, paired McNemar:** 33 URLs only the n-gram model gets right vs 26 only XGBoost gets right, exact p = **0.435**. Owner-bootstrap 95% CI for the accuracy difference: **[−0.023, +0.067] pp**, which includes 0. Wilson CIs: n-gram 99.70% [99.63, 99.75]; XGBoost 99.68% [99.61, 99.73]. With 0 errors on 35,130 URLs, the 95% lower bound is about **99.991%** (the "rule of three": 3/35,130), so 0, 1 and 2 errors are indistinguishable. Domain-only: Wilson [88.26, 88.92], owner bootstrap **[87.44, 89.79]**. The averaging model's "improvement" is 4 errors (`guide/02_models.md:100–101`). | Add a 95% CI column (owner bootstrap) to every results table. Add a `mcnemar(a, b)` helper. Rewrite "Averaging helped" as "Averaging was not significantly different". Say "all three all-column models are indistinguishable from perfect". | M |
| M2 | major | Domain-only view | The domain-only view still carries a collection shortcut: the legitimate list is popular registered domains, so "has a subdomain" or "is on a hosting platform" almost means phishing. | **Domain view, share with a subdomain left after stripping `www.`:** legitimate 2.8%, phishing 33.2%. **On a PSL-private hosting platform:** 0.08% vs 26.5%. **Rule** "subdomain or platform or IP or digit or `xn--` → phishing": **79.4%** test accuracy (recall 60%, FPR 6.2%), no learning at all. The model scores **95.7%** on the 8,897 test rows with such structure (91.7% of which are phishing) but **86.2%** on the 26,233 bare registered domains, where 74.3% are legitimate. **Registered-domain-only view** (eTLD+1, subdomains removed): **84.6%**, AUC 0.907. | Rename it "shortcut-reduced", not "honest". Add a 4th view, `registered domain` (eTLD+1), and report accuracy separately for bare domains vs subdomain/platform hosts. Add a sentence to D10 and to guide 03. | M |
| M3 | major | Domain-only view / metrics | The test set has heavy repeats. Metrics count each URL once, so a few phishing domains dominate. | **Test phishing:** 14,907 rows but only **11,256 unique domains**. `ipfs.io` ×1,115, `mail.deliverylifesupport.com` ×370, `gateway.pinata.cloud` ×232. All of those are predicted correctly. Top-5 owners = **23.9%** of test phishing (`ipfs.io` owner alone 1,441 rows). Phishing recall: **78.0% row-weighted vs 74.1% over unique domains**. No domain string is shared between splits (0 train∩test, 0 train∩val), so this isn't leakage, but it shrinks the effective sample size. | Report domain-only metrics both row-weighted and **deduplicated by domain**, and use owner-cluster CIs (M1). State the top-owner shares in 01b. | S |
| M4 | major | Code: URL parsing | `urls.hostnames()` uses a regex that **keeps the `user@` part**, breaks on IPv6, and returns NaN for scheme-less input. NaN then **crashes** `site_owner()`. `features.py` parses the same URLs correctly with `urlsplit`, so the two modules disagree. | `urls.py:11` regex `([^/:?#]+)` doesn't stop at `@`. **10 phishing URLs** in the data get the wrong host or owner, e.g. `http://www.paypal.com:911@xn--80aaa…xn--p1ai/fix-account` → owner **`paypal.com`** (the real host is `xn--80aaa….xn--p1ai`). `host_only("http://paypal.com@evil.top/login")` = `paypal.com@evil.top`. `host_only("http://[2001:db8::1]/login")` = `[2001`. `host_only("www.example.com:8080/a")` = NaN. `site_owner(pd.Series(["www.example.com/a"]))` → `AttributeError: 'float' object has no attribute 'find'`. | One `parse_host(url)` in `urls.py`: prepend `//` if there's no scheme, use `urlsplit(...).hostname`, and handle `ValueError` like `features._split`. Use it in `hostnames`, `host_only` and `site_owner`. Add tests for `@`, IPv6, scheme-less and uppercase input. **Warning:** this changes 10 owners, so the shuffled owner order and therefore the **whole split** change. First freeze the current split to disk (see §3), then fix, re-split once on purpose, re-run 01b–07 (a few minutes) and log it as a decision. | M |
| M5 | major | Metrics / base rate | `evaluate()` (`evaluate.py:21–39`) reports accuracy, precision, recall, F1 and ROC-AUC only: no PR-AUC, MCC, TPR at low FPR or calibration. Precision at a realistic phishing rate is never discussed. | **Domain-only model (test):** PR-AUC 0.935, MCC 0.770. TPR at FPR ≤ 0.1% / 1% / 5%: **50.2% / 70.1% / 79.4%**. Brier 0.091, ECE 0.031 (well calibrated; slightly under-confident at the top: the 0.9–1.0 bin averages p = 0.975 but is 99.5% phishing). FPR at the chosen threshold = 734/20,223 = **3.6%**. If 1% of URLs were phishing, precision would be **17.8%**; at 0.1%, **2.1%**. **Raw n-gram:** PR-AUC 0.9989, MCC 0.9938, TPR at 0.1% FPR 0.994. | Add `pr_auc`, `mcc`, `balanced_accuracy`, `tpr_at_fpr_1pct`, `tpr_at_fpr_0_1pct`, `brier` and `log_loss` to `evaluate()`. Add one paragraph to guide 02 on base rate, with the 17.8% example. | S |
| M6 | major | Reproducibility | Results can't be re-analysed without retraining, and the environment isn't pinned. | `save_result` (`evaluate.py:55–61`) stores only test metrics and the threshold. There are no hyperparameters (e.g. which `C` won in notebook 07 isn't recorded), no seed, no validation metrics, no probabilities, no library versions, no git commit and no run time. `pyproject.toml:6–14` pins only tldextract; the venv has pandas **3.0.6**, whose string dtype changes outputs (`dtype: str` in notebook 03 cell 15). A teammate on pandas 2.x would see different outputs. | `save_result(..., params=best, seed=..., val_metrics=..., runtime_s=...)` and add versions plus `git rev-parse HEAD` automatically. Save val/test probabilities to `results/<group>/<slug>.npz`; every later significance test, ensemble and M7 comparison needs them. Commit a lock file (`uv pip compile pyproject.toml -o requirements.lock`) and state `requires-python = "==3.12.*"`. | M |
| M7 | major | Data / latent leak | The dataset's own URL columns were computed on the **raw** URL, so the trailing-slash shortcut D3 removed is still recoverable from them. Mixing provided columns with features computed on the cleaned URL brings the leak back. | `URLLength − len(cleaned URL)`: legitimate **always −1** (134,849/134,849); phishing −1 for 38,707, but 0 for 37,800 and +1 for 22,842. So "≠ −1 → phishing" has **100% precision and 61% recall**. Also, `URLLength = len(raw) − 1` for 79% of rows: the column is off by one in an inconsistent way. This probably helps explain why the dataset's 19 URL columns (99.75%) beat our 39 features (99.68%). D3 (`DECISIONS.md:22`) only says "may be off by one". | Update D3 and D9 to say so. **Rule for M5:** the hybrid uses only our own features, never provided columns. Add a test asserting `phishguard.features` never reads dataset columns. | S |
| M8 | major | Repo hygiene / docs | Legacy files and the README contradict the project's own decisions. | `notebooks/ML_EDA.ipynb` cell 0 uses `google.colab.files.upload()` and **195,340 rows**. `procssesed_dataset/ML_PhishGuard_proceesed_dataset.ipynb` loads **196,403 rows**, the truncated file from D1. Both are tracked in git. `README.md:1–2` has two names (PhishGuard / PhishResist). `README.md:21` says "run them in order (01 → 05)" (01b, 06 and 07 exist). `README.md:638–650` marks CNN, BiLSTM, hybrid, SHAP and robustness as "✅ Current". `README.md:224` is a lone `*`. `notebooks/readme.md` is empty (1 byte). The README metrics list (`README.md:471–480`) doesn't match what's reported. | Move the legacy notebooks to `archive/` with a README that says "superseded by 01/01b; uses the truncated file, see D1", or delete them. Pick one name. Put a short status table at the top of the README (✅ done / ⏳ planned), fix the run order, remove the stray `*`, and write a one-paragraph `notebooks/readme.md`. | S |
| m1 | minor | Threshold protocol | `best_threshold` picks the **lowest** threshold among ties (`np.argmin`, `evaluate.py:52`). With flat error curves that gives extreme thresholds. The "0.16 is better than 0.5" story (D6, `guide/02_models.md:92`) is within noise. | XGBoost all-columns: 64 thresholds (0.02–0.65) tie at 1 validation error, and 0.02 was chosen. Raw n-gram: 13 thresholds within 3 validation errors of the best. Test accuracy at 0.5 is 99.678% vs 99.695% at 0.16 (6 URLs). **No leakage:** for the domain-only model, the validation-chosen 0.56 equals the best threshold on test, and the validation plateau is 0.51–0.59. | Break ties by taking the middle of the tied range (or the one closest to 0.5). Rephrase D6: "we choose it on validation; here it changed only a handful of test predictions". For M7 and M8, a stable threshold matters more than one extra correct validation URL. | S |
| m2 | minor | Hyperparameter search | The grids are tiny (2–6 settings), several hyperparameters are fixed, and the winner is often decided by **one** validation URL. | XGBoost URL features (03 cell 11): depth 10 / 800 trees won with 0.997324 vs 0.997296 (1 URL) while having the **lowest** val AUC of the top settings. `learning_rate` is fixed at 0.1, with no `subsample`/`colsample_bytree`/`min_child_weight` and no early stopping. Raw n-gram best `C=10` sits at the **edge** of `[3, 10]` (04 cell 3). Notebook 07 re-tunes C for n-grams but **not** XGBoost. **Check:** re-tuning XGBoost on the domain view (depth 4/6/10 × min_child_weight 1/10, lr 0.05, subsample 0.8) → 81.6% vs 81.4%, so the "n-grams beat features" conclusion holds. Domain-only C: 1 → 88.28, **3 → 88.58**, 10 → 88.49, 30 → 87.98, 100 → 87.05 val accuracy, so 3 is a genuine interior optimum. | Select on validation **log-loss or ROC-AUC** (smooth) instead of accuracy (step-like), or use the "1-SE rule". Widen grids when the winner is on the edge. Use XGBoost early stopping on val. Re-tune every model per view and say so. | M |
| m3 | minor | Features | Edge-case bugs and false positives in `features.py`. | `has_ip` (`features.py:67`) misses decimal (`http://3232235521/`), hex (`0x7f.0x0.0x0.0x1`) and IPv6. `has_port` (`:68`) = 1 for `http://[2001:db8::1]/login`. `suspicious_words` (`:75`) uses substring matching, so `freedomhouse.org` → "free" and `supportgroup.org` → "support". There's no punycode (`xn--`) feature, which M7's homoglyph test needs. `url_entropy` and `url_length` include the scheme and `www.`, so they encode the prefix shortcut. | Fix `has_ip` with `ipaddress.ip_address` after decoding decimal/hex. Make `has_port` use `parts.port`. Match words on token boundaries (`_TOKEN.findall`). Add `has_punycode` and `brand_in_subdomain`. Compute the length/entropy features on `strip_prefix(url)`. Re-run 03/07 afterwards; numbers will move slightly. | M |
| m4 | minor | Averaging model | The model averages **probabilities** from two differently calibrated models (thresholds 0.16 and 0.70). | `04` cell 9; JSON thresholds 0.16 vs 0.70; the gain is 4 errors (see M1). | Average logits, or calibrated probabilities, or rank-average. For M5, train any stacker on **out-of-fold** predictions from train, never on val (val is already used for the threshold). | S |
| m5 | minor | Published comparisons | Results from different splits are compared as if they were equivalent, and citations are incomplete. | `05` cell 4: "matches or beats the published results"; "The Frontiers paper also used a host-disjoint split, like ours" (no citation or page). "Frontiers in Computer Science, 2026" has no authors or title anywhere (`DECISIONS.md:64`, `05` cell 3). | Create `docs/references.md` with full citations and **each paper's split protocol** (random / stratified / host-disjoint / unknown). Reword to "comparable to; not directly comparable because the splits differ". | S |
| m6 | minor | Doc accuracy | Small factual slips a careful examiner could find. | `guide/03_shortcuts.md:41` and `DECISIONS.md:74` say legitimate leftover is "always/exactly 12". It's 12 for all but **4** of 134,849 (max 16, e.g. `https://www.dzwww.com`). `guide/02_models.md:103` "validation score was 99.78%" doesn't appear in any notebook output (the n-gram val was 99.77%; the averaged model's val accuracy is never printed). `guide/03_shortcuts.md:35` and 06 cell 15 state "phishing pages were tiny or already taken down" as fact; it's a hypothesis. `guide/02_models.md:58–63` snippet omits `min_df=2`, `solver="liblinear"` and the tuned `C`. | Correct each one. Mark hypotheses as "likely, not tested" or test them (e.g. `LineOfCode` distribution by class). | S |
| m7 | minor | Tests | Coverage gaps where the bugs above live. | `tests/test_urls.py:31–33` uses `return` instead of `pytest.skip`, so it silently passes without data. No tests for `@`/IPv6/scheme-less hosts (M4) or `best_threshold` ties (m1). `test_tune_returns_a_setting_from_the_grid` only checks `best in grid`, not that it is the *best*. Nothing proves the TF-IDF is fitted on train only. Nothing checks that numbers in the docs match `results/*.json`. | Add those tests. Leakage test: wrap a vectorizer that records the rows it was fitted on, run `fit_and_test` and assert the rows equal `train_idx`. Add a `tests/test_docs_numbers.py` that greps the headline numbers in README/ROADMAP against the JSON. | M |
| m8 | minor | Split design (undocumented) | Some owners and domain strings carry both labels. This is fine, but undocumented, and it sets an error floor. | **105 owners (560 rows)** have both labels and are assigned by majority class (`split.py:37`), e.g. `pinata.cloud` (233 rows, 99.6% phishing), `washington.edu` (12 rows, 8% phishing), `adobe.com`, `office.com`. **57 domain strings** have both labels (42 train, 6 val, 9 test = 18 test rows): a domain-only model *must* get some of them wrong. | One paragraph in D5 and guide 01. These are mostly compromised or abused sub-pages of real sites, which is a good viva talking point. | S |
| m9 | minor | Protocol hygiene | Some "test once" decisions were influenced by the whole dataset or by test results. | `06` cell 5 computes single-column AUCs on **all** rows, including test. The one-line rules were found in `01` §5 on all rows, then scored on test. D10 picked "domain only" as the headline after seeing test scores for all views. The same test set has now been scored for about 20 variants (m2, m3_columns, m3_views). | Say so honestly in the report ("exploratory, on the full data"). From M4 on, **pre-register** what will be reported (raw + domain-only + registered-domain, mean over seeds) before running. Multi-seed evaluation (C1) also reduces reliance on one test set. | S |
| p1 | polish | Notebooks | Feature-view XGBoost silently trains on constant columns. | 15 of 39 features are constant in the domain view (`is_https`, `path_length`, `num_slashes`, …). | One sentence in 07 that the domain-only XGBoost effectively uses 24 features. | S |
| p2 | polish | Notebook 05 | The main table is a pandas `Styler`; its saved `text/plain` output is just `<Styler at 0x…>`. | `05` cell 1 output. | Also `print(table.to_string())` so the table appears in any viewer and in diffs. | S |
| p3 | polish | Naming | Misspelled folder `procssesed_dataset/…proceesed…`; mixed `DataSets/`. | repo tree | Handled by M8's archive move. | S |

### 2b. Suspicions I could not confirm

| # | Suspicion | Why I think so | How to check |
|---|---|---|---|
| S1 | `TLDLegitimateProb` and `URLCharProb` may have been computed from PhiUSIIL's own labels over the whole dataset, a form of target encoding that leaks test labels into the all-column models. | The 3 statistic columns alone give 99.69% (`m3_columns`). | Read the PhiUSIIL paper's definition of these columns. Recompute the TLD probability on train only and compare. |
| S2 | The Frontiers 2026 paper used a host-disjoint split. | Stated in `05` cell 4 without a quote. | Find the sentence in the paper and cite it with a page number, or remove the claim. |
| S3 | The seed spread in C1 is slightly overstated or understated. | I fixed C=3 for the other seeds instead of re-tuning. | `evaluate_over_seeds()` with per-seed tuning (C1 fix). |
| S4 | Model-seed variance (RF/XGBoost `random_state`) is tiny compared with split variance. | Tree ensembles with 300–800 trees are usually stable. | 3 model seeds × the reference split, report the SD. |
| S5 | "Missed phishing like `cama.it` are compromised legitimate sites." | Plausible, and `m8` shows mixed-label owners exist. | Spot-check 20 missed domains against an archive or a blocklist and report the share. |

### 2c. Things I checked that are fine (cite these in the viva)

- **Split integrity:** 0 owners and 0 domain strings shared between any two splits. All 234,204 rows are assigned exactly once. Class share is 42.42/42.43/42.43%.
- **No TF-IDF leakage:** the vectorizer is inside the pipeline and fitted in `tune()`/`fit_and_test()` on `train_idx` only.
- **No threshold leakage:** the threshold is chosen on validation. For the domain-only model it equals the best threshold on test.
- **Numbers match:** README, ROADMAP, D8–D10 and the guide tables match the JSON files to the stated precision (e.g. 99.707, 99.695, 99.681, 99.675; error counts 103/107/112/114; views 99.70/90.89/88.59 and 99.68/87.97/81.42). The only exceptions are listed in m6.
- **Reproducible:** I re-ran the raw n-gram model (99.6954%, 107 errors, threshold 0.16), raw XGBoost (99.6755%, 114 errors, threshold 0.70) and domain-only n-gram (88.5938%, threshold 0.56). All are bit-for-bit equal to the JSON.
- **Cleaning:** no case-insensitive duplicate URLs remain. Every URL has an `http(s)://` scheme, so no host is NaN *in this dataset* (M4 bites only on new input).
- **Imbalance:** 58/42 is mild. No resampling is needed, and none was done. That's correct.
- **Run time:** every model trains in 4–15 s on the M2 (domain n-gram ≈ 6 s, raw n-gram ≈ 8 s, XGBoost ≈ 4 s, feature extraction ≈ 4 s). **Multi-seed evaluation is cheap**, so there's no reason to skip it.

---

## 3. Before Milestone 4: checklist (in priority order)

1. **Freeze the reference split.** Save `URL → split` for seed 42 to `results/splits/seed42.csv` and add a test that `get_splits(df)` still reproduces it. Then any later change to parsing is a *visible* decision, not a silent reshuffle.
2. **Fix URL parsing (M4)** with one `parse_host()`, and add tests for `@`, IPv6, scheme-less and uppercase input. Re-split once on purpose, re-run 01b–07 and update the numbers. Log it as **D11**.
3. **Multi-seed evaluation (C1).** Add `phishguard.modeling.evaluate_over_seeds(make_model, grid, X, y, df, seeds=(42, 0, 1, 2, 3))`. It returns per-seed metrics plus mean ± SD and re-tunes per seed. Re-run notebook 07 with it. The headline becomes "domain-only n-gram: 86.3 ± 1.3% (5 splits)" (or whatever the re-tuned run gives).
4. **Uncertainty helpers (M1, M3):** `owner_bootstrap_ci(correct, owners)`, `mcnemar(correct_a, correct_b)`, and a deduplicated-by-domain metric. Put CI columns in notebook 05 and 07 tables.
5. **Extend `evaluate()` (M5)** with PR-AUC, MCC, balanced accuracy, TPR at FPR 1% / 0.1%, Brier and log-loss. Add a base-rate paragraph to guide 02.
6. **Richer `save_result()` (M6):** params, seed, val metrics, run time, versions, git SHA, plus an `.npz` of val/test probabilities. Commit a lock file.
7. **Threshold ties and selection criterion (m1, m2):** take the middle of tied thresholds, and select hyperparameters on val log-loss or ROC-AUC.
8. **Add the `registered domain` view (M2)** to `URL_VIEWS`, reword "honest" to "shortcut-reduced" everywhere, and add the bare-vs-structured breakdown.
9. **Repo hygiene (M8):** archive the legacy notebooks, fix the README (name, status, run order, stray bullet), write `notebooks/readme.md`.
10. **Docs corrections (m5, m6, m7-M7, m8):** `docs/references.md` with each paper's split, "12 for all but 4", remove "99.78%", soften the "taken down" claim, the URLLength residue in D3, mixed-label owners in D5.
11. **Feature fixes (m3)**, then re-run 03, 04 and 07.
12. **Tests (m7):** leakage test, threshold-tie test, docs-vs-JSON test, `pytest.skip` fix.

Items 1–6 take roughly a day for one person and should be done before any PyTorch code. M4–M7 will all reuse these helpers.

---

## 4. Cross-questions

Answer out loud first, then open the model answer. Questions get harder as you go down. The "what would you do if…" questions have no single right answer; the model answer shows one good line of reasoning.

### Manan: data, cleaning, split, evaluation protocol, overall design

**Q1. Walk me through `load_clean()`. Why does `load_raw()` demand the shape be *exactly* (235,795, 55)? Isn't that brittle?**
<details><summary>Model answer</summary>

`load_clean()` = `clean(add_target(load_raw()))` (`data.py:79–82`).

1. `load_raw` reads the zip and checks the exact shape, no missing values and labels ∈ {0, 1}.
2. `add_target` creates `is_phishing = (label == 0)` and drops `label`.
3. `clean` strips whitespace and trailing `/`, drops every copy of a URL with conflicting labels (2 rows), then drops duplicates (1,589 rows), leaving 234,204.

The exact-shape check is *deliberately* brittle. We were burned once: a Colab upload silently gave 196,403 rows and nobody noticed (D1). If the authors released a new version, the code should stop. We would then update `EXPECTED_SHAPE` on purpose, record it in DECISIONS, and re-run everything. Failing loudly is the point.
</details>

**Q2. You strip the trailing `/` because it leaks the label. Does that remove the leak from *every* model you trained?**
<details><summary>Model answer</summary>

No. It removes the leak from anything computed on the cleaned `URL` string: our 39 features, the n-gram models and the URL views. The dataset's own columns, like `URLLength`, were computed on the **raw** URL. Comparing `URLLength` with the length of the cleaned URL gives −1 for *every* legitimate URL, but 0 or +1 for 61% of phishing URLs. So if anyone combined provided columns with features from the cleaned URL, a model could rebuild the shortcut with 100% precision. That's one more reason the all-column results are only a reproduction, and why the M5 hybrid will use only our own features.
</details>

**Q3. Explain "site owner". Why option C rather than B (plain registered domain)? What's still wrong with C?**
<details><summary>Model answer</summary>

The site owner is the registered domain (the "eTLD+1", e.g. `evil.co.uk`). Customers of hosting platforms listed in the *private* section of the Public Suffix List count separately (`foo.web.app` ≠ `bar.web.app`). It's computed with `tldextract` 5.4.0, pinned because its bundled list defines the groups.

B would treat all of `web.app` (5,699 rows) as one owner, so whole platforms would vanish from training. That tests "unseen platforms", a different and harder question. C only stops one attacker's subdomains from being split across train and test.

What's still wrong:
- Platforms missing from the PSL act as one giant owner. `weeblysite.com` (3,079 rows) went wholly to **val**, and `ipfs.io` (1,441 rows) wholly to **test**, which is 9.7% of test phishing. The top 5 test owners make up 23.9% of test phishing, so the test set is "lumpy". That's exactly why the score swings between split seeds.
- 105 owners have both labels and are assigned by majority.
- `hostnames()` mis-parses `user@host` URLs (10 rows).
</details>

**Q4. Your test set has 35,130 URLs. How confident are you in "88.59%"?**
<details><summary>Model answer</summary>

There are three layers of uncertainty:

1. **Sampling noise, assuming independent URLs:** the Wilson 95% CI is [88.26, 88.92].
2. **URLs aren't independent.** URLs from one owner rise or fall together (e.g. 1,115 identical `ipfs.io` rows). An owner-cluster bootstrap gives [87.44, 89.79].
3. **The split itself is random.** Re-splitting with five other seeds gave 85.0–87.1%. Seed 42 was the best of six; the mean is 86.3% ± 1.3.

So the honest statement is "about 86%, ± 1–2 points, depending on which owners land in test". The second decimal place in 88.59 carries no information.
</details>

**Q5. You say "test once". Is there any way test information still influenced what you report?**
<details><summary>Model answer</summary>

Hyperparameters and thresholds are chosen on validation only, and the TF-IDF is fitted on train only. That part is clean, and for the domain-only model the validation threshold even equals the test-optimal one.

But there are softer channels:
- The shortcut rules and single-column AUCs were explored on the *full* dataset, including test.
- D10 chose "domain only" as the headline *after* seeing every view's test score.
- The same test set has been scored for about 20 model variants.

None of this tunes a parameter on test, but it is "researcher degrees of freedom". To control it, from M4 we fix in advance what we'll report (raw, domain-only and registered-domain, mean over 5 split seeds) and make model choices on validation only.
</details>

**Q6. Look at the threshold chosen for XGBoost on all columns: 0.02. Is that a sensible threshold? Why did the code pick it?**
<details><summary>Model answer</summary>

On validation, 64 thresholds (0.02–0.65) all give the minimum of 1 error. `best_threshold` uses `np.argmin`, which returns the *first*, i.e. lowest, tied candidate (`evaluate.py:52`). So 0.02 is an artefact of tie-breaking, not a meaningful choice. A threshold at the edge of a plateau is fragile: a tiny shift in the probabilities on new data flips predictions. Fix: take the middle of the tied range (or the tied value closest to 0.5), or pick the threshold by a cost (e.g. "FPR ≤ 1%").
</details>

**Q7. What would you do if the professor said: "In real browsing maybe 1 in 100 URLs is phishing. What's your domain-only model's precision then?"**
<details><summary>Model answer</summary>

At our threshold the domain-only model has TPR ≈ 78% and FPR ≈ 3.6% (734 of 20,223 legitimate test URLs). At 1% prevalence:

precision = 0.78 × 0.01 / (0.78 × 0.01 + 0.036 × 0.99) ≈ **18%**

So about 4 of every 5 alarms would be false. At 0.1% prevalence, precision is about 2%. Accuracy and F1 on a 42%-phishing test set hide this. That's why we should report **TPR at a fixed low FPR**: at FPR ≤ 1% we still catch 70% of phishing, and at FPR ≤ 0.1%, 50%. We should also report PR-AUC, and set the app's threshold for a target false-alarm rate, not for minimum errors on a 42% set. Resampling or class weights don't fix this, because it's about the deployment base rate, not training balance.
</details>

**Q8. What would you do if you found a bug in `hostnames()` that changes the owner of just 10 URLs, the week before the viva?**
<details><summary>Model answer</summary>

(That bug exists; see M4.) Even 10 changed owners change the *set* of groups. `get_splits` sorts the groups and applies one random permutation, so the whole order shifts and almost every owner can move to a different split. Every number would change slightly.

The right process:
1. Save the current split to disk so old results stay reproducible.
2. Fix the parser, with tests.
3. Re-split deliberately and re-run all notebooks (a few minutes total).
4. Check the changes fall within the seed-to-seed spread we already measured.
5. Log it as a decision with old vs new numbers.

What we must *not* do is quietly keep the buggy parser because the numbers look nice, or patch only the app and leave training inconsistent with it.
</details>

### Shaurya: classical models, tuning, the all-columns results, shortcut audit

**Q1. Why does the Logistic Regression pipeline need `StandardScaler`, but Random Forest and XGBoost don't?**
<details><summary>Model answer</summary>

Logistic regression multiplies each column by a weight and penalises large weights (L2, controlled by `C`). If one column is in thousands (`LineOfCode`) and another is 0/1, the penalty affects them unequally and the optimiser converges badly. Standardising puts them on the same scale. Trees split on thresholds of single columns ("`LineOfCode` < 120?"). Rescaling a column doesn't change which split is best, so trees are scale-invariant. The scaler sits inside the pipeline, so it is fitted on train only.
</details>

**Q2. XGBoost made 0 errors and Random Forest 1 error on 35,130 test URLs. Is XGBoost the better model?**
<details><summary>Model answer</summary>

No. With 0 errors in 35,130, the 95% upper bound on the error rate is about 3/35,130 ≈ 0.0085% (the "rule of three"), so the true accuracy could be anywhere from about 99.991% to 100%. Random Forest's 1 error and LogReg's 2 are inside that range. A McNemar test with 1 or 2 disagreements can't come close to significance. All three are "indistinguishable from perfect on this test set". And the point of M3 is that "perfect" is explained by shortcut columns anyway.
</details>

**Q3. How big were your hyperparameter searches, what did you leave fixed, and how was the winner chosen when settings differed by one validation URL?**
<details><summary>Model answer</summary>

The searches were small:
- LogReg: 4 values of C.
- Random Forest: 4 settings (depth × min_samples_leaf).
- XGBoost: 6 settings (depth × number of trees), with `learning_rate=0.1` fixed and no subsample/colsample/min_child_weight or early stopping.

`tune()` ranks by validation accuracy at each setting's best threshold, then breaks ties by AUC. On URL features, depth 10 / 800 trees beat depth 6 / 300 by **one** validation URL (0.997324 vs 0.997296), even though it had a *lower* val AUC. That's noise deciding the winner. Better options:
- Select on a smooth metric (val log-loss or AUC).
- Use the "one-standard-error rule": prefer the simplest model within noise of the best.
- Use early stopping on val.

I also checked that this doesn't change our conclusions. Re-tuning XGBoost on the domain view with a different grid moved it from 81.4% to 81.6%.
</details>

**Q4. The dataset's 19 URL columns give 99.75%, but your own 39 URL features give 99.68%. Why would hand-made features with more columns do *worse*?**
<details><summary>Model answer</summary>

Two reasons:

1. **The provided columns were computed on the raw URL**, before we stripped the trailing slash. So they still contain the trailing-slash shortcut that D3 removed from ours (`URLLength` vs cleaned length: legitimate always −1, phishing often 0 or +1).
2. **"`URLLength` minus counted characters" = 12** exactly encodes the `https://www.` prefix (notebook 06 §4). Trees can compute that from the provided columns.

So "better" here partly means "more shortcut". For a fair comparison you'd recompute those 19 columns on the cleaned URL. The 0.07-point difference is also near the noise level (M1).
</details>

**Q5. In notebook 06 you rank columns by single-column ROC-AUC with `max(AUC, 1−AUC)`. What does that measure, and what can it *not* detect?**
<details><summary>Model answer</summary>

It measures how well one column, used directly as a score, ranks phishing above legitimate. Taking `max(AUC, 1−AUC)` counts columns that point "the other way" too.

What it can't detect:
- **Combinations.** `URLLength` and the character counts are each moderate (the best URL column is about 0.82), but their difference separates the classes almost perfectly.
- **Non-monotonic effects.** A column that's suspicious when very high *or* very low would score near 0.5.

It was also computed on the full dataset, including test. That's fine for description, but say so.
</details>

**Q6. You wrote that page columns score 99.64% "because many phishing pages were tiny or already taken down". How would you *test* that rather than assert it?**
<details><summary>Model answer</summary>

Make it a falsifiable prediction. If takedowns explain it, many phishing rows should have near-empty pages: `LineOfCode` ≈ 0, `HasTitle` = 0, `NoOfImage` = 0, `NoOfJS` = 0. To check:
- Compare the share of such "empty" pages by class.
- Retrain the page-only model on non-empty pages only and see whether accuracy collapses.
- Read 20 random phishing rows' page columns.

If accuracy stays high on non-empty pages, the explanation is something else (e.g. homepages are simply much bigger than login pages). Until then, the guide should say "likely".
</details>

**Q7. What would you do if the professor said: "Fine, the all-column result is a shortcut. Show me a fair version"?**
<details><summary>Model answer</summary>

There are three levels, from cheapest to most convincing:

1. **Remove what the app can't have and what encodes the prefix.** Use only URL-string features computed on the cleaned URL, with scheme and `www.` removed. That's essentially our domain-only and registered-domain views (about 85–86% across splits).
2. **Re-evaluate under a stricter split**, e.g. registered domain (option B) or holding out whole hosting platforms, to test generalisation to unseen platforms.
3. **Cross-dataset test.** Train on PhiUSIIL and test on a different source: recent PhishTank/OpenPhish URLs plus legitimate URLs *with paths* from Tranco top sites. That breaks "legitimate = homepage" entirely and is the only real answer to "would this work in the wild?". Expect a big drop. That's a finding, not a failure.
</details>

**Q8. The validation-chosen thresholds were 0.12 (LogReg), 0.17 (RF) and 0.02 (XGBoost) on all columns, and 0.66 (RF) / 0.70 (XGBoost) on URL features. Why so different, and what does that mean for the app's "confidence" number?**
<details><summary>Model answer</summary>

Different models produce differently *calibrated* scores. A random forest's score is the share of trees voting phishing, which tends to stay away from 0 and 1. Boosted trees and regularised LogReg have their own distortions. On all columns the error curve is flat over a huge range (64 tied thresholds for XGBoost), so the exact value is mostly tie-breaking (Manan Q6). So "0.7" from one model and "0.7" from another don't mean the same thing.

Before the app shows "97% phishing", we should calibrate on validation (Platt scaling or isotonic regression, e.g. `CalibratedClassifierCV(..., cv="prefit")` fitted on val) and check the Brier score and reliability curve on test. The domain-only n-gram LR is already fairly well calibrated (ECE 0.031).
</details>

### Abhishek: URL features, n-gram model, URL views, M4 deep models

**Q1. Explain TF-IDF with `sublinear_tf=True` and `min_df=2`. Where exactly is the vectorizer fitted, and why does that matter?**
<details><summary>Model answer</summary>

Each URL becomes a sparse vector with one entry per character n-gram (1–4 characters).
- **TF:** how often the n-gram occurs in this URL. With `sublinear_tf`, it is 1 + log(count), so repeats count less.
- **IDF:** log of (number of URLs / number of URLs containing the n-gram), so common chunks like `com` are down-weighted.
- **`min_df=2`:** drops n-grams seen in only one training URL. That's noise, and it saves memory (308,150 n-grams remain on raw URLs).

The vectorizer is the first step of the sklearn pipeline, so `.fit()` in `tune()`/`fit_and_test()` fits it on `train_idx` only (`modeling.py:25,37`). If it were fitted on all data, the vocabulary and IDF weights would include test URLs. That's a leak, mild for TF-IDF, but still a leak.
</details>

**Q2. On raw URLs, `p:`, `tp:`, `p:/`, `tp:/`, `ttp:` and `p://` all have exactly the same weight, 25.8. Why? What does that tell you about reading single n-gram weights?**
<details><summary>Model answer</summary>

Those n-grams always occur together: every `http://` contains all of them, and they're absent in `https://`. Perfectly co-occurring features are interchangeable to the model. With L2 regularisation, the cheapest solution spreads the total weight equally across them. So you can't interpret one n-gram's weight alone. Interpret *groups* ("the `http://` prefix has a combined weight of about 150") or use contributions for a specific URL (weight × TF-IDF value). The same applies to the `www` family on the legitimate side.
</details>

**Q3. What does your code do with `http://paypal.com@evil.top/login`? Which parts get it right and which get it wrong?**
<details><summary>Model answer</summary>

Everything before `@` is "user info". The real host is `evil.top`. It's a classic phishing trick, because users read `paypal.com`.
- `features.py` uses `urlsplit`, which correctly gives host `evil.top`.
- `urls.hostnames()` uses a regex that doesn't stop at `@`, so it gives `paypal.com@evil.top`. That makes `host_only()` return `paypal.com@evil.top`, and `site_owner()` return `paypal.com` for a real dataset row like this one.

There are 10 such phishing URLs in PhiUSIIL. IPv6 hosts become `[2001`, and scheme-less input (`paypal.com/login`, which users *will* type into the app) gives NaN and crashes `site_owner()`. Fix: one `parse_host()` based on `urlsplit`, used everywhere, with tests. M7's "subdomain injection" and M8's input box both depend on it.
</details>

**Q4. In the domain-only view, can the same domain string appear in train and test? Can a legitimate and a phishing URL share a domain string? Does either make the domain-only test unclean?**
<details><summary>Model answer</summary>

- **Across splits: no.** The owner is a function of the host, so one host's URLs share one split. I checked: 0 domain strings are shared between train and test, or train and val.
- **Within a split, yes.** A domain can repeat, and some repeat heavily: `ipfs.io` appears 1,115 times among 14,907 test phishing rows. That's not leakage, but it means test metrics are dominated by a few domains. Recall is 78.0% counting rows vs 74.1% counting unique domains.
- **Both labels, yes:** 57 domain strings (18 test rows), e.g. under `washington.edu` and `office.com`. Usually it's a real site with a compromised or abused page. With only the domain as input those rows are identical, so some errors are unavoidable.

So the test is clean of leakage, but we should report deduplicated metrics and owner-clustered confidence intervals.
</details>

**Q5. The docs call 88.59% "the honest number". Defend or attack that word.**
<details><summary>Model answer</summary>

**Defend:** it removes the two big collection artefacts (the `https://www.` prefix and "has a path"). The remaining top n-grams (`.cf`, `.ga`, `.top`, digits, `.gov`, `.edu`) are plausible real signals.

**Attack:**
1. **Another artefact remains.** The legitimate list is mostly bare registered domains of popular sites. Only 2.8% of legitimate hosts have a subdomain beyond `www`, and 0.08% are on a hosting platform, against 33% and 26.5% of phishing hosts. A rule with no learning gets 79.4%, and the model scores 95.7% on structured hosts but 86.2% on bare domains. In real browsing, `docs.google.com` and `someone.github.io` are legitimate.
2. **Repeats inflate it** (Q4).
3. **It's one split.** Across six split seeds the mean is 86.3% ± 1.3.

Better wording: "shortcut-reduced", and also report a registered-domain view (84.6%), which asks the purest question: can you tell from the *name* alone?
</details>

**Q6. N-grams beat XGBoost on the domain view, 88.6% vs 81.4%. Was it a fair fight?**
<details><summary>Model answer</summary>

Not quite. In notebook 07, C was re-tuned per view for n-grams, but XGBoost reused the raw-URL settings. Also, 15 of the 39 features are constant on bare domains (no path, query or scheme), so XGBoost really had 24 features. I re-tuned XGBoost on the domain view (depth 4/6/10, min_child_weight 1/10, lr 0.05, subsampling) and got 81.6%, so the conclusion holds.

The reason is information, not tuning. Counts like `host_digits=1` or `tld_length=2` can't distinguish `.cf` from `.de` or `paypa1` from `paypal`. The n-gram model sees the actual characters. That's the motivation for M4: a char-CNN learns which character patterns matter, instead of us choosing counts.
</details>

**Q7. For M4 on an 8 GB M2, what input encoding, maximum length and architecture will you use? How will you show the CNN beats the n-gram baseline?**
<details><summary>Model answer</summary>

- **Encoding:** integer character IDs (PhiUSIIL uses only 71 distinct characters) with an `nn.Embedding` (dimension 32–64), plus PAD = 0 and UNK for unseen characters. Build the vocabulary from **train only**. **Never one-hot the whole dataset:** 164k × 160 × 71 × 4 bytes ≈ 7.4 GB, which would exhaust the RAM on its own. As uint8 IDs it's about 26 MB.
- **Max length:** raw URLs, 160 (the 99th percentile is 145). Domain-only, 64 (99th percentile 56, max 110).
- **char-CNN:** parallel `Conv1d` with kernel sizes 3/4/5/7 and 128 filters each, ReLU, global max-pool, dropout 0.5, linear → 1 logit. `BCEWithLogitsLoss`, AdamW at lr 1e-3, batch 512, early stopping on val loss, 3 model seeds.
- **BiLSTM:** embedding 32, 1 layer, hidden 64–128 per direction, masked max-pool over time.
- **Proof it's better:** the same split seeds as the n-gram baseline, threshold on val, then on test a paired McNemar test per seed, plus mean ± SD across seeds. "Better" means a consistent gain on the domain-only and registered-domain views, not 0.05 points on raw URLs.
</details>

**Q8. What would you do if your char-CNN scores 99.8% on raw URLs but only 85% on domain-only?**
<details><summary>Model answer</summary>

That's the *expected* result, not a failure. On raw URLs the CNN finds the `https://www.` prefix in its first filters, just as the n-gram model found `p://`. To prove it, use character attributions: Integrated Gradients or occlusion should light up the first 12 characters.

Then:
- Report domain-only as the main number, against the n-gram's 86.3% ± 1.3. If the CNN is within that band, say honestly that a deep model adds nothing over n-grams here, which is itself a publishable-style finding.
- Run the M7 perturbation "prepend `https://www.` and drop the path" on phishing URLs. The raw model should collapse, which is a clean demonstration of the shortcut.
- Try the hybrid (M5) on the domain view to see whether structure features add anything on top.
</details>

---

## 5. Guidance for M4–M9

### M4: char-CNN and BiLSTM (PyTorch on an M2 Air, 8 GB)
- **Setup:** `torch` (MPS build). Use `device = "mps" if torch.backends.mps.is_available() else "cpu"`. Set `torch.manual_seed`, `np.random.seed` and `random.seed`. MPS isn't fully deterministic, so run **3 model seeds** and report mean ± SD. Use `DataLoader(num_workers=0)` inside notebooks, because macOS multiprocessing from Jupyter is fragile. Put the training loop in `src/phishguard/deep.py` so the app can import it, and test it on a 1,000-row slice.
- **Data:** a vocabulary from train characters (index 0 = PAD, 1 = UNK), stored as JSON next to the weights. Encode to `uint8`/`int16` arrays and keep them in memory (tens of MB). Lowercase the host. Case in paths can stay.
- **char-CNN (start here):** embedding 48 → 4 parallel `Conv1d` (kernels 3, 4, 5, 7; 128 filters) → ReLU → global max-pool → concat (512) → dropout 0.5 → linear(1). About 0.3M parameters. AdamW (lr 1e-3, weight decay 1e-4), batch 512, at most 15 epochs, early stopping on **val log-loss** (patience 3). Expect seconds to a minute per epoch on domain-only.
- **BiLSTM:** embedding 32 → BiLSTM (hidden 96, 1 layer) → **masked** max-pool (ignore PAD positions) → dropout → linear. If `pack_padded_sequence` misbehaves on MPS, bucket batches by length or run the LSTM on CPU for the domain view (sequences ≤ 64). It's slower, so tune lightly.
- **Evaluation (tied to C1, M1, M3, M5):** reuse `get_splits`, the threshold from validation and test once, through the *extended* `evaluate()`. Report raw, domain-only and registered-domain views over **5 split seeds × 3 model seeds**. Save probabilities. Compare against the n-gram LR on the same splits with McNemar. Pre-register this table before training.
- **Pitfalls:**
  - Early stopping and the threshold both use val. That's fine, but say so.
  - Don't tune the architecture on test.
  - Padding side matters: right-pad and mask, so the model can't learn "lots of padding = short = legitimate". Length differs by class (domain mean 15.2 vs 22.8), and that part is a real signal.
  - Watch for UNK-heavy inputs at app time.

### M5: hybrid
- Fuse the CNN's pooled vector (e.g. 512 → dense 64) with **our own** features (standardised, scaler fitted on train) → dense 32 → concat → dense → logit. **Never** feed the dataset's provided columns (M7 trailing-slash residue; the app can't compute them).
- Compute features on the **same view** as the CNN input. On the domain view, drop the constant columns (p1).
- **Baselines to beat:** CNN alone; features alone; logit-average of the two; a stacker trained on **out-of-fold** train predictions (5-fold over owners with `GroupKFold`). Don't train the stacker on val, which is already used for thresholds.
- **Ablation table:** each component removed, mean ± SD over seeds. If the hybrid gain is inside the noise, say so.

### M6: explainability
- **XGBoost on own features:** `shap.TreeExplainer` on about 2,000 test rows (seconds). Group correlated features (`url_length`/`num_letters`/`host_length`), or SHAP will split credit and confuse readers. Show the raw-URL model leaning on `is_https`/`has_www`/`path_length`. That's good evidence for M3.
- **n-gram LR:** exact per-URL contributions = coefficient × TF-IDF value. Free and perfect for the app. Aggregate co-occurring n-grams (Abhishek Q2).
- **char-CNN:** Integrated Gradients (`captum`) or occlusion (mask one character at a time) for per-character heatmaps. Avoid KernelSHAP on the CNN (too slow for 8 GB / Air thermals).
- **Sanity check:** randomise the model weights and confirm the explanations change (the "sanity checks for saliency maps" test). Professors like it.

### M7: robustness (model not retrained)
- First fix parsing (M4) and add `has_punycode` (m3).
- **Transforms** (each a tested pure function `perturb_x(url) -> url`), applied to **phishing test URLs that the model currently detects**:
  - homoglyph/digit swap (`l→1`, `o→0`, `rn→m`)
  - Unicode homoglyph → IDNA/punycode
  - subdomain injection (`paypal.com.` + host)
  - `@` user-info trick
  - path or query padding
  - TLD swap (`.com → .top`/`.xyz`)
  - prefix normalisation (`http://` → `https://www.`, drop the path)
- **Metrics:**
  - **evasion rate** = share of detected phishing URLs that flip to "legitimate"
  - recall before/after with owner-bootstrap CIs
  - per-transform tables for raw vs domain-only models
- Also perturb **legitimate** URLs (add a path) to measure *induced false alarms*. The raw model will flag them, which is another shortcut demonstration.
- Keep the **validation threshold fixed**; edge thresholds (m1) make results fragile.
- Note invariance by construction: path transforms can't affect a domain-only model. That isn't "robustness", and you should say so.

### M8: Streamlit app
- **Input normalisation must equal training:**
  - strip whitespace and the trailing `/`
  - add `http://` if there's no scheme
  - lowercase the host
  - IDN → punycode
  - parse with the fixed `parse_host()`

  Reject garbage with a friendly message.
- Load saved artefacts (model, vocabulary, scaler, threshold, calibrator) with `st.cache_resource`. Never train at startup. Pin versions (pickles break across scikit-learn versions; M6).
- **Default to the domain-only model**, and show the raw-URL model only with a warning: "trained where every legitimate URL is a bare homepage; any URL with a path will look suspicious".
- Show a **calibrated** probability (fit on val), the top contributing n-grams or characters, and a disclaimer that this is a course project, not a security product.
- Test with `streamlit.testing.v1.AppTest` (types a URL, checks for an output). Have a 2-minute demo script ready: a bare legitimate domain, an obvious `.cf` phish, an `@`-trick URL, and a compromised-looking `cama.it`-style miss.

### M9: report and slides
- Structure: research question → dataset audit (shortcuts) → protocol (split, threshold, test once, seeds) → results with **CIs and seed SDs** → explainability → robustness → limitations and threats to validity → reproducibility appendix (versions, hardware, run times, seeds, commit).
- The story is your differentiator: "We reproduced 99.99%, showed why it's an artefact, and measured what's left: about 86% from the domain, plus how fragile models are to simple edits". Lead with it.
- Every comparison with a paper should carry its split protocol (m5). Every "A beats B" claim should carry a significance test (M1).
- One figure to rule them all: the views chart (raw → no scheme → domain → registered domain) with error bars over split seeds, one bar per model (n-gram, XGBoost, CNN, BiLSTM, hybrid).

---

## Appendix: how I checked the key claims

All checks used the repo's own functions. Each runs in under a minute.

```python
import numpy as np, pandas as pd
from phishguard.data import load_clean, TARGET
from phishguard.split import get_splits, site_owner
from phishguard.urls import host_only, TLD_EXTRACT
from phishguard.evaluate import best_threshold, evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = load_clean(); y = df[TARGET]; dom = host_only(df["URL"])
make = lambda: make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2, sublinear_tf=True, dtype=np.float32),
    LogisticRegression(C=3.0, solver="liblinear", max_iter=3000))

# C1: split-seed variance of the domain-only n-gram model
for seed in [42, 0, 1, 2, 3, 7]:
    tr, va, te = get_splits(df, seed=seed)
    m = make().fit(dom[tr], y[tr])
    t = best_threshold(y[va], m.predict_proba(dom[va])[:, 1])
    print(seed, evaluate(y[te], m.predict_proba(dom[te])[:, 1], t)["accuracy"])
# -> 0.8859, 0.8497, 0.8564, 0.8571, 0.8706, 0.8600

# M1: owner-cluster bootstrap CI (correct = 0/1 per test row, owners = site_owner of test rows)
def owner_bootstrap_ci(correct, owners, n=2000, seed=0):
    u, inv = np.unique(owners, return_inverse=True)
    s, c = np.bincount(inv, weights=correct), np.bincount(inv)
    rng = np.random.default_rng(seed)
    accs = [s[k].sum() / c[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    return np.percentile(accs, [2.5, 97.5])

# M2: residual structure in the domain view
ext = dom.map(TLD_EXTRACT)
has_sub = ext.map(lambda e: e.subdomain != "")
on_platform = ext.map(lambda e: e.is_private)
print(has_sub.groupby(y).mean(), on_platform.groupby(y).mean())  # 0.028 vs 0.332; 0.0008 vs 0.265

# M4: parser bug
print(host_only(pd.Series(["http://paypal.com@evil.top/login"])))  # 'paypal.com@evil.top'
site_owner(pd.Series(["www.example.com/a"]))                         # AttributeError

# M7: trailing-slash residue in provided columns
print(pd.crosstab(df["URLLength"] - df["URL"].str.len(), y))        # legit always -1
```

The McNemar comparison (raw n-gram vs raw XGBoost) refits both models with their notebook settings (n-gram `C=10`; XGBoost depth 10, 800 trees). It counts test URLs where exactly one model is right (33 vs 26) and applies `scipy.stats.binomtest(26, 59, 0.5)`, giving p = 0.435.

Good work so far. The foundations are solid enough that fixing the items in §3 is a day's work, and it turns a good project into one that holds up under hard questioning.
