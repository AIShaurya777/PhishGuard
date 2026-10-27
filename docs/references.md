# References

Each paper we compare against or cite, **with how it split the data**. Results from different splits aren't directly comparable: a random split lets one website appear in both training and test, which usually raises scores.

## Dataset
| Reference | What we use it for | Split / protocol |
|---|---|---|
| A. Prasad, S. Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning", *Computers & Security* 136 (2024) 103545. [link](https://www.sciencedirect.com/science/article/abs/pii/S0167404823004558) · data: [UCI 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | The dataset; reported 99.24% (incremental) / 99.79% (pre-trained) accuracy | All features, including `URLSimilarityIndex`. Split not verified: the paper is paywalled and the numbers come from its abstract |

## Results on PhiUSIIL we compare with
| Reference | Reported | Split / protocol |
|---|---|---|
| Bao et al., *Computers, Materials & Continua* 87(2), 2026. [link](https://www.techscience.com/cmc/v87n2/66606/html) | Random Forest 99.99% (all columns) | Stratified random 80/20; leakage not discussed (from our literature notes, not re-verified) |
| T. Ahamed, S. C. Kakon, F. Al Farid, J. Uddin, H. B. A. Karim, "An integrated evaluation protocol for adversarial robustness, generalization, and explanation stability in URL-based phishing detection", *Frontiers in Computer Science* 8 (2026). [doi:10.3389/fcomp.2026.1834407](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1834407/full) | URL only: LR 99.62%, XGBoost 99.74%, CharCNN 99.81%, BERT 99.84% (Table 6) | **Host-disjoint ~70/15/15**: "no host appears in more than one split" (Sec. 3.4); 235,370 URLs after deduplication; normalised URL strings (Sec. 3.3). Closest to our protocol: they group by *host*, we group by *site owner* (registered domain) |

## Independent audits of PhiUSIIL (checked 27 Oct 2026)
These are practitioner write-ups (GitHub READMEs, a model card, a preprint), **not peer-reviewed**. We quote the numbers they report and haven't re-run their code. We list them because they found the same shortcuts independently of us.

| Source | What they report | Matches our |
|---|---|---|
| ollivianguy, "phishing-url-detection" ([GitHub](https://github.com/ollivianguy/phishing-url-detection)) | `URLSimilarityIndex` = 100 for all legitimate rows (rule: 99.67%); every legitimate URL starts with `https://www.`; HTTPS recoverable from `URLLength` minus counted characters for 99.7% of URLs; last character cut from every legitimate URL before feature extraction but from only 52% of phishing; page columns mostly empty for phishing (`LineOfCode` 66 vs 1,947). **Attacks on the raw-URL model:** switching to HTTPS drops detection 99.4% → 67.7%, five edits → 54.5% | D3, D9, M7 |
| Yoyagm, "phishing-url-detector" ([GitHub](https://github.com/Yoyagm/phishing-url-detector)) | Two label leaks (`URLSimilarityIndex`; legitimate class normalised at collection: 100% HTTPS + `www.`, 0% paths). Host-name-only model, split by registered domain: ROC-AUC 0.919 internally, 0.937 on Tranco + later PhishTank; 70.5% recall at 1% FPR. Subdomain depth is its top SHAP feature | D9, D10, D14 |
| awesomedudeworld13, "phish-drift" ([GitHub](https://github.com/awesomedudeworld13/phish-drift)) | A zero-parameter `https://www.` regex: 99.56% accuracy on PhiUSIIL. A PhiUSIIL-trained Random Forest scores TSS 0.000 on live URLs (small sample, 568 URLs); live legitimate URLs mostly *have* paths | D9 |
| valezion, "url-phishing-detector" model card ([Hugging Face](https://huggingface.co/valezion/url-phishing-detector)) | Calls PhiUSIIL unusable ("a model trained on it learns length"); "a known false positive: `accounts.google.com/signin` still scores as phishing" | D14 |
| NclRsi46, "phishing-detection-ml" ([GitHub](https://github.com/NclRsi46/phishing-detection-ml)) | (Different data: PhishTank vs Tranco) a host model flagged `login.microsoftonline.com` because the legitimate list contained only main domains | D14 |
| Nadi-989, "Shortcut Features and Fair Evaluation in Phishing URL Detection" ([GitHub](https://github.com/Nadi-989/phishing-shortcut-evaluation)) | Single-feature ROC-AUC screen (> 0.95 = shortcut candidate); `URLSimilarityIndex` 0.996. Publication venue not verified | D9 (same method) |
| Ramesh L, SSRN preprint ([link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7399699)) | Full-feature PhiUSIIL accuracies sit at the ceiling and "should not be read as evidence of a deployable detector" | D8 |

**What we add beyond these:** a 5-split site-owner evaluation with McNemar tests; char-CNN/BiLSTM/hybrid models on three views; explanations checked with a deletion test; and the measured cause of the subdomain bias (6.2% vs 56.9% structured domains).

## Methodology
| Reference | Used for |
|---|---|
| D. Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", *USENIX Security* 2022 | Pitfalls we check: sampling bias, data snooping, spurious correlations (shortcuts), base-rate fallacy |
| F. Pendlebury et al., "TESSERACT: Eliminating Experimental Bias in Malware Classification", *USENIX Security* 2019 | Why related samples must not be split across train and test |
| C. Whittaker, B. Ryner, M. Nazif, "Large-Scale Automatic Classification of Phishing Pages", *NDSS* 2010 | Why low false-positive rates matter at realistic phishing base rates |
| H. Le et al., "URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection", arXiv:1802.03162, 2018 | Character-level CNN design for Milestone 4 |
| K. Sabir et al., "An Evasion Attack against ML-based Phishing URL Detectors" (URLBUG), arXiv:2005.08454 (*IEEE TDSC*) | Perturbation families for the Milestone 7 robustness test |

## Tools
| Tool | Used for |
|---|---|
| scikit-learn, XGBoost, pandas, NumPy, SciPy | Models, metrics, data handling |
| tldextract 5.4.0 (bundled Public Suffix List) | Site owners for the split, URL views |
