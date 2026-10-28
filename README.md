# 🛡️ PhishGuard

**Phishing URL detection on the PhiUSIIL dataset:** classical ML, deep learning on raw URLs, a hybrid model, explanations, and a robustness test against modified URLs. A university ML course project by Manan, Shaurya and Abhishek.

## 📍 Status

| # | Milestone | Status |
|---|---|---|
| 1 | Data loading, label direction, cleaning, split by site owner | ✅ Done |
| 2 | Reproduce published ≥99.9%; URL-only models | ✅ Done |
| 3 | Shortcut analysis: why the scores are so high | ✅ Done |
| 4 | Char-CNN and BiLSTM on raw URLs and shortcut-reduced views | ✅ Done |
| 5 | Hybrid model (char-CNN + handcrafted features) | ✅ Done |
| 6 | Explainability (SHAP, character attributions) | ✅ Done |
| 7 | Robustness: original vs modified URLs, popular real websites | ✅ Done |
| 8 | Project website: link checker with explanations and warnings | ✅ Done |
| 9 | Report and slides | ⏳ Next |

Details: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## ⚙️ Setup

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements.lock      # exact versions we used
uv pip install --python .venv --no-deps -e .
.venv/bin/pytest                                  # all tests should pass
```

Every notebook loads, splits and evaluates data the same way:

```python
from phishguard.data import load_clean, TARGET   # TARGET = "is_phishing", 1 = phishing
from phishguard.experiment import run            # tune → threshold on validation → test, over 5 splits
df = load_clean()
```

**Run the website** (after setup):

```bash
.venv/bin/python -m phishguard.web               # then open http://127.0.0.1:8000
```

It needs the files in `models/`, which are in the repo. To rebuild them from the data: `.venv/bin/python -m phishguard.serving` (a few minutes).

- **New to the project?** Read [`docs/guide/`](docs/guide/README.md), which explains every part simply, with likely viva questions.
- **Every decision and why:** [`docs/DECISIONS.md`](docs/DECISIONS.md). **Sources:** [`docs/references.md`](docs/references.md).
- **Notebooks:** run in the order listed in [`notebooks/readme.md`](notebooks/readme.md).

## 📊 Results so far

<!-- RESULTS:START -->
Test accuracy, **mean ± sd over 5 site-owner splits**, and the seed-42 split with its 95% confidence interval.

| Model | Input | Accuracy (5 splits) | Seed 42 [95% CI] | ROC-AUC |
|---|---|---|---|---|
| XGBoost (all columns) | all provided columns | 100.00% ± 0.00% | 100.00% [100.00%, 100.00%] | 1.0000 |
| Random Forest (all columns) | all provided columns | 100.00% ± 0.00% | 100.00% [100.00%, 100.00%] | 1.0000 |
| LogReg (all columns) | all provided columns | 99.99% ± 0.00% | 99.99% [99.98%, 100.00%] | 1.0000 |
| Average: n-gram + XGBoost (URL only) | raw URL + URL features | 99.70% ± 0.03% | 99.75% [99.68%, 99.80%] | 0.9986 |
| Char n-gram LogReg (raw URL) | raw URL | 99.69% ± 0.05% | 99.75% [99.69%, 99.80%] | 0.9986 |
| XGBoost (URL features) | raw URL → 40 features | 99.63% ± 0.03% | 99.68% [99.62%, 99.74%] | 0.9973 |
| Char n-gram LogReg | **domain only** (shortcut-reduced) | 86.27% ± 0.89% | 87.33% [86.14%, 88.57%] | 0.9162 |
| Char n-gram LogReg | **registered domain** (shortcut-reduced) | 82.49% ± 1.93% | 85.03% [83.38%, 86.71%] | 0.8945 |
| XGBoost URL features | **domain only** (shortcut-reduced) | 81.43% ± 1.15% | 80.19% [74.55%, 83.56%] | 0.8473 |
| Char-CNN | raw URL | 99.68% ± 0.03% | 99.73% [99.66%, 99.78%] | 0.9982 |
| Char-CNN | **domain only** (shortcut-reduced) | 86.58% ± 0.92% | 87.83% [86.83%, 89.06%] | 0.9165 |
| Char-CNN | **registered domain** (shortcut-reduced) | 80.85% ± 1.53% | 83.14% [81.21%, 85.04%] | 0.8793 |
| BiLSTM | **domain only** (shortcut-reduced) | 85.55% ± 1.35% | 87.21% [85.88%, 88.57%] | 0.9093 |
| Hybrid CNN + features | **domain only** (shortcut-reduced) | 86.72% ± 0.99% | 88.05% [87.06%, 89.26%] | 0.9173 |
| Hybrid CNN + features | **registered domain** (shortcut-reduced) | 82.01% ± 0.85% | 83.39% [81.41%, 85.32%] | 0.8853 |

Every legitimate URL in PhiUSIIL is a bare `https://www.` homepage, so the near-perfect scores mostly reflect how the dataset was built ([shortcut analysis](docs/guide/03_shortcuts.md)). The shortcut-reduced rows show what a URL-only model can do from the domain name.
<!-- RESULTS:END -->

---

### Explainable Hybrid Machine Learning and Deep Learning for Phishing URL Detection

**PhishGuard** is a Machine Learning and Deep Learning based system for detecting phishing URLs by combining **handcrafted URL features** with **deep representations learned directly from raw URLs**.

The project compares traditional ML models, deep learning models, and a hybrid ML-DL model. It also investigates **why URLs are classified as phishing** and evaluates the model's robustness against **realistic URL manipulations** by comparing predictions on original and modified URLs.

---

# 🚨 Problem Statement

Phishing URLs are designed to deceive users into visiting malicious websites and revealing sensitive information such as passwords, financial details, and personal data.

Traditional phishing detection systems often depend on manually engineered URL features, while deep learning approaches can learn patterns directly from raw URL sequences.

Both approaches have limitations:

* Handcrafted features may fail to capture complex URL patterns.
* Deep learning models may be difficult to interpret.
* A model that performs well on normal test data may behave differently when a phishing URL is slightly modified.

Therefore, this project develops a **hybrid ML-DL phishing URL detection system** and evaluates both its classification performance and its robustness to URL-level modifications.

---

# 🎯 Objectives

* Detect whether a URL is **phishing or legitimate**.
* Perform exploratory data analysis on phishing URL datasets.
* Extract lexical and structural URL features.
* Train traditional ML models such as:

  * Random Forest
  * XGBoost
* Train deep learning models on raw URL sequences:

  * CNN
  * BiLSTM
* Develop a hybrid ML-DL phishing detection model.
* Compare ML, DL, and hybrid model performance.
* Understand why a URL is classified as phishing.
* Apply explainability techniques such as SHAP and feature importance.
* Generate realistic modified versions of phishing URLs.
* Compare model performance on **original vs modified URLs**.
* Measure the performance degradation caused by URL manipulation.
* Deploy the final detection system using Streamlit.

---

# 🔄 Project Pipeline

```text
                         PHISHING URL DATASET
                                  ↓
                         Data Preprocessing
                                  ↓
              ┌───────────────────┴───────────────────┐
              ↓                                       ↓
      Feature Engineering                         Raw URL
              ↓                                       ↓
    URL Lexical / Structural                  Character
          Features                            Tokenization
              ↓                                       ↓
       ┌──────┴──────┐                            Embedding
       ↓             ↓                                ↓
 Random Forest    XGBoost                       CNN / BiLSTM
       ↓             ↓                                ↓
       └──────┬──────┘                                │
              │                                       │
              └────────────────┬──────────────────────┘
                               ↓
                         HYBRID MODEL
                               ↓
                      Phishing Prediction
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
             Explainability        Robustness Testing
                    ↓                     ↓
             Why phishing?        Original vs Modified
                    ↓                     ↓
             SHAP / Feature        Performance Drop
                Analysis
```

---

# 🧠 System Architecture

The system contains two main branches.

## 1. Handcrafted Feature Branch

Relevant lexical and structural characteristics are extracted from each URL.

Possible features include:

* URL length
* Number of dots
* Number of hyphens
* Number of digits
* Number of special characters
* Number of subdirectories
* Number of query parameters
* Number of subdomains
* Presence of an IP address
* Suspicious keywords
* URL shortening indicators
* Character entropy
* HTTPS-related characteristics
* Domain-related characteristics

These features are passed to traditional machine learning models.

```text
URL
 ↓
Feature Extraction
 ↓
Lexical / Structural Features
 ↓
Random Forest / XGBoost
 ↓
Phishing / Legitimate
```

---

# 🤖 2. Raw URL Deep Learning Branch

Instead of relying only on manually extracted features, the raw URL is directly processed using deep learning.

```text
Raw URL
 ↓
Character Tokenization
 ↓
Embedding
 ↓
CNN / BiLSTM
 ↓
Deep URL Representation
 ↓
Prediction
```

The objective is to allow the deep learning model to learn useful patterns directly from URL character sequences.

---

# 🔗 3. Hybrid ML-DL Model

The main model combines information from both branches.

```text
                    Raw URL
                       ↓
              Character Tokenization
                       ↓
                   Embedding
                       ↓
                 CNN / BiLSTM
                       ↓
                Deep Representation
                       │
                       │
Handcrafted Features ──┤
       ↓               │
    Dense Layer        │
       ↓               │
       └───────┬───────┘
               ↓
          Feature Fusion
               ↓
          Dense Layers
               ↓
       Phishing Probability
```

The hybrid model combines:

**Handcrafted URL features**

+

**Deep features learned from raw URLs**

↓

**Final phishing prediction**

---

# 📊 Model Comparison

The project evaluates three main approaches.

## Traditional ML

```text
URL
 ↓
Feature Engineering
 ↓
Random Forest / XGBoost
 ↓
Prediction
```

## Deep Learning

```text
Raw URL
 ↓
Tokenization
 ↓
Embedding
 ↓
CNN / BiLSTM
 ↓
Prediction
```

## Hybrid ML-DL

```text
Raw URL + Handcrafted Features
              ↓
       ML + Deep Learning
              ↓
        Feature Fusion
              ↓
          Prediction
```

This comparison helps determine whether combining manually engineered features with automatically learned URL representations improves phishing detection.

---

# 🔍 Explainability

The system aims to answer:

> **"Why was this URL classified as phishing?"**

Instead of only showing:

```text
Prediction: PHISHING
Confidence: 97.4%
```

the system can identify influential characteristics such as:

```text
✓ Unusually long URL
✓ High number of subdomains
✓ Suspicious keyword detected
✓ High special-character frequency
✓ Presence of an IP address
✓ Unusual character distribution
```

### Explainability techniques

* SHAP
* Feature Importance
* Feature contribution analysis
* URL characteristic analysis
* Character/token-level analysis where applicable

---

# ⚔️ Robustness Testing: Original vs Modified URLs

A key experimental component of the project is evaluating how the trained model behaves when phishing URLs are realistically modified.

For example:

```text
Original URL:
paypal-login.com

Modified URL:
paypa1-login.com
```

The model is **not retrained** for this experiment.

Instead, the already-trained model is tested on both versions.

```text
                 Trained Model
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
    Original URL            Modified URL
          ↓                       ↓
      Prediction              Prediction
          ↓                       ↓
          └───────────┬───────────┘
                      ↓
             Robustness Analysis
                      ↓
              Performance Drop
```

Possible URL transformations include:

* Character substitution
* Digit/character substitution
* Homoglyph-style substitution
* Subdomain injection
* URL padding
* Path modification
* Query modification
* TLD modification

The goal is to determine whether small changes to a phishing URL can cause the model's prediction performance to degrade.

---

# 📈 Robustness Evaluation

The model is evaluated separately on:

### Clean URLs

```text
Original Test URLs
       ↓
     Model
       ↓
Performance Metrics
```

### Modified URLs

```text
Modified Test URLs
       ↓
     Model
       ↓
Performance Metrics
```

The results are compared to calculate performance degradation.

```text
Performance Drop =
Clean Performance - Modified Performance
```

A smaller performance drop indicates better robustness.

---

# 🧪 Experimental Design

## Experiment 1 — Exploratory Data Analysis

Analyze:

* Class distribution
* URL length
* Character distributions
* Feature distributions
* Phishing vs legitimate URL characteristics
* Feature correlations

---

## Experiment 2 — Traditional ML

Train and evaluate:

* Random Forest
* XGBoost

using engineered URL features.

---

## Experiment 3 — Deep Learning

Train and evaluate:

* CNN
* BiLSTM

using raw URL sequences.

---

## Experiment 4 — Hybrid Model

Combine:

```text
Handcrafted URL Features
          +
Raw URL Deep Representation
          ↓
      Feature Fusion
          ↓
      Hybrid Model
```

---

## Experiment 5 — Explainability

Analyze which URL characteristics contribute most strongly to phishing predictions.

---

## Experiment 6 — Robustness Testing

Generate modified versions of phishing URLs and compare:

```text
Original URL
      vs
Modified URL
```

The objective is to measure how much the model's performance changes after URL manipulation.

---

# 📏 Evaluation Metrics

The models will be evaluated using:

* Accuracy, Precision, Recall, F1-score, Confusion Matrix
* ROC-AUC and PR-AUC (threshold-free)
* MCC (Matthews correlation coefficient)
* Share of phishing caught at 1% and 0.1% false alarms
* Mean ± standard deviation over 5 site-owner splits, and a 95% confidence interval that resamples whole site owners

For robustness evaluation:

* Clean performance
* Modified URL performance
* Performance degradation
* Robustness comparison across models

---

# 🖥️ Streamlit Application

The final model will be integrated into a Streamlit application.

### Workflow

```text
User enters URL
       ↓
Preprocessing
       ↓
Feature Extraction
       ↓
Raw URL Processing
       ↓
Hybrid Model
       ↓
Phishing / Legitimate
       ↓
Confidence Score
       ↓
Explainability
```


# 🔮 Future Scope

The current version focuses on:

* Phishing URL classification
* ML/DL comparison
* Hybrid ML-DL detection
* Explainability
* Original vs modified URL robustness testing

Future extensions can include:

## 1. Adversarial Training

The modified URLs generated during robustness testing can be incorporated into the training dataset.

```text
Original Training URLs
          +
Modified Training URLs
          ↓
    Retrain Model
          ↓
  More Robust Model
```

The performance of the original model and adversarially trained model can then be compared.

---

## 2. HTML/DOM-Based Detection

Extend the system beyond URLs by analyzing:

* HTML structure
* Forms
* JavaScript
* External resources
* DOM characteristics

---

## 3. Screenshot-Based Detection

Use computer vision/deep learning to analyze website screenshots.

---

## 4. Domain and DNS Intelligence

Incorporate:

* WHOIS information
* DNS records
* Domain age
* Certificate information
* Domain reputation

---

## 5. Threat Intelligence Integration

Integrate external threat intelligence sources for real-time phishing information.

---

## 6. Browser Extension

Deploy the model as a browser extension that can analyze URLs before users visit websites.

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest
* XGBoost
* Logistic Regression / SVM

### Deep Learning

* PyTorch or TensorFlow
* CNN
* BiLSTM

### Explainability

* SHAP
* Feature Importance

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---


# 📌 Scope: in this project vs future work

| Component                  |   Status  |
| -------------------------- | :-------: |
| URL preprocessing          | ✅ Done |
| URL feature engineering    | ✅ Done |
| Random Forest              | ✅ Done |
| XGBoost                    | ✅ Done |
| CNN                        | ⏳ Planned |
| BiLSTM                     | ⏳ Planned |
| Hybrid ML-DL model         | ⏳ Planned |
| Explainability / SHAP      | ⏳ Planned |
| Original vs Modified URL   | ⏳ Planned |
| Robustness testing         | ⏳ Planned |
| Performance drop analysis  | ⏳ Planned |
| Adversarial training       | 🔮 Future |
| HTML/DOM analysis          | 🔮 Future |
| Screenshot-based detection | 🔮 Future |
| DNS/WHOIS intelligence     | 🔮 Future |
| Threat intelligence        | 🔮 Future |
| Browser extension          | 🔮 Future |

---

# 🚀 Expected Outcome

PhishGuard aims to develop a phishing URL detection system that:

1. Accurately distinguishes phishing URLs from legitimate URLs.
2. Compares traditional ML and deep learning approaches.
3. Combines handcrafted URL features with deep representations.
4. Provides interpretable insights into model predictions.
5. Evaluates how URL modifications affect model performance.
6. Establishes a foundation for future adversarially robust phishing detection.

