# PhishGuard
# 🛡️ PhishResist

### Explainable Hybrid Machine Learning and Deep Learning for Phishing URL Detection

**PhishResist** is a Machine Learning and Deep Learning based system for detecting phishing URLs by combining **handcrafted URL features** with **deep representations learned directly from raw URLs**.

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

*

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

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

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


# 📌 Current Scope vs Future Scope

| Component                  |   Status  |
| -------------------------- | :-------: |
| URL preprocessing          | ✅ Current |
| URL feature engineering    | ✅ Current |
| Random Forest              | ✅ Current |
| XGBoost                    | ✅ Current |
| CNN                        | ✅ Current |
| BiLSTM                     | ✅ Current |
| Hybrid ML-DL model         | ✅ Current |
| Explainability / SHAP      | ✅ Current |
| Original vs Modified URL   | ✅ Current |
| Robustness testing         | ✅ Current |
| Performance drop analysis  | ✅ Current |
| Adversarial training       | 🔮 Future |
| HTML/DOM analysis          | 🔮 Future |
| Screenshot-based detection | 🔮 Future |
| DNS/WHOIS intelligence     | 🔮 Future |
| Threat intelligence        | 🔮 Future |
| Browser extension          | 🔮 Future |

---

# 🚀 Expected Outcome

PhishResist aims to develop a phishing URL detection system that:

1. Accurately distinguishes phishing URLs from legitimate URLs.
2. Compares traditional ML and deep learning approaches.
3. Combines handcrafted URL features with deep representations.
4. Provides interpretable insights into model predictions.
5. Evaluates how URL modifications affect model performance.
6. Establishes a foundation for future adversarially robust phishing detection.

