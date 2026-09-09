Shape & structure

235,795 rows × 55 columns, no missing values anywhere
Confirmed from official docs: label = 1 → legitimate, label = 0 → phishing

Class distribution

Legitimate: 134,850 (57.2%)
Phishing: 100,945 (42.8%)
Ratio ~1.34:1 — mild imbalance, doesn't need SMOTE/undersampling, just keep it in mind for metrics (favor F1/ROC-AUC over raw accuracy)
