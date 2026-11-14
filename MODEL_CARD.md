# Model Card: CardioSense AI XGBoost Classifier

Following the framework proposed by Mitchell et al. (2019) (*Model Cards for Model Reporting*), this document details the architectural specifications, training procedures, evaluation benchmarks, clinical limitations, and algorithmic fairness boundaries for the serialized CardioSense model pipeline.

---

## 1. Model Details

- **Model Name**: CardioSense CAD Stratification Classifier
- **Model Type**: Supervised binary classification pipeline chaining a `ColumnTransformer` preprocessor and a tree-based ensemble (`xgboost.XGBClassifier`).
- **Model Version**: 1.0.0 (Production Release)
- **Serialized Artifact**: `models/best_model.joblib` (Artifact size: ~260 KB)
- **Release Date**: November 2026
- **License**: MIT License (Code) / CC BY 4.0 (Dataset Citation)
- **Software Dependencies**:
  - `python==3.14.10` (tested runtime)
  - `xgboost==3.4.1`
  - `scikit-learn==1.9.1`
  - `shap==0.54.1`
  - `joblib==1.6.0`
  - `pandas==3.1.0`
  - `numpy==2.5.4`
- **Optimal Hyperparameters** (tuned via 5-fold cross-validation on `X_train`):
  - `classifier__n_estimators`: 100
  - `classifier__max_depth`: 2
  - `classifier__learning_rate`: 0.03
  - `classifier__subsample`: 0.8
  - `classifier__colsample_bytree`: 0.8
  - `classifier__eval_metric`: "logloss"
  - `classifier__random_state`: 42

---

## 2. Intended Use

### Intended Use Cases
- **Educational Demonstrator**: Designed strictly for academic engineering education, machine learning interpretability research, and demonstrating safety architectures in health AI.
- **Explainability Research**: Illustrating how `shap.TreeExplainer` decomposes physiological risk factors on tabular clinical records.
- **Clinical AI Safety Exploration**: Benchmarking keyword-based emergency escalation and pre-LLM guardrail systems.

### Out-of-Scope & Prohibited Uses
- **Clinical Diagnostics**: The model must **never** be used as a standalone diagnostic tool or medical triage device.
- **Acute Patient Care**: Prohibited from use in active emergency rooms, hospital wards, or clinical outpatient offices.
- **Pediatric Populations**: Not designed or evaluated on individuals under 18 years of age (training cohort age range: 29 to 77 years).
- **Out-of-Distribution Populations**: Individuals whose physiological parameters fall outside the ranges observed in the Cleveland dataset or who have congenital cardiovascular disorders.

---

## 3. Training & Preprocessing Data

- **Data Source**: Cleveland Clinic Foundation Heart Disease Dataset, donor David W. Aha, UC Irvine Machine Learning Repository.
- **Reference**: Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. The American Journal of Cardiology, 64(5), 304–310.
- **Total Dataset Size**: $N = 303$ patient records.
- **Target Variable**: Binary angiographic coronary disease (`0` = absence of angiographic stenosis, `1` = presence of $\ge 50\%$ diameter stenosis in at least one major coronary artery).
- **Split Strategy**: Stratified 80/20 train/test split (`random_state=42`):
  - **Training Partition**: $N = 242$ (111 diseased [45.87%], 131 healthy [54.13%]).
  - **Held-Out Test Partition**: $N = 61$ (28 diseased [45.90%], 33 healthy [54.10%]).
- **Leakage Prevention**: All imputation statistics (numeric medians, categorical modes) and scaling parameters (`StandardScaler` means and standard deviations) were fitted **strictly on `X_train`** and applied downstream to `X_test`.
- **Model Selection Integrity**: **The held-out test set ($N=61$) was never accessed during model exploration, algorithm comparison, or hyperparameter grid searching.** Tuning was conducted exclusively within 5-fold cross-validation loops on `X_train`.

---

## 4. Evaluation Data & Quantitative Performance

### A. Held-Out Test Set Metrics ($N=61$)
Evaluated once on the final serialized pipeline. Confidence intervals were computed via $B = 2,000$ bootstrap resamples (percentile method):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.9578** | [0.9022, 0.9925] |
| **Accuracy** | **0.8852** (54 / 61) | [0.8033, 0.9508] |
| **Precision** | **0.8621** (25 / 29) | [0.7241, 0.9688] |
| **Recall (Sensitivity)** | **0.8929** (25 / 28) | [0.7600, 1.0000] |
| **F1-Score** | **0.8772** | [0.7666, 0.9552] |
| **Brier Score** | **0.0970** | [0.0666, 0.1336] |

*Clinical Trade-Off Analysis: At the default 0.50 decision threshold, the model achieved a sensitivity of 89.29%, missing 3 diseased patients out of 28 (false negatives), while yielding 4 false positives out of 33 healthy patients (specificity = 87.88%).*

### B. Probability Calibration Analysis
Probability calibration measures whether predicted probabilities correspond to true empirical disease prevalence:

| Calibration Parameter | 5-Fold Cross-Validation ($N=242$, Primary) | Held-Out Test Set ($N=61$, Indicative) |
| :--- | :---: | :---: |
| **Model Brier Loss** | **0.1332** | **0.0970** |
| **Prevalence Baseline Brier** | 0.2483 | 0.2483 |
| **Brier Skill Score (BSS)** | **+46.36%** | **+60.95%** |
| **Expected Calibration Error (ECE)** | **8.19%** (8 uniform bins) | **11.25%** (5 uniform bins) |
| **Maximum Calibration Error (MCE)** | **22.46%** | **20.44%** |

*Tree Ensemble Probability Behavior: Tree-based gradient boosting models often exhibit slight sigmoidal probability distortions due to leaf-weight regularization. In our experiments, constraining tree depth to `max_depth=2` and applying conservative shrinkage (`learning_rate=0.03`) may have helped retain smooth, monotonic risk stratification without requiring post-hoc Platt scaling.*

---

## 5. Subgroup Performance & Fairness Analysis

To evaluate algorithmic fairness and identify demographic vulnerabilities, the model was evaluated across biological sex and age brackets. Any subgroup containing fewer than 20 patients ($N < 20$) or fewer than 10 positive cases is explicitly flagged as statistically unreliable:

### A. 5-Fold Out-of-Fold Subgroups on Training Data ($N=242$, Out-of-Sample)

| Demographic Subgroup | Sample Size ($N$) | Positives (Prevalence) | Accuracy | Precision | Recall | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Male (`sex=1`)** | $N=165$ | 93 / 165 (56.4%) | 0.8121 | 0.8523 | 0.8065 | 0.8792 | 0.1425 |
| **Female (`sex=0`)** | $N=77$ | 18 / 77 (23.4%) | 0.8701 | 0.7500 | 0.6667 | 0.8653 | 0.1133 |
| **Age: $< 55$ years** | $N=115$ | 37 / 115 (32.2%) | 0.8696 | 0.8438 | 0.7297 | 0.9182 | 0.1068 |
| **Age: $55 - 64$ years** | $N=92$ | 57 / 92 (62.0%) | 0.7609 | 0.8302 | 0.7719 | 0.8261 | 0.1693 |
| **Age: $\ge 65$ years** | $N=35$ | 17 / 35 (48.6%) | 0.8857 | 0.8421 | 0.9412 | 0.9346 | 0.1251 |

### B. Held-Out Test Set Subgroups ($N=61$)

| Demographic Subgroup | Sample Size ($N$) | Positives (Prevalence) | Accuracy | Precision | Recall | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Male (`sex=1`)** | $N=41$ | 21 / 41 (51.2%) | 0.8780 | 0.8333 | 0.9524 | 0.9476 | 0.1113 |
| **Female (`sex=0`)** `*` | $N=20$ | 7 / 20 (35.0%) | 0.9000 | 1.0000 | 0.7143 | 0.9890 | 0.0676 |
| **Age: $< 55$ years** `*` | $N=28$ | 7 / 28 (25.0%) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0552 |
| **Age: $55 - 64$ years** | $N=27$ | 18 / 27 (66.7%) | 0.8148 | 0.8824 | 0.8333 | 0.9074 | 0.1239 |
| **Age: $\ge 65$ years** `*` | $N=6$ | 3 / 6 (50.0%) | 0.6667 | 0.6000 | 1.0000 | 1.0000 | 0.1707 |

`*` *Flagged as statistically unreliable due to low subgroup sample size ($N < 20$) or low positive event count ($N_{\text{pos}} < 10$).*

### Ethical & Clinical Fairness Observations:
- **Sex Representation Disparity**: Women comprise only 31.8% of the training partition ($N=77/242$) and 32.8% of the test partition ($N=20/61$), reflecting 1980s cardiac catheterization referral patterns.
- **Recall Discrepancy**: Slicing out-of-fold predictions on training data shows a possible lower recall in women ($66.67\%$ in women vs $80.65\%$ in men; $71.43\%$ vs $95.24\%$ on the test set), though this is not statistically conclusive given the small sample of diseased women ($N=18$ in CV, $N=7$ in test). In clinical practice, any potential under-detection in women reinforces the imperative that automated screening tools must never override clinical assessment.

---

## 6. Risk Tier Definitions & Clinical Disclaimer

The application displays a 3-segment risk gauge based on predicted probabilities:
- **Low Risk Tier**: Predicted probability $< 35\%$
- **Intermediate Risk Tier**: Predicted probability $35\% \le p < 65\%$
- **Elevated Risk Tier**: Predicted probability $\ge 65\%$

### Critical Policy & Operational Notice:
> **THESE THRESHOLDS ARE HEURISTIC ARTIFACTS OF THIS APPLICATION.**
> The 35% and 65% cut-offs were established solely for user interface partitioning and educational demonstration. **They are NOT clinical guideline thresholds (such as AHA/ACC, ESC, or ASCVD guidelines).** They do not represent validated medical decision boundaries and must never be used to initiate, alter, or withhold medical treatment.

---

## 7. Caveats, Limitations & Recommendations

1. **Retrospective Single-Center Bias**: The data reflects a single hospital's patient demographics from the 1980s. Changes in dietary patterns, smoking rates, and medical technology limit generalizability to contemporary populations.
2. **Angiographic Narrowing vs. Clinical Risk**: The ground truth reflects anatomical coronary luminal diameter narrowing $\ge 50\%$. It does **not** evaluate microvascular angina, vulnerable non-stenotic plaque, or 10-year major adverse cardiac event (MACE) risk.
3. **Conversational Safety**: Pre-LLM safety guardrails rely on deterministic keyword filters. While effective for common emergency and diagnostic prompts, keyword systems cannot detect all adversarial prompt injections or nuanced, colloquial clinical descriptions.
4. **Mandatory Recommendation**: Any real-world deployment requires prospective recalibration, validation across multi-center electronic health records, and algorithmic fairness mitigation to address possible sex-based recall disparities.
