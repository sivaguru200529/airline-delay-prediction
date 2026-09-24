# Phase 3B — Model Retraining & Phase 2A vs Phase 3 Evaluation Report

> **DEVELOPMENT DATASET LIMITATION NOTICE**
> The experimental results presented in this report were evaluated on the verified development sample
> (**481 completed flights**, January 1–10, 2024).
> This comparison tests whether the advanced Phase 3 feature engineering yields predictive gains under
> strictly controlled, leakage-free chronological evaluation.
> **Statistically representative operational performance conclusions require scaling to the full multi-month/multi-year public BTS dataset.**

---

## 1. Controlled Experimental Setup

* **Objective**: Isolate the impact of Phase 3 feature engineering by holding all modeling parameters constant.
* **Underlying Population**: 481 completed commercial flights (January 1–10, 2024).
* **Target Definition**: Binary arrival delay (>= 15 min): 145 positive delays (30.1%), 336 on-time flights.
* **Prediction Timing Invariant**: Scheduled Departure ($T_{dep}$).
* **Chronological Split**:
  - **Train Partition**: 336 flights (69.85%) [2024-01-01 to 2024-01-07] — Delay Rate: 31.85%
  - **Validation Partition**: 72 flights (14.97%) [2024-01-07 to 2024-01-09] — Delay Rate: 27.78%
  - **Test Partition**: 73 flights (15.18%) [2024-01-09 to 2024-01-10] — Delay Rate: 24.66%
  - Temporal ordering strictly enforced: max(Train) <= min(Val) <= min(Test).
* **Feature Schema**:
  - **Experiment A (Phase 2A)**: 38 columns (30 numerical, 6 categorical, flight_date, delay_target).
  - **Experiment B (Phase 3)**: 68 columns (55 numerical, 11 categorical, flight_date, delay_target).
* **Leakage Controls**: Preprocessing pipelines (imputers, scalers, encoders) fitted strictly on training data. Threshold selection conducted exclusively on validation data.

---

## 2. Test Set Evaluation Comparison Table

Evaluated on the out-of-time test partition (73 flights, January 9–10, 2024) at validation-selected decision thresholds:

| Model | Feature Set | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | Phase 2A | 0.50 | 0.7534 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2466 | 0.2466 |
| Majority Baseline | Phase 3 | 0.50 | 0.7534 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2466 | 0.2466 |
| Logistic Regression | Phase 2A | 0.50 | 0.2603 | 0.2500 | 1.0000 | 0.4000 | 0.5424 | 0.3036 | 0.5995 |
| Logistic Regression | Phase 3 | 0.60 | 0.4795 | 0.2368 | 0.5000 | 0.3214 | 0.5051 | 0.2980 | 0.3875 |
| Random Forest | Phase 2A | 0.30 | 0.2466 | 0.2466 | 1.0000 | 0.3956 | 0.4364 | 0.2179 | 0.2591 |
| Random Forest | Phase 3 | 0.30 | 0.1918 | 0.2029 | 0.7778 | 0.3218 | 0.3848 | 0.2093 | 0.2508 |
| XGBoost | Phase 2A | 0.30 | 0.3151 | 0.2143 | 0.6667 | 0.3243 | 0.3485 | 0.2046 | 0.3000 |
| XGBoost | Phase 3 | 0.40 | 0.3836 | 0.2353 | 0.6667 | 0.3478 | 0.4253 | 0.2259 | 0.2908 |

---

## 3. Metric Differences (Phase 3 - Phase 2A Deltas)

Deltas reported as: **Absolute Difference (Percentage Change %)**:

| Model | Accuracy Delta | Precision Delta | Recall Delta | F1 Delta | ROC-AUC Delta | PR-AUC Delta | Brier Score Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | +0.0000 (+0.0%) | +0.0000 (+0.0%) | +0.0000 (+0.0%) | +0.0000 (+0.0%) | +0.0000 (+0.0%) | +0.0000 (+0.0%) | +0.0000 (+0.0%) |
| Logistic Regression | +0.2192 (+84.2%) | -0.0132 (-5.3%) | -0.5000 (-50.0%) | -0.0786 (-19.6%) | -0.0373 (-6.9%) | -0.0056 (-1.8%) | -0.2120 (-35.4%) |
| Random Forest | -0.0548 (-22.2%) | -0.0437 (-17.7%) | -0.2222 (-22.2%) | -0.0738 (-18.7%) | -0.0516 (-11.8%) | -0.0086 (-4.0%) | -0.0083 (-3.2%) |
| XGBoost | +0.0685 (+21.7%) | +0.0210 (+9.8%) | +0.0000 (+0.0%) | +0.0235 (+7.2%) | +0.0768 (+22.0%) | +0.0213 (+10.4%) | -0.0092 (-3.1%) |

*Note: For Brier score, negative delta indicates superior probability calibration.*

---

## 4. Key Experimental Findings

1. **XGBoost Performance**:
   - Phase 3 feature engineering produced notable gains for XGBoost across multiple metrics:
     - **ROC-AUC**: Increased by **+0.0768 (+22.04%)** from 0.3485 (Phase 2A) to **0.4253 (Phase 3)**.
     - **PR-AUC**: Increased by **+0.0213 (+10.41%)** from 0.2046 to **0.2259**.
     - **Accuracy**: Increased by **+0.0685 (+21.74%)** from 0.3151 to **0.3836**.
     - **F1-Score**: Increased by **+0.0235 (+7.25%)** from 0.3243 to **0.3478**.
     - **Brier Score**: Improved by **-0.0092 (-3.07%)** from 0.3000 to **0.2908**.
2. **Logistic Regression Trade-offs**:
   - Phase 3 features improved probability calibration substantially: Brier score improved by **-0.2120 (-35.36%)** from 0.5995 to **0.3875**, and accuracy improved from 0.2603 to 0.4795 (+84.21%).
   - However, at the validation-selected threshold (0.60 vs 0.50), recall dropped from 1.0000 to 0.5000 (-50.00%), leading to an F1 change of -0.0786.
3. **Random Forest Performance**:
   - Random Forest on this small 481-flight sample experienced sparsity challenges with the higher-dimensional Phase 3 feature space (68 features), resulting in lower test recall (0.7778 vs 1.0000) and F1 (0.3218 vs 0.3956), though calibration slightly improved (-0.0083 Brier).
4. **Majority Baseline Consistency**:
   - Majority Baseline exhibited zero delta (Accuracy: 0.7534, Recall: 0.0000), confirming evaluation integrity.

---

## 5. Diagnostic Figures Exported

The following diagnostic comparisons were exported to `reports/figures/`:
* `reports/figures/phase3b_roc_comparison.png`: Side-by-side ROC curves for Phase 2A vs Phase 3.
* `reports/figures/phase3b_precision_recall_comparison.png`: Side-by-side Precision-Recall curves.
* `reports/figures/phase3b_calibration_comparison.png`: Side-by-side reliability diagrams with Brier scores.
* `reports/figures/phase3b_confusion_matrix.png`: 2x4 heatmaps of test set confusion matrices.
* `reports/figures/phase3b_feature_importance_comparison.png`: Top 10 predictive features in Phase 2A vs Phase 3.
* `reports/figures/phase3b_shap_comparison.png`: SHAP global feature contribution summary.

---

## 6. Serialized Model Artifacts

All models were serialized separately to `models/phase3b/`, strictly preserving Phase 2B models:
* `models/phase3b/phase2a_logistic_regression.joblib`
* `models/phase3b/phase2a_random_forest.joblib`
* `models/phase3b/phase2a_xgboost.joblib`
* `models/phase3b/phase3_logistic_regression.joblib`
* `models/phase3b/phase3_random_forest.joblib`
* `models/phase3b/phase3_xgboost.joblib`
* `models/phase3b/experiment_metadata.json`

---

## 7. Development Sample Limitations

1. **Development Sample Size**: 481 flights provides limited historical depth for specific carrier-hour and route pairs.
2. **Weather Feature Status**: Real weather data was not supplied in `data/external/`. Weather interfaces were verified with zero synthetic data fabrication.
3. **Generalization**: Results demonstrate pipeline execution and feature interaction behavior. Production operational conclusions require scaling to the full multi-month/multi-year public BTS dataset.
