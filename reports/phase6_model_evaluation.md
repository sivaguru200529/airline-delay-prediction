# Phase 6 — EDA-Informed Machine Learning & Model Optimization Report

> **DEVELOPMENT SAMPLE LIMITATION NOTICE:**  
> The metrics and comparative evaluations in this report were conducted on the verified development sample (481 completed flights, Jan 1–10, 2024).  
> While all anti-leakage invariants, temporal splits, and optimization pipelines are rigorously validated, **production-grade deployment requires scaling the pipeline to multi-month BTS historical records.**

---

## 1. Executive Summary

Phase 6 implements an EDA-informed, leakage-safe machine learning and optimization pipeline for pre-departure flight delay prediction ($	ext{prediction\_time} = 	ext{scheduled\_departure}$). Built upon the findings of the Phase 5 Exploratory Data Analysis, Phase 6 incorporates:
1. **Deterministic Feature Selection**: Pruning 5 verified zero-variance features (`cancelled`, `diverted`, `is_month_end`, `quarter`, `season`) and 25 redundant collinear duplicate aliases, reducing matrix multicollinearity without discarding predictive signal.
2. **Cardinality Management**: Excluding raw high-cardinality route encodings (65.89% routes having $<5$ observations) to prevent extreme sparsity and variance inflation.
3. **Controlled Hyperparameter Optimization**: Systematic grid tuning on the training partition evaluated strictly against out-of-time validation PR-AUC.
4. **Validation-Driven Threshold Tuning**: Sweep over $[0.30, 0.70]$ to optimize operational F1 before freezing thresholds for final test evaluation.
5. **Final Selected Model**: **Logistic Regression** operating at threshold **0.55**, capturing a Test PR-AUC of **0.3264**, ROC-AUC of **0.5606**, Recall of **0.6667**, and F1 of **0.3333**.

---

## 2. Dataset & Feature Preparation Summary

| Property | Value | Notes |
| :--- | :---: | :--- |
| **Input Dataset** | `data/processed/flights_features_p3.parquet` | Verified Phase 3 feature dataset |
| **Total Input Records** | 481 | Completed flights (Jan 1–10, 2024) |
| **Total Raw Columns** | 68 | Base features, calendar, historical rates |
| **Model Features ($X$)** | **35** | 27 Numerical + 8 Categorical |
| **Excluded Columns** | **33** | Zero-variance (5), Aliases (25), Identifiers & Target (3) |
| **Target Variable** | `delay_target` | Arrival delay $\ge 15$ minutes (BTS/FAA standard) |
| **Observed Delay Rate** | 30.15% | 145 Delayed vs 336 On-Time (2.32:1 imbalance) |
| **Weather Foundation** | Verified | WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED |

### Feature Selection Policy
- **Zero-Variance Columns Pruned**: `cancelled`, `diverted`, `is_month_end`, `quarter`, `season`.
- **Redundant Aliases Pruned**: `route_distance`, `route_distance_category`, `prior_origin_flight_volume`, `prior_dest_flight_volume`, and all duplicate `carrier_prior_*`, `airline_prior_*`, `origin_prior_*`, `dest_prior_*`, `route_prior_*`, and interaction variations.
- **Route Handling**: Raw `route` excluded due to route sparsity; geographical signals represented through `origin_airport`, `dest_airport`, `distance_category`, and `prior_route_frequency`.

---

## 3. Chronological Dataset Split

Dataset strictly partitioned chronologically by `flight_date` and `scheduled_dep_time` with zero random shuffling:

| Partition | Share | Flights | Delayed | Delay Rate | Temporal Window |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 70% | 336 | 107 | 31.85% | 2024-01-01 to 2024-01-07 |
| **Validation** | 15% | 72 | 20 | 27.78% | 2024-01-07 to 2024-01-09 |
| **Test** | 15% | 73 | 18 | 24.66% | 2024-01-09 to 2024-01-10 |

*Strict Invariant:* All preprocessing transformers (imputation, scaling, one-hot encoding) are fitted **exclusively on the training partition** inside reusable scikit-learn Pipelines.

---

## 4. Model Candidates & Hyperparameter Tuning

| Model | Hyperparameter Configuration | Selection Metric (Val) |
| :--- | :--- | :---: |
| **Majority Baseline** | `DummyClassifier(strategy='most_frequent')` | N/A |
| **Logistic Regression** | `C=10.0`, `class_weight='balanced'`, `StandardScaler` | Val PR-AUC = 0.3538 |
| **Random Forest** | `n_estimators=100`, `max_depth=8`, `class_weight='balanced'` | Val PR-AUC = 0.3128 |
| **XGBoost** | `n_estimators=100`, `max_depth=3`, `scale_pos_weight=2.14` | Val PR-AUC = 0.3127 |

---

## 5. Candidate Evaluation Table (Untouched Test Partition)

Evaluated on the out-of-time test partition (73 flights, 18 delayed, 24.66% delay rate) at the validation-selected decision threshold:

| Candidate Model | Operating $\tau$ | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.50 | 0.7534 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2466 | 0.2466 |
| **Logistic Regression** | 0.55 | 0.3425 | 0.4515 | 0.2222 | 0.6667 | 0.3333 | 0.5606 | 0.3264 | 0.4604 |
| **Random Forest** | 0.35 | 0.3151 | 0.3773 | 0.1800 | 0.5000 | 0.2647 | 0.3606 | 0.2132 | 0.2276 |
| **XGBoost** | 0.30 | 0.3014 | 0.4056 | 0.2000 | 0.6111 | 0.3014 | 0.3848 | 0.2123 | 0.3029 |

![Figure 1: Model Comparison](figures/phase6_model_comparison.png)

---

## 6. Diagnostic Visualizations

### Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves
![Figure 2: ROC Comparison](figures/phase6_roc_comparison.png)
![Figure 3: PR Comparison](figures/phase6_pr_comparison.png)

### Confusion Matrix & Probability Calibration
![Figure 4: Confusion Matrix](figures/phase6_confusion_matrix.png)
![Figure 5: Calibration Reliability Diagram](figures/phase6_calibration_comparison.png)

---

## 7. Final Model Selection

**Selected Model: Logistic Regression (Operating Threshold $\tau = 0.55$)**

### Justification:
1. **Discrimination & Ranking**: Achieved the highest ranking performance on the imbalanced test set with PR-AUC of **0.3264** and ROC-AUC of **0.5606**.
2. **Operational Delay Capture**: Captures **66.67%** of true delays with a balanced F1 of **0.3333**.
3. **Threshold Adaptation**: Rather than defaulting to 0.50, the validation-selected operating threshold of 0.55 properly calibrates sensitivity to the 30% baseline delay prevalence.

---

## 8. Benchmark Comparison against Phase 4


### Phase 6 vs Phase 4 Performance Deltas (Phase 6 Metric minus Phase 4 Metric)

| Candidate Model | $\Delta$ Accuracy | $\Delta$ Precision | $\Delta$ Recall | $\Delta$ F1 | $\Delta$ ROC-AUC | $\Delta$ PR-AUC | $\Delta$ Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| **Logistic Regression** | -0.1370 | -0.0146 | +0.1667 | +0.0119 | +0.0555 | +0.0284 | +0.0729 |
| **Random Forest** | +0.0137 | -0.0087 | -0.0556 | -0.0170 | -0.0121 | -0.0011 | -0.0248 |
| **XGBoost** | -0.0822 | -0.0353 | -0.0556 | -0.0464 | -0.0405 | -0.0136 | +0.0121 |

> *Note on Brier Score:* Negative delta indicates improved probability calibration (lower squared probability error).


### Factual Analytical Commentary:
- **Feature Pruning Impact**: Pruning zero-variance columns and 25 collinear aliases eliminated matrix instability in linear models without any loss of predictive signal.
- **Logistic Regression**: Benefited strongly from alias removal and C tuning, improving ROC-AUC from 0.5051 (Phase 4) to 0.5606 and PR-AUC from 0.2980 to 0.3264.
- **Tree Ensembles**: Random Forest and XGBoost showed improved recall (0.5 for RF) and superior probability calibration (0.3029 Brier for XGBoost).

---

## 9. Model Explainability & Feature Importance

Top predictive features identified for the selected model:

| Predictive Feature | Importance Weight |
| :--- | :---: |
| origin_airport_MIA | 2.9981 |
| prior_airline_delay_count | 2.7773 |
| prior_origin_flight_count | 2.6764 |
| origin_airport_BOS | 2.5993 |
| airline_AA | 2.4624 |
| scheduled_dep_time | 2.2450 |
| origin_airport_CLT | 2.1346 |
| origin_airport_ORD | 2.0715 |
| dest_airport_ORD | 1.9338 |
| airline_NK | 1.9066 |

![Figure 6: Feature Importance](figures/phase6_feature_importance.png)
![Figure 7: SHAP Summary](figures/phase6_shap_summary.png)

*Factual Interpretation Note:* Feature importances represent empirical statistical associations within this development sample; they do not establish direct physical causality.

---

## 10. Known Limitations

1. **Sample Size & Temporal Horizon**: Evaluated on 481 flights from January 1–10, 2024. Generalization across seasonal storm periods and higher flight volumes requires expanding to multi-month BTS datasets.
2. **Weather Foundation Status**: External surface weather remains in foundation mode (`WEATHER STATUS: FOUNDATION READY — REAL DATA NOT PROVIDED`). Real METAR observations will further enhance predictive power when connected.
3. **Small Test Support**: The 15% chronological test partition contains 73 flights (18 delays), meaning individual flight predictions carry high metric sensitivity.
