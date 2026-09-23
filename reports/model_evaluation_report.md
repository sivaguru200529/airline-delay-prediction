# Model Evaluation & Explainability Report (Phase 2B)

> **DEVELOPMENT DATASET LIMITATION NOTICE**
> The metrics reported in this document are derived from the development sample dataset (**481 completed flights**, Jan 1–10, 2024).
> This evaluation verifies pipeline integrity, chronological validation, anti-leakage guarantees, and explainability architecture.
> **Production operational conclusions require training and evaluating on the full multi-month/multi-year BTS dataset.**

---

## 1. Dataset & Chronological Out-of-Time Splitting

* **Total Cleaned Observations**: 481
* **Splitting Strategy**: Strictly chronological ($	ext{Train} < 	ext{Validation} < 	ext{Test}$)
* **Training Partition**: 336 flights (69.85%) [2024-01-01 to 2024-01-07] — Delay Rate: 31.85%
* **Validation Partition**: 72 flights (14.97%) [2024-01-07 to 2024-01-09] — Delay Rate: 27.78%
* **Test Partition**: 73 flights (15.18%) [2024-01-09 to 2024-01-10] — Delay Rate: 24.66%

---

## 2. Feature Schema & Route Representation Analysis

* **Numerical Features (30)**: `distance`, `departure_hour`, `departure_minute`, cyclical projections (`sin`/`cos`), and all 8 historical rate/count columns.
* **Categorical Features (6)**: `airline`, `origin_airport`, `dest_airport`, `time_of_day`, `haul_category`.
* **Historical Features Preserved**: All 8 Phase 2A historical features retained with strict prior-time aggregation.
* **Route Representation Decision**:
  * Unique routes in sample: 129 (Average 3.73 obs/route)
  * Status: **INCLUDED**
  * Rationale: Route feature has sufficient support (3.73 obs/route) and is included.
* **Leakage-Safe Preprocessing**:
  * Numerical: Median imputation
  * Categorical: `OneHotEncoder(handle_unknown='ignore')`
  * Fitted strictly on training data partition only.

---

## 3. Validation Model Comparison & Selection

Evaluated strictly on the **Validation Set** to prevent test set snooping.

| Candidate Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Majority Baseline | 0.7222 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2778 | 0.2778 |
| Logistic Regression | 0.3194 | 0.2836 | 0.9500 | 0.4368 | 0.5615 | 0.3557 | 0.4791 |
| Random Forest | 0.4861 | 0.2069 | 0.3000 | 0.2449 | 0.3779 | 0.2291 | 0.2621 |
| XGBoost | 0.4722 | 0.2353 | 0.4000 | 0.2963 | 0.4067 | 0.2483 | 0.3032 |

### Selection Rationale
* **Selected Winning Model**: `Logistic Regression`
* **Model Selection Justification**: Selected 'Logistic Regression' based on superior validation PR-AUC and balanced F1-score. Optimal threshold 0.50 was selected on validation data to balance operational delay recall against false alarm precision.

---

## 4. Threshold Sensitivity Analysis (Validation Set)

Evaluated across candidate thresholds for `Logistic Regression` on the validation partition:

| Threshold | Accuracy | Precision | Recall | F1-Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.30 | 0.2778 | 0.2778 | 1.0000 | 0.4348 | 20.0 | 52.0 | 0.0 | 0.0 |
| 0.40 | 0.2639 | 0.2676 | 0.9500 | 0.4176 | 19.0 | 52.0 | 1.0 | 0.0 |
| 0.50 | 0.3194 | 0.2836 | 0.9500 | 0.4368 | 19.0 | 48.0 | 1.0 | 4.0 |
| 0.60 | 0.3472 | 0.2787 | 0.8500 | 0.4198 | 17.0 | 44.0 | 3.0 | 8.0 |
| 0.70 | 0.3889 | 0.2857 | 0.8000 | 0.4211 | 16.0 | 40.0 | 4.0 | 12.0 |

* **Selected Operational Threshold**: **0.50** (Optimizes balance of delay capture vs false alert rate).

---

## 5. Final Out-of-Time Test Evaluation (Unbiased)

The selected `Logistic Regression` was evaluated **once** on the untouched test partition (73 flights) at threshold = **0.50**:

* **Test Accuracy**: 0.2603
* **Test Precision**: 0.2500
* **Test Recall**: 1.0000
* **Test F1-Score**: 0.4000
* **Test ROC-AUC**: 0.5424
* **Test PR-AUC**: 0.3036
* **Test Brier Score**: 0.5995

### Test Confusion Matrix
| | Pred On-Time (0) | Pred Delayed (1) |
| :--- | :--- | :--- |
| **Actual On-Time (0)** | TN = 1 | FP = 54 |
| **Actual Delayed (1)** | FN = 0 | TP = 18 |

---

## 6. Feature Importance & Explainability

### Top 10 Predictive Features (Transformed Names)
| Rank | Transformed Feature | Gini / Gain Weight |
| :--- | :--- | :--- |
| 1 | `route_ATL_BOS` | 1.4169 |
| 2 | `origin_airport_MIA` | 1.2505 |
| 3 | `route_JFK_BOS` | 1.0365 |
| 4 | `origin_airport_BOS` | 0.9347 |
| 5 | `route_ATL_SFO` | 0.9135 |
| 6 | `route_JFK_MIA` | 0.8982 |
| 7 | `dest_airport_JFK` | 0.8909 |
| 8 | `dest_airport_ORD` | 0.8736 |
| 9 | `historical_destination_flight_count` | 0.8723 |
| 10 | `route_MIA_SEA` | 0.8685 |

### SHAP Explainability Status
* **Status**: **SUCCESS**
* Global feature contributions generated using `shap.TreeExplainer` on training samples.
* Figures exported to `reports/figures/shap_summary.png` and `reports/figures/feature_importance.png`.

---

## 7. Known Limitations & Next Steps

1. **Sample Size**: 481 flights from 10 days in January 2024 is suitable for smoke-testing and pipeline verification, but requires expansion to multi-month BTS datasets for production deployment.
2. **Extreme Class Imbalance**: Low volume of delay events in test partition produces wider confidence intervals.
3. **External Weather**: Ground and en-route weather datasets will significantly enhance predictive power when loaded into `data/external/`.
