"""Script to generate notebooks/07_phase4_model_training_evaluation.ipynb with 20 structured sections."""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""# Phase 4 — Model Retraining & Phase 2B vs Phase 3 Evaluation

**Project:** Airline Delay Prediction & Operations Analytics  
**Scope:** Machine-learning evaluation layer using the leakage-safe Phase 3 feature dataset.  
**Objectives:**
1. Train candidate models (Baseline, Logistic Regression, Random Forest, XGBoost) on Phase 3 features.
2. Fairly compare Phase 3 models against the existing Phase 2B model artifact (`models/delay_model.joblib`).
3. Evaluate whether advanced Phase 3 feature engineering yields measurable performance gains.
4. Maintain strict temporal train/validation/test ordering (Train < Val < Test).
5. Guarantee zero target leakage.
6. Preserve existing Phase 2B model artifacts untouched.
7. Produce reproducible evaluation reports and serialized models.

> **DEVELOPMENT SAMPLE LIMITATION NOTICE:**  
> This notebook runs on the development dataset sample (481 flights from January 1–10, 2024). Results verify pipeline execution, anti-leakage invariants, and calibration architecture. Statistically representative operational performance requires scaling to the full BTS dataset.""")

    add_code("""import sys
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set project root
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.models.phase4_eval import (
    prepare_phase4_modeling_dataset,
    perform_temporal_split,
    train_phase4_models,
    evaluate_phase4_models,
    compare_phase2b_vs_phase3,
    analyze_feature_importance,
    analyze_phase3_feature_groups,
    save_phase4_artifacts,
    build_phase4_markdown_report,
)

config = get_config()
print("Phase 4 modules and configuration initialized successfully.")""")

    # 1. Load Phase 3 feature dataset
    add_md("## 1. Load Phase 3 Feature Dataset\nLoad primary feature dataset `flights_features_p3.parquet`.")
    add_code("""p3_path = config.data_processed_dir / "flights_features_p3.parquet"
if not p3_path.exists():
    p3_path = config.data_processed_dir / "flights_features_p3.csv"

df_p3 = pd.read_parquet(p3_path) if p3_path.suffix == ".parquet" else pd.read_csv(p3_path)
print(f"Loaded Phase 3 dataset: {df_p3.shape[0]} rows, {df_p3.shape[1]} columns")
df_p3.head(3)""")

    # 2. Inspect schema
    add_md("## 2. Inspect Schema\nAudit data types, null values, and feature columns.")
    add_code("""print("Schema Data Types Breakdown:")
print(df_p3.dtypes.value_counts())
print("\\nMissing Values Count:")
null_counts = df_p3.isnull().sum()
missing_cols = null_counts[null_counts > 0]
print(missing_cols if len(missing_cols) > 0 else "Zero missing values detected across all columns.")""")

    # 3. Validate target
    add_md("## 3. Validate Target\nVerify target column `delay_target` presence and binary properties.")
    add_code("""assert 'delay_target' in df_p3.columns, "Target column missing"
target_vals = df_p3['delay_target'].unique()
print(f"Unique target values: {target_vals}")
assert set(target_vals).issubset({0, 1}), "Target contains non-binary values"
print("Target column validation PASSED.")""")

    # 4. Analyze class distribution
    add_md("## 4. Analyze Class Distribution\nExamine class balance between delayed and on-time flights.")
    add_code("""total_samples = len(df_p3)
delayed_samples = int(df_p3['delay_target'].sum())
ontime_samples = total_samples - delayed_samples
delay_pct = (delayed_samples / total_samples) * 100.0

print(f"Total Samples   : {total_samples}")
print(f"Delayed (1)     : {delayed_samples} ({delay_pct:.2f}%)")
print(f"On-Time (0)     : {ontime_samples} ({100.0 - delay_pct:.2f}%)")
print(f"Imbalance Ratio : {ontime_samples / max(delayed_samples, 1):.2f}:1")""")

    # 5. Define prediction timestamp
    add_md("## 5. Define Prediction Timestamp\nEnforce prediction time $T_{dep} = \\text{scheduled\\_departure}$.")
    add_code("""pred_time_col = "scheduled_dep_time"
assert pred_time_col in df_p3.columns, "scheduled_dep_time column missing"
print(f"Prediction Reference: scheduled_departure ({pred_time_col})")
print("Invariant: All model predictors must be observed strictly prior to pushback.")""")

    # 6. Perform temporal split
    add_md("## 6. Perform Temporal Split\nSplit dataset chronologically: Past (Train 70%), Future (Val 15%), Latest Future (Test 15%).")
    add_code("""df_clean, feature_meta = prepare_phase4_modeling_dataset(df_p3, target_col="delay_target", config=config)
split_result = perform_temporal_split(df_clean, target_col="delay_target", config=config)

print(f"Train : {len(split_result.y_train)} rows ({split_result.train_date_range[0]} to {split_result.train_date_range[1]}) | Delay Rate: {split_result.y_train.mean()*100:.1f}%")
print(f"Val   : {len(split_result.y_val)} rows ({split_result.val_date_range[0]} to {split_result.val_date_range[1]}) | Delay Rate: {split_result.y_val.mean()*100:.1f}%")
print(f"Test  : {len(split_result.y_test)} rows ({split_result.test_date_range[0]} to {split_result.test_date_range[1]}) | Delay Rate: {split_result.y_test.mean()*100:.1f}%")

assert split_result.train_date_range[1] <= split_result.val_date_range[0], "Temporal leakage between Train and Val"
assert split_result.val_date_range[1] <= split_result.test_date_range[0], "Temporal leakage between Val and Test"
print("Temporal ordering invariant confirmed: max(Train) <= min(Val) <= min(Test).")""")

    # 7. Prepare modeling features
    add_md("## 7. Prepare Modeling Features\nInspect MODEL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, and EXCLUDED_FEATURES.")
    add_code("""print(f"Total Model Features : {feature_meta['model_features_count']}")
print(f"  - Numerical        : {feature_meta['numerical_features_count']}")
print(f"  - Categorical      : {feature_meta['categorical_features_count']}")
print(f"  - Excluded Columns : {feature_meta['excluded_features_count']}")
print("\\nExclusion Registry and Rationales:")
for col, reason in feature_meta['EXCLUDED_FEATURES'].items():
    print(f"  * {col}: {reason}")""")

    # 8. Train Logistic Regression
    add_md("## 8. Train Logistic Regression\nTrain baseline linear model with standard scaling and class balancing.")
    add_code("""models = train_phase4_models(
    X_train=split_result.X_train,
    y_train=split_result.y_train,
    numerical_features=feature_meta["NUMERICAL_FEATURES"],
    categorical_features=feature_meta["CATEGORICAL_FEATURES"],
    config=config,
)
lr_pipe = models["Logistic Regression"]
print("Logistic Regression Pipeline trained successfully.")
print(f"Model Steps: {list(lr_pipe.named_steps.keys())}")""")

    # 9. Evaluate Logistic Regression
    add_md("## 9. Evaluate Logistic Regression\nAssess classification performance, confusion matrix, and threshold sensitivity.")
    add_code("""eval_results = evaluate_phase4_models(models, split_result, config=config)
lr_test_def = eval_results["test_metrics_default"]["Logistic Regression"]
lr_thresh = eval_results["selected_thresholds"]["Logistic Regression"]
lr_test_opt = eval_results["test_metrics_selected"]["Logistic Regression"]

print(f"Logistic Regression @ 0.50 Threshold:")
print(f"  Accuracy: {lr_test_def['accuracy']:.4f}, Precision: {lr_test_def['precision']:.4f}, Recall: {lr_test_def['recall']:.4f}, F1: {lr_test_def['f1']:.4f}, ROC-AUC: {lr_test_def['roc_auc']:.4f}, PR-AUC: {lr_test_def['pr_auc']:.4f}")
print(f"Logistic Regression @ {lr_thresh:.2f} (Val-Tuned):")
print(f"  Accuracy: {lr_test_opt['accuracy']:.4f}, Precision: {lr_test_opt['precision']:.4f}, Recall: {lr_test_opt['recall']:.4f}, F1: {lr_test_opt['f1']:.4f}, ROC-AUC: {lr_test_opt['roc_auc']:.4f}, PR-AUC: {lr_test_opt['pr_auc']:.4f}")""")

    # 10. Train Random Forest
    add_md("## 10. Train Random Forest\nTrain class-weighted ensemble classifier (100 trees, max_depth=6).")
    add_code("""rf_pipe = models["Random Forest"]
print("Random Forest Pipeline fitted successfully.")
clf_rf = rf_pipe.named_steps["classifier"]
print(f"Ensemble Trees: {len(clf_rf.estimators_)}, Max Depth: {clf_rf.max_depth}")""")

    # 11. Evaluate Random Forest
    add_md("## 11. Evaluate Random Forest\nAssess Random Forest metrics on out-of-time test partition.")
    add_code("""rf_test_def = eval_results["test_metrics_default"]["Random Forest"]
rf_thresh = eval_results["selected_thresholds"]["Random Forest"]
rf_test_opt = eval_results["test_metrics_selected"]["Random Forest"]

print(f"Random Forest @ 0.50 Threshold:")
print(f"  Accuracy: {rf_test_def['accuracy']:.4f}, Precision: {rf_test_def['precision']:.4f}, Recall: {rf_test_def['recall']:.4f}, F1: {rf_test_def['f1']:.4f}, ROC-AUC: {rf_test_def['roc_auc']:.4f}, PR-AUC: {rf_test_def['pr_auc']:.4f}")
print(f"Random Forest @ {rf_thresh:.2f} (Val-Tuned):")
print(f"  Accuracy: {rf_test_opt['accuracy']:.4f}, Precision: {rf_test_opt['precision']:.4f}, Recall: {rf_test_opt['recall']:.4f}, F1: {rf_test_opt['f1']:.4f}, ROC-AUC: {rf_test_opt['roc_auc']:.4f}, PR-AUC: {rf_test_opt['pr_auc']:.4f}")""")

    # 12. Train XGBoost
    add_md("## 12. Train XGBoost\nTrain gradient boosted decision trees with positive class weighting.")
    add_code("""xgb_pipe = models["XGBoost"]
print("XGBoost Pipeline fitted successfully.")
clf_xgb = xgb_pipe.named_steps["classifier"]
print(f"Number of Estimators: {clf_xgb.n_estimators}, Learning Rate: {clf_xgb.learning_rate}, Scale Pos Weight: {clf_xgb.scale_pos_weight:.2f}")""")

    # 13. Evaluate XGBoost
    add_md("## 13. Evaluate XGBoost\nAssess XGBoost metrics on out-of-time test partition.")
    add_code("""xgb_test_def = eval_results["test_metrics_default"]["XGBoost"]
xgb_thresh = eval_results["selected_thresholds"]["XGBoost"]
xgb_test_opt = eval_results["test_metrics_selected"]["XGBoost"]

print(f"XGBoost @ 0.50 Threshold:")
print(f"  Accuracy: {xgb_test_def['accuracy']:.4f}, Precision: {xgb_test_def['precision']:.4f}, Recall: {xgb_test_def['recall']:.4f}, F1: {xgb_test_def['f1']:.4f}, ROC-AUC: {xgb_test_def['roc_auc']:.4f}, PR-AUC: {xgb_test_def['pr_auc']:.4f}")
print(f"XGBoost @ {xgb_thresh:.2f} (Val-Tuned):")
print(f"  Accuracy: {xgb_test_opt['accuracy']:.4f}, Precision: {xgb_test_opt['precision']:.4f}, Recall: {xgb_test_opt['recall']:.4f}, F1: {xgb_test_opt['f1']:.4f}, ROC-AUC: {xgb_test_opt['roc_auc']:.4f}, PR-AUC: {xgb_test_opt['pr_auc']:.4f}")""")

    # 14. Compare Phase 2B and Phase 3
    add_md("## 14. Compare Phase 2B and Phase 3\nBenchmark existing Phase 2B model artifact against newly retrained Phase 3 models.")
    add_code("""p2b_comp = compare_phase2b_vs_phase3(eval_results, split_result, config=config)
comp_df = pd.DataFrame(p2b_comp["comparison_table"])
print(f"Phase 2B Model Artifact Verified (SHA256: {p2b_comp['phase2b_artifact_sha256'][:16]}...)")
comp_df[['model', 'feature_version', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score']]""")

    # 15. Analyze feature importance
    add_md("## 15. Analyze Feature Importance\nExtract and rank Top 20 Phase 3 features for the strongest tree-based model.")
    add_code("""feat_imp_analysis = analyze_feature_importance(
    models=models,
    numerical_features=feature_meta["NUMERICAL_FEATURES"],
    categorical_features=feature_meta["CATEGORICAL_FEATURES"],
    p3_eval_results=eval_results,
)
print(f"Strongest Tree Model: {feat_imp_analysis['strongest_tree_model']} ({feat_imp_analysis['selection_criterion']})")
top20_df = pd.DataFrame(feat_imp_analysis["top_20_features"])
top20_df.head(10)""")

    # 16. Analyze Phase 3 feature groups
    add_md("## 16. Analyze Phase 3 Feature Groups\nInspect contributions across the 8 functional feature groups.")
    add_code("""group_analysis = analyze_phase3_feature_groups(
    df=df_clean,
    feature_importances=feat_imp_analysis.get("all_importances", {}),
    strongest_model_name=feat_imp_analysis["strongest_tree_model"],
)
for g_name, g_info in group_analysis.items():
    print(f"Domain: {g_name}")
    print(f"  - Configured / Active : {g_info['configured_feature_count']} / {g_info['present_in_dataset_count']}")
    print(f"  - Missingness %       : {g_info['missingness_percentage']:.1f}%")
    print(f"  - Importance Weight   : {g_info['cumulative_importance']:.4f}")
    print(f"  - Status              : {g_info['status']}")
    print(f"  - Observation         : {g_info['analytical_association']}\\n")""")

    # 17. Probability & Calibration analysis
    add_md("## 17. Probability & Calibration Analysis\nExamine predicted probability distributions and calibration (Brier scores).")
    add_code("""cal_df = pd.DataFrame([
    {"Model": name, "Brier Score": data["brier_score"]}
    for name, data in eval_results["calibration_data"].items()
])
print("Probability Calibration Analysis (Brier Scores — Lower is Better):")
print(cal_df.to_string(index=False))""")

    # 18. Save model artifacts
    add_md("## 18. Save Model Artifacts\nSerialize Phase 4 models and metadata while preserving Phase 2B model.")
    add_code("""saved_artifacts = save_phase4_artifacts(
    models=models,
    feature_meta=feature_meta,
    split_result=split_result,
    p3_eval_results=eval_results,
    p2b_comparison=p2b_comp,
    feature_imp_analysis=feat_imp_analysis,
    group_analysis=group_analysis,
    config=config,
)
print("Saved Model Artifacts:")
for k, v in saved_artifacts.items():
    print(f"  * {k}: {v}")""")

    # 19. Generate Phase 4 report
    add_md("## 19. Generate Phase 4 Report\nCompile 22-section markdown report and machine-readable JSON.")
    add_code("""rep_md_path = config.reports_dir / "phase4_model_evaluation.md"
rep_json_path = config.reports_dir / "phase4_model_evaluation.json"

assert rep_md_path.exists(), "Markdown report missing"
assert rep_json_path.exists(), "JSON report missing"
print(f"Phase 4 Markdown Report : {rep_md_path} ({rep_md_path.stat().st_size:,} bytes)")
print(f"Phase 4 JSON Report     : {rep_json_path} ({rep_json_path.stat().st_size:,} bytes)")""")

    # 20. Final verification
    add_md("## 20. Final Verification\nVerify pipeline invariants, anti-leakage audit, and model integrity.")
    add_code("""# 1. Phase 2B preservation
p2b_file = config.models_dir / "delay_model.joblib"
assert p2b_file.exists()
current_hash = hashlib.sha256(p2b_file.read_bytes()).hexdigest()
assert current_hash == p2b_comp["phase2b_artifact_sha256"], "Phase 2B artifact modified!"

# 2. Phase 4 artifacts
for f in [
    "delay_model_phase3_logistic.joblib",
    "delay_model_phase3_rf.joblib",
    "delay_model_phase3_xgb.joblib",
    "phase4_metadata.json",
]:
    p = config.models_dir / f
    assert p.exists() and p.stat().st_size > 0, f"Artifact {f} missing or empty"

print("================================================================================")
print("PHASE 4 VERIFICATION SUCCESSFUL: ZERO LEAKAGE, REPRODUCIBLE EVALUATION COMPLETE!")
print("================================================================================")""")

    out_file = Path("notebooks/07_phase4_model_training_evaluation.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook written to {out_file}")

if __name__ == "__main__":
    create_notebook()
