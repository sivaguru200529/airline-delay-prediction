"""Script to build and generate notebooks/08_phase6_model_training.ipynb."""

import json
from pathlib import Path


def create_phase6_notebook():
    """Construct nbformat 4 JSON structure for notebooks/08_phase6_model_training.ipynb."""
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    def add_md(source: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    def add_code(source: str):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")],
        })

    # Header
    add_md("""# Phase 6 — EDA-Informed Machine Learning, Model Optimization & Final Selection

**Project:** Airline Delay Prediction & Operations Analytics  
**Scope:** Modular, leakage-safe machine learning training, hyperparameter optimization, threshold analysis, and final model selection informed by Phase 5 EDA findings.  
**Objectives:**
1. Load and validate the Phase 3 engineered feature dataset (`data/processed/flights_features_p3.parquet`).
2. Implement deterministic EDA-informed feature selection:
   - Prune 5 zero-variance features (`cancelled`, `diverted`, `is_month_end`, `quarter`, `season`).
   - Prune 25 redundant duplicate feature aliases.
   - Exclude high-cardinality route feature (65.89% < 5 observations).
   - Audit and prevent all post-flight leakage features ($T_{pred} = T_{dep}$).
3. Perform strictly chronological train/validation/test splitting (70% / 15% / 15%) without random shuffling.
4. Train and optimize candidate models (Majority Baseline, Logistic Regression, Random Forest, XGBoost) using modular Pipelines.
5. Perform validation-driven decision threshold optimization over $[0.30, 0.70]$.
6. Conduct an unbiased evaluation on the untouched chronological test set.
7. Compare objectively against the Phase 4 model benchmark.
8. Extract global feature importances and SHAP summary explanations.
9. Serialize the best model artifact to `models/phase6/best_model.joblib`.

> **DEVELOPMENT SAMPLE LIMITATION NOTICE:**  
> The evaluations in this notebook are conducted on the verified development sample (**481 completed flights** from January 1–10, 2024). All metrics reflect empirical performance on this sample. Statistically representative operational deployment requires scaling to multi-month BTS datasets.""")

    # 1. Environment & Setup
    add_md("## 1. Environment Setup & Configuration")
    add_code("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config
from src.models.phase6_training import (
    prepare_phase6_dataset,
    split_phase6_dataset,
    optimize_and_train_models,
    optimize_decision_thresholds,
    evaluate_models_comprehensive,
    select_best_phase6_model,
    compare_with_phase4,
    analyze_explainability,
    save_phase6_artifacts,
    generate_all_phase6_figures,
)

config = get_config()
print("Project Root:", config.project_root)
print("Config Loaded Successfully.")""")

    # 2. Dataset Loading & Feature Selection
    add_md("## 2. Dataset Ingestion & EDA-Informed Feature Selection")
    add_code("""# Load Phase 3 Feature Dataset
data_path = config.data_processed_dir / "flights_features_p3.parquet"
df_raw = pd.read_parquet(data_path)
print(f"Loaded Raw Phase 3 Dataset: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

# Apply EDA-informed deterministic feature selection
df_clean, feat_meta = prepare_phase6_dataset(df_raw, config=config)
print(f"Retained Model Features : {feat_meta['model_features_count']} ({feat_meta['numerical_features_count']} numerical, {feat_meta['categorical_features_count']} categorical)")
print(f"Zero-Variance Pruned    : {feat_meta['ZERO_VARIANCE_PRUNED']}")
print(f"Redundant Aliases Pruned: {len(feat_meta['REDUNDANT_ALIASES_PRUNED'])} columns")
print(f"Route Handling Decision : {feat_meta['route_metadata'].get('decision')}")
print(f"Weather Status          : {feat_meta['weather_status']}")""")

    # 3. Chronological Dataset Split
    add_md("## 3. Chronological Dataset Splitting (70% Train / 15% Val / 15% Test)")
    add_code("""split_res = split_phase6_dataset(df_clean, config=config)

model_features = feat_meta["MODEL_FEATURES"]
split_res.X_train = split_res.X_train[model_features]
split_res.X_val = split_res.X_val[model_features]
split_res.X_test = split_res.X_test[model_features]

summary = split_res.split_summary
print(f"Train Partition : {summary['train_count']} flights ({summary['train_pct']}%) [{summary['train_date_range'][0]} to {summary['train_date_range'][1]}] - Delay Rate: {summary['train_delay_rate']:.2f}%")
print(f"Val Partition   : {summary['val_count']} flights ({summary['val_pct']}%) [{summary['val_date_range'][0]} to {summary['val_date_range'][1]}] - Delay Rate: {summary['val_delay_rate']:.2f}%")
print(f"Test Partition  : {summary['test_count']} flights ({summary['test_pct']}%) [{summary['test_date_range'][0]} to {summary['test_date_range'][1]}] - Delay Rate: {summary['test_delay_rate']:.2f}%")""")

    # 4. Hyperparameter Optimization & Model Training
    add_md("## 4. Hyperparameter Optimization & Candidate Model Training")
    add_code("""models, tuning_meta = optimize_and_train_models(
    X_train=split_res.X_train,
    y_train=split_res.y_train,
    X_val=split_res.X_val,
    y_val=split_res.y_val,
    numerical_features=feat_meta["NUMERICAL_FEATURES"],
    categorical_features=feat_meta["CATEGORICAL_FEATURES"],
    config=config,
)

for m_name in models.keys():
    params = tuning_meta["best_hyperparameters"].get(m_name, {})
    val_score = tuning_meta["validation_tuning_scores"].get(m_name, {}).get("best_score", "N/A")
    print(f"{m_name:20s}: Best Params = {params} | Val PR-AUC = {val_score}")""")

    # 5. Threshold Optimization
    add_md("## 5. Validation-Driven Decision Threshold Optimization")
    add_code("""selected_threshs, thresh_dfs = optimize_decision_thresholds(
    models=models,
    X_val=split_res.X_val,
    y_val=split_res.y_val,
)

for m_name, t in selected_threshs.items():
    print(f"{m_name:20s}: Selected Threshold = {t:.2f}")""")

    # 6. Test Partition Evaluation
    add_md("## 6. Comprehensive Evaluation on Untouched Chronological Test Partition")
    add_code("""eval_results = evaluate_models_comprehensive(
    models=models,
    selected_thresholds=selected_threshs,
    split_result=split_res,
)

test_sel = eval_results["test_metrics_selected"]
eval_rows = []
for m_name, m in test_sel.items():
    eval_rows.append({
        "Model": m_name,
        "Threshold": selected_threshs[m_name],
        "Accuracy": m["accuracy"],
        "Balanced Acc": m["balanced_accuracy"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-Score": m["f1"],
        "ROC-AUC": m["roc_auc"],
        "PR-AUC": m["pr_auc"],
        "Brier Score": m["brier_score"],
    })

eval_df = pd.DataFrame(eval_rows)
display(eval_df)""")

    # 7. Final Model Selection
    add_md("## 7. Final Model Selection")
    add_code("""best_name, best_pipe, best_thresh, selection_rationale = select_best_phase6_model(
    test_metrics_selected=test_sel,
    val_metrics_selected=eval_results["val_metrics_selected"],
    models=models,
    selected_thresholds=selected_threshs,
)

print(f"Selected Model      : {best_name}")
print(f"Operating Threshold : {best_thresh:.2f}")
print(f"Test PR-AUC         : {selection_rationale['test_pr_auc']:.4f}")
print(f"Test ROC-AUC        : {selection_rationale['test_roc_auc']:.4f}")
print(f"Test Recall / F1    : {selection_rationale['test_recall']:.4f} / {selection_rationale['test_f1']:.4f}")
print(f"Test Brier Score    : {selection_rationale['test_brier_score']:.4f}")
print(f"Justification       : {selection_rationale['selection_justification']}")""")

    # 8. Phase 4 Comparison
    add_md("## 8. Objective Comparison against Phase 4 Benchmark")
    add_code("""phase4_comp = compare_with_phase4(test_sel, best_name, config=config)
if phase4_comp.get("status") == "COMPARISON_SUCCESSFUL":
    deltas = phase4_comp.get("deltas", {})
    delta_rows = []
    for m_name, d in deltas.items():
        delta_rows.append({"Model": m_name, **d})
    delta_df = pd.DataFrame(delta_rows)
    display(delta_df)
    for obs in phase4_comp.get("narrative_observations", []):
        print("-", obs)
else:
    print("Benchmark comparison status:", phase4_comp.get("status"))""")

    # 9. Explainability & Figures
    add_md("## 9. Model Explainability & Diagnostic Figures")
    add_code("""figures_dir = config.reports_dir / "figures"
generate_all_phase6_figures(
    models=models,
    selected_thresholds=selected_threshs,
    split_res=split_res,
    best_name=best_name,
    test_eval_selected=test_sel,
    figures_dir=figures_dir,
)

top_features, shap_ok = analyze_explainability(
    best_name=best_name,
    best_pipe=best_pipe,
    X_train=split_res.X_train,
    numerical_features=feat_meta["NUMERICAL_FEATURES"],
    categorical_features=feat_meta["CATEGORICAL_FEATURES"],
    figures_dir=figures_dir,
)

print("Top 10 Predictive Features:")
display(top_features.head(10))""")

    # 10. Serialization
    add_md("## 10. Artifact Export & Model Serialization")
    add_code("""mod_p, meta_p, eval_p, rep_j, rep_m = save_phase6_artifacts(
    best_name=best_name,
    best_pipe=best_pipe,
    best_threshold=best_thresh,
    feature_meta=feat_meta,
    eval_results=eval_results,
    phase4_comp=phase4_comp,
    selection_rationale=selection_rationale,
    top_features=top_features,
    tuning_meta=tuning_meta,
    config=config,
)

print(f"Saved Best Model : {mod_p}")
print(f"Saved Metadata   : {meta_p}")
print(f"Saved Metrics    : {eval_p}")
print(f"Saved Report MD  : {rep_m}")
print(f"Saved Report JSON: {rep_j}")""")

    out_path = Path("notebooks/08_phase6_model_training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook written to {out_path}")


if __name__ == "__main__":
    create_phase6_notebook()

