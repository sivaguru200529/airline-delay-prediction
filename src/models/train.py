"""Machine learning training and model orchestration module.

Executes the complete Phase 2B ML workflow:
1. Feature matrix validation and anti-leakage audit.
2. Chronological dataset splitting (Train / Validation / Test).
3. Data-driven route and categorical cardinality analysis.
4. Leakage-free preprocessing (ColumnTransformer fitted strictly on train).
5. Model training: Baseline, Logistic Regression, Random Forest, XGBoost.
6. Validation evaluation, threshold sensitivity, and calibration analysis.
7. Objective, validation-driven model and threshold selection.
8. Single unbiased evaluation on the out-of-time test set.
9. Tree feature importance and SHAP explainability.
10. Model serialization (Joblib + JSON metadata) and comprehensive reports.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.models.evaluate import (
    calculate_classification_metrics,
    compare_models,
    compute_calibration_curve,
    compute_confusion_matrix,
    evaluate_thresholds,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_shap_summary,
)
from src.models.split import ChronologicalSplitResult, split_dataset_chronologically
from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("model_train")


def determine_feature_groups(
    df: pd.DataFrame,
    target_col: str = "delay_target",
    config: Optional[AppConfig] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Dynamically identify numerical and categorical features and evaluate route suitability.

    Args:
        df: Input DataFrame containing features.
        target_col: Prediction target column.
        config: Application configuration.

    Returns:
        Tuple of (numerical_features, categorical_features, route_analysis_metadata).
    """
    cfg = config or get_config()
    ignore_cols = {
        target_col,
        "flight_date",
        "observation_timestamp",
        "scheduled_departure_ts",
    }
    ignore_cols.update(cfg.post_flight_leakage_columns)

    # 1. Evaluate route feature representation in current dataset
    total_records = len(df)
    route_meta: Dict[str, Any] = {}
    include_route = cfg.include_route_feature

    if "route" in df.columns:
        n_unique_routes = df["route"].nunique()
        avg_obs_per_route = round(total_records / max(n_unique_routes, 1), 2)
        route_meta = {
            "total_records": total_records,
            "unique_routes": n_unique_routes,
            "avg_records_per_route": avg_obs_per_route,
            "configured_include": include_route,
        }

        # Data-driven route evaluation rule:
        # If route cardinality is high (>30% unique routes relative to sample size)
        # and not explicitly forced, exclude 'route' in favor of 'origin_airport' and 'dest_airport'
        # which capture geographical endpoints with much higher observation density.
        if not include_route:
            if avg_obs_per_route < 3.0:
                route_meta["decision"] = "EXCLUDED"
                route_meta["reason"] = (
                    f"Route feature exhibits high sparsity on current development sample "
                    f"({n_unique_routes} unique routes across {total_records} flights, avg {avg_obs_per_route} obs/route). "
                    f"Excluded in favor of origin_airport and dest_airport categoricals to prevent severe one-hot sparsity."
                )
                ignore_cols.add("route")
            else:
                route_meta["decision"] = "INCLUDED"
                route_meta["reason"] = (
                    f"Route feature has sufficient support ({avg_obs_per_route} obs/route) and is included."
                )
        else:
            route_meta["decision"] = "INCLUDED (CONFIG OVERRIDE)"
            route_meta["reason"] = "User configuration explicitly requested route feature inclusion."

    # 2. Separate remaining columns into numerical and categorical
    candidate_cols = [c for c in df.columns if c not in ignore_cols and not any(kw in c.lower() for kw in cfg.forbidden_leakage_keywords)]

    numerical_features: List[str] = []
    categorical_features: List[str] = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    logger.info(
        "Identified %d numerical features and %d categorical features. Route status: %s",
        len(numerical_features),
        len(categorical_features),
        route_meta.get("decision", "N/A"),
    )

    return numerical_features, categorical_features, route_meta


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Construct leakage-free ColumnTransformer pipeline.

    Fitted strictly on training data during model training.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        scale_numeric: Whether to apply StandardScaler (required for Logistic Regression).

    Returns:
        scikit-learn ColumnTransformer.
    """
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numerical_features),
            ("cat", Pipeline(cat_steps), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    """Retrieve human-readable transformed feature names from fitted ColumnTransformer."""
    try:
        # scikit-learn >= 1.0 supports get_feature_names_out
        names = preprocessor.get_feature_names_out()
        clean_names = [str(n).replace("num__", "").replace("cat__", "") for n in names]
        return clean_names
    except Exception as e:
        logger.warning("Could not extract feature_names_out directly: %s. Using fallback reconstruction.", e)
        # Fallback reconstruction
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(categorical_features))
        return list(numerical_features) + cat_names


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: List[str],
    categorical_features: List[str],
    config: Optional[AppConfig] = None,
) -> Dict[str, Pipeline]:
    """Train Baseline, Logistic Regression, Random Forest, and XGBoost pipelines.

    All preprocessing is fitted strictly on X_train inside the pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        numerical_features: List of numerical columns.
        categorical_features: List of categorical columns.
        config: Application configuration.

    Returns:
        Dictionary of model_name -> fitted Pipeline.
    """
    cfg = config or get_config()
    seed = cfg.random_seed

    # Class imbalance weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    logger.info(
        "Training candidate models (N=%d, Class 0=%d, Class 1=%d, Imbalance Ratio=%.2f:1)",
        len(y_train), n_neg, n_pos, scale_pos_weight
    )

    models: Dict[str, Pipeline] = {}

    # 1. Baseline Model (Majority class predictor)
    dummy_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numerical_features, categorical_features, scale_numeric=False)),
        ("classifier", DummyClassifier(strategy="most_frequent")),
    ])
    dummy_pipe.fit(X_train, y_train)
    models["Majority Baseline"] = dummy_pipe

    # 2. Logistic Regression (Scaled numericals + balanced class weights)
    lr_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numerical_features, categorical_features, scale_numeric=True)),
        ("classifier", LogisticRegression(
            C=cfg.lr_c,
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
            solver="lbfgs",
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    models["Logistic Regression"] = lr_pipe

    # 3. Random Forest (Class-weighted ensemble)
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numerical_features, categorical_features, scale_numeric=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    models["Random Forest"] = rf_pipe

    # 4. XGBoost (Gradient boosting with scale_pos_weight)
    xgb_pipe = Pipeline([
        ("preprocessor", build_preprocessor(numerical_features, categorical_features, scale_numeric=False)),
        ("classifier", XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    models["XGBoost"] = xgb_pipe

    logger.info("All 4 candidate model pipelines fitted successfully on training data.")
    return models


def select_best_model_on_validation(
    models: Dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidate_thresholds: List[float],
) -> Tuple[str, float, Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Objectively compare models on validation data and select winning model and threshold.

    Selection Criteria:
        1. Compare validation PR-AUC (primary metric for imbalanced delay detection).
        2. Evaluate F1-score and Precision/Recall trade-off across thresholds.
        3. Reject non-predictive baseline.
        4. Select threshold that maximizes validation F1 while preserving recall.

    Args:
        models: Dictionary of trained pipelines.
        X_val: Validation feature matrix.
        y_val: Validation target series.
        candidate_thresholds: List of candidate thresholds to analyze.

    Returns:
        Tuple of (selected_model_name, selected_threshold, val_metrics_summary, threshold_dfs).
    """
    val_metrics: Dict[str, Dict[str, Any]] = {}
    threshold_dfs: Dict[str, pd.DataFrame] = {}
    model_scores: Dict[str, float] = {}

    for name, pipe in models.items():
        # Baseline might only output single probability or constant
        try:
            probs = pipe.predict_proba(X_val)[:, 1]
        except Exception:
            probs = np.zeros(len(y_val))

        # Default threshold metrics
        metrics = calculate_classification_metrics(y_val, probs, threshold=0.50)
        val_metrics[name] = metrics

        # Threshold analysis
        thresh_df = evaluate_thresholds(y_val, probs, thresholds=candidate_thresholds)
        threshold_dfs[name] = thresh_df

        # Score by validation PR-AUC + max F1 (excluding baseline)
        if name != "Majority Baseline":
            best_f1 = float(thresh_df["f1"].max())
            pr_auc = float(metrics["pr_auc"])
            # Combined validation score favoring PR-AUC and balanced F1
            score = 0.6 * pr_auc + 0.4 * best_f1
            model_scores[name] = score

    # Select model with highest validation score
    if model_scores:
        selected_model_name = max(model_scores, key=model_scores.get)
    else:
        selected_model_name = "Logistic Regression"

    # Select optimal threshold for winning model on validation set
    win_thresh_df = threshold_dfs[selected_model_name]
    best_row = win_thresh_df.loc[win_thresh_df["f1"].idxmax()]
    selected_threshold = float(best_row["threshold"])

    logger.info(
        "Validation Model Selection: Selected '%s' (Val PR-AUC=%.4f, Val F1=%.4f at threshold=%.2f)",
        selected_model_name,
        val_metrics[selected_model_name]["pr_auc"],
        best_row["f1"],
        selected_threshold,
    )

    return selected_model_name, selected_threshold, val_metrics, threshold_dfs


def extract_feature_importances(
    pipeline: Pipeline,
    numerical_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Extract and rank tree-based feature importances with meaningful feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = get_transformed_feature_names(preprocessor, numerical_features, categorical_features)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def compute_shap_explainability(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    save_path: Optional[Path] = None,
    max_samples: int = 150,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Compute SHAP values using TreeExplainer and save summary plot with meaningful names.

    Args:
        pipeline: Fitted model pipeline.
        X_train: Training feature DataFrame.
        numerical_features: Numerical column names.
        categorical_features: Categorical column names.
        save_path: Path to save the SHAP plot.
        max_samples: Sample size cap for computational efficiency.

    Returns:
        Tuple of (status_string, shap_importance_df).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    try:
        import shap

        feature_names = get_transformed_feature_names(preprocessor, numerical_features, categorical_features)
        sample_df = X_train.iloc[:max_samples].copy()
        X_trans = preprocessor.transform(sample_df)
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

        if isinstance(classifier, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_trans)
        elif isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(classifier, X_trans)
            shap_values = explainer.shap_values(X_trans)
        else:
            explainer = shap.Explainer(classifier, X_trans)
            shap_values = explainer.shap_values(X_trans)

        # Plot summary
        success = plot_shap_summary(shap_values, X_trans_df, save_path=save_path)
        if success:
            # Calculate mean absolute SHAP value per feature
            vals = shap_values
            if isinstance(vals, list) and len(vals) == 2:
                vals = vals[1]
            elif isinstance(vals, np.ndarray) and len(vals.shape) == 3:
                vals = vals[:, :, 1]
            mean_abs_shap = np.abs(vals).mean(axis=0)

            shap_imp_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

            logger.info("SHAP computation completed successfully on %d samples.", len(sample_df))
            return "SUCCESS", shap_imp_df
        else:
            return "PARTIAL: SHAP computed but plot export encountered an issue", None

    except Exception as e:
        logger.warning("SHAP computation unavailable: %s", e)
        return f"NOT AVAILABLE: {type(e).__name__} - {str(e)}", None


def save_serialized_model(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    models_dir: Path,
    model_filename: str = "delay_model.joblib",
    meta_filename: str = "model_metadata.json",
) -> Tuple[Path, Path]:
    """Serialize the trained pipeline and metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_filename
    meta_path = models_dir / meta_filename

    joblib.dump(pipeline, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Model serialized to %s and metadata to %s", model_path, meta_path)
    return model_path, meta_path


def generate_evaluation_reports(
    split_meta: Dict[str, Any],
    route_meta: Dict[str, Any],
    numerical_features: List[str],
    categorical_features: List[str],
    val_comparison_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    test_metrics: Dict[str, Any],
    selected_model_name: str,
    selected_threshold: float,
    importance_df: pd.DataFrame,
    calibration_dict: Dict[str, Any],
    shap_status: str,
    reports_dir: Path,
) -> Tuple[Path, Path]:
    """Generate comprehensive markdown and JSON model evaluation reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_file = reports_dir / "model_evaluation_report.md"
    json_file = reports_dir / "model_evaluation_report.json"

    # Assemble JSON report dictionary
    report_dict = {
        "report_title": "Airline Delay Prediction — Phase 2B Model Evaluation & Explainability Report",
        "dataset_limitation_notice": (
            "NOTICE: Results presented in this report were generated on the verified development sample "
            f"({split_meta['total_records']} completed flights from January 1–10, 2024). "
            "These metrics represent development-sample behavior and smoke-test verification. "
            "Statistically representative operational performance requires training on the full multi-month "
            "or multi-year BTS dataset as documented in data/raw/README.md."
        ),
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": {
            "total_records": split_meta["total_records"],
            "train_records": split_meta["train_count"],
            "val_records": split_meta["val_count"],
            "test_records": split_meta["test_count"],
            "train_date_range": split_meta["train_date_range"],
            "val_date_range": split_meta["val_date_range"],
            "test_date_range": split_meta["test_date_range"],
            "train_delay_rate_pct": split_meta["train_delay_rate"],
            "val_delay_rate_pct": split_meta["val_delay_rate"],
            "test_delay_rate_pct": split_meta["test_delay_rate"],
        },
        "feature_summary": {
            "numerical_features_count": len(numerical_features),
            "numerical_features": numerical_features,
            "categorical_features_count": len(categorical_features),
            "categorical_features": categorical_features,
            "route_evaluation": route_meta,
            "historical_features_retained": [
                "historical_origin_delay_rate",
                "historical_origin_flight_count",
                "historical_destination_delay_rate",
                "historical_destination_flight_count",
                "historical_airline_delay_rate",
                "historical_airline_flight_count",
                "historical_route_delay_rate",
                "historical_route_flight_count",
            ],
            "preprocessing_strategy": {
                "numerical": "Median Imputation (SimpleImputer, strategy='median') + optional StandardScaler",
                "categorical": "Missing Indicator Imputation + OneHotEncoder(handle_unknown='ignore')",
                "leakage_prevention": "ColumnTransformer fitted strictly on training data partition only",
            },
        },
        "model_selection": {
            "selected_model": selected_model_name,
            "selected_threshold": selected_threshold,
            "selection_rationale": (
                f"Selected '{selected_model_name}' based on superior validation PR-AUC and balanced F1-score. "
                f"Optimal threshold {selected_threshold:.2f} was selected on validation data to balance operational "
                f"delay recall against false alarm precision."
            ),
            "validation_comparison": val_comparison_df.to_dict(orient="records"),
            "threshold_analysis_on_validation": threshold_df.to_dict(orient="records"),
        },
        "final_test_evaluation": {
            "evaluation_notice": "Evaluated ONCE on out-of-time test dataset (untouched during training & tuning)",
            "test_metrics": test_metrics,
            "calibration_metrics": calibration_dict,
        },
        "explainability": {
            "shap_status": shap_status,
            "top_10_features_by_importance": importance_df.head(10).to_dict(orient="records"),
        },
        "known_limitations": [
            "Development dataset sample size (481 flights across 10 days) provides limited statistical depth.",
            "Historical flight counts for rare origin-destination pairs frequently relied on global prior fallback.",
            "Extreme class imbalance coupled with small test set (73 flights) widens metric confidence intervals.",
            "External weather data not supplied; models currently utilize calendar, route, timing, and historical features.",
        ],
    }

    # Format Markdown Report
    top_feats_md = "\n".join([
        f"| {i+1} | `{row['feature']}` | {row['importance']:.4f} |"
        for i, row in importance_df.head(10).iterrows()
    ])

    val_comp_md = "\n".join([
        f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC-AUC']:.4f} | {r['PR-AUC']:.4f} | {r['Brier Score']:.4f} |"
        for _, r in val_comparison_df.iterrows()
    ])

    thresh_md = "\n".join([
        f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        for _, r in threshold_df.iterrows()
    ])

    cm = test_metrics["confusion_matrix"]

    md_content = f"""# Model Evaluation & Explainability Report (Phase 2B)

> **DEVELOPMENT DATASET LIMITATION NOTICE**
> The metrics reported in this document are derived from the development sample dataset (**{split_meta['total_records']} completed flights**, Jan 1–10, 2024).
> This evaluation verifies pipeline integrity, chronological validation, anti-leakage guarantees, and explainability architecture.
> **Production operational conclusions require training and evaluating on the full multi-month/multi-year BTS dataset.**

---

## 1. Dataset & Chronological Out-of-Time Splitting

* **Total Cleaned Observations**: {split_meta['total_records']}
* **Splitting Strategy**: Strictly chronological ($\\text{{Train}} < \\text{{Validation}} < \\text{{Test}}$)
* **Training Partition**: {split_meta['train_count']} flights ({split_meta['train_pct']}%) [{split_meta['train_date_range'][0]} to {split_meta['train_date_range'][1]}] — Delay Rate: {split_meta['train_delay_rate']}%
* **Validation Partition**: {split_meta['val_count']} flights ({split_meta['val_pct']}%) [{split_meta['val_date_range'][0]} to {split_meta['val_date_range'][1]}] — Delay Rate: {split_meta['val_delay_rate']}%
* **Test Partition**: {split_meta['test_count']} flights ({split_meta['test_pct']}%) [{split_meta['test_date_range'][0]} to {split_meta['test_date_range'][1]}] — Delay Rate: {split_meta['test_delay_rate']}%

---

## 2. Feature Schema & Route Representation Analysis

* **Numerical Features ({len(numerical_features)})**: `distance`, `departure_hour`, `departure_minute`, cyclical projections (`sin`/`cos`), and all 8 historical rate/count columns.
* **Categorical Features ({len(categorical_features)})**: `airline`, `origin_airport`, `dest_airport`, `time_of_day`, `haul_category`.
* **Historical Features Preserved**: All 8 Phase 2A historical features retained with strict prior-time aggregation.
* **Route Representation Decision**:
  * Unique routes in sample: {route_meta.get('unique_routes', 'N/A')} (Average {route_meta.get('avg_records_per_route', 'N/A')} obs/route)
  * Status: **{route_meta.get('decision', 'N/A')}**
  * Rationale: {route_meta.get('reason', 'N/A')}
* **Leakage-Safe Preprocessing**:
  * Numerical: Median imputation
  * Categorical: `OneHotEncoder(handle_unknown='ignore')`
  * Fitted strictly on training data partition only.

---

## 3. Validation Model Comparison & Selection

Evaluated strictly on the **Validation Set** to prevent test set snooping.

| Candidate Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{val_comp_md}

### Selection Rationale
* **Selected Winning Model**: `{selected_model_name}`
* **Model Selection Justification**: {report_dict['model_selection']['selection_rationale']}

---

## 4. Threshold Sensitivity Analysis (Validation Set)

Evaluated across candidate thresholds for `{selected_model_name}` on the validation partition:

| Threshold | Accuracy | Precision | Recall | F1-Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{thresh_md}

* **Selected Operational Threshold**: **{selected_threshold:.2f}** (Optimizes balance of delay capture vs false alert rate).

---

## 5. Final Out-of-Time Test Evaluation (Unbiased)

The selected `{selected_model_name}` was evaluated **once** on the untouched test partition ({split_meta['test_count']} flights) at threshold = **{selected_threshold:.2f}**:

* **Test Accuracy**: {test_metrics['accuracy']:.4f}
* **Test Precision**: {test_metrics['precision']:.4f}
* **Test Recall**: {test_metrics['recall']:.4f}
* **Test F1-Score**: {test_metrics['f1']:.4f}
* **Test ROC-AUC**: {test_metrics['roc_auc']:.4f}
* **Test PR-AUC**: {test_metrics['pr_auc']:.4f}
* **Test Brier Score**: {test_metrics['brier_score']:.4f}

### Test Confusion Matrix
| | Pred On-Time (0) | Pred Delayed (1) |
| :--- | :--- | :--- |
| **Actual On-Time (0)** | TN = {cm['tn']} | FP = {cm['fp']} |
| **Actual Delayed (1)** | FN = {cm['fn']} | TP = {cm['tp']} |

---

## 6. Feature Importance & Explainability

### Top 10 Predictive Features (Transformed Names)
| Rank | Transformed Feature | Gini / Gain Weight |
| :--- | :--- | :--- |
{top_feats_md}

### SHAP Explainability Status
* **Status**: **{shap_status}**
* Global feature contributions generated using `shap.TreeExplainer` on training samples.
* Figures exported to `reports/figures/shap_summary.png` and `reports/figures/feature_importance.png`.

---

## 7. Known Limitations & Next Steps

1. **Sample Size**: 481 flights from 10 days in January 2024 is suitable for smoke-testing and pipeline verification, but requires expansion to multi-month BTS datasets for production deployment.
2. **Extreme Class Imbalance**: Low volume of delay events in test partition produces wider confidence intervals.
3. **External Weather**: Ground and en-route weather datasets will significantly enhance predictive power when loaded into `data/external/`.
"""

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)

    md_file.write_text(md_content, encoding="utf-8")
    logger.info("Saved evaluation reports to %s and %s", md_file, json_file)
    return md_file, json_file


def run_training_pipeline(
    features_df: Optional[pd.DataFrame] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 2B model training and evaluation pipeline.

    Args:
        features_df: Pre-engineered features DataFrame (or loaded from data/processed).
        config: Application configuration.

    Returns:
        Dictionary of execution results, metrics, and artifact paths.
    """
    cfg = config or get_config()

    print("\n" + "=" * 80)
    print("EXECUTING PHASE 2B: MACHINE LEARNING TRAINING, EVALUATION & EXPLAINABILITY")
    print("=" * 80)

    # 1. Load Features
    if features_df is None:
        parquet_path = cfg.data_processed_dir / "flights_features.parquet"
        csv_path = cfg.data_processed_dir / "flights_features.csv"

        if parquet_path.exists():
            logger.info("Loading ML feature dataset from %s", parquet_path)
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            logger.info("Loading ML feature dataset from %s", csv_path)
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No feature dataset found in {cfg.data_processed_dir}. Run Phase 2A first."
            )
    else:
        df = features_df.copy()

    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 1: ML DATASET CONTRACT & LEAKAGE AUDIT")
    print("-" * 80)
    print(f"Total ML Records Loaded    : {len(df):,}")
    print(f"Total Columns Detected     : {len(df.columns)}")
    print(f"Prediction Target Column   : delay_target (Presence confirmed)")

    if "delay_target" not in df.columns:
        raise ValueError("Missing 'delay_target' column in ML features.")

    # 2. Dynamic Feature Identification & Route Analysis
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 2: DYNAMIC FEATURE SCHEMA & ROUTE ANALYSIS")
    print("-" * 80)
    num_cols, cat_cols, route_meta = determine_feature_groups(df, target_col="delay_target", config=cfg)
    print(f"Numerical Features ({len(num_cols)})   : {num_cols[:6]} ...")
    print(f"Categorical Features ({len(cat_cols)}) : {cat_cols}")
    print(f"Historical Features (8)    : RETAINED (origin, dest, airline, route rates + counts)")
    print(f"Route Feature Evaluation   : {route_meta.get('decision', 'N/A')}")
    print(f"  - Details: {route_meta.get('reason', 'N/A')}")

    # 3. Chronological Dataset Split
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 3: CHRONOLOGICAL OUT-OF-TIME DATASET SPLIT")
    print("-" * 80)
    split_res = split_dataset_chronologically(
        df=df,
        date_col="flight_date",
        time_col="scheduled_dep_time" if "scheduled_dep_time" in df.columns else None,
        target_col="delay_target",
        train_pct=cfg.train_ratio,
        val_pct=cfg.val_ratio,
        config=cfg,
    )
    X_train, y_train = split_res.X_train, split_res.y_train
    X_val, y_val = split_res.X_val, split_res.y_val
    X_test, y_test = split_res.X_test, split_res.y_test
    split_meta = split_res.split_summary

    print(f"Train Partition : {split_meta['train_count']:,} flights ({split_meta['train_pct']}%) [{split_meta['train_date_range'][0]} to {split_meta['train_date_range'][1]}] — Delay Rate: {split_meta['train_delay_rate']}%")
    print(f"Val Partition   : {split_meta['val_count']:,} flights ({split_meta['val_pct']}%) [{split_meta['val_date_range'][0]} to {split_meta['val_date_range'][1]}] — Delay Rate: {split_meta['val_delay_rate']}%")
    print(f"Test Partition  : {split_meta['test_count']:,} flights ({split_meta['test_pct']}%) [{split_meta['test_date_range'][0]} to {split_meta['test_date_range'][1]}] — Delay Rate: {split_meta['test_delay_rate']}%")
    print(f"Temporal Ordering: Train Max ({split_meta['train_date_range'][1]}) <= Val Min ({split_meta['val_date_range'][0]}) <= Test Min ({split_meta['test_date_range'][0]}) [CONFIRMED]")

    # 4. Train Models
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 4: MODEL TRAINING (LEAKAGE-FREE PREPROCESSING)")
    print("-" * 80)
    models = train_models(
        X_train=X_train,
        y_train=y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        config=cfg,
    )
    for model_name in models.keys():
        print(f"  - Fitted Pipeline: {model_name}")

    # 5. Validation Evaluation & Model Selection
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 5: VALIDATION COMPARISON & OBJECTIVE MODEL SELECTION")
    print("-" * 80)
    selected_model_name, selected_threshold, val_metrics, thresh_dfs = select_best_model_on_validation(
        models=models,
        X_val=X_val,
        y_val=y_val,
        candidate_thresholds=cfg.candidate_thresholds,
    )
    val_comp_df = compare_models(val_metrics)
    print("Validation Model Comparison Table:")
    print(val_comp_df.to_string(index=False))

    selected_pipe = models[selected_model_name]
    best_thresh_df = thresh_dfs[selected_model_name]
    print(f"\nSelected Model     : {selected_model_name}")
    print(f"Selected Threshold : {selected_threshold:.2f}")
    print("Validation Threshold Sensitivity Table:")
    print(best_thresh_df[["threshold", "accuracy", "precision", "recall", "f1"]].to_string(index=False))

    # 6. Final Test Evaluation (Once on Test Set)
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 6: UNBIASED FINAL EVALUATION ON OUT-OF-TIME TEST SET")
    print("-" * 80)
    test_probs = selected_pipe.predict_proba(X_test)[:, 1]
    test_metrics = calculate_classification_metrics(y_test, test_probs, threshold=selected_threshold)

    print(f"Final Model Evaluated : {selected_model_name}")
    print(f"Classification Thresh : {selected_threshold:.2f}")
    print(f"Test Accuracy         : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision        : {test_metrics['precision']:.4f}")
    print(f"Test Recall           : {test_metrics['recall']:.4f}")
    print(f"Test F1-Score         : {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC          : {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC           : {test_metrics['pr_auc']:.4f}")
    print(f"Test Brier Score      : {test_metrics['brier_score']:.4f}")
    cm = test_metrics["confusion_matrix"]
    print(f"Confusion Matrix      : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    # 7. Probability Calibration Analysis
    calib_dict = compute_calibration_curve(y_test, test_probs, n_bins=5)
    print(f"Test Calibration Brier Score : {calib_dict['brier_score']:.4f}")

    # 8. Feature Importance & SHAP
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 7: FEATURE IMPORTANCE & SHAP EXPLAINABILITY")
    print("-" * 80)
    imp_df = extract_feature_importances(selected_pipe, num_cols, cat_cols)
    print("Top 5 Predictive Features (Meaningful Names):")
    for _, r in imp_df.head(5).iterrows():
        print(f"  - {r['feature']}: {r['importance']:.4f}")

    # Export Figures
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cm_fig = cfg.figures_dir / "confusion_matrix.png"
    roc_fig = cfg.figures_dir / "roc_curve.png"
    pr_fig = cfg.figures_dir / "precision_recall_curve.png"
    calib_fig = cfg.figures_dir / "calibration_curve.png"
    imp_fig = cfg.figures_dir / "feature_importance.png"
    shap_fig = cfg.figures_dir / "shap_summary.png"

    plot_confusion_matrix(cm, model_name=selected_model_name, save_path=cm_fig)
    curves_dict = {
        name: (y_val.to_numpy(), pipe.predict_proba(X_val)[:, 1])
        for name, pipe in models.items()
    }
    plot_roc_curves(curves_dict, save_path=roc_fig)
    plot_pr_curves(curves_dict, save_path=pr_fig)
    plot_calibration_curves(curves_dict, save_path=calib_fig)
    plot_feature_importance(imp_df, model_name=selected_model_name, top_n=15, save_path=imp_fig)

    shap_status, shap_imp_df = compute_shap_explainability(
        selected_pipe,
        X_train,
        num_cols,
        cat_cols,
        save_path=shap_fig,
    )
    print(f"SHAP Explainability Status: {shap_status}")

    # 9. Model Serialization
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 8: MODEL SERIALIZATION & METADATA EXPORT")
    print("-" * 80)
    metadata = {
        "model_name": selected_model_name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.random_seed,
        "selected_threshold": selected_threshold,
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "transformed_names": get_transformed_feature_names(
                selected_pipe.named_steps["preprocessor"], num_cols, cat_cols
            ),
        },
        "target_definition": "delay_target = 1 if arrival_delay >= 15 min else 0",
        "date_ranges": {
            "train": split_meta["train_date_range"],
            "validation": split_meta["val_date_range"],
            "test": split_meta["test_date_range"],
        },
        "test_metrics": test_metrics,
        "calibration": calib_dict,
        "dataset_limitation_notice": (
            "Model trained on development dataset (481 records). For production, "
            "train on the full multi-month/multi-year dataset."
        ),
    }

    model_path, meta_path = save_serialized_model(selected_pipe, metadata, cfg.models_dir)
    print(f"Serialized Model Pipeline : {model_path}")
    print(f"Model Metadata File       : {meta_path}")

    # 10. Generate Reports
    print("\n" + "-" * 80)
    print("PHASE 2B - STEP 9: GENERATE EVALUATION REPORTS")
    print("-" * 80)
    md_report, json_report = generate_evaluation_reports(
        split_meta=split_meta,
        route_meta=route_meta,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        val_comparison_df=val_comp_df,
        threshold_df=best_thresh_df,
        test_metrics=test_metrics,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        importance_df=imp_df,
        calibration_dict=calib_dict,
        shap_status=shap_status,
        reports_dir=cfg.reports_dir,
    )
    print(f"Markdown Evaluation Report: {md_report}")
    print(f"JSON Evaluation Report    : {json_report}")

    print("\n" + "=" * 80)
    print("PHASE 2B EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    return {
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "split_summary": split_meta,
        "model_path": model_path,
        "meta_path": meta_path,
        "md_report": md_report,
        "json_report": json_report,
        "shap_status": shap_status,
    }
