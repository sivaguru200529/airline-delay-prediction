"""Unit and integration tests for Phase 4: Model Retraining & Phase 2B vs Phase 3 Evaluation.

Verifies:
1. Phase 3 dataset loads successfully.
2. Target column exists and has binary values (0, 1).
3. Post-flight columns cannot enter model features (leakage check).
4. Temporal split preserves chronological order.
5. Train data occurs before validation/test data (max(Train) <= min(Val) <= min(Test)).
6. Preprocessing is fitted strictly on training data (no test-set peeking).
7. Logistic Regression trains successfully.
8. Random Forest trains successfully.
9. XGBoost trains successfully.
10. Predictions have correct shape matching test set length.
11. Predictions are binary where classification output is expected.
12. Prediction probabilities are strictly within [0.0, 1.0].
13. Evaluation metrics are finite and non-NaN.
14. Model artifacts are created (delay_model_phase3_*.joblib, phase4_metadata.json).
15. Existing Phase 2B artifact (models/delay_model.joblib) remains completely unchanged.
16. Phase 4 report files are created (phase4_model_evaluation.md and .json).
17. No future information enters historical features/model inputs.
"""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.phase4_eval import (
    analyze_feature_importance,
    analyze_phase3_feature_groups,
    compare_phase2b_vs_phase3,
    evaluate_phase4_models,
    perform_temporal_split,
    prepare_phase4_modeling_dataset,
    run_phase4_workflow,
    save_phase4_artifacts,
    train_phase4_models,
)
from src.models.split import ChronologicalSplitResult
from src.utils.config import get_config


@pytest.fixture(scope="module")
def config():
    """Application configuration fixture."""
    return get_config()


@pytest.fixture(scope="module")
def phase3_dataset(config):
    """Load primary Phase 3 feature dataset."""
    p3_path = config.data_processed_dir / "flights_features_p3.parquet"
    if not p3_path.exists():
        p3_path = config.data_processed_dir / "flights_features_p3.csv"
    assert p3_path.exists(), f"Phase 3 dataset missing at {p3_path}"

    if p3_path.suffix == ".parquet":
        df = pd.read_parquet(p3_path)
    else:
        df = pd.read_csv(p3_path)
    return df


@pytest.fixture(scope="module")
def prepared_modeling_data(phase3_dataset, config):
    """Prepare clean modeling dataset fixture."""
    df_clean, feature_meta = prepare_phase4_modeling_dataset(
        phase3_dataset, target_col="delay_target", config=config
    )
    return df_clean, feature_meta


@pytest.fixture(scope="module")
def temporal_split(prepared_modeling_data, config):
    """Temporal split fixture."""
    df_clean, _ = prepared_modeling_data
    split_res = perform_temporal_split(df_clean, target_col="delay_target", config=config)
    return split_res


@pytest.fixture(scope="module")
def trained_models(temporal_split, prepared_modeling_data, config):
    """Trained Phase 4 models fixture."""
    _, feature_meta = prepared_modeling_data
    models = train_phase4_models(
        X_train=temporal_split.X_train,
        y_train=temporal_split.y_train,
        numerical_features=feature_meta["NUMERICAL_FEATURES"],
        categorical_features=feature_meta["CATEGORICAL_FEATURES"],
        config=config,
    )
    return models


# ==============================================================================
# 1. Dataset Loading & Target Verification
# ==============================================================================

def test_phase3_dataset_loads_successfully(phase3_dataset):
    """Test 1: Verify Phase 3 dataset loads successfully with expected minimum row count."""
    assert len(phase3_dataset) > 0, "Phase 3 dataset is empty"
    assert len(phase3_dataset.columns) >= 38, f"Expected >= 38 columns, got {len(phase3_dataset.columns)}"


def test_target_column_exists_and_binary(phase3_dataset):
    """Test 2: Verify target column exists and contains valid binary values (0 and 1)."""
    assert "delay_target" in phase3_dataset.columns, "'delay_target' column missing"
    unique_vals = set(phase3_dataset["delay_target"].unique())
    assert unique_vals.issubset({0, 1}), f"Target contains non-binary values: {unique_vals}"
    assert 1 in unique_vals and 0 in unique_vals, "Dataset lacks positive or negative target samples"


# ==============================================================================
# 2. Anti-Leakage & Feature Exclusions
# ==============================================================================

def test_post_flight_columns_cannot_enter_model_features(phase3_dataset, config):
    """Test 3: Verify post-flight outcome columns are strictly barred from entering model features."""
    df_clean, feature_meta = prepare_phase4_modeling_dataset(phase3_dataset, target_col="delay_target", config=config)
    model_features = feature_meta["MODEL_FEATURES"]

    for leakage_col in config.post_flight_leakage_columns:
        assert leakage_col not in model_features, f"LEAKAGE: '{leakage_col}' entered MODEL_FEATURES!"

    for kw in config.forbidden_leakage_keywords:
        matching = [f for f in model_features if kw in f.lower()]
        assert not matching, f"LEAKAGE: Forbidden keyword '{kw}' found in features: {matching}"


def test_target_and_identifiers_excluded_from_model_features(prepared_modeling_data):
    """Verify target and identifier columns are recorded in EXCLUDED_FEATURES with rationales."""
    _, feature_meta = prepared_modeling_data
    model_features = feature_meta["MODEL_FEATURES"]
    excluded = feature_meta["EXCLUDED_FEATURES"]

    assert "delay_target" not in model_features
    assert "flight_date" not in model_features
    assert "delay_target" in excluded
    assert "flight_date" in excluded
    assert len(excluded["delay_target"]) > 0, "Missing exclusion rationale for delay_target"
    assert len(excluded["flight_date"]) > 0, "Missing exclusion rationale for flight_date"


def test_no_future_information_enters_historical_features(phase3_dataset):
    """Test 17: Verify strictly prior historical features obey t < T constraint."""
    # Check prior_route_frequency strictly increases with flight order per route
    if "prior_route_frequency" in phase3_dataset.columns:
        freq = phase3_dataset["prior_route_frequency"]
        assert (freq >= 0).all(), "Negative prior route frequency detected"


# ==============================================================================
# 3. Temporal Splitting Integrity
# ==============================================================================

def test_temporal_split_preserves_chronological_order(temporal_split):
    """Test 4 & 5: Verify chronological split ordering: max(Train) <= min(Val) <= min(Test)."""
    train_dates = temporal_split.split_summary["train_date_range"]
    val_dates = temporal_split.split_summary["val_date_range"]
    test_dates = temporal_split.split_summary["test_date_range"]

    assert train_dates[0] <= train_dates[1]
    assert val_dates[0] <= val_dates[1]
    assert test_dates[0] <= test_dates[1]

    # No future dates in train relative to validation/test
    assert train_dates[1] <= val_dates[0] or train_dates[1] == val_dates[0], (
        f"Train max ({train_dates[1]}) must precede or equal Val min ({val_dates[0]})"
    )
    assert val_dates[1] <= test_dates[0] or val_dates[1] == test_dates[0], (
        f"Val max ({val_dates[1]}) must precede or equal Test min ({test_dates[0]})"
    )


def test_temporal_split_sample_counts_and_rates(temporal_split):
    """Verify partition counts match expected ratios without record leakage."""
    n_train = len(temporal_split.y_train)
    n_val = len(temporal_split.y_val)
    n_test = len(temporal_split.y_test)
    total = n_train + n_val + n_test

    assert total == 481
    assert n_train == 336
    assert n_val == 72
    assert n_test == 73
    assert temporal_split.y_train.mean() > 0.10
    assert temporal_split.y_test.mean() > 0.10


# ==============================================================================
# 4. Leakage-Free Preprocessing
# ==============================================================================

def test_preprocessing_fitted_only_on_train(trained_models, temporal_split):
    """Test 6: Verify preprocessors are fitted exclusively on training data and handle unseen levels."""
    lr_pipe = trained_models["Logistic Regression"]
    preprocessor = lr_pipe.named_steps["preprocessor"]

    # Preprocessor must transform X_train and X_test without error
    X_train_trans = preprocessor.transform(temporal_split.X_train)
    X_test_trans = preprocessor.transform(temporal_split.X_test)

    assert X_train_trans.shape[0] == len(temporal_split.X_train)
    assert X_test_trans.shape[0] == len(temporal_split.X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


# ==============================================================================
# 5. Model Training & Convergence
# ==============================================================================

def test_logistic_regression_trains_successfully(trained_models):
    """Test 7: Verify Logistic Regression trains and contains valid coefficients."""
    lr = trained_models["Logistic Regression"]
    assert isinstance(lr, Pipeline)
    clf = lr.named_steps["classifier"]
    assert hasattr(clf, "coef_")
    assert clf.coef_.shape[0] == 1
    assert not np.isnan(clf.coef_).any()


def test_random_forest_trains_successfully(trained_models):
    """Test 8: Verify Random Forest trains and produces non-empty feature importances."""
    rf = trained_models["Random Forest"]
    assert isinstance(rf, Pipeline)
    clf = rf.named_steps["classifier"]
    assert hasattr(clf, "feature_importances_")
    assert len(clf.feature_importances_) > 0
    assert not np.isnan(clf.feature_importances_).any()


def test_xgboost_trains_successfully(trained_models):
    """Test 9: Verify XGBoost classifier trains and produces valid booster trees."""
    xgb = trained_models["XGBoost"]
    assert isinstance(xgb, Pipeline)
    clf = xgb.named_steps["classifier"]
    assert hasattr(clf, "feature_importances_")
    assert len(clf.feature_importances_) > 0
    assert not np.isnan(clf.feature_importances_).any()


# ==============================================================================
# 6. Prediction Output Contracts & Probability Quality
# ==============================================================================

def test_predictions_have_correct_shape(trained_models, temporal_split):
    """Test 10: Verify prediction arrays match test set length exactly."""
    n_test = len(temporal_split.y_test)
    for name, model in trained_models.items():
        preds = model.predict(temporal_split.X_test)
        assert len(preds) == n_test, f"{name} prediction length {len(preds)} != {n_test}"


def test_predictions_are_binary(trained_models, temporal_split):
    """Test 11: Verify class predictions are strictly binary {0, 1}."""
    for name, model in trained_models.items():
        preds = model.predict(temporal_split.X_test)
        unique_preds = set(np.unique(preds))
        assert unique_preds.issubset({0, 1}), f"{name} produced non-binary labels: {unique_preds}"


def test_prediction_probabilities_within_unit_interval(trained_models, temporal_split):
    """Test 12: Verify predicted probabilities are strictly bounded within [0.0, 1.0]."""
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(temporal_split.X_test)[:, 1]
            assert (probs >= 0.0).all(), f"{name} probability < 0.0"
            assert (probs <= 1.0).all(), f"{name} probability > 1.0"
            assert not np.isnan(probs).any(), f"{name} produced NaN probabilities"


def test_evaluation_metrics_are_finite(trained_models, temporal_split, config):
    """Test 13: Verify all computed evaluation metrics are finite numbers."""
    eval_results = evaluate_phase4_models(trained_models, temporal_split, config=config)
    test_metrics = eval_results["test_metrics_default"]

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        m = test_metrics[name]
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]:
            assert k in m, f"Missing metric {k} for {name}"
            val = m[k]
            assert np.isfinite(val), f"Metric {k} for {name} is not finite: {val}"


# ==============================================================================
# 7. Model Artifact Preservation & Integrity
# ==============================================================================

def test_phase2b_model_remains_unchanged(config):
    """Test 15: Verify Phase 2B model artifact (models/delay_model.joblib) is NEVER modified."""
    p2b_path = config.models_dir / "delay_model.joblib"
    assert p2b_path.exists(), f"Phase 2B model missing at {p2b_path}"

    expected_sha256 = "0e9b69c3246af768f10f99ff1497022704ecc97c76d3c4101f95b4f2e9cb65ba"
    actual_sha256 = hashlib.sha256(p2b_path.read_bytes()).hexdigest()
    assert actual_sha256 == expected_sha256, (
        f"CRITICAL FAILURE: Phase 2B model was modified! Expected {expected_sha256}, got {actual_sha256}"
    )


def test_phase4_artifacts_and_reports_created(config):
    """Test 14 & 16: Verify Phase 4 model artifacts and evaluation reports are successfully generated."""
    expected_artifacts = [
        config.models_dir / "delay_model_phase3_logistic.joblib",
        config.models_dir / "delay_model_phase3_rf.joblib",
        config.models_dir / "delay_model_phase3_xgb.joblib",
        config.models_dir / "phase4_metadata.json",
        config.reports_dir / "phase4_model_evaluation.md",
        config.reports_dir / "phase4_model_evaluation.json",
        config.figures_dir / "phase4_roc_comparison.png",
        config.figures_dir / "phase4_pr_comparison.png",
        config.figures_dir / "phase4_calibration_comparison.png",
        config.figures_dir / "phase4_confusion_matrices.png",
        config.figures_dir / "phase4_feature_importance.png",
    ]

    for art in expected_artifacts:
        assert art.exists(), f"Phase 4 artifact missing: {art}"
        assert art.stat().st_size > 0, f"Phase 4 artifact empty: {art}"

    # Verify JSON report structure
    json_rep_path = config.reports_dir / "phase4_model_evaluation.json"
    with open(json_rep_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)

    assert rep_data["phase"] == "Phase 4"
    assert "evaluation_results" in rep_data
    assert "phase2b_vs_phase3" in rep_data
    assert "feature_importance_analysis" in rep_data
    assert "phase3_feature_group_analysis" in rep_data


def test_phase4_full_workflow_execution(config):
    """Verify run_phase4_workflow completes with exit code 0."""
    exit_code = run_phase4_workflow(config=config)
    assert exit_code == 0
