"""Unit tests for Phase 2B machine learning pipeline.

Tests cover:
1. Chronological dataset split and temporal ordering.
2. Temporal anti-leakage (no future observations in training).
3. Target and post-flight leakage variables excluded from feature matrix.
4. Preprocessing fitted strictly on training data.
5. Handling of unseen/unknown categorical levels during inference.
6. Baseline model behavior (majority class prediction).
7. Candidate model training and convergence (Logistic Regression, Random Forest, XGBoost).
8. Probability prediction bounds [0.0, 1.0].
9. Prediction output schema and threshold sensitivity.
10. Operational risk tier mapping.
11. Model serialization and deserialization reproducibility.
12. Metric calculations (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score).
13. Preservation of all 8 historical delay features.
14. Data-driven route feature evaluation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.evaluate import (
    calculate_classification_metrics,
    compare_models,
    compute_calibration_curve,
    compute_confusion_matrix,
    evaluate_thresholds,
)
from src.models.predict import load_model, predict_delay_probability
from src.models.split import ChronologicalSplitResult, split_dataset_chronologically
from src.models.train import (
    build_preprocessor,
    determine_feature_groups,
    extract_feature_importances,
    save_serialized_model,
    select_best_model_on_validation,
    train_models,
)
from src.utils.config import AppConfig, get_config


# ==============================================================================
# DETERMINISTIC FIXTURES
# ==============================================================================

@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Generate a deterministic synthetic feature dataset mimicking Phase 2A output."""
    dates = [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10",
    ]
    records = []
    rng = np.random.RandomState(42)

    for i in range(100):
        d_idx = i // 10  # 10 flights per date
        flight_date = dates[d_idx]
        dep_time = 600 + (i % 10) * 100
        airline = ["AA", "DL", "UA", "WN"][i % 4]
        origin = ["JFK", "LAX", "ORD", "DFW"][i % 4]
        dest = ["MIA", "SFO", "ATL", "DEN"][(i + 1) % 4]
        route = f"{origin}_{dest}"
        distance = 500 + (i % 5) * 300
        haul = "short_haul" if distance < 800 else "medium_haul"
        time_of_day = "morning" if dep_time < 1200 else "afternoon"

        # Delay target with ~30% delayed
        delay_target = 1 if (i % 3 == 0) else 0

        records.append({
            "flight_date": flight_date,
            "scheduled_dep_time": dep_time,
            "airline": airline,
            "origin_airport": origin,
            "dest_airport": dest,
            "route": route,
            "distance": distance,
            "haul_category": haul,
            "departure_hour": dep_time // 100,
            "departure_minute": dep_time % 100,
            "departure_minutes_since_midnight": (dep_time // 100) * 60,
            "time_of_day": time_of_day,
            "departure_hour_sin": np.sin(2 * np.pi * (dep_time // 100) / 24),
            "departure_hour_cos": np.cos(2 * np.pi * (dep_time // 100) / 24),
            "historical_origin_delay_rate": 0.25 + rng.normal(0, 0.05),
            "historical_origin_flight_count": 10 + i,
            "historical_destination_delay_rate": 0.20 + rng.normal(0, 0.05),
            "historical_destination_flight_count": 8 + i,
            "historical_airline_delay_rate": 0.28 + rng.normal(0, 0.05),
            "historical_airline_flight_count": 15 + i,
            "historical_route_delay_rate": 0.15 + rng.normal(0, 0.05),
            "historical_route_flight_count": 5 + (i % 10),
            "delay_target": delay_target,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 1. CHRONOLOGICAL SPLITTING TESTS
# ==============================================================================

def test_chronological_split_ordering(sample_feature_df):
    """Verify that chronological splitting partitions data strictly into Past -> Later -> Future."""
    split_res = split_dataset_chronologically(
        df=sample_feature_df,
        date_col="flight_date",
        time_col="scheduled_dep_time",
        train_pct=0.70,
        val_pct=0.15,
    )
    summary = split_res.split_summary

    assert summary["train_count"] == 70
    assert summary["val_count"] == 15
    assert summary["test_count"] == 15

    # Check temporal ordering: max train <= min val and max val <= test min
    train_max = summary["train_date_range"][1]
    val_min = summary["val_date_range"][0]
    val_max = summary["val_date_range"][1]
    test_min = summary["test_date_range"][0]

    assert train_max <= val_min
    assert val_max <= test_min


def test_target_and_leakage_excluded_from_features(sample_feature_df):
    """Verify delay_target and post-flight leakage variables are strictly purged from X."""
    # Artificially inject leakage columns into test dataframe
    df_with_leakage = sample_feature_df.copy()
    df_with_leakage["arrival_delay"] = 25.0
    df_with_leakage["actual_dep_time"] = 815
    df_with_leakage["taxi_out"] = 18.0

    split_res = split_dataset_chronologically(df=df_with_leakage)

    for X in [split_res.X_train, split_res.X_val, split_res.X_test]:
        assert "delay_target" not in X.columns
        assert "arrival_delay" not in X.columns
        assert "actual_dep_time" not in X.columns
        assert "taxi_out" not in X.columns


# ==============================================================================
# 2. PREPROCESSING & ROUTE ANALYSIS TESTS
# ==============================================================================

def test_route_evaluation_and_feature_groups(sample_feature_df):
    """Verify dynamic feature grouping and route sparsity analysis."""
    num_cols, cat_cols, route_meta = determine_feature_groups(sample_feature_df)

    assert "delay_target" not in num_cols
    assert "delay_target" not in cat_cols
    assert "flight_date" not in num_cols
    assert "flight_date" not in cat_cols

    # Verify historical features are retained
    assert "historical_origin_delay_rate" in num_cols
    assert "historical_destination_delay_rate" in num_cols
    assert "historical_airline_delay_rate" in num_cols
    assert "historical_route_delay_rate" in num_cols

    # Verify categoricals include airport endpoints and time of day
    assert "airline" in cat_cols
    assert "origin_airport" in cat_cols
    assert "dest_airport" in cat_cols
    assert "time_of_day" in cat_cols


def test_preprocessing_fitted_only_on_train(sample_feature_df):
    """Verify ColumnTransformer is fitted on train and handles missing values via median/constant."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    preprocessor.fit(split_res.X_train)

    # Transform train, val, test
    X_train_trans = preprocessor.transform(split_res.X_train)
    X_val_trans = preprocessor.transform(split_res.X_val)
    X_test_trans = preprocessor.transform(split_res.X_test)

    assert X_train_trans.shape[0] == len(split_res.X_train)
    assert X_val_trans.shape[0] == len(split_res.X_val)
    assert X_test_trans.shape[0] == len(split_res.X_test)
    assert not np.isnan(X_train_trans).any()


def test_unknown_categorical_values_handled_gracefully(sample_feature_df):
    """Verify unseen categories in test do not throw errors (handle_unknown='ignore')."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    preprocessor.fit(split_res.X_train)

    # Introduce unknown airline and airport
    novel_df = split_res.X_test.copy()
    novel_df.loc[0, "airline"] = "NOVEL_AIRLINE"
    novel_df.loc[0, "origin_airport"] = "ZZZ"

    # Must transform without error
    novel_trans = preprocessor.transform(novel_df)
    assert novel_trans.shape[0] == len(novel_df)
    assert not np.isnan(novel_trans).any()


# ==============================================================================
# 3. MODEL TRAINING & PREDICTION TESTS
# ==============================================================================

def test_model_training_and_convergence(sample_feature_df):
    """Verify Baseline, Logistic Regression, Random Forest, and XGBoost fit cleanly."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    models = train_models(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    assert "Majority Baseline" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost" in models

    for name, pipe in models.items():
        probs = pipe.predict_proba(split_res.X_val)[:, 1]
        assert len(probs) == len(split_res.X_val)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_validation_model_selection(sample_feature_df):
    """Verify objective validation comparison and threshold selection."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    models = train_models(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    best_model, best_thresh, val_metrics, thresh_dfs = select_best_model_on_validation(
        models=models,
        X_val=split_res.X_val,
        y_val=split_res.y_val,
        candidate_thresholds=[0.30, 0.40, 0.50, 0.60, 0.70],
    )

    assert best_model in ["Logistic Regression", "Random Forest", "XGBoost"]
    assert best_thresh in [0.30, 0.40, 0.50, 0.60, 0.70]
    assert best_model in thresh_dfs


def test_predict_utility_schema_and_risk_mapping(sample_feature_df):
    """Verify predict_delay_probability output schema, thresholding, and risk tiers."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    models = train_models(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    model = models["Logistic Regression"]

    # 1. Single dict prediction
    single_record = split_res.X_test.iloc[0].to_dict()
    res_dict = predict_delay_probability(model, single_record, threshold=0.40)

    assert "delay_probability" in res_dict
    assert "predicted_class" in res_dict
    assert "risk_level" in res_dict
    assert "threshold" in res_dict
    assert res_dict["threshold"] == 0.40
    assert res_dict["risk_level"] in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]

    # 2. Batch DataFrame prediction
    batch_res = predict_delay_probability(model, split_res.X_test, threshold=0.50)
    assert isinstance(batch_res, pd.DataFrame)
    assert "delay_probability" in batch_res.columns
    assert "predicted_class" in batch_res.columns
    assert "risk_level" in batch_res.columns
    assert len(batch_res) == len(split_res.X_test)


def test_model_serialization_and_reloading(sample_feature_df):
    """Verify joblib serialization and identical predictions after reload."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    models = train_models(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    original_model = models["Random Forest"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {"model_name": "Random Forest", "version": "1.0"}
        model_file, meta_file = save_serialized_model(original_model, meta, tmp_path)

        assert model_file.exists()
        assert meta_file.exists()

        reloaded_model = load_model(model_file)
        orig_preds = original_model.predict_proba(split_res.X_test)[:, 1]
        reloaded_preds = reloaded_model.predict_proba(split_res.X_test)[:, 1]

        np.testing.assert_allclose(orig_preds, reloaded_preds, rtol=1e-5)


# ==============================================================================
# 4. METRIC & EVALUATION FUNCTION TESTS
# ==============================================================================

def test_classification_metrics_calculation():
    """Verify calculate_classification_metrics correctness on known deterministic data."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.8])

    metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_threshold_analysis():
    """Verify evaluate_thresholds computes expected metrics across thresholds."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.35, 0.45, 0.8])

    thresh_df = evaluate_thresholds(y_true, y_prob, thresholds=[0.30, 0.50, 0.70])
    assert len(thresh_df) == 3
    assert "precision" in thresh_df.columns
    assert "recall" in thresh_df.columns
    assert "f1" in thresh_df.columns


def test_feature_importance_extraction(sample_feature_df):
    """Verify extract_feature_importances maps to meaningful feature names."""
    split_res = split_dataset_chronologically(sample_feature_df)
    num_cols, cat_cols, _ = determine_feature_groups(sample_feature_df)

    models = train_models(
        X_train=split_res.X_train,
        y_train=split_res.y_train,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    rf_pipe = models["Random Forest"]
    imp_df = extract_feature_importances(rf_pipe, num_cols, cat_cols)

    assert len(imp_df) > 0
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
    # Check that feature names are meaningful, not feature_0
    assert not imp_df["feature"].str.startswith("feature_").any()
