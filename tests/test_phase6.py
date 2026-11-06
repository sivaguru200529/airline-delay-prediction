"""Comprehensive test suite for Phase 6: EDA-Informed Machine Learning & Model Optimization.

Covers all 14 required Phase 6 verification points:
1. Dataset loading
2. Required column validation
3. Constant (zero-variance) feature removal
4. Redundant feature handling
5. Leakage prevention
6. Chronological split correctness
7. Preprocessing isolation (fitted strictly on train)
8. Class weighting
9. Model training
10. Probability prediction
11. Threshold optimization
12. Metric calculation
13. Model serialization and reload
14. Artifact generation & CLI integration
15. Notebook validity
"""

import json
from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.phase6_training import (
    ZERO_VARIANCE_FEATURES,
    REDUNDANT_FEATURE_ALIASES,
    prepare_phase6_dataset,
    split_phase6_dataset,
    calculate_phase6_metrics,
    optimize_and_train_models,
    optimize_decision_thresholds,
    evaluate_models_comprehensive,
    select_best_phase6_model,
    compare_with_phase4,
    save_phase6_artifacts,
    run_phase6_workflow,
)
from src.models.split import ChronologicalSplitResult
from src.models.train import build_preprocessor
from src.utils.config import get_config


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def module_feature_df() -> pd.DataFrame:
    """Module-scoped deterministic synthetic fixture mimicking flights_features_p3.parquet schema."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    records = []

    np.random.seed(42)
    for i in range(n):
        dt = dates[i % len(dates)]
        dep_time = 600 + (i * 15) % 1600
        delay_target = 1 if (i % 3 == 0) else 0

        row = {
            "flight_date": dt.strftime("%Y-%m-%d"),
            "airline": np.random.choice(["AA", "DL", "UA", "WN"]),
            "origin_airport": np.random.choice(["ATL", "ORD", "DFW", "DEN"]),
            "dest_airport": np.random.choice(["LAX", "JFK", "SFO", "MIA"]),
            "scheduled_dep_time": dep_time,
            "scheduled_arr_time": (dep_time + 200) % 2400,
            "distance": float(np.random.randint(300, 2500)),
            "is_month_start": 1 if dt.day == 1 else 0,
            "dep_time_of_day": "morning" if dep_time < 1200 else "afternoon",
            "dep_time_bucket": "08:00-11:59",
            "dep_minute_sin": np.sin(2 * np.pi * (dep_time % 60) / 60),
            "dep_minute_cos": np.cos(2 * np.pi * (dep_time % 60) / 60),
            "arr_time_of_day": "afternoon",
            "arr_time_bucket": "12:00-15:59",
            "arr_hour_sin": np.sin(2 * np.pi * (dep_time // 100) / 24),
            "arr_hour_cos": np.cos(2 * np.pi * (dep_time // 100) / 24),
            "route": f"ATL_LAX",
            "distance_category": "medium_haul",
            "prior_route_frequency": float(np.random.randint(1, 10)),
            # Primary historical delay features
            "prior_airline_flight_count": float(i + 1),
            "prior_airline_delay_count": float(i // 3),
            "prior_airline_delay_rate": float((i // 3) / max(i + 1, 1)),
            "prior_origin_flight_count": float(i + 2),
            "prior_origin_delay_count": float(i // 4),
            "prior_origin_delay_rate": float((i // 4) / max(i + 2, 1)),
            "prior_dest_flight_count": float(i + 1),
            "prior_dest_delay_count": float(i // 5),
            "prior_dest_delay_rate": float((i // 5) / max(i + 1, 1)),
            "prior_route_flight_count": float(i + 1),
            "prior_route_delay_count": float(i // 6),
            "prior_route_delay_rate": float((i // 6) / max(i + 1, 1)),
            "prior_airline_dep_hour_flight_count": float(i + 1),
            "prior_airline_dep_hour_delay_count": float(i // 4),
            "prior_airline_dep_hour_delay_rate": float((i // 4) / max(i + 1, 1)),
            "prior_origin_dep_hour_flight_count": float(i + 1),
            "prior_origin_dep_hour_delay_count": float(i // 4),
            "prior_origin_dep_hour_delay_rate": float((i // 4) / max(i + 1, 1)),
            # Zero-variance columns
            "cancelled": 0,
            "diverted": 0,
            "is_month_end": 0,
            "quarter": 1,
            "season": "winter",
            # Redundant aliases
            "route_distance": float(np.random.randint(300, 2500)),
            "route_distance_category": "medium_haul",
            "prior_origin_flight_volume": float(i + 2),
            "prior_dest_flight_volume": float(i + 1),
            "carrier_prior_flight_count": float(i + 1),
            "airline_prior_flight_count": float(i + 1),
            "carrier_prior_delay_count": float(i // 3),
            "airline_prior_delay_count": float(i // 3),
            "carrier_prior_delay_rate": float((i // 3) / max(i + 1, 1)),
            "airline_prior_delay_rate": float((i // 3) / max(i + 1, 1)),
            "origin_prior_flight_count": float(i + 2),
            "origin_prior_delay_count": float(i // 4),
            "origin_prior_delay_rate": float((i // 4) / max(i + 2, 1)),
            "dest_prior_flight_count": float(i + 1),
            "dest_prior_delay_count": float(i // 5),
            "dest_prior_delay_rate": float((i // 5) / max(i + 1, 1)),
            "route_prior_flight_count": float(i + 1),
            "route_prior_delay_count": float(i // 6),
            "route_prior_delay_rate": float((i // 6) / max(i + 1, 1)),
            "carrier_origin_hour_prior_flight_count": float(i + 1),
            "airline_dep_hour_prior_flight_count": float(i + 1),
            "carrier_origin_hour_prior_delay_count": float(i // 4),
            "airline_dep_hour_prior_delay_count": float(i // 4),
            "carrier_origin_hour_prior_delay_rate": float((i // 4) / max(i + 1, 1)),
            "airline_dep_hour_prior_delay_rate": float((i // 4) / max(i + 1, 1)),
            "delay_target": delay_target,
        }
        records.append(row)

    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def trained_bundle(module_feature_df):
    """Module-scoped fitted models bundle to avoid redundant tuning in tests."""
    df, meta = prepare_phase6_dataset(module_feature_df)
    split_res = split_phase6_dataset(df)
    models, tuning_meta = optimize_and_train_models(
        X_train=split_res.X_train[meta["MODEL_FEATURES"]],
        y_train=split_res.y_train,
        X_val=split_res.X_val[meta["MODEL_FEATURES"]],
        y_val=split_res.y_val,
        numerical_features=meta["NUMERICAL_FEATURES"],
        categorical_features=meta["CATEGORICAL_FEATURES"],
    )
    return {
        "models": models,
        "tuning_meta": tuning_meta,
        "split_res": split_res,
        "meta": meta,
    }


# ==============================================================================
# TESTS
# ==============================================================================

def test_1_dataset_loading():
    """Verify loading and schema of primary Phase 3 feature dataset."""
    cfg = get_config()
    p3_path = cfg.data_processed_dir / "flights_features_p3.parquet"
    assert p3_path.exists(), f"Phase 3 dataset missing at {p3_path}"

    df = pd.read_parquet(p3_path)
    assert len(df) == 481, f"Expected 481 records, got {len(df)}"
    assert len(df.columns) == 68, f"Expected 68 columns, got {len(df.columns)}"
    assert "delay_target" in df.columns


def test_2_required_column_validation(module_feature_df):
    """Verify validation fails if required target column is missing."""
    df_missing = module_feature_df.drop(columns=["delay_target"])
    with pytest.raises(ValueError, match="Required target column 'delay_target' not found"):
        prepare_phase6_dataset(df_missing)


def test_3_constant_feature_removal(module_feature_df):
    """Verify all 5 zero-variance features are strictly removed from MODEL_FEATURES."""
    _, meta = prepare_phase6_dataset(module_feature_df)
    model_feats = meta["MODEL_FEATURES"]

    for zv in ZERO_VARIANCE_FEATURES:
        assert zv not in model_feats, f"Zero-variance column {zv} was not pruned from MODEL_FEATURES!"
        assert zv in meta["ZERO_VARIANCE_PRUNED"]


def test_4_redundant_feature_handling(module_feature_df):
    """Verify redundant feature aliases identified in Phase 5 are pruned."""
    _, meta = prepare_phase6_dataset(module_feature_df)
    model_feats = meta["MODEL_FEATURES"]

    for alias in REDUNDANT_FEATURE_ALIASES:
        assert alias not in model_feats, f"Redundant alias {alias} was not pruned!"

    # Verify primary features are retained
    assert "distance" in model_feats
    assert "distance_category" in model_feats
    assert "prior_airline_delay_rate" in model_feats
    assert "prior_origin_delay_rate" in model_feats
    assert "prior_dest_delay_rate" in model_feats


def test_5_leakage_prevention(module_feature_df):
    """Verify post-flight outcome columns trigger errors and delay_target is excluded."""
    # Check that ground-truth target is not in X features
    _, meta = prepare_phase6_dataset(module_feature_df)
    assert "delay_target" not in meta["MODEL_FEATURES"]

    # Inject a post-flight leakage column
    df_leak = module_feature_df.copy()
    df_leak["arrival_delay"] = 25.0
    with pytest.raises(ValueError, match="TARGET LEAKAGE DETECTED"):
        prepare_phase6_dataset(df_leak)


def test_6_chronological_split_correctness(module_feature_df):
    """Verify chronological split enforces strict temporal boundaries without random shuffling."""
    split_res = split_phase6_dataset(module_feature_df)

    assert len(split_res.X_train) == 70
    assert len(split_res.X_val) == 15
    assert len(split_res.X_test) == 15

    # Check temporal ordering
    train_max_date = split_res.train_date_range[1]
    val_min_date = split_res.val_date_range[0]
    val_max_date = split_res.val_date_range[1]
    test_min_date = split_res.test_date_range[0]

    assert train_max_date <= val_min_date, f"Temporal leak: train max {train_max_date} > val min {val_min_date}"
    assert val_max_date <= test_min_date, f"Temporal leak: val max {val_max_date} > test min {test_min_date}"


def test_7_preprocessing_isolation(module_feature_df):
    """Verify preprocessor is fitted strictly on training data and transforms val/test safely."""
    _, meta = prepare_phase6_dataset(module_feature_df)
    split_res = split_phase6_dataset(module_feature_df)

    num_cols = meta["NUMERICAL_FEATURES"]
    cat_cols = meta["CATEGORICAL_FEATURES"]

    pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols, scale_numeric=True)),
        ("classifier", LogisticRegression()),
    ])

    # Fit strictly on train
    pipe.fit(split_res.X_train[meta["MODEL_FEATURES"]], split_res.y_train)

    # Transform val and test
    val_trans = pipe.named_steps["preprocessor"].transform(split_res.X_val[meta["MODEL_FEATURES"]])
    test_trans = pipe.named_steps["preprocessor"].transform(split_res.X_test[meta["MODEL_FEATURES"]])

    assert val_trans.shape[0] == len(split_res.X_val)
    assert test_trans.shape[0] == len(split_res.X_test)
    assert not np.isnan(val_trans).any()
    assert not np.isnan(test_trans).any()


def test_8_class_weighting(trained_bundle):
    """Verify class weighting and scale_pos_weight are properly configured."""
    models = trained_bundle["models"]

    # Check Logistic Regression class_weight
    lr_clf = models["Logistic Regression"].named_steps["classifier"]
    assert lr_clf.class_weight == "balanced"

    # Check Random Forest class_weight
    rf_clf = models["Random Forest"].named_steps["classifier"]
    assert rf_clf.class_weight == "balanced"

    # Check XGBoost scale_pos_weight
    xgb_clf = models["XGBoost"].named_steps["classifier"]
    assert xgb_clf.scale_pos_weight > 0.0


def test_9_model_training_all_candidates(trained_bundle):
    """Verify all candidate models (Baseline, LR, RF, XGB) train cleanly."""
    models = trained_bundle["models"]
    expected_models = ["Majority Baseline", "Logistic Regression", "Random Forest", "XGBoost"]
    for m in expected_models:
        assert m in models
        assert hasattr(models[m], "predict")


def test_10_probability_prediction(trained_bundle):
    """Verify predicted probabilities are strictly bounded within [0, 1]."""
    models = trained_bundle["models"]
    split_res = trained_bundle["split_res"]
    meta = trained_bundle["meta"]

    X_test_m = split_res.X_test[meta["MODEL_FEATURES"]]
    for name, pipe in models.items():
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(X_test_m)
            assert probs.shape == (len(X_test_m), 2)
            assert (probs >= 0.0).all()
            assert (probs <= 1.0).all()
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_11_threshold_optimization(trained_bundle):
    """Verify validation-driven decision threshold optimization between 0.30 and 0.70."""
    models = trained_bundle["models"]
    split_res = trained_bundle["split_res"]
    meta = trained_bundle["meta"]

    threshs, dfs = optimize_decision_thresholds(
        models=models,
        X_val=split_res.X_val[meta["MODEL_FEATURES"]],
        y_val=split_res.y_val,
    )

    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        assert 0.30 <= threshs[name] <= 0.70
        assert not dfs[name].empty
        assert "threshold" in dfs[name].columns
        assert "f1" in dfs[name].columns


def test_12_metric_calculation():
    """Verify calculate_phase6_metrics returns all required metrics accurately."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.3, 0.7, 0.4, 0.2, 0.9, 0.5])

    metrics = calculate_phase6_metrics(y_true, y_prob, threshold=0.50)

    required_keys = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "roc_auc", "pr_auc", "brier_score", "confusion_matrix"
    ]
    for k in required_keys:
        assert k in metrics, f"Missing metric {k}"

    cm = metrics["confusion_matrix"]
    assert "tn" in cm and "fp" in cm and "fn" in cm and "tp" in cm
    assert cm["tp"] == 3
    assert cm["tn"] == 4
    assert cm["fp"] == 1
    assert cm["fn"] == 0


def test_13_model_serialization_and_reload(trained_bundle, tmp_path):
    """Verify serialized model pipeline can be reloaded and yields identical predictions."""
    models = trained_bundle["models"]
    split_res = trained_bundle["split_res"]
    meta = trained_bundle["meta"]

    pipe = models["Logistic Regression"]
    save_file = tmp_path / "test_model.joblib"
    joblib.dump(pipe, save_file)

    reloaded = joblib.load(save_file)
    X_test_m = split_res.X_test[meta["MODEL_FEATURES"]]
    orig_probs = pipe.predict_proba(X_test_m)
    reloaded_probs = reloaded.predict_proba(X_test_m)

    np.testing.assert_allclose(orig_probs, reloaded_probs, atol=1e-6)


def test_14_phase6_artifacts_and_cli():
    """Verify that Phase 6 CLI execution creates all expected artifacts."""
    cfg = get_config()
    res = run_phase6_workflow(config=cfg)
    assert res == 0, "Phase 6 workflow failed!"

    # Check model artifacts
    p6_dir = cfg.project_root / "models" / "phase6"
    assert (p6_dir / "best_model.joblib").exists()
    assert (p6_dir / "model_metadata.json").exists()
    assert (p6_dir / "evaluation_metrics.json").exists()

    # Check report artifacts
    assert (cfg.reports_dir / "phase6_model_evaluation.json").exists()
    assert (cfg.reports_dir / "phase6_model_evaluation.md").exists()

    # Check 7 figures
    fig_dir = cfg.reports_dir / "figures"
    expected_figures = [
        "phase6_model_comparison.png",
        "phase6_roc_comparison.png",
        "phase6_pr_comparison.png",
        "phase6_confusion_matrix.png",
        "phase6_calibration_comparison.png",
        "phase6_feature_importance.png",
        "phase6_shap_summary.png",
    ]
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing at {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} appears empty/corrupt"


def test_15_phase6_notebook_validity():
    """Verify that notebooks/08_phase6_model_training.ipynb exists and is valid."""
    cfg = get_config()
    nb_path = cfg.project_root / "notebooks" / "08_phase6_model_training.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_json = json.load(f)

    assert "cells" in nb_json
    assert len(nb_json["cells"]) >= 10
    cell_types = [c["cell_type"] for c in nb_json["cells"]]
    assert "code" in cell_types
    assert "markdown" in cell_types
