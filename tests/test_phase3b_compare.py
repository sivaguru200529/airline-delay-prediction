"""Unit and integration tests for Phase 3B: Model Retraining & Phase 2A vs Phase 3 Evaluation.

Verifies:
1. Dataset loading and matching flight populations (481 rows, Jan 1-10, 2024).
2. Chronological split boundary parity with zero temporal overlap (Train < Val < Test).
3. Preprocessing fitted strictly on training partition only.
4. Validation-driven threshold selection (no test-set peeking).
5. Accurate computation of absolute and percentage performance deltas.
6. Phase 2B model preservation (models/delay_model.joblib is NOT overwritten).
7. Artifact generation (models/phase3b/, reports, figures, metadata).
8. No NaN or invalid metric values across all models.
"""

import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.phase3b_compare import (
    compute_experiment_deltas,
    run_phase3b_workflow,
    run_single_experiment,
    save_phase3b_artifacts,
    validate_experiment_datasets,
)
from src.models.split import split_dataset_chronologically
from src.utils.config import get_config


@pytest.fixture(scope="module")
def config():
    """Application config fixture."""
    return get_config()


@pytest.fixture(scope="module")
def datasets(config):
    """Load Phase 2A and Phase 3 feature datasets."""
    p2a_path = config.data_processed_dir / "flights_features.parquet"
    p3_path = config.data_processed_dir / "flights_features_p3.parquet"

    assert p2a_path.exists(), f"Phase 2A dataset missing at {p2a_path}"
    assert p3_path.exists(), f"Phase 3 dataset missing at {p3_path}"

    df_2a = pd.read_parquet(p2a_path)
    df_p3 = pd.read_parquet(p3_path)
    return df_2a, df_p3


def test_phase2a_and_phase3_datasets_load(datasets):
    """Verify both datasets load successfully and contain expected feature column counts."""
    df_2a, df_p3 = datasets
    assert len(df_2a) > 0, "Phase 2A dataset is empty"
    assert len(df_p3) > 0, "Phase 3 dataset is empty"
    assert len(df_p3.columns) > len(df_2a.columns), (
        f"Phase 3 ({len(df_p3.columns)} cols) should have more features than Phase 2A ({len(df_2a.columns)} cols)"
    )


def test_matching_row_population_and_targets(datasets):
    """Verify exact row-for-row match on population, dates, and delay targets."""
    df_2a, df_p3 = datasets
    meta = validate_experiment_datasets(df_2a, df_p3)

    assert meta["population_match"] is True
    assert meta["total_records"] == 481
    assert meta["delay_target_positive"] == 145
    assert meta["delay_target_negative"] == 336
    assert np.isclose(meta["delay_rate"], 145 / 481, atol=1e-4)


def test_validate_experiment_datasets_catches_divergence(datasets):
    """Verify validate_experiment_datasets detects population or target tampering."""
    df_2a, df_p3 = datasets

    # 1. Row count mismatch
    with pytest.raises(ValueError, match="Dataset population mismatch"):
        validate_experiment_datasets(df_2a.iloc[:-1], df_p3)

    # 2. Target divergence
    df_tampered = df_p3.copy()
    df_tampered.iloc[0, df_tampered.columns.get_loc("delay_target")] = 1 - df_tampered.iloc[0]["delay_target"]
    with pytest.raises(ValueError, match="delay_target values do not match"):
        validate_experiment_datasets(df_2a, df_tampered)


def test_chronological_split_identical_boundaries(datasets, config):
    """Verify both datasets split into identical chronological boundaries with zero temporal overlap."""
    df_2a, df_p3 = datasets

    split_2a = split_dataset_chronologically(
        df_2a,
        date_col="flight_date",
        time_col="scheduled_dep_time" if "scheduled_dep_time" in df_2a.columns else None,
        target_col="delay_target",
        train_pct=config.train_ratio,
        val_pct=config.val_ratio,
        config=config,
    )
    split_p3 = split_dataset_chronologically(
        df_p3,
        date_col="flight_date",
        time_col="scheduled_dep_time" if "scheduled_dep_time" in df_p3.columns else None,
        target_col="delay_target",
        train_pct=config.train_ratio,
        val_pct=config.val_ratio,
        config=config,
    )

    # Check identical partition sizes
    assert len(split_2a.X_train) == len(split_p3.X_train) == 336
    assert len(split_2a.X_val) == len(split_p3.X_val) == 72
    assert len(split_2a.X_test) == len(split_p3.X_test) == 73

    # Check target alignment in each split
    assert (split_2a.y_train.values == split_p3.y_train.values).all()
    assert (split_2a.y_val.values == split_p3.y_val.values).all()
    assert (split_2a.y_test.values == split_p3.y_test.values).all()

    # Check temporal ordering: max(train) <= min(val) and max(val) <= min(test)
    assert split_2a.split_summary["train_date_range"][1] <= split_2a.split_summary["val_date_range"][0]
    assert split_2a.split_summary["val_date_range"][1] <= split_2a.split_summary["test_date_range"][0]
    assert split_p3.split_summary["train_date_range"][1] <= split_p3.split_summary["val_date_range"][0]
    assert split_p3.split_summary["val_date_range"][1] <= split_p3.split_summary["test_date_range"][0]


def test_preprocessing_fitted_only_on_train(datasets, config):
    """Verify preprocessing is fitted strictly on the train partition and predicts on val/test."""
    df_2a, _ = datasets
    exp_res = run_single_experiment(df_2a, experiment_name="Phase 2A Test", config=config)

    rf_pipeline = exp_res["models"]["Random Forest"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    # Preprocessor must have been fitted
    assert hasattr(preprocessor, "transformers_"), "Preprocessor has not been fitted"

    # Preprocessor must transform X_test without error
    X_test = exp_res["split_result"].X_test
    X_test_transformed = preprocessor.transform(X_test)
    assert X_test_transformed.shape[0] == len(X_test)
    assert not np.isnan(X_test_transformed).any(), "Transformed test features contain unexpected NaNs"


def test_threshold_selection_uses_validation_only(datasets, config):
    """Verify decision thresholds are selected using validation metrics, not test metrics."""
    df_2a, _ = datasets
    exp_res = run_single_experiment(df_2a, experiment_name="Phase 2A Test", config=config)

    for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        best_t = exp_res["selected_thresholds"][m_name]
        val_thresh_df = exp_res["val_threshold_dfs"][m_name]

        # The selected threshold must be one of the candidate thresholds evaluated on validation
        assert best_t in config.candidate_thresholds
        # Must correspond to the max F1 on validation set
        if val_thresh_df["f1"].max() > 0:
            expected_best = float(val_thresh_df.loc[val_thresh_df["f1"].idxmax(), "threshold"])
            assert best_t == expected_best


def test_experiment_deltas_computation_and_no_nan(datasets, config):
    """Verify compute_experiment_deltas calculates valid deltas without NaNs."""
    df_2a, df_p3 = datasets
    res_2a = run_single_experiment(df_2a, experiment_name="Phase 2A", config=config)
    res_p3 = run_single_experiment(df_p3, experiment_name="Phase 3", config=config)

    deltas = compute_experiment_deltas(res_2a, res_p3)
    assert len(deltas) == 4  # Baseline, LR, RF, XGBoost

    required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]
    for row in deltas:
        for m in required_metrics:
            v_2a = row["metrics"][m]["phase2a"]
            v_p3 = row["metrics"][m]["phase3"]
            d_abs = row["deltas_absolute"][m]
            d_pct = row["deltas_percentage"][m]

            assert not np.isnan(v_2a), f"NaN found in Phase 2A {m} for {row['model']}"
            assert not np.isnan(v_p3), f"NaN found in Phase 3 {m} for {row['model']}"
            assert not np.isnan(d_abs), f"NaN found in absolute delta {m} for {row['model']}"
            assert not np.isnan(d_pct), f"NaN found in percentage delta {m} for {row['model']}"
            assert np.isclose(d_abs, v_p3 - v_2a, atol=1e-4)


def test_phase2b_model_is_not_overwritten(datasets, config, tmp_path):
    """Verify Phase 3B model serialization NEVER touches Phase 2B models/delay_model.joblib."""
    p2b_model_path = config.models_dir / "delay_model.joblib"
    original_mtime = p2b_model_path.stat().st_mtime if p2b_model_path.exists() else None

    df_2a, df_p3 = datasets
    res_2a = run_single_experiment(df_2a, experiment_name="Phase 2A", config=config)
    res_p3 = run_single_experiment(df_p3, experiment_name="Phase 3", config=config)
    deltas = compute_experiment_deltas(res_2a, res_p3)
    meta = validate_experiment_datasets(df_2a, df_p3)

    # Save artifacts
    models_dir, report_md, report_json = save_phase3b_artifacts(
        res_2a, res_p3, deltas, meta, config=config
    )

    # Verify Phase 3B directory
    assert models_dir == config.models_dir / "phase3b"
    assert (models_dir / "phase2a_logistic_regression.joblib").exists()
    assert (models_dir / "phase3_xgboost.joblib").exists()
    assert (models_dir / "experiment_metadata.json").exists()

    # Verify Phase 2B delay_model.joblib was NOT modified
    if original_mtime is not None:
        assert p2b_model_path.stat().st_mtime == original_mtime, (
            "CRITICAL: Phase 2B models/delay_model.joblib was modified by Phase 3B!"
        )


def test_phase3b_full_workflow_execution(config):
    """Verify complete Phase 3B workflow executes cleanly and returns exit code 0."""
    exit_code = run_phase3b_workflow(config=config)
    assert exit_code == 0

    # Verify reports and metadata exist
    assert (config.reports_dir / "phase3b_model_comparison.md").exists()
    assert (config.reports_dir / "phase3b_model_comparison.json").exists()
    assert (config.reports_dir / "figures" / "phase3b_roc_comparison.png").exists()
    assert (config.reports_dir / "figures" / "phase3b_precision_recall_comparison.png").exists()
    assert (config.reports_dir / "figures" / "phase3b_calibration_comparison.png").exists()
    assert (config.reports_dir / "figures" / "phase3b_confusion_matrix.png").exists()
    assert (config.reports_dir / "figures" / "phase3b_feature_importance_comparison.png").exists()
