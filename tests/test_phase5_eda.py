"""Unit and integration tests for Phase 5: Exploratory Data Analysis (EDA).

Verifies:
1. Phase 3 feature dataset loads successfully with expected dimensions (481 rows, 68 columns).
2. Target variable exists, is binary {0, 1}, and exhibits the exact empirical delay rate (30.15%).
3. Data quality checks verify 0 nulls, 0 duplicate rows, and exactly 5 constant features.
4. Strict anti-leakage verification: zero post-flight outcome columns exist in predictive features.
5. All 9 Phase 5 publication-grade figures exist in reports/figures/ and are non-empty.
6. The comprehensive Phase 5 EDA report exists and contains all 16 required sections.
7. The Phase 5 notebook exists, conforms to nbformat 4, and contains structured analytical sections.
8. The EDAAnalyzer engine runs end-to-end and returns structured metrics.
9. CLI execution via main.py --phase5 executes successfully with exit code 0.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eda.eda_analyzer import EDAAnalyzer, run_eda_analysis
from src.utils.config import get_config


@pytest.fixture(scope="module")
def config():
    """Application configuration fixture."""
    return get_config()


@pytest.fixture(scope="module")
def phase3_df(config):
    """Load primary Phase 3 feature dataset."""
    p3_path = config.data_processed_dir / "flights_features_p3.parquet"
    if not p3_path.exists():
        p3_path = config.data_processed_dir / "flights_features_p3.csv"
    assert p3_path.exists(), f"Phase 3 dataset missing at {p3_path}"
    if p3_path.suffix == ".parquet":
        return pd.read_parquet(p3_path)
    return pd.read_csv(p3_path)


def test_phase3_dataset_loads_for_eda(phase3_df):
    """Verify Phase 3 dataset exists and has expected dimensions."""
    assert len(phase3_df) == 481, f"Expected 481 rows, got {len(phase3_df)}"
    assert phase3_df.shape[1] == 68, f"Expected 68 columns, got {phase3_df.shape[1]}"


def test_target_distribution_and_balance(phase3_df):
    """Verify target variable delay_target is binary and matches base rate."""
    assert "delay_target" in phase3_df.columns, "delay_target column missing"
    unique_vals = set(phase3_df["delay_target"].unique())
    assert unique_vals.issubset({0, 1}), f"delay_target contains non-binary values: {unique_vals}"

    counts = phase3_df["delay_target"].value_counts().to_dict()
    assert counts[0] == 336, f"Expected 336 on-time flights, got {counts.get(0)}"
    assert counts[1] == 145, f"Expected 145 delayed flights, got {counts.get(1)}"

    delay_rate = phase3_df["delay_target"].mean()
    assert round(delay_rate, 4) == 0.3015, f"Expected delay rate 0.3015, got {delay_rate}"


def test_data_quality_audit_metrics(phase3_df):
    """Verify zero missing values, zero duplicates, and identify constant columns."""
    # Zero missing values
    null_count = phase3_df.isnull().sum().sum()
    assert null_count == 0, f"Expected 0 null cells, got {null_count}"

    # Zero exact duplicates
    dup_count = phase3_df.duplicated().sum()
    assert dup_count == 0, f"Expected 0 duplicate rows, got {dup_count}"

    # Constant columns check
    constant_cols = [c for c in phase3_df.columns if phase3_df[c].nunique(dropna=False) <= 1]
    assert len(constant_cols) == 5, f"Expected 5 constant columns, got {len(constant_cols)}: {constant_cols}"
    expected_constants = {"cancelled", "diverted", "is_month_end", "quarter", "season"}
    assert set(constant_cols) == expected_constants


def test_no_post_flight_leakage_in_eda_features(config, phase3_df):
    """Verify no post-flight outcome columns exist in the predictive Phase 3 feature matrix."""
    forbidden_cols = config.post_flight_leakage_columns
    found_forbidden = [c for c in forbidden_cols if c in phase3_df.columns]
    assert len(found_forbidden) == 0, f"Forbidden leakage columns found in Phase 3 features: {found_forbidden}"


def test_eda_figures_generated_and_non_empty(config):
    """Verify that all 9 Phase 5 publication-grade figures exist and are non-empty."""
    expected_figures = [
        "phase5_target_distribution.png",
        "phase5_temporal_delay_patterns.png",
        "phase5_airline_delay_performance.png",
        "phase5_airport_hotspots.png",
        "phase5_route_delay_analysis.png",
        "phase5_historical_delay_features.png",
        "phase5_numerical_distributions.png",
        "phase5_correlation_heatmap.png",
        "phase5_outlier_boxplots.png",
    ]
    for fig_name in expected_figures:
        fig_path = config.figures_dir / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_path} is suspiciously small ({fig_path.stat().st_size} bytes)"


def test_phase5_eda_report_exists_and_contains_required_sections(config):
    """Verify reports/phase5_eda_report.md exists and contains all 16 required sections."""
    rep_path = config.reports_dir / "phase5_eda_report.md"
    assert rep_path.exists(), f"Phase 5 EDA report missing at {rep_path}"
    assert rep_path.stat().st_size > 5000, f"Report {rep_path} is too small"

    content = rep_path.read_text(encoding="utf-8")
    required_sections = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Data Quality Assessment",
        "4. Target Variable Analysis",
        "5. Temporal Analysis",
        "6. Airline Analysis",
        "7. Airport Analysis",
        "8. Route Analysis",
        "9. Historical Delay Analysis",
        "10. Weather Analysis",
        "11. Numerical Feature Analysis",
        "12. Correlation Analysis",
        "13. Outlier and Suspicious Value Analysis",
        "14. Key EDA Findings",
        "15. Implications for Phase 6 Machine Learning",
        "16. Limitations",
    ]
    for sec in required_sections:
        assert sec in content, f"Missing required section in EDA report: '{sec}'"


def test_phase5_notebook_validity(config):
    """Verify notebooks/07_phase5_eda.ipynb exists and is valid nbformat 4."""
    nb_path = config.project_root / "notebooks" / "07_phase5_eda.ipynb"
    assert nb_path.exists(), f"Phase 5 notebook missing at {nb_path}"

    nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_json.get("nbformat") == 4, "Notebook is not nbformat version 4"
    assert "cells" in nb_json and len(nb_json["cells"]) >= 15, "Notebook has insufficient cells"

    # Verify code cells are present
    code_cells = [c for c in nb_json["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 10, f"Expected at least 10 code cells, found {len(code_cells)}"


def test_eda_analyzer_full_workflow(config):
    """Verify EDAAnalyzer runs full analysis and produces all expected metric keys."""
    analyzer = EDAAnalyzer(config=config)
    results = analyzer.run_full_analysis()

    expected_keys = [
        "overview",
        "data_quality",
        "target",
        "temporal",
        "airlines",
        "airports",
        "routes",
        "historical_delay",
        "weather",
        "numerical_features",
        "correlations",
        "outliers",
    ]
    for k in expected_keys:
        assert k in results, f"Missing key '{k}' in EDA analysis results"

    assert results["target"]["delayed_count"] == 145
    assert results["target"]["on_time_count"] == 336
    assert results["overview"]["total_features"] == 68
    assert results["data_quality"]["columns_with_missing_values_count"] == 0


def test_main_cli_phase5_execution(config):
    """Verify main.py run_pipeline(do_phase5=True) executes with exit code 0."""
    from main import run_pipeline

    exit_code = run_pipeline(do_phase5=True)
    assert exit_code == 0, f"run_pipeline(do_phase5=True) failed with exit code {exit_code}"
