"""Unit and integration tests for Phase 3 Advanced Feature Engineering."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.config import get_config
from src.features.feature_engineering import assert_no_target_leakage
from src.features.advanced_features import (
    build_phase3_date_features,
    build_phase3_time_features,
    build_phase3_route_and_airport_features,
    build_phase3_feature_pipeline,
)
from src.features.historical_features import (
    HistoricalFeatureCalculator,
    construct_departure_timestamp,
)
from src.features.weather_features import (
    WeatherIntegrator,
    WeatherValidationResult,
)
from src.features.phase3_reporter import generate_phase3_reports
from main import run_phase3


@pytest.fixture
def sample_flight_df() -> pd.DataFrame:
    """Create a minimal clean pre-departure flight DataFrame spanning multiple days and routes."""
    return pd.DataFrame({
        "flight_date": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-31"
        ]),
        "carrier": ["AA", "DL", "AA", "UA", "AA"],
        "origin": ["JFK", "ORD", "JFK", "LAX", "JFK"],
        "dest": ["LAX", "ATL", "LAX", "SFO", "ORD"],
        "crs_dep_time": ["0630", "1215", "1430", "2045", "0030"],
        "crs_arr_time": ["0945", "1430", "1750", "2200", "0215"],
        "distance": [2475.0, 606.0, 2475.0, 337.0, 740.0],
        "delay_target": [0, 1, 1, 0, 1],
    })


# ==============================================================================
# 1. ADVANCED DATE FEATURES TESTS
# ==============================================================================

def test_phase3_date_features():
    """Verify is_month_start, is_month_end, quarter, and season."""
    df = pd.DataFrame({
        "flight_date": pd.to_datetime([
            "2024-01-01",  # Month start, Winter, Q1
            "2024-01-31",  # Month end, Winter, Q1
            "2024-04-15",  # Mid-month, Spring, Q2
            "2024-07-20",  # Summer, Q3
            "2024-10-10",  # Fall, Q4
        ])
    })

    date_feats = build_phase3_date_features(df)

    # Month start & end
    assert date_feats["is_month_start"].tolist() == [1, 0, 0, 0, 0]
    assert date_feats["is_month_end"].tolist() == [0, 1, 0, 0, 0]

    # Quarter
    assert date_feats["quarter"].tolist() == [1, 1, 2, 3, 4]

    # Season
    assert date_feats["season"].tolist() == ["winter", "winter", "spring", "summer", "fall"]


# ==============================================================================
# 2. ADVANCED TIME FEATURES TESTS
# ==============================================================================

def test_phase3_time_features():
    """Verify time of day, 4-hour buckets, and sin/cos cyclical encodings."""
    df = pd.DataFrame({
        "crs_dep_time": ["0215", "0830", "1445", "2100"],
        "crs_arr_time": ["0500", "1115", "1800", "2330"],
    })

    time_feats = build_phase3_time_features(df)

    # Time of day
    assert time_feats["dep_time_of_day"].tolist() == ["night", "morning", "afternoon", "evening"]
    assert time_feats["arr_time_of_day"].tolist() == ["night", "morning", "evening", "night"]

    # 4-hour buckets
    assert time_feats["dep_time_bucket"].tolist() == ["00-04", "08-12", "12-16", "20-24"]
    assert time_feats["arr_time_bucket"].tolist() == ["04-08", "08-12", "16-20", "20-24"]

    # Cyclical bounded in [-1, 1]
    for col in ["arr_hour_sin", "arr_hour_cos", "dep_minute_sin", "dep_minute_cos"]:
        assert col in time_feats.columns
        assert (time_feats[col] >= -1.0).all() and (time_feats[col] <= 1.0).all()

    # Trigonometric identity check sin^2 + cos^2 == 1
    arr_trig_sum = time_feats["arr_hour_sin"] ** 2 + time_feats["arr_hour_cos"] ** 2
    assert np.allclose(arr_trig_sum, 1.0, atol=1e-5)

    dep_min_trig_sum = time_feats["dep_minute_sin"] ** 2 + time_feats["dep_minute_cos"] ** 2
    assert np.allclose(dep_min_trig_sum, 1.0, atol=1e-5)


# ==============================================================================
# 3. ROUTE & AIRPORT FEATURES TESTS
# ==============================================================================

def test_route_distance_categories():
    """Verify distance category binning: Short-haul (<500), Medium-haul (500-1499), Long-haul (>=1500)."""
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "origin": ["JFK", "ORD", "LAX"],
        "dest": ["BOS", "ATL", "JFK"],
        "distance": [187.0, 606.0, 2475.0],
        "crs_dep_time": ["0800", "0900", "1000"],
    })

    feats, _ = build_phase3_route_and_airport_features(df)
    assert feats["distance_category"].tolist() == ["short_haul", "medium_haul", "long_haul"]


def test_prior_route_frequency_strictly_prior():
    """Verify prior_route_frequency only counts strictly prior flights on the route (t < T)."""
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
        "origin": ["JFK", "JFK", "ORD", "JFK"],
        "dest": ["LAX", "LAX", "ATL", "LAX"],
        "crs_dep_time": ["0600", "1200", "1300", "1800"],
        "distance": [2475.0, 2475.0, 606.0, 2475.0],
    })

    feats, _ = build_phase3_route_and_airport_features(df)

    # JFK_LAX appears at 06:00, 12:00, and 18:00
    # Flight 0 (06:00): prior = 0
    # Flight 1 (12:00): prior = 1
    # Flight 3 (18:00): prior = 2
    # Flight 2 (ORD_ATL 13:00): prior = 0
    assert feats.loc[0, "prior_route_frequency"] == 0
    assert feats.loc[1, "prior_route_frequency"] == 1
    assert feats.loc[2, "prior_route_frequency"] == 0
    assert feats.loc[3, "prior_route_frequency"] == 2


def test_airport_features_strictly_prior():
    """Verify airport prior volume and delay rates are strictly prior (t < T)."""
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        "origin": ["JFK", "JFK", "JFK"],
        "dest": ["LAX", "ORD", "MIA"],
        "crs_dep_time": ["0800", "1200", "1600"],
        "distance": [2475.0, 740.0, 1090.0],
        "delay_target": [1, 0, 1],
    })

    feats, _ = build_phase3_route_and_airport_features(df)

    # Flight 0 (08:00): First flight at JFK -> prior volume = 0, prior delay rate = 0.0
    assert feats.loc[0, "prior_origin_flight_volume"] == 0
    assert feats.loc[0, "prior_origin_delay_rate"] == 0.0

    # Flight 1 (12:00): 1 prior flight (delayed=1) -> prior volume = 1, prior delay rate = 1.0
    assert feats.loc[1, "prior_origin_flight_volume"] == 1
    assert feats.loc[1, "prior_origin_delay_rate"] == 1.0

    # Flight 2 (16:00): 2 prior flights (1 delayed, 1 on-time) -> prior volume = 2, delay rate = 0.5
    assert feats.loc[2, "prior_origin_flight_volume"] == 2
    assert feats.loc[2, "prior_origin_delay_rate"] == 0.5


# ==============================================================================
# 4. MULTI-GRANULAR HISTORICAL DELAY FEATURES TESTS
# ==============================================================================

def test_phase3_historical_features_strictly_prior():
    """Verify multi-granularity historical features enforce strict prior contract and global fallback."""
    config = get_config()
    calc = HistoricalFeatureCalculator(config=config)

    # Construct a sequence of 5 flights for carrier "AA"
    df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]),
        "carrier": ["AA", "AA", "AA", "AA", "AA"],
        "origin": ["JFK", "JFK", "JFK", "JFK", "JFK"],
        "dest": ["LAX", "LAX", "LAX", "LAX", "LAX"],
        "crs_dep_time": ["0800", "1200", "0800", "1400", "0900"],
        "delay_target": [1, 1, 0, 1, 0],
    })

    hist_df, notes, audit = calc.compute_phase3_historical_features(
        df, target_col="delay_target", min_history=3
    )

    # Check that required columns exist
    expected_cols = [
        "carrier_prior_flight_count", "carrier_prior_delay_count", "carrier_prior_delay_rate",
        "origin_prior_flight_count", "origin_prior_delay_count", "origin_prior_delay_rate",
        "dest_prior_flight_count", "dest_prior_delay_count", "dest_prior_delay_rate",
        "route_prior_flight_count", "route_prior_delay_count", "route_prior_delay_rate",
        "carrier_origin_hour_prior_flight_count", "carrier_origin_hour_prior_delay_count",
        "carrier_origin_hour_prior_delay_rate",
    ]
    for c in expected_cols:
        assert c in hist_df.columns

    # Flight 0: First flight, prior count = 0, delay_count = 0, rate = 0.0 (no prior flights exist)
    assert hist_df.loc[0, "carrier_prior_flight_count"] == 0
    assert hist_df.loc[0, "carrier_prior_delay_count"] == 0
    assert hist_df.loc[0, "carrier_prior_delay_rate"] == 0.0

    # Flight 1: 1 prior flight (delayed). Group count < 3, so falls back to strictly prior global rate (1.0)
    assert hist_df.loc[1, "carrier_prior_flight_count"] == 1
    assert hist_df.loc[1, "carrier_prior_delay_count"] == 1
    assert hist_df.loc[1, "carrier_prior_delay_rate"] == 1.0

    # Flight 2: 2 prior flights (both delayed). Group count < 3, global fallback = 2/2 = 1.0
    assert hist_df.loc[2, "carrier_prior_flight_count"] == 2
    assert hist_df.loc[2, "carrier_prior_delay_count"] == 2
    assert hist_df.loc[2, "carrier_prior_delay_rate"] == 1.0

    # Flight 3: 3 prior flights (2 delayed, 1 on-time). Group count == 3 >= min_history (3),
    # so group rate is used: 2 / 3 = 0.6667
    assert hist_df.loc[3, "carrier_prior_flight_count"] == 3
    assert hist_df.loc[3, "carrier_prior_delay_count"] == 2
    assert np.isclose(hist_df.loc[3, "carrier_prior_delay_rate"], 2.0 / 3.0, atol=1e-3)

    # Flight 4: 4 prior flights (3 delayed, 1 on-time). Group rate: 3 / 4 = 0.75
    assert hist_df.loc[4, "carrier_prior_flight_count"] == 4
    assert hist_df.loc[4, "carrier_prior_delay_count"] == 3
    assert np.isclose(hist_df.loc[4, "carrier_prior_delay_rate"], 0.75, atol=1e-3)

    # Check coverage audit
    assert "carrier" in audit
    assert audit["carrier"]["total_records"] == 5


# ==============================================================================
# 5. WEATHER INTEGRATION & SEVERE WEATHER FLAGS TESTS
# ==============================================================================

def test_weather_temporal_availability_contract():
    """Verify that weather integration strictly satisfies observation_time <= scheduled_dep_time."""
    config = get_config()
    integrator = WeatherIntegrator(config=config)

    flight_df = pd.DataFrame({
        "flight_date": pd.to_datetime(["2024-01-01"]),
        "origin": ["JFK"],
        "crs_dep_time": ["1000"],
    })

    # Weather observations: one before (09:00), one after (11:00)
    weather_df = pd.DataFrame({
        "station_id": ["JFK", "JFK"],
        "weather_timestamp": ["2024-01-01 09:00:00", "2024-01-01 11:00:00"],
        "temperature_c": [5.0, 12.0],
        "precipitation_inches": [0.0, 0.5],
        "visibility_miles": [10.0, 2.0],
        "wind_speed_knots": [15.0, 30.0],
    })

    joined = integrator.join_nearest_prior_weather(flight_df, weather_df)
    assert len(joined) == 1
    # Must pick the 09:00 observation (5.0 C), NOT the 11:00 future observation (12.0 C)
    assert joined.loc[0, "origin_temperature_c"] == 5.0
    assert joined.loc[0, "origin_weather_observed_missing"] == 0


def test_severe_weather_indicators():
    """Verify that severe weather indicator flags are accurately derived without fabricating data."""
    config = get_config()
    integrator = WeatherIntegrator(config=config)

    weather_features = pd.DataFrame({
        "origin_precipitation_inches": [0.25, 0.0, 0.0, 0.0],
        "origin_snow_inches": [0.0, 2.5, 0.0, 0.0],
        "origin_visibility_miles": [10.0, 5.0, 1.5, 10.0],
        "origin_weather_condition": ["rain", "snow", "fog", "clear"],
        "origin_wind_speed_knots": [10.0, 15.0, 5.0, 45.0],
    })

    derived = integrator.derive_severe_weather_indicators(weather_features)

    # Check flags
    assert derived["origin_is_rain"].tolist() == [1, 0, 0, 0]
    assert derived["origin_is_snow"].tolist() == [0, 1, 0, 0]
    assert derived["origin_is_fog"].tolist() == [0, 0, 1, 0]
    assert derived["origin_is_low_visibility"].tolist() == [0, 0, 1, 0]
    assert derived["origin_is_storm"].tolist() == [0, 0, 0, 1]  # 45 knots wind trigger


# ==============================================================================
# 6. LEAKAGE AUDIT TESTS
# ==============================================================================

def test_leakage_audit_phase3_passing(sample_flight_df):
    """Verify that a legitimate Phase 3 pipeline dataframe passes the centralized leakage audit."""
    config = get_config()
    base_df, _ = build_phase3_feature_pipeline(sample_flight_df, config=config)
    calc = HistoricalFeatureCalculator(config=config)
    hist_df, _, _ = calc.compute_phase3_historical_features(base_df, target_col="delay_target")

    combined = pd.concat([base_df.drop(columns=["delay_target"]), hist_df], axis=1)
    combined["delay_target"] = sample_flight_df["delay_target"]

    # Should pass without exception
    assert_no_target_leakage(combined, config=config)


def test_leakage_audit_catches_derived_leakage(sample_flight_df):
    """Verify that derived target or temporal leakage columns are caught by assert_no_target_leakage."""
    config = get_config()
    base_df, _ = build_phase3_feature_pipeline(sample_flight_df, config=config)

    # Inject forbidden full-dataset aggregation
    leaky_df = base_df.copy()
    leaky_df["full_dataset_delay_rate"] = 0.35

    with pytest.raises(ValueError) as exc:
        assert_no_target_leakage(leaky_df, config=config)
    assert "Derived Temporal Leakage Detected" in str(exc.value)

    # Inject post-flight column
    leaky_df2 = base_df.copy()
    leaky_df2["actual_elapsed_time"] = 120.0

    with pytest.raises(ValueError) as exc:
        assert_no_target_leakage(leaky_df2, config=config)
    assert "Direct Target/Post-Flight Leakage Detected" in str(exc.value)


# ==============================================================================
# 7. END-TO-END PIPELINE & REPORT GENERATION TESTS
# ==============================================================================

def test_phase3_pipeline_and_report_generation(tmp_path):
    """Verify end-to-end Phase 3 pipeline execution and report output generation."""
    config = get_config()

    # Run Phase 3
    exit_code = run_phase3(config=config)
    assert exit_code == 0

    # Verify output datasets exist
    p3_parquet = config.data_processed_dir / "flights_features_p3.parquet"
    p3_csv = config.data_processed_dir / "flights_features_p3.csv"
    assert p3_parquet.exists()
    assert p3_csv.exists()

    df_p3 = pd.read_parquet(p3_parquet)
    assert len(df_p3) > 0
    assert "delay_target" in df_p3.columns
    assert "is_month_start" in df_p3.columns
    assert "carrier_prior_flight_count" in df_p3.columns
    assert "prior_route_frequency" in df_p3.columns
    assert "distance_category" in df_p3.columns

    # Verify reports exist
    report_md = config.reports_dir / "phase3_feature_report.md"
    report_json = config.reports_dir / "phase3_feature_report.json"
    assert report_md.exists()
    assert report_json.exists()

    with open(report_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["feature_version"] == "phase3"
    assert meta["total_records"] == len(df_p3)
    assert meta["total_features"] == len(df_p3.columns)
    assert "historical_coverage_audit" in meta
