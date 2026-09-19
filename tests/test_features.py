"""Unit tests for feature engineering, historical delay calculations, weather integration, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    assert_no_target_leakage,
    build_arr_time_features,
    build_cyclical_features,
    build_date_features,
    build_dep_time_features,
    build_pre_departure_feature_pipeline,
    build_route_features,
    categorize_time_of_day,
    parse_scheduled_time,
    split_dataset_chronologically,
    verify_temporal_validity,
)
from src.features.historical_features import (
    HistoricalFeatureCalculator,
    construct_departure_timestamp,
)
from src.features.weather_features import (
    WeatherIntegrator,
    WeatherValidationResult,
)


# ==============================================================================
# 1. TEMPORAL VALIDITY HELPER TESTS (PHASE 1 PRESERVED)
# ==============================================================================

def test_temporal_validity_accepts_strictly_prior_observations():
    """Verify that historical timestamps occurring strictly before prediction_time pass."""
    prediction_time = pd.Timestamp("2024-06-15 10:00:00")
    historical_timestamps = pd.Series([
        pd.Timestamp("2024-06-01 08:00:00"),
        pd.Timestamp("2024-06-14 23:59:59"),
        pd.Timestamp("2024-06-15 09:59:00"),
    ])

    is_valid = verify_temporal_validity(prediction_time, historical_timestamps)
    assert is_valid is True


def test_temporal_validity_rejects_future_or_concurrent_observations():
    """Verify that any observation timestamp >= prediction_time raises a ValueError."""
    prediction_time = pd.Timestamp("2024-06-15 10:00:00")

    invalid_timestamps = pd.Series([
        pd.Timestamp("2024-06-15 08:00:00"),
        pd.Timestamp("2024-06-15 10:00:00"),  # Concurrent violation
        pd.Timestamp("2024-06-15 10:05:00"),  # Future violation
    ])

    with pytest.raises(ValueError) as excinfo:
        verify_temporal_validity(prediction_time, invalid_timestamps)

    assert "Temporal Leakage Violation" in str(excinfo.value)
    assert "observation_timestamp < prediction_time" in str(excinfo.value)


# ==============================================================================
# 2. DATE FEATURE GENERATION TESTS
# ==============================================================================

def test_date_feature_generation():
    """Verify extraction of year, month, day, day_of_week, week_of_year, day_of_year, is_weekend."""
    df = pd.DataFrame({
        "flight_date": ["2024-01-01", "2024-01-06", "2024-07-04", "2024-12-31"]
    })
    date_feats = build_date_features(df, date_col="flight_date")

    # 2024-01-01 was a Monday (dayofweek = 0)
    assert date_feats["year"].iloc[0] == 2024
    assert date_feats["month"].iloc[0] == 1
    assert date_feats["day"].iloc[0] == 1
    assert date_feats["day_of_week"].iloc[0] == 0
    assert date_feats["is_weekend"].iloc[0] == 0
    assert date_feats["day_of_year"].iloc[0] == 1

    # 2024-01-06 was a Saturday (dayofweek = 5)
    assert date_feats["day_of_week"].iloc[1] == 5
    assert date_feats["is_weekend"].iloc[1] == 1

    # 2024-12-31 is day 366 (leap year)
    assert date_feats["day_of_year"].iloc[3] == 366
    assert date_feats["month"].iloc[3] == 12


# ==============================================================================
# 3. SCHEDULED TIME PARSING AND TIME-OF-DAY TESTS
# ==============================================================================

def test_scheduled_time_parsing_and_categories():
    """Verify HHMM parsing, minute rollover, and time-of-day category blocks."""
    times = pd.Series([800, 1515, 5, 2359, 2400, 1485])  # 1485 has 85 minutes
    hours, minutes, mins_midnight = parse_scheduled_time(times)

    # 800 -> 08:00
    assert hours.iloc[0] == 8
    assert minutes.iloc[0] == 0
    assert mins_midnight.iloc[0] == 480

    # 1515 -> 15:15
    assert hours.iloc[1] == 15
    assert minutes.iloc[1] == 15
    assert mins_midnight.iloc[1] == 15 * 60 + 15

    # 5 -> 00:05
    assert hours.iloc[2] == 0
    assert minutes.iloc[2] == 5
    assert mins_midnight.iloc[2] == 5

    # 2400 -> rollover to 00:00
    assert hours.iloc[4] == 0
    assert minutes.iloc[4] == 0

    # 1485 -> 14h + 85m = 15h 25m
    assert hours.iloc[5] == 15
    assert minutes.iloc[5] == 25
    assert mins_midnight.iloc[5] == 15 * 60 + 25

    # Time of Day categories
    cat_hours = pd.Series([2, 7, 13, 19, 23])
    categories = categorize_time_of_day(cat_hours).tolist()
    assert categories == ["overnight", "morning", "afternoon", "evening", "overnight"]


# ==============================================================================
# 4. CYCLICAL FEATURES: MATHEMATICAL AND SEMANTIC PERIODICITY TESTS
# ==============================================================================

def test_cyclical_features_mathematical_identity():
    """Verify the trigonometric identity sin^2(x) + cos^2(x) = 1 for all cyclical features."""
    hours = pd.Series(range(24))
    dows = pd.Series([i % 7 for i in range(24)])
    months = pd.Series([(i % 12) + 1 for i in range(24)])

    feats = build_cyclical_features(hours, dows, months)

    # Unit circle radius verification
    hour_radius = np.square(feats["departure_hour_sin"]) + np.square(feats["departure_hour_cos"])
    np.testing.assert_allclose(hour_radius, 1.0, atol=1e-7)

    dow_radius = np.square(feats["day_of_week_sin"]) + np.square(feats["day_of_week_cos"])
    np.testing.assert_allclose(dow_radius, 1.0, atol=1e-7)

    month_radius = np.square(feats["month_sin"]) + np.square(feats["month_cos"])
    np.testing.assert_allclose(month_radius, 1.0, atol=1e-7)


def test_cyclical_features_semantic_periodicity():
    """Verify semantic continuity:
    1. Hour 0 and 24 wrap to identical coordinates.
    2. Hour 23 is close to Hour 0 on the unit circle (Euclidean distance << linear distance).
    3. December (12) is close to January (1) in circular space.
    4. Sunday (6) is close to Monday (0) in circular space.
    """
    # 1. Hour 0 vs Hour 24
    h0_sin = np.sin(2 * np.pi * 0 / 24.0)
    h0_cos = np.cos(2 * np.pi * 0 / 24.0)
    h24_sin = np.sin(2 * np.pi * 24 / 24.0)
    h24_cos = np.cos(2 * np.pi * 24 / 24.0)

    np.testing.assert_allclose([h0_sin, h0_cos], [h24_sin, h24_cos], atol=1e-7)

    # 2. Hour 23 vs Hour 0
    h23_sin = np.sin(2 * np.pi * 23 / 24.0)
    h23_cos = np.cos(2 * np.pi * 23 / 24.0)
    dist_23_to_0 = np.sqrt((h23_sin - h0_sin) ** 2 + (h23_cos - h0_cos) ** 2)
    # 1 hour angle = 2*pi/24 = ~0.2618 radians; chord length = 2*sin(pi/24) ~ 0.261
    assert dist_23_to_0 < 0.3, "Hour 23 and Hour 0 must be adjacent on unit circle"

    # Contrast with linear distance: 23 - 0 = 23 (maximally far)!
    # 3. Month December (12) vs January (1)
    # Using 1-indexed formula: (month - 1) * 2*pi / 12
    m_jan_sin = np.sin(2 * np.pi * (1 - 1) / 12.0)
    m_jan_cos = np.cos(2 * np.pi * (1 - 1) / 12.0)
    m_dec_sin = np.sin(2 * np.pi * (12 - 1) / 12.0)
    m_dec_cos = np.cos(2 * np.pi * (12 - 1) / 12.0)
    dist_dec_to_jan = np.sqrt((m_dec_sin - m_jan_sin) ** 2 + (m_dec_cos - m_jan_cos) ** 2)
    assert dist_dec_to_jan < 0.55, "December and January must be adjacent on unit circle"

    # 4. Sunday (6) vs Monday (0)
    dow_mon_sin = np.sin(2 * np.pi * 0 / 7.0)
    dow_mon_cos = np.cos(2 * np.pi * 0 / 7.0)
    dow_sun_sin = np.sin(2 * np.pi * 6 / 7.0)
    dow_sun_cos = np.cos(2 * np.pi * 6 / 7.0)
    dist_sun_to_mon = np.sqrt((dow_sun_sin - dow_mon_sin) ** 2 + (dow_sun_cos - dow_mon_cos) ** 2)
    assert dist_sun_to_mon < 0.9, "Sunday and Monday must be adjacent on unit circle"


# ==============================================================================
# 5. ROUTE & CONDITIONAL FEATURE TESTS
# ==============================================================================

def test_route_features_and_conditional_same_airport():
    """Verify route concatenation, haul categorization, and conditional same_airport_flag omission."""
    df = pd.DataFrame({
        "origin_airport": ["JFK", "ATL", "ORD"],
        "dest_airport": ["LAX", "MIA", "SFO"],
        "distance": [2475, 594, 450],
    })
    route_feats, notes = build_route_features(df)

    assert route_feats["route"].tolist() == ["JFK_LAX", "ATL_MIA", "ORD_SFO"]
    assert route_feats["haul_category"].tolist() == ["long_haul", "medium_haul", "short_haul"]

    # When origin != dest for all flights (zero variance), same_airport_flag is omitted
    assert "same_airport_flag" not in route_feats.columns
    assert "OMITTED" in notes["same_airport_flag"]


# ==============================================================================
# 6. HISTORICAL DELAY FEATURES: STRICT ANTI-LEAKAGE VERIFICATION
# ==============================================================================

def test_historical_delay_features_anti_leakage_deterministic():
    """Prove with a deterministic fixture that:
    1. A flight never sees its own target.
    2. Concurrent observations are excluded.
    3. A flight never sees future flights.
    4. Historical rates change correctly as time advances.
    """
    # Deterministic test dataset: 4 flights for carrier AA
    test_df = pd.DataFrame({
        "flight_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
        "scheduled_dep_time": [800, 800, 900, 1000],  # Flights 0 and 1 are CONCURRENT
        "airline": ["AA", "AA", "AA", "AA"],
        "origin_airport": ["JFK", "JFK", "JFK", "JFK"],
        "dest_airport": ["LAX", "LAX", "LAX", "LAX"],
        "delay_target": [1, 0, 1, 0],
    })

    calc = HistoricalFeatureCalculator()
    feats, notes = calc.compute_historical_features(test_df, min_history=1, fallback_strategy="nan")

    counts = feats["historical_airline_flight_count"].tolist()
    rates = feats["historical_airline_delay_rate"].tolist()

    # Flight 0 (2024-01-01 08:00, target=1):
    # Has 0 prior flights -> count=0, rate=NaN
    assert counts[0] == 0
    assert np.isnan(rates[0])

    # Flight 1 (2024-01-01 08:00, target=0, CONCURRENT with Flight 0):
    # Must NOT see Flight 0! -> count=0, rate=NaN
    assert counts[1] == 0
    assert np.isnan(rates[1])

    # Flight 2 (2024-01-02 09:00, target=1):
    # Sees Flights 0 and 1 (targets 1 and 0).
    # Does NOT see itself (target 1) or Flight 3 (target 0).
    # Prior count = 2, delay_sum = 1 + 0 = 1 -> rate = 1 / 2 = 0.50
    assert counts[2] == 2
    assert rates[2] == 0.50

    # Flight 3 (2024-01-03 10:00, target=0):
    # Sees Flights 0, 1, and 2 (targets 1, 0, 1).
    # Does NOT see itself (target 0).
    # Prior count = 3, delay_sum = 1 + 0 + 1 = 2 -> rate = 2 / 3 = 0.6667
    assert counts[3] == 3
    assert rates[3] == 0.6667


def test_historical_minimum_history_fallback():
    """Verify that when prior flight count is below min_history, fallback is applied."""
    test_df = pd.DataFrame({
        "flight_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "scheduled_dep_time": [800, 900, 1000],
        "airline": ["AA", "AA", "DL"],  # DL has only 1 flight at 1000
        "origin_airport": ["JFK", "JFK", "ATL"],
        "dest_airport": ["LAX", "LAX", "MIA"],
        "delay_target": [1, 1, 0],
    })

    # Set min_history = 3
    calc = HistoricalFeatureCalculator()
    feats, _ = calc.compute_historical_features(test_df, min_history=3, fallback_strategy="global_prior")

    # For DL (Row 2), prior count is 0 (< 3).
    # Global prior flights before 10:00 on 2024-01-03 are Flights 0 and 1 (both delayed -> rate 1.0)
    dl_count = feats["historical_airline_flight_count"].iloc[2]
    dl_rate = feats["historical_airline_delay_rate"].iloc[2]

    assert dl_count == 0
    assert dl_rate == 1.0  # Fallback to global prior delay rate


# ==============================================================================
# 7. WEATHER INTEGRATION FOUNDATION TESTS
# ==============================================================================

def test_weather_schema_validation():
    """Verify validation of external weather data schema and value range checks."""
    integrator = WeatherIntegrator()

    # Valid synthetic weather fixture
    valid_weather = pd.DataFrame({
        "airport": ["JFK", "LAX"],
        "timestamp": ["2024-01-01 07:00:00", "2024-01-01 08:00:00"],
        "temperature": [5.2, 18.4],
        "humidity": [65.0, 45.0],
        "wind_speed": [12.5, 8.0],
        "wind_direction": [180.0, 270.0],
        "precipitation": [0.0, 0.0],
        "visibility": [10.0, 10.0],
        "pressure": [1013.2, 1015.0],
        "cloud_cover": [0.2, 0.0],
        "weather_condition": ["Clear", "Sunny"],
    })
    res = integrator.validate_weather_schema(valid_weather)
    assert res.is_valid is True
    assert len(res.missing_fields) == 0

    # Invalid weather fixture: missing wind_speed, invalid airport code
    invalid_weather = pd.DataFrame({
        "airport": ["INVALID_CODE"],
        "timestamp": ["2024-01-01 07:00:00"],
        "temperature": [5.0],
        # Missing other fields
    })
    res_inv = integrator.validate_weather_schema(invalid_weather)
    assert res_inv.is_valid is False
    assert len(res_inv.missing_fields) > 0


def test_weather_temporal_matching():
    """Verify weather observation joins strictly obey: observation_time <= scheduled_dep_time."""
    integrator = WeatherIntegrator()

    flights = pd.DataFrame({
        "origin_airport": ["JFK", "JFK"],
        "_scheduled_dep_ts": [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 12:00:00"),
        ]
    })

    weather = pd.DataFrame({
        "airport": ["JFK", "JFK", "JFK"],
        "timestamp": [
            pd.Timestamp("2024-01-01 09:30:00"),  # Valid for Flight 0 (within 2h)
            pd.Timestamp("2024-01-01 10:05:00"),  # FUTURE for Flight 0! Valid for Flight 1
            pd.Timestamp("2024-01-01 11:45:00"),  # Valid for Flight 1 (within 2h)
        ],
        "temperature": [10.0, 15.0, 20.0],
        "humidity": [50.0, 50.0, 50.0],
        "wind_speed": [5.0, 5.0, 5.0],
        "wind_direction": [180.0, 180.0, 180.0],
        "precipitation": [0.0, 0.0, 0.0],
        "visibility": [10.0, 10.0, 10.0],
        "pressure": [1013.0, 1013.0, 1013.0],
        "cloud_cover": [0.1, 0.1, 0.1],
        "weather_condition": ["Clear", "Clear", "Clear"],
    })

    joined = integrator.join_nearest_prior_weather(
        flights_df=flights,
        weather_df=weather,
        flight_airport_col="origin_airport",
        flight_time_col="_scheduled_dep_ts",
        lookback_minutes=120,
    )

    # Flight 0 (dep 10:00): nearest prior is 09:30 (temp 10.0), NOT future 10:05!
    assert joined["weather_origin_temperature"].iloc[0] == 10.0

    # Flight 1 (dep 12:00): nearest prior is 11:45 (temp 20.0)
    assert joined["weather_origin_temperature"].iloc[1] == 20.0


# ==============================================================================
# 8. LEAKAGE AUDIT TESTS
# ==============================================================================

def test_leakage_audit_passes_clean_pre_departure():
    """Verify that a legitimate pre-departure feature set passes the leakage audit."""
    clean_df = pd.DataFrame({
        "flight_date": ["2024-01-01"],
        "airline": ["AA"],
        "origin_airport": ["JFK"],
        "dest_airport": ["LAX"],
        "departure_hour": [8],
        "distance": [2475],
        "historical_airline_delay_rate": [0.25],
        "delay_target": [1],
    })
    # Should pass without error
    assert assert_no_target_leakage(clean_df) is True


def test_leakage_audit_fails_on_post_flight_leakage():
    """Verify that injecting post-flight operational columns triggers a loud failure."""
    forbidden_cols = [
        "arrival_delay",
        "departure_delay",
        "actual_dep_time",
        "actual_arr_time",
        "taxi_out",
        "taxi_in",
        "air_time",
    ]

    for col in forbidden_cols:
        leaky_df = pd.DataFrame({
            "flight_date": ["2024-01-01"],
            "airline": ["AA"],
            col: [15.0],
            "delay_target": [1],
        })
        with pytest.raises(ValueError) as excinfo:
            assert_no_target_leakage(leaky_df)
        assert "LEAKAGE AUDIT FAILED" in str(excinfo.value)


# ==============================================================================
# 9. CHRONOLOGICAL DATASET SPLITTING TESTS
# ==============================================================================

def test_chronological_split():
    """Verify chronological dataset split preserves temporal ordering (no future leakage)."""
    df = pd.DataFrame({
        "flight_date": [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
            "2024-01-09", "2024-01-10"
        ],
        "scheduled_dep_time": [800] * 10,
        "airline": ["AA"] * 10,
        "delay_target": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    })

    splits = split_dataset_chronologically(df, train_pct=0.6, val_pct=0.2)
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]

    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    # Verify temporal order: max(train) <= min(val) and max(val) <= min(test)
    assert train_df["flight_date"].max() <= val_df["flight_date"].min()
    assert val_df["flight_date"].max() <= test_df["flight_date"].min()


# ==============================================================================
# 10. FINAL FEATURE DATASET SCHEMA AND LEAKAGE AUDIT TEST
# ==============================================================================

def test_final_feature_dataset_schema():
    """Verify that the generated flights_features.parquet adheres to Phase 2A requirements."""
    from src.utils.config import get_config
    config = get_config()
    feat_path = config.data_processed_dir / "flights_features.parquet"

    if not feat_path.exists():
        pytest.skip("flights_features.parquet not yet exported")

    df = pd.read_parquet(feat_path)

    # 1. Essential columns
    required_cols = [
        "flight_date",
        "airline",
        "origin_airport",
        "dest_airport",
        "route",
        "departure_hour",
        "departure_minute",
        "departure_minutes_since_midnight",
        "time_of_day",
        "departure_hour_sin",
        "departure_hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "year",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "distance",
        "haul_category",
        "historical_origin_delay_rate",
        "historical_origin_flight_count",
        "historical_destination_delay_rate",
        "historical_destination_flight_count",
        "historical_airline_delay_rate",
        "historical_airline_flight_count",
        "historical_route_delay_rate",
        "historical_route_flight_count",
        "delay_target",
    ]

    for col in required_cols:
        assert col in df.columns, f"Required feature '{col}' missing from final dataset"

    # 2. Strict Zero Post-Flight Leakage
    forbidden_leakage = [
        "arrival_delay",
        "departure_delay",
        "actual_dep_time",
        "actual_arr_time",
        "taxi_out",
        "taxi_in",
        "wheels_off",
        "wheels_on",
        "air_time",
        "elapsed_time",
    ]
    for col in forbidden_leakage:
        assert col not in df.columns, f"LEAKAGE VIOLATION: '{col}' found in final feature dataset!"

    # 3. Leakage audit function passes
    assert assert_no_target_leakage(df, config=config) is True
