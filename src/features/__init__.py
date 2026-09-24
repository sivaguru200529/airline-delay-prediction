"""Feature engineering module for schedule, historical, route, airport, and weather features."""

from src.features.advanced_features import (
    build_phase3_date_features,
    build_phase3_feature_pipeline,
    build_phase3_route_and_airport_features,
    build_phase3_time_features,
    categorize_operational_time_of_day,
    categorize_time_bucket_4h,
)
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
from src.features.phase3_reporter import generate_phase3_reports
from src.features.weather_features import (
    WeatherIntegrator,
    WeatherValidationResult,
)

__all__ = [
    "verify_temporal_validity",
    "parse_scheduled_time",
    "categorize_time_of_day",
    "categorize_operational_time_of_day",
    "categorize_time_bucket_4h",
    "build_date_features",
    "build_dep_time_features",
    "build_arr_time_features",
    "build_cyclical_features",
    "build_route_features",
    "assert_no_target_leakage",
    "build_pre_departure_feature_pipeline",
    "split_dataset_chronologically",
    "HistoricalFeatureCalculator",
    "construct_departure_timestamp",
    "WeatherIntegrator",
    "WeatherValidationResult",
    "build_phase3_date_features",
    "build_phase3_time_features",
    "build_phase3_route_and_airport_features",
    "build_phase3_feature_pipeline",
    "generate_phase3_reports",
]
