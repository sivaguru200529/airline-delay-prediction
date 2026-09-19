"""Centralized configuration management for the Airline Delay Prediction project.

Handles environment loading, project directory resolution, threshold settings,
risk categories, and strict definitions of post-flight leakage features.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Resolve project root directory (two levels above src/utils)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Automatically load environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class RiskCategoryThresholds:
    """Configurable risk probability thresholds.
    
    Tiers:
      - LOW: [0.0, low_upper)
      - MEDIUM: [low_upper, medium_upper)
      - HIGH: [medium_upper, high_upper)
      - VERY_HIGH: [high_upper, 1.0]
    """
    low_upper: float = float(os.getenv("RISK_THRESHOLD_LOW", "0.30"))
    medium_upper: float = float(os.getenv("RISK_THRESHOLD_MEDIUM", "0.60"))
    high_upper: float = float(os.getenv("RISK_THRESHOLD_HIGH", "0.80"))

    def classify_risk(self, probability: float) -> str:
        """Classify a predicted delay probability into an operational risk tier."""
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Probability must be within [0.0, 1.0], got {probability}")
        if probability < self.low_upper:
            return "LOW"
        if probability < self.medium_upper:
            return "MEDIUM"
        if probability < self.high_upper:
            return "HIGH"
        return "VERY HIGH"


@dataclass
class AppConfig:
    """Master application configuration."""

    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Project directories
    project_root: Path = field(default=PROJECT_ROOT)
    data_raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
    )
    data_external_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("DATA_EXTERNAL_DIR", "data/external")
    )
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports"))
    figures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "figures")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / os.getenv("MODELS_DIR", "models"))
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    # Prediction scenario and target definition
    # FAA / BTS Standard: Arrival delay >= 15 minutes defines a delayed flight
    arrival_delay_threshold: int = field(
        default_factory=lambda: int(os.getenv("ARRIVAL_DELAY_THRESHOLD", "15"))
    )

    # Reference timestamp for prediction:
    # All prediction features MUST be available before scheduled departure
    prediction_time_reference: str = "scheduled_departure"

    # Risk tiers
    risk_thresholds: RiskCategoryThresholds = field(default_factory=RiskCategoryThresholds)

    # Essential conceptual fields required in any raw dataset
    required_conceptual_fields: List[str] = field(
        default_factory=lambda: [
            "flight_date",
            "airline",
            "origin_airport",
            "dest_airport",
            "scheduled_dep_time",
            "arrival_delay",
        ]
    )

    # ==========================================================================
    # DATA LEAKAGE PREVENTION SPECIFICATION:
    # Any column representing post-departure or post-flight information is strictly
    # forbidden from entering the pre-departure prediction feature matrix.
    # ==========================================================================
    post_flight_leakage_columns: List[str] = field(
        default_factory=lambda: [
            "arrival_delay",       # Prediction target ground truth
            "departure_delay",     # Post-pushback operational outcome
            "actual_departure",    # Actual pushback timestamp
            "actual_arrival",      # Actual arrival timestamp
            "actual_dep_time",     # Actual departure time representation
            "actual_arr_time",     # Actual arrival time representation
            "taxi_out",            # Taxi out duration (post-pushback)
            "taxi_in",             # Taxi in duration (post-landing)
            "wheels_off",          # Takeoff timestamp
            "wheels_on",           # Touchdown timestamp
            "air_time",            # Airborne flight duration
            "elapsed_time",        # Actual door-to-door duration
        ]
    )

    # Derived or pattern-based leakage column identifiers to audit against
    forbidden_leakage_keywords: List[str] = field(
        default_factory=lambda: [
            "arr_delay",
            "dep_delay",
            "actual_dep",
            "actual_arr",
            "actual_departure",
            "actual_arrival",
            "taxi_out",
            "taxi_in",
            "wheels_off",
            "wheels_on",
            "air_time",
            "elapsed_time",
            "cancellation_code",
        ]
    )

    # Time of Day Categories: [start_hour, end_hour)
    # overnight: [22, 6), morning: [6, 12), afternoon: [12, 18), evening: [18, 22)
    time_of_day_definitions: Dict[str, str] = field(
        default_factory=lambda: {
            "overnight": "22:00 to 05:59",
            "morning": "06:00 to 11:59",
            "afternoon": "12:00 to 17:59",
            "evening": "18:00 to 21:59",
        }
    )

    # Historical Delay Rate calculation parameters
    historical_min_history: int = field(
        default_factory=lambda: int(os.getenv("HISTORICAL_MIN_HISTORY", "3"))
    )
    historical_fallback_strategy: str = field(
        default_factory=lambda: os.getenv("HISTORICAL_FALLBACK_STRATEGY", "global_prior")
    )

    # Weather foundation parameters
    weather_lookback_minutes: int = field(
        default_factory=lambda: int(os.getenv("WEATHER_LOOKBACK_MINUTES", "120"))
    )
    weather_required_fields: List[str] = field(
        default_factory=lambda: [
            "airport",
            "timestamp",
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "precipitation",
            "visibility",
            "pressure",
            "cloud_cover",
            "weather_condition",
        ]
    )

    # Database settings (Phase 3)
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "airline_analytics"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    def ensure_directories(self) -> None:
        """Create project directories if they do not exist."""
        for directory in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_external_dir,
            self.reports_dir,
            self.figures_dir,
            self.models_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Factory function returning initialized application configuration."""
    cfg = AppConfig()
    cfg.ensure_directories()
    return cfg
