"""Feature engineering module and temporal leakage validation.

================================================================================
CRITICAL DESIGN SPECIFICATION & TEMPORAL CONSTRAINTS:
================================================================================

1. Prediction Scenario:
   The system predicts whether a flight will experience an arrival delay of >= 15 min
   using ONLY information available strictly at or before scheduled departure:
       prediction_time = scheduled_departure

2. Strict Temporal Integrity Policy:
   All future historical and rolling features (such as airline historical delay rate,
   origin airport delay rate, route delay rate, and recent inbound aircraft delay)
   MUST obey:
       observation_timestamp < prediction_time

   NO FUTURE FLIGHT RECORDS MAY CONTRIBUTE TO ANY FEATURE CALCULATION.
   In Phase 2, rolling windows and historical statistics must be computed using
   expanding or sliding time windows evaluated strictly up to prediction_time.

3. Weather Integration Strategy (Phase 2):
   Weather features (temperature, precipitation, wind speed, visibility, etc.)
   will be joined using an exact or nearest-prior match strategy:
       (origin_airport, weather_observation_timestamp)
   with explicit local-to-UTC timezone reconciliation. Forecast or METAR weather
   must be dated strictly BEFORE scheduled departure.
================================================================================
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")


def verify_temporal_validity(
    prediction_time: pd.Timestamp,
    historical_timestamps: pd.Series,
) -> bool:
    """Verify that all historical records strictly precede the prediction timestamp.

    This helper enforces the anti-leakage invariant:
        historical_timestamp < prediction_time

    Args:
        prediction_time: Scheduled departure timestamp for the flight being predicted.
        historical_timestamps: Series of observation timestamps used in historical aggregation.

    Returns:
        True if all historical timestamps are strictly before prediction_time.

    Raises:
        ValueError: If any historical timestamp >= prediction_time is detected.
    """
    ts_series = pd.to_datetime(historical_timestamps)
    pred_ts = pd.to_datetime(prediction_time)

    future_records = ts_series >= pred_ts
    if future_records.any():
        num_future = int(future_records.sum())
        max_future_ts = ts_series[future_records].max()
        error_msg = (
            f"Temporal Leakage Violation! Detected {num_future} historical observations "
            f"occurring at or after prediction_time ({pred_ts}). Max future timestamp: {max_future_ts}. "
            f"All features must satisfy: observation_timestamp < prediction_time."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return True


# ==============================================================================
# PHASE 2A FEATURE ENGINEERING IMPLEMENTATION
# ==============================================================================

def parse_scheduled_time(time_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Parse integer, float, or string scheduled times into hour, minute, and minutes since midnight.

    Handles standard 3-digit and 4-digit HHMM values (e.g., 800 -> 08:00, 1515 -> 15:15, 5 -> 00:05, 2400 -> 00:00).
    Safely adjusts any anomalous minutes >= 60 by rolling over into hours:
        hour = (raw_hour + raw_minute // 60) % 24
        minute = raw_minute % 60
        minutes_since_midnight = hour * 60 + minute

    Args:
        time_series: Series containing scheduled times (e.g. CRS_DEP_TIME).

    Returns:
        Tuple of (hour_series, minute_series, minutes_since_midnight_series).
    """
    numeric_time = pd.to_numeric(time_series, errors="coerce").fillna(0).astype(int)

    raw_hours = numeric_time // 100
    raw_minutes = numeric_time % 100

    # Rollover logic for minutes >= 60 or hours >= 24
    extra_hours = raw_minutes // 60
    clean_minutes = raw_minutes % 60
    clean_hours = (raw_hours + extra_hours) % 24

    minutes_since_midnight = clean_hours * 60 + clean_minutes

    return clean_hours, clean_minutes, minutes_since_midnight


def categorize_time_of_day(hour_series: pd.Series) -> pd.Series:
    """Categorize departure hour into standard aviation operational time-of-day blocks.

    Definitions:
        - overnight: [22:00, 06:00) (hours 22, 23, 0, 1, 2, 3, 4, 5)
        - morning:   [06:00, 12:00) (hours 6, 7, 8, 9, 10, 11)
        - afternoon: [12:00, 18:00) (hours 12, 13, 14, 15, 16, 17)
        - evening:   [18:00, 22:00) (hours 18, 19, 20, 21)

    Args:
        hour_series: Series of integer hours (0 to 23).

    Returns:
        pd.Series of string categories.
    """
    conditions = [
        (hour_series >= 22) | (hour_series < 6),
        (hour_series >= 6) & (hour_series < 12),
        (hour_series >= 12) & (hour_series < 18),
        (hour_series >= 18) & (hour_series < 22),
    ]
    choices = ["overnight", "morning", "afternoon", "evening"]
    return pd.Series(np.select(conditions, choices, default="morning"), index=hour_series.index)


def build_date_features(df: pd.DataFrame, date_col: str = "flight_date") -> pd.DataFrame:
    """Extract calendar features from flight_date.

    Features generated:
        - year (int)
        - month (int: 1 to 12)
        - day (int: 1 to 31)
        - day_of_week (int: 0=Monday to 6=Sunday)
        - week_of_year (int: 1 to 53)
        - day_of_year (int: 1 to 366)
        - is_weekend (int: 1 if Saturday/Sunday else 0)

    Args:
        df: Input DataFrame containing date_col.
        date_col: Column name representing flight date.

    Returns:
        DataFrame with calendar features.
    """
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    dt_series = pd.to_datetime(df[date_col], errors="coerce")
    if dt_series.isna().all():
        raise ValueError(f"All values in date column '{date_col}' failed to parse as datetime.")

    features = pd.DataFrame(index=df.index)
    features["year"] = dt_series.dt.year.astype("int32")
    features["month"] = dt_series.dt.month.astype("int32")
    features["day"] = dt_series.dt.day.astype("int32")
    features["day_of_week"] = dt_series.dt.dayofweek.astype("int32")  # 0=Monday, 6=Sunday
    features["week_of_year"] = dt_series.dt.isocalendar().week.astype("int32")
    features["day_of_year"] = dt_series.dt.dayofyear.astype("int32")
    features["is_weekend"] = (features["day_of_week"] >= 5).astype("int32")

    return features


def build_dep_time_features(df: pd.DataFrame, time_col: str = "scheduled_dep_time") -> pd.DataFrame:
    """Extract scheduled departure time components and time-of-day category.

    Features generated:
        - departure_hour (0-23)
        - departure_minute (0-59)
        - departure_minutes_since_midnight (0-1439)
        - time_of_day (overnight, morning, afternoon, evening)

    Args:
        df: Input DataFrame containing time_col.
        time_col: Column name representing scheduled departure time.

    Returns:
        DataFrame with departure timing features.
    """
    if time_col not in df.columns:
        raise KeyError(f"Departure time column '{time_col}' not found in DataFrame.")

    hours, minutes, mins_midnight = parse_scheduled_time(df[time_col])

    features = pd.DataFrame(index=df.index)
    features["departure_hour"] = hours.astype("int32")
    features["departure_minute"] = minutes.astype("int32")
    features["departure_minutes_since_midnight"] = mins_midnight.astype("int32")
    features["time_of_day"] = categorize_time_of_day(features["departure_hour"])

    return features


def build_arr_time_features(df: pd.DataFrame, time_col: str = "scheduled_arr_time") -> Optional[pd.DataFrame]:
    """Extract scheduled arrival time components if available.

    Features generated:
        - arrival_hour (0-23)
        - arrival_minute (0-59)
        - arrival_minutes_since_midnight (0-1439)

    Args:
        df: Input DataFrame.
        time_col: Column name representing scheduled arrival time.

    Returns:
        DataFrame with arrival timing features, or None if time_col not present.
    """
    if time_col not in df.columns:
        logger.info("Column '%s' not present; skipping arrival schedule features.", time_col)
        return None

    hours, minutes, mins_midnight = parse_scheduled_time(df[time_col])

    features = pd.DataFrame(index=df.index)
    features["arrival_hour"] = hours.astype("int32")
    features["arrival_minute"] = minutes.astype("int32")
    features["arrival_minutes_since_midnight"] = mins_midnight.astype("int32")

    return features


def build_cyclical_features(
    hour_series: pd.Series,
    day_of_week_series: pd.Series,
    month_series: pd.Series,
) -> pd.DataFrame:
    """Generate mathematically correct periodic cyclical transformations.

    Linear time features incorrectly imply that 23:59 is maximally distant from 00:01,
    or that December (12) is far from January (1). Sine and cosine projections wrap around
    the unit circle, preserving chronological periodicity.

    Transformations:
        - departure_hour:  period = 24  -> sin/cos(2 * pi * hour / 24)
        - day_of_week:     period = 7   -> sin/cos(2 * pi * day_of_week / 7)
        - month:           period = 12  -> sin/cos(2 * pi * (month - 1) / 12)

    Args:
        hour_series: Integer hour Series (0-23).
        day_of_week_series: Integer day of week Series (0-6).
        month_series: Integer month Series (1-12).

    Returns:
        DataFrame with 6 cyclical features.
    """
    features = pd.DataFrame(index=hour_series.index)

    # Hour of day (period 24)
    features["departure_hour_sin"] = np.sin(2.0 * np.pi * hour_series / 24.0)
    features["departure_hour_cos"] = np.cos(2.0 * np.pi * hour_series / 24.0)

    # Day of week (period 7, 0=Monday, 6=Sunday)
    features["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week_series / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week_series / 7.0)

    # Month of year (period 12, 1-indexed so subtract 1 for 0-indexed angle)
    features["month_sin"] = np.sin(2.0 * np.pi * (month_series - 1) / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * (month_series - 1) / 12.0)

    return features


def build_route_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Construct route identifier and flight distance haul category.

    Also evaluates conditional features such as `same_airport_flag`:
    If origin_airport == dest_airport has zero variance (constant across entire dataset),
    it is intentionally omitted from the feature matrix to avoid zero-variance noise,
    and the omission is documented.

    Haul Category Definitions (FAA Standard):
        - short_haul:  < 500 miles
        - medium_haul: 500 to 1,500 miles
        - long_haul:   > 1,500 miles

    Args:
        df: Input DataFrame with origin_airport, dest_airport, distance.

    Returns:
        Tuple of (features_df, metadata_notes).
    """
    features = pd.DataFrame(index=df.index)
    notes: Dict[str, str] = {}

    if "origin_airport" in df.columns and "dest_airport" in df.columns:
        origin = df["origin_airport"].astype(str).str.strip().str.upper()
        dest = df["dest_airport"].astype(str).str.strip().str.upper()
        features["route"] = origin + "_" + dest

        # Check same_airport_flag condition
        is_same = (origin == dest).astype(int)
        if is_same.nunique() > 1 and is_same.var() > 0:
            features["same_airport_flag"] = is_same
            notes["same_airport_flag"] = "INCLUDED: Meaningful variation detected between origin and dest."
        else:
            notes["same_airport_flag"] = (
                "OMITTED: Zero variance detected (100% flights have origin != dest). "
                "Feature omitted to avoid dead zero-variance noise."
            )

    if "distance" in df.columns:
        dist_numeric = pd.to_numeric(df["distance"], errors="coerce")
        features["distance"] = dist_numeric

        conditions = [
            dist_numeric < 500,
            (dist_numeric >= 500) & (dist_numeric <= 1500),
            dist_numeric > 1500,
        ]
        choices = ["short_haul", "medium_haul", "long_haul"]
        features["haul_category"] = np.select(conditions, choices, default="medium_haul")

    return features, notes


def assert_no_target_leakage(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> bool:
    """Strictly assert that no post-flight operational outcome columns exist in the feature set.

    Inspects both:
        1. Direct column names against centralized config.post_flight_leakage_columns
        2. Substring/keyword matches against config.forbidden_leakage_keywords

    Args:
        df: Feature DataFrame to audit.
        config: Application configuration.

    Returns:
        True if the feature set is completely free of leakage.

    Raises:
        ValueError: If any post-flight leakage column or keyword is detected.
    """
    cfg = config or get_config()
    detected_violations: List[str] = []

    columns_lower = {col.lower(): col for col in df.columns}

    # 1. Check exact forbidden post-flight column names
    for forbidden in cfg.post_flight_leakage_columns:
        if forbidden.lower() in columns_lower:
            # Note: delay_target is allowed as the ground-truth label column
            if forbidden.lower() == "arrival_delay":
                detected_violations.append(f"Direct Target/Post-Flight Leakage Detected: Direct Target '{columns_lower[forbidden.lower()]}'")
            else:
                detected_violations.append(f"Direct Target/Post-Flight Leakage Detected: Post-Flight Operational Column '{columns_lower[forbidden.lower()]}'")

    # 2. Check derived/keyword patterns and temporal lookahead leakage
    for kw in cfg.forbidden_leakage_keywords:
        for col_lower, original_col in columns_lower.items():
            if original_col in ["delay_target", "flight_date"]:
                continue
            if original_col.startswith("historical_") or original_col.startswith("prior_"):
                # Strictly prior features are vetted by temporal inequality tests
                continue
            if kw in col_lower:
                detected_violations.append(f"Direct Target/Post-Flight Leakage Detected: Forbidden Pattern '{kw}' in feature '{original_col}'")

    # 3. Check for derived temporal leakage prefixes/keywords
    derived_temporal_leakage_keywords = ["future_", "full_dataset_", "post_flight_"]
    for d_kw in derived_temporal_leakage_keywords:
        for col_lower, original_col in columns_lower.items():
            if original_col in ["delay_target", "flight_date"]:
                continue
            if d_kw in col_lower:
                detected_violations.append(f"Derived Temporal Leakage Detected: '{d_kw}' in feature '{original_col}'")

    if detected_violations:
        unique_violations = sorted(list(set(detected_violations)))
        error_msg = (
            f"LEAKAGE AUDIT FAILED! The following forbidden post-flight fields were detected:\n"
            + "\n".join(f"  - {v}" for v in unique_violations)
            + "\nPre-departure ML features must strictly use information available at scheduled departure."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("LEAKAGE AUDIT: PASS - Feature matrix contains 0 post-flight leakage columns.")
    return True


def build_pre_departure_feature_pipeline(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Execute complete modular feature engineering pipeline for pre-departure prediction.

    Pipeline:
        1. Date features (year, month, day, day_of_week, week_of_year, day_of_year, is_weekend)
        2. Scheduled departure time features (departure_hour, departure_minute, departure_minutes_since_midnight, time_of_day)
        3. Scheduled arrival time features (if present)
        4. Cyclical periodic representations (sin/cos for hour, day_of_week, month)
        5. Route & distance haul features
        6. Base identifiers preserved: flight_date, airline, origin_airport, dest_airport, delay_target

    Args:
        df: Input pre-departure DataFrame.
        config: Application configuration.

    Returns:
        Tuple of (engineered_features_df, feature_notes).
    """
    logger.info("Starting feature engineering pipeline on %d records", len(df))
    out_df = pd.DataFrame(index=df.index)

    # 1. Base Identifiers
    base_cols = ["flight_date", "airline", "origin_airport", "dest_airport"]
    for col in base_cols:
        if col in df.columns:
            out_df[col] = df[col].copy()

    # 2. Date features
    date_feats = build_date_features(df, date_col="flight_date")
    for col in date_feats.columns:
        out_df[col] = date_feats[col]

    # 3. Scheduled departure features
    dep_feats = build_dep_time_features(df, time_col="scheduled_dep_time")
    out_df["scheduled_dep_time"] = df["scheduled_dep_time"].copy()
    for col in dep_feats.columns:
        out_df[col] = dep_feats[col]

    # 4. Scheduled arrival features
    if "scheduled_arr_time" in df.columns:
        out_df["scheduled_arr_time"] = df["scheduled_arr_time"].copy()
        arr_feats = build_arr_time_features(df, time_col="scheduled_arr_time")
        if arr_feats is not None:
            for col in arr_feats.columns:
                out_df[col] = arr_feats[col]

    # 5. Cyclical features
    cyclical_feats = build_cyclical_features(
        hour_series=out_df["departure_hour"],
        day_of_week_series=out_df["day_of_week"],
        month_series=out_df["month"],
    )
    for col in cyclical_feats.columns:
        out_df[col] = cyclical_feats[col]

    # 6. Route and Haul features
    route_feats, notes = build_route_features(df)
    for col in route_feats.columns:
        out_df[col] = route_feats[col]

    # 7. Preserve Target
    if "delay_target" in df.columns:
        out_df["delay_target"] = df["delay_target"].astype("int32")

    logger.info(
        "Engineered %d total features across date, timing, cyclical, and route groups",
        len(out_df.columns),
    )
    return out_df, notes


def split_dataset_chronologically(
    df: pd.DataFrame,
    date_col: str = "flight_date",
    time_col: Optional[str] = "scheduled_dep_time",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Dict[str, Any]:
    """Split dataset chronologically into train, validation, and test sets.

    Strictly chronological:
        Train (Earliest records) -> Validation (Intermediate records) -> Test (Future records)
    No random shuffling is permitted in time-series operations modeling.

    Args:
        df: Input DataFrame.
        date_col: Column containing flight dates.
        time_col: Optional column containing scheduled departure times.
        train_pct: Fraction of records for training (e.g. 0.70).
        val_pct: Fraction of records for validation (e.g. 0.15).

    Returns:
        Dict with 'train_df', 'val_df', 'test_df', and 'split_summary'.
    """
    total_records = len(df)
    if total_records == 0:
        raise ValueError("Cannot split empty DataFrame.")

    # Sort strictly chronologically
    sort_cols = [date_col]
    if time_col and time_col in df.columns:
        sort_cols.append(time_col)

    sorted_df = df.sort_values(sort_cols).reset_index(drop=True)

    n_train = int(total_records * train_pct)
    n_val = int(total_records * val_pct)

    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train:n_train + n_val].copy()
    test_df = sorted_df.iloc[n_train + n_val:].copy()

    split_summary = {
        "total_records": total_records,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_pct": round(len(train_df) / total_records * 100, 2),
        "val_pct": round(len(val_df) / total_records * 100, 2),
        "test_pct": round(len(test_df) / total_records * 100, 2),
        "train_date_range": [str(train_df[date_col].min()), str(train_df[date_col].max())] if not train_df.empty else [],
        "val_date_range": [str(val_df[date_col].min()), str(val_df[date_col].max())] if not val_df.empty else [],
        "test_date_range": [str(test_df[date_col].min()), str(test_df[date_col].max())] if not test_df.empty else [],
        "sample_size_limitation": (
            "The current development dataset contains only a limited number of records. "
            "Chronological splits should be scaled to multi-month or multi-year public extracts for Phase 2B production modeling."
            if total_records < 2000 else "Sufficient sample size for temporal evaluation."
        ),
    }

    logger.info(
        "Chronological split complete: Train=%d (%s to %s), Val=%d (%s to %s), Test=%d (%s to %s)",
        len(train_df), split_summary["train_date_range"][0] if split_summary["train_date_range"] else "N/A",
        split_summary["train_date_range"][1] if split_summary["train_date_range"] else "N/A",
        len(val_df), split_summary["val_date_range"][0] if split_summary["val_date_range"] else "N/A",
        split_summary["val_date_range"][1] if split_summary["val_date_range"] else "N/A",
        len(test_df), split_summary["test_date_range"][0] if split_summary["test_date_range"] else "N/A",
        split_summary["test_date_range"][1] if split_summary["test_date_range"] else "N/A",
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_summary": split_summary,
    }
