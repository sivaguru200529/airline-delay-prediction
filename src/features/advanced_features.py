"""Phase 3 Advanced Feature Engineering module.

Expands upon Phase 2A pre-departure features with:
1. Advanced date features (is_month_start, is_month_end, quarter, season).
2. Advanced time features (operational time blocks, 4-hour time buckets, arrival cyclical projections).
3. Data-aware route features and strictly prior route frequency (prior_route_frequency).
4. Strictly prior airport operational volume features (prior_origin_flight_volume, prior_dest_flight_volume).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.feature_engineering import parse_scheduled_time
from src.features.historical_features import construct_departure_timestamp
from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("advanced_features")


# ==============================================================================
# 1. ADVANCED DATE FEATURES
# ==============================================================================

def build_phase3_date_features(df: pd.DataFrame, date_col: str = "flight_date") -> pd.DataFrame:
    """Extract Phase 3 calendar and seasonal features from flight_date.

    Features generated:
        - is_month_start: 1 if first day of the calendar month else 0
        - is_month_end: 1 if last day of the calendar month else 0
        - quarter: Integer calendar quarter (1 to 4)
        - season: Astronomical/meteorological season string:
            'winter' (Dec, Jan, Feb), 'spring' (Mar, Apr, May),
            'summer' (Jun, Jul, Aug), 'fall' (Sep, Oct, Nov)

    Args:
        df: Input DataFrame containing date_col.
        date_col: Column name for flight date.

    Returns:
        DataFrame containing Phase 3 date features.
    """
    dt_series = pd.to_datetime(df[date_col], errors="coerce")

    is_month_start = (dt_series.dt.day == 1).astype(int)
    is_month_end = dt_series.dt.is_month_end.astype(int)
    quarter = dt_series.dt.quarter.fillna(1).astype(int)

    # Season mapping based on calendar month
    month_series = dt_series.dt.month.fillna(1).astype(int)
    season_conditions = [
        month_series.isin([12, 1, 2]),
        month_series.isin([3, 4, 5]),
        month_series.isin([6, 7, 8]),
        month_series.isin([9, 10, 11]),
    ]
    season_choices = ["winter", "spring", "summer", "fall"]
    season = pd.Series(
        np.select(season_conditions, season_choices, default="winter"),
        index=df.index,
    )

    out_df = pd.DataFrame(
        {
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "quarter": quarter,
            "season": season,
        },
        index=df.index,
    )
    return out_df


# ==============================================================================
# 2. ADVANCED TIME FEATURES & CYCLICAL ENCODINGS
# ==============================================================================

def categorize_operational_time_of_day(hour_series: pd.Series) -> pd.Series:
    """Categorize departure or arrival hour into standard aviation operational blocks.

    Blocks:
        - night:     [22:00, 06:00) (hours 22, 23, 0, 1, 2, 3, 4, 5)
        - morning:   [06:00, 12:00) (hours 6, 7, 8, 9, 10, 11)
        - afternoon: [12:00, 18:00) (hours 12, 13, 14, 15, 16, 17)
        - evening:   [18:00, 22:00) (hours 18, 19, 20, 21)

    Args:
        hour_series: Series of integer hours (0 to 23).

    Returns:
        pd.Series of string categories.
    """
    clean_hours = pd.to_numeric(hour_series, errors="coerce").fillna(12).astype(int)
    conditions = [
        (clean_hours >= 22) | (clean_hours < 6),
        (clean_hours >= 6) & (clean_hours < 12),
        (clean_hours >= 12) & (clean_hours < 18),
        (clean_hours >= 18) & (clean_hours < 22),
    ]
    choices = ["night", "morning", "afternoon", "evening"]
    return pd.Series(np.select(conditions, choices, default="morning"), index=hour_series.index)


def categorize_time_bucket_4h(hour_series: pd.Series) -> pd.Series:
    """Categorize hour of day into 4-hour operational dispatch blocks.

    Buckets:
        - '00-04': [0, 4)
        - '04-08': [4, 8)
        - '08-12': [8, 12)
        - '12-16': [12, 16)
        - '16-20': [16, 20)
        - '20-24': [20, 24)
    """
    clean_hours = pd.to_numeric(hour_series, errors="coerce").fillna(12).astype(int)
    conditions = [
        (clean_hours >= 0) & (clean_hours < 4),
        (clean_hours >= 4) & (clean_hours < 8),
        (clean_hours >= 8) & (clean_hours < 12),
        (clean_hours >= 12) & (clean_hours < 16),
        (clean_hours >= 16) & (clean_hours < 20),
        (clean_hours >= 20) & (clean_hours < 24),
    ]
    choices = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
    return pd.Series(np.select(conditions, choices, default="08-12"), index=hour_series.index)


def build_phase3_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct Phase 3 time categorization and cyclical arrival features.

    Features generated:
        - dep_time_of_day: Operational time block for departure ('night', 'morning', 'afternoon', 'evening')
        - arr_time_of_day: Operational time block for arrival (if scheduled arrival available)
        - dep_time_bucket: 4-hour operational block for departure ('00-04', '04-08', etc.)
        - arr_time_bucket: 4-hour operational block for arrival
        - arr_hour_sin, arr_hour_cos: Unit-circle periodic projections of scheduled arrival hour
        - dep_minute_sin, dep_minute_cos: Unit-circle periodic projections of scheduled departure minute

    Args:
        df: Input DataFrame containing scheduled_dep_time and optional scheduled_arr_time.

    Returns:
        DataFrame containing Phase 3 timing features.
    """
    out_df = pd.DataFrame(index=df.index)

    # 1. Scheduled departure timing
    dep_col = None
    if "departure_hour" in df.columns:
        dep_hours = df["departure_hour"]
    elif "scheduled_dep_time" in df.columns:
        dep_col = "scheduled_dep_time"
        dep_hours, _, _ = parse_scheduled_time(df[dep_col])
    elif "crs_dep_time" in df.columns:
        dep_col = "crs_dep_time"
        dep_hours, _, _ = parse_scheduled_time(df[dep_col])
    else:
        dep_hours = pd.Series(12, index=df.index)

    if "departure_minute" in df.columns:
        dep_minutes = df["departure_minute"]
    elif dep_col:
        _, dep_minutes, _ = parse_scheduled_time(df[dep_col])
    else:
        dep_minutes = pd.Series(0, index=df.index)

    out_df["dep_time_of_day"] = categorize_operational_time_of_day(dep_hours)
    out_df["dep_time_bucket"] = categorize_time_bucket_4h(dep_hours)

    # Departure minute cyclical projections (period 60)
    dep_min_rad = 2.0 * np.pi * dep_minutes.astype(float) / 60.0
    out_df["dep_minute_sin"] = np.sin(dep_min_rad).round(6)
    out_df["dep_minute_cos"] = np.cos(dep_min_rad).round(6)

    # 2. Scheduled arrival timing
    arr_col = None
    if "arrival_hour" in df.columns:
        arr_hours = df["arrival_hour"]
    elif "scheduled_arr_time" in df.columns:
        arr_col = "scheduled_arr_time"
        arr_hours, _, _ = parse_scheduled_time(df[arr_col])
    elif "crs_arr_time" in df.columns:
        arr_col = "crs_arr_time"
        arr_hours, _, _ = parse_scheduled_time(df[arr_col])
    else:
        arr_hours = dep_hours  # Default fallback if arrival schedule not present

    out_df["arr_time_of_day"] = categorize_operational_time_of_day(arr_hours)
    out_df["arr_time_bucket"] = categorize_time_bucket_4h(arr_hours)

    # Arrival hour cyclical projections (period 24)
    arr_hour_rad = 2.0 * np.pi * arr_hours.astype(float) / 24.0
    out_df["arr_hour_sin"] = np.sin(arr_hour_rad).round(6)
    out_df["arr_hour_cos"] = np.cos(arr_hour_rad).round(6)

    return out_df


# ==============================================================================
# 3. ROUTE & AIRPORT OPERATIONAL FEATURES (STRICTLY PRIOR)
# ==============================================================================

def build_phase3_route_and_airport_features(
    df: pd.DataFrame,
    date_col: str = "flight_date",
    dep_time_col: str = "scheduled_dep_time",
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute data-aware route distance categories, strictly prior route frequency, and airport volumes.

    CRITICAL ANTI-LEAKAGE INVARIANT:
        prior_route_frequency and prior_airport_volumes for a flight at time T
        are computed ONLY over flights where:
            observation_timestamp < T
        The current flight and future flights are STRICTLY EXCLUDED.

    Features generated:
        - route: Origin_Destination pair
        - route_distance: Actual flight distance in statute miles
        - route_distance_category / distance_category: Categorical haul bucket ('short_haul', 'medium_haul', 'long_haul')
        - prior_route_frequency: Strictly prior flight volume on this route (0 for 1st flight, 1 for 2nd, etc.)
        - prior_origin_flight_volume: Strictly prior departures from origin airport
        - prior_dest_flight_volume: Strictly prior arrivals at destination airport
        - prior_origin_delay_rate / prior_dest_delay_rate: Strictly prior airport delay rates (if delay_target present)

    Args:
        df: Input DataFrame.
        date_col: Flight date column.
        dep_time_col: Scheduled departure time column.
        config: Application configuration.

    Returns:
        Tuple of (route_and_airport_features_df, distance_audit_metadata).
    """
    cfg = config or get_config()
    out_df = pd.DataFrame(index=df.index)

    # 1. Route assemble
    orig_col = "origin_airport" if "origin_airport" in df.columns else ("origin" if "origin" in df.columns else None)
    dest_col = "dest_airport" if "dest_airport" in df.columns else ("dest" if "dest" in df.columns else None)

    if "route" in df.columns:
        route_series = df["route"].astype(str)
    elif orig_col and dest_col:
        route_series = df[orig_col].astype(str) + "_" + df[dest_col].astype(str)
    else:
        route_series = pd.Series("UNKNOWN_UNKNOWN", index=df.index)
    out_df["route"] = route_series

    # 2. Distance and Distance Category
    if "distance" in df.columns:
        dist_series = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int)
    else:
        dist_series = pd.Series(500, index=df.index)
    out_df["route_distance"] = dist_series

    dist_meta = {
        "min_distance": int(dist_series.min()) if not dist_series.empty else 0,
        "max_distance": int(dist_series.max()) if not dist_series.empty else 0,
        "median_distance": float(dist_series.median()) if not dist_series.empty else 0.0,
        "missing_distance": int(dist_series.isna().sum()),
        "configured_categories": cfg.distance_categories,
    }

    # Data-aware distance categorization
    short_max = cfg.distance_categories.get("short_haul", (0, 500))[1]
    med_max = cfg.distance_categories.get("medium_haul", (500, 1500))[1]

    dist_conditions = [
        (dist_series < short_max),
        (dist_series >= short_max) & (dist_series < med_max),
        (dist_series >= med_max),
    ]
    dist_choices = ["short_haul", "medium_haul", "long_haul"]
    cat_series = pd.Series(
        np.select(dist_conditions, dist_choices, default="medium_haul"),
        index=df.index,
    )
    out_df["route_distance_category"] = cat_series
    out_df["distance_category"] = cat_series

    # 3. Construct unified scheduled departure timestamp
    resolved_dep_time = dep_time_col
    if resolved_dep_time not in df.columns and "crs_dep_time" in df.columns:
        resolved_dep_time = "crs_dep_time"

    resolved_date = date_col
    if resolved_date not in df.columns and "fl_date" in df.columns:
        resolved_date = "fl_date"

    dep_timestamps = construct_departure_timestamp(df, date_col=resolved_date, dep_time_col=resolved_dep_time)

    # 4. Strictly Prior Route Frequency Calculation
    orig_series = df[orig_col].astype(str) if orig_col else pd.Series("UNK", index=df.index)
    dest_series = df[dest_col].astype(str) if dest_col else pd.Series("UNK", index=df.index)

    temp_df = pd.DataFrame({
        "route": route_series,
        "origin": orig_series,
        "dest": dest_series,
        "ts": dep_timestamps,
        "orig_idx": df.index,
    })

    prior_route_freq = np.zeros(len(df), dtype=int)
    prior_origin_vol = np.zeros(len(df), dtype=int)
    prior_dest_vol = np.zeros(len(df), dtype=int)

    # Route Frequency
    for _, grp_indices in temp_df.groupby("route").groups.items():
        grp = temp_df.loc[grp_indices]
        grp_ts = grp["ts"].to_numpy()
        grp_orig_idx = grp["orig_idx"].to_numpy()

        for j in range(len(grp)):
            current_t = grp_ts[j]
            prior_count = int(np.sum(grp_ts < current_t))
            prior_route_freq[grp_orig_idx[j]] = prior_count

    # Origin Airport Volume
    for _, grp_indices in temp_df.groupby("origin").groups.items():
        grp = temp_df.loc[grp_indices]
        grp_ts = grp["ts"].to_numpy()
        grp_orig_idx = grp["orig_idx"].to_numpy()

        for j in range(len(grp)):
            current_t = grp_ts[j]
            prior_count = int(np.sum(grp_ts < current_t))
            prior_origin_vol[grp_orig_idx[j]] = prior_count

    # Destination Airport Volume
    for _, grp_indices in temp_df.groupby("dest").groups.items():
        grp = temp_df.loc[grp_indices]
        grp_ts = grp["ts"].to_numpy()
        grp_orig_idx = grp["orig_idx"].to_numpy()

        for j in range(len(grp)):
            current_t = grp_ts[j]
            prior_count = int(np.sum(grp_ts < current_t))
            prior_dest_vol[grp_orig_idx[j]] = prior_count

    out_df["prior_route_frequency"] = pd.Series(prior_route_freq, index=df.index)
    out_df["prior_origin_flight_volume"] = pd.Series(prior_origin_vol, index=df.index)
    out_df["prior_dest_flight_volume"] = pd.Series(prior_dest_vol, index=df.index)

    # 5. Airport delay rates if delay_target present
    if "delay_target" in df.columns:
        targets = pd.to_numeric(df["delay_target"], errors="coerce").fillna(0).to_numpy()
        prior_orig_delays = np.zeros(len(df), dtype=float)
        prior_dest_delays = np.zeros(len(df), dtype=float)

        for _, grp_indices in temp_df.groupby("origin").groups.items():
            grp = temp_df.loc[grp_indices]
            grp_ts = grp["ts"].to_numpy()
            grp_orig_idx = grp["orig_idx"].to_numpy()
            grp_tgt = targets[grp_orig_idx]

            for j in range(len(grp)):
                current_t = grp_ts[j]
                mask = grp_ts < current_t
                if np.any(mask):
                    prior_orig_delays[grp_orig_idx[j]] = float(np.sum(grp_tgt[mask])) / float(np.sum(mask))

        for _, grp_indices in temp_df.groupby("dest").groups.items():
            grp = temp_df.loc[grp_indices]
            grp_ts = grp["ts"].to_numpy()
            grp_orig_idx = grp["orig_idx"].to_numpy()
            grp_tgt = targets[grp_orig_idx]

            for j in range(len(grp)):
                current_t = grp_ts[j]
                mask = grp_ts < current_t
                if np.any(mask):
                    prior_dest_delays[grp_orig_idx[j]] = float(np.sum(grp_tgt[mask])) / float(np.sum(mask))

        out_df["prior_origin_delay_rate"] = pd.Series(prior_orig_delays, index=df.index).round(4)
        out_df["prior_dest_delay_rate"] = pd.Series(prior_dest_delays, index=df.index).round(4)

    logger.info(
        "Computed strictly prior route frequency and airport volume features for %d flights.",
        len(df),
    )
    return out_df, dist_meta


# ==============================================================================
# 4. MASTER PHASE 3 ADVANCED PIPELINE ASSEMBLER
# ==============================================================================

def build_phase3_feature_pipeline(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Assemble complete Phase 3 feature dataset.

    Integrates:
        - Base identifiers & Phase 2A features (preserved)
        - Phase 3 Date features
        - Phase 3 Time features & Cyclical encodings
        - Phase 3 Route & Airport operational volume features
        - delay_target ground truth label (preserved)

    Args:
        df: Input DataFrame (e.g. Phase 2A feature matrix or clean pre-departure data).
        config: Application configuration.

    Returns:
        Tuple of (combined_phase3_df, pipeline_metadata).
    """
    cfg = config or get_config()
    logger.info("Executing Phase 3 advanced feature pipeline on %d flights", len(df))

    # 1. Date features
    date_df = build_phase3_date_features(df, date_col="flight_date")

    # 2. Time features
    time_df = build_phase3_time_features(df)

    # 3. Route & Airport volume features
    route_airport_df, dist_meta = build_phase3_route_and_airport_features(
        df, date_col="flight_date", dep_time_col="scheduled_dep_time", config=cfg
    )

    # 4. Combine with existing features without creating duplicate columns
    combined = df.copy()

    for col in date_df.columns:
        combined[col] = date_df[col]

    for col in time_df.columns:
        combined[col] = time_df[col]

    for col in route_airport_df.columns:
        combined[col] = route_airport_df[col]

    # Ensure target remains as the final column
    if "delay_target" in combined.columns:
        target_series = combined.pop("delay_target")
        combined["delay_target"] = target_series

    metadata = {
        "feature_version": cfg.feature_version,
        "total_records": len(combined),
        "total_features": len(combined.columns),
        "distance_metadata": dist_meta,
        "new_features_added": list(date_df.columns) + list(time_df.columns) + list(route_airport_df.columns),
    }

    logger.info(
        "Phase 3 feature engineering complete: %d total columns (version: %s)",
        len(combined.columns),
        cfg.feature_version,
    )
    return combined, metadata
