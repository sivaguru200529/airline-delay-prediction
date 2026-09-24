"""Time-aware historical feature calculation module (Phase 2A & Phase 3).

CRITICAL ANTI-LEAKAGE SPECIFICATION:
For every flight at scheduled departure time T:
    historical_observation_timestamp < T

Guarantees:
    1. A flight never sees its own delay outcome.
    2. A flight never sees future flights.
    3. Concurrent flights (same departure timestamp) are strictly excluded from each other.
    4. First observations and low-history flights use a strictly prior global fallback.
    5. Observation counts and delay counts are retained alongside rates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("historical_features")


def construct_departure_timestamp(
    df: pd.DataFrame,
    date_col: str = "flight_date",
    dep_time_col: str = "scheduled_dep_time",
) -> pd.Series:
    """Construct a single unified datetime timestamp for scheduled departure.

    Args:
        df: Input DataFrame.
        date_col: Flight date column (YYYY-MM-DD).
        dep_time_col: Scheduled departure time column (HHMM).

    Returns:
        pd.Series of datetime64[ns] timestamps.
    """
    resolved_date = date_col if date_col in df.columns else ("fl_date" if "fl_date" in df.columns else date_col)
    resolved_dep = dep_time_col if dep_time_col in df.columns else ("crs_dep_time" if "crs_dep_time" in df.columns else dep_time_col)

    dates = pd.to_datetime(df[resolved_date], errors="coerce")
    raw_times = pd.to_numeric(df[resolved_dep], errors="coerce").fillna(0).astype(int)

    raw_hours = raw_times // 100
    raw_minutes = raw_times % 100

    clean_hours = (raw_hours + raw_minutes // 60) % 24
    clean_minutes = raw_minutes % 60

    dep_timestamps = dates + pd.to_timedelta(clean_hours, unit="h") + pd.to_timedelta(clean_minutes, unit="m")
    return dep_timestamps


class HistoricalFeatureCalculator:
    """Calculates strictly prior historical delay statistics with configurable fallbacks."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize calculator with configuration."""
        self.config = config or get_config()
        self.min_history = self.config.historical_min_history
        self.fallback_strategy = self.config.historical_fallback_strategy

    def _compute_strictly_prior_group_stats(
        self,
        df: pd.DataFrame,
        group_col: str,
        target_col: str,
        time_col: str,
        window_days: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute strictly prior target sum and observation count for a group column.

        Enforces:
            observation_timestamp < T

        Args:
            df: DataFrame containing group_col, target_col, time_col.
            group_col: Categorical entity column (e.g. 'airline', 'origin_airport').
            target_col: Binary target column (delay_target).
            time_col: Timestamp column for scheduled departure.
            window_days: Optional rolling window in days. If None, uses expanding history.

        Returns:
            Tuple of (prior_target_sum_series, prior_count_series) matching df index.
        """
        temp = pd.DataFrame({
            "group": df[group_col].astype(str),
            "target": pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(float),
            "ts": pd.to_datetime(df[time_col]),
            "orig_idx": df.index,
        })

        prior_sums = np.zeros(len(df), dtype=float)
        prior_counts = np.zeros(len(df), dtype=int)

        # Process each unique group independently
        for _, grp_indices in temp.groupby("group").groups.items():
            grp_data = temp.loc[grp_indices]
            unique_ts = np.sort(grp_data["ts"].unique())

            # Aggregate target sum and count per unique timestamp within the group
            ts_summary = grp_data.groupby("ts")["target"].agg(["sum", "count"]).reindex(unique_ts, fill_value=0)

            if window_days is None:
                # Expanding window strictly prior to current timestamp:
                # cumulative sum shifted by 1 excludes the current timestamp entirely
                shifted_sum = ts_summary["sum"].cumsum().shift(1).fillna(0.0)
                shifted_count = ts_summary["count"].cumsum().shift(1).fillna(0)
            else:
                # Rolling window of size window_days:
                # Shifted by 1 so current timestamp is never included
                window_str = f"{window_days}D"
                rolled_sum = ts_summary["sum"].rolling(window_str, closed="left").sum().fillna(0.0)
                rolled_count = ts_summary["count"].rolling(window_str, closed="left").sum().fillna(0)
                shifted_sum = rolled_sum
                shifted_count = rolled_count

            # Map historical metrics back to individual flight rows
            sum_map = dict(zip(unique_ts, shifted_sum))
            count_map = dict(zip(unique_ts, shifted_count))

            for idx in grp_indices:
                row_ts = temp.at[idx, "ts"]
                prior_sums[idx] = sum_map[row_ts]
                prior_counts[idx] = count_map[row_ts]

        return pd.Series(prior_sums, index=df.index), pd.Series(prior_counts, index=df.index)

    def _compute_strictly_prior_global_rate(
        self,
        df: pd.DataFrame,
        target_col: str,
        time_col: str,
    ) -> pd.Series:
        """Compute strictly prior overall global delay rate across all flights (observation_timestamp < T)."""
        temp = pd.DataFrame({
            "target": pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(float),
            "ts": pd.to_datetime(df[time_col]),
        })

        unique_ts = np.sort(temp["ts"].unique())
        ts_summary = temp.groupby("ts")["target"].agg(["sum", "count"]).reindex(unique_ts, fill_value=0)

        shifted_sum = ts_summary["sum"].cumsum().shift(1).fillna(0.0)
        shifted_count = ts_summary["count"].cumsum().shift(1).fillna(0)

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_series = np.where(shifted_count > 0, shifted_sum / shifted_count, np.nan)

        rate_map = dict(zip(ts_summary.index, rate_series))
        global_rates = [rate_map[t] for t in temp["ts"]]
        return pd.Series(global_rates, index=df.index)

    def compute_historical_features(
        self,
        df: pd.DataFrame,
        target_col: str = "delay_target",
        time_col: Optional[str] = None,
        window_days: Optional[int] = None,
        min_history: Optional[int] = None,
        fallback_strategy: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Generate Phase 2A backward-compatible historical delay features."""
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found for historical feature calculation.")

        effective_min_history = min_history if min_history is not None else self.min_history
        effective_fallback = fallback_strategy if fallback_strategy is not None else self.fallback_strategy

        working_df = df.copy()
        if time_col is None or time_col not in working_df.columns:
            time_col = "_scheduled_dep_ts"
            working_df[time_col] = construct_departure_timestamp(working_df)

        global_prior_rates = self._compute_strictly_prior_global_rate(
            working_df, target_col=target_col, time_col=time_col
        )

        if "route" not in working_df.columns and "origin_airport" in working_df.columns and "dest_airport" in working_df.columns:
            working_df["route"] = (
                working_df["origin_airport"].astype(str) + "_" + working_df["dest_airport"].astype(str)
            )

        dimensions = [
            ("origin_airport", "historical_origin"),
            ("dest_airport", "historical_destination"),
            ("airline", "historical_airline"),
            ("route", "historical_route"),
        ]

        historical_features = pd.DataFrame(index=df.index)
        notes: Dict[str, str] = {}

        for group_col, prefix in dimensions:
            if group_col not in working_df.columns:
                logger.warning("Column '%s' not present; skipping %s features.", group_col, prefix)
                continue

            prior_sums, prior_counts = self._compute_strictly_prior_group_stats(
                working_df,
                group_col=group_col,
                target_col=target_col,
                time_col=time_col,
                window_days=window_days,
            )

            rate_col = f"{prefix}_delay_rate"
            count_col = f"{prefix}_flight_count"

            with np.errstate(divide="ignore", invalid="ignore"):
                group_rates = np.where(prior_counts > 0, prior_sums / prior_counts, np.nan)

            final_rates = np.full(len(working_df), np.nan, dtype=float)
            sufficient_history_mask = prior_counts >= effective_min_history
            final_rates[sufficient_history_mask] = group_rates[sufficient_history_mask]

            insufficient_history_mask = ~sufficient_history_mask
            if effective_fallback == "global_prior":
                final_rates[insufficient_history_mask] = global_prior_rates[insufficient_history_mask]
            elif effective_fallback == "nan":
                final_rates[insufficient_history_mask] = np.nan
            else:
                final_rates[insufficient_history_mask] = float(effective_fallback)

            historical_features[rate_col] = np.round(final_rates, 4)
            historical_features[count_col] = prior_counts.astype("int32")

            pct_with_history = (sufficient_history_mask.sum() / max(1, len(working_df))) * 100.0
            notes[rate_col] = (
                f"Strictly prior delay rate (observation_timestamp < T). "
                f"{pct_with_history:.1f}% records met min_history={effective_min_history}; "
                f"remainder fell back to {effective_fallback}."
            )

        return historical_features, notes

    def compute_phase3_historical_features(
        self,
        df: pd.DataFrame,
        target_col: str = "delay_target",
        time_col: Optional[str] = None,
        min_history: int = 3,
        fallback_strategy: str = "global_prior",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate comprehensive Phase 3 historical delay features with counts, rates, and coverage audit.

        Entities covered:
            1. Airline: prior_airline_flight_count, prior_airline_delay_count, prior_airline_delay_rate
            2. Origin Airport: prior_origin_flight_count, prior_origin_delay_count, prior_origin_delay_rate
            3. Destination Airport: prior_dest_flight_count, prior_dest_delay_count, prior_dest_delay_rate
            4. Route: prior_route_flight_count, prior_route_delay_count, prior_route_delay_rate
            5. Airline + Departure Hour: prior_airline_dep_hour_flight_count, prior_airline_dep_hour_delay_rate
            6. Origin Airport + Departure Hour: prior_origin_dep_hour_flight_count, prior_origin_dep_hour_delay_rate

        Anti-Leakage Invariants:
            - Strictly prior: observation_timestamp < T
            - Current flight excluded
            - Future flights excluded
            - Fallback global_prior calculated strictly prior to T

        Args:
            df: Input DataFrame containing flight records and delay_target.
            target_col: Binary delay target column name.
            time_col: Departure timestamp column. If None, constructed from flight_date and scheduled_dep_time.
            min_history: Minimum required prior observations (default: 3).
            fallback_strategy: 'global_prior' or 'nan'.

        Returns:
            Tuple of (phase3_historical_features_df, coverage_audit_dict).
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found for historical feature calculation.")

        working_df = df.copy()
        if time_col is None or time_col not in working_df.columns:
            time_col = "_scheduled_dep_ts"
            working_df[time_col] = construct_departure_timestamp(working_df)

        # Strictly prior global rate for fallback (t < T)
        global_prior_rates = self._compute_strictly_prior_global_rate(
            working_df, target_col=target_col, time_col=time_col
        )

        # Resolve entity column names
        airline_col = "airline" if "airline" in working_df.columns else ("carrier" if "carrier" in working_df.columns else "airline")
        orig_col = "origin_airport" if "origin_airport" in working_df.columns else ("origin" if "origin" in working_df.columns else "origin_airport")
        dest_col = "dest_airport" if "dest_airport" in working_df.columns else ("dest" if "dest" in working_df.columns else "dest_airport")

        # Ensure route exists
        if "route" not in working_df.columns:
            working_df["route"] = (
                working_df[orig_col].astype(str) + "_" + working_df[dest_col].astype(str)
            )

        # Ensure departure hour exists for interaction features
        if "departure_hour" in working_df.columns:
            dep_hour_series = working_df["departure_hour"].astype(str)
        elif "scheduled_dep_time" in working_df.columns:
            raw_h = pd.to_numeric(working_df["scheduled_dep_time"], errors="coerce").fillna(0).astype(int) // 100
            dep_hour_series = raw_h.astype(str)
        elif "crs_dep_time" in working_df.columns:
            raw_h = pd.to_numeric(working_df["crs_dep_time"], errors="coerce").fillna(0).astype(int) // 100
            dep_hour_series = raw_h.astype(str)
        else:
            dep_hour_series = pd.Series("12", index=working_df.index)

        working_df["_airline_dep_hour"] = working_df[airline_col].astype(str) + "_h" + dep_hour_series
        working_df["_origin_dep_hour"] = working_df[orig_col].astype(str) + "_h" + dep_hour_series

        # Define all Phase 3 historical dimensions
        dimensions = [
            (airline_col, "prior_airline"),
            (orig_col, "prior_origin"),
            (dest_col, "prior_dest"),
            ("route", "prior_route"),
            ("_airline_dep_hour", "prior_airline_dep_hour"),
            ("_origin_dep_hour", "prior_origin_dep_hour"),
        ]

        historical_df = pd.DataFrame(index=df.index)
        coverage_audit: Dict[str, Dict[str, Any]] = {}
        total_records = len(working_df)

        for group_col, prefix in dimensions:
            prior_sums, prior_counts = self._compute_strictly_prior_group_stats(
                working_df,
                group_col=group_col,
                target_col=target_col,
                time_col=time_col,
            )

            count_col = f"{prefix}_flight_count"
            delay_count_col = f"{prefix}_delay_count"
            rate_col = f"{prefix}_delay_rate"

            with np.errstate(divide="ignore", invalid="ignore"):
                raw_rates = np.where(prior_counts > 0, prior_sums / prior_counts, np.nan)

            # Apply minimum history threshold
            final_rates = np.full(total_records, np.nan, dtype=float)
            sufficient_mask = prior_counts >= min_history
            insufficient_mask = (prior_counts > 0) & (prior_counts < min_history)
            no_history_mask = prior_counts == 0

            # Sufficient history -> use entity group rate
            final_rates[sufficient_mask] = raw_rates[sufficient_mask]

            # Insufficient or zero history -> apply strictly prior global fallback
            fallback_mask = ~sufficient_mask
            if fallback_strategy == "global_prior":
                final_rates[fallback_mask] = np.nan_to_num(global_prior_rates[fallback_mask], nan=0.0)
            elif fallback_strategy == "nan":
                final_rates[fallback_mask] = np.nan
            else:
                final_rates[fallback_mask] = float(fallback_strategy)

            historical_df[count_col] = prior_counts.astype("int32")
            historical_df[delay_count_col] = prior_sums.astype("int32")
            historical_df[rate_col] = np.round(final_rates, 4)

            # Provide standard alternate naming aliases for flexible downstream access
            if prefix == "prior_airline":
                historical_df["carrier_prior_flight_count"] = historical_df["airline_prior_flight_count"] = historical_df[count_col]
                historical_df["carrier_prior_delay_count"] = historical_df["airline_prior_delay_count"] = historical_df[delay_count_col]
                historical_df["carrier_prior_delay_rate"] = historical_df["airline_prior_delay_rate"] = historical_df[rate_col]
            elif prefix == "prior_origin":
                historical_df["origin_prior_flight_count"] = historical_df[count_col]
                historical_df["origin_prior_delay_count"] = historical_df[delay_count_col]
                historical_df["origin_prior_delay_rate"] = historical_df[rate_col]
            elif prefix == "prior_dest":
                historical_df["dest_prior_flight_count"] = historical_df[count_col]
                historical_df["dest_prior_delay_count"] = historical_df[delay_count_col]
                historical_df["dest_prior_delay_rate"] = historical_df[rate_col]
            elif prefix == "prior_route":
                historical_df["route_prior_flight_count"] = historical_df[count_col]
                historical_df["route_prior_delay_count"] = historical_df[delay_count_col]
                historical_df["route_prior_delay_rate"] = historical_df[rate_col]
            elif prefix == "prior_airline_dep_hour":
                historical_df["carrier_origin_hour_prior_flight_count"] = historical_df["airline_dep_hour_prior_flight_count"] = historical_df[count_col]
                historical_df["carrier_origin_hour_prior_delay_count"] = historical_df["airline_dep_hour_prior_delay_count"] = historical_df[delay_count_col]
                historical_df["carrier_origin_hour_prior_delay_rate"] = historical_df["airline_dep_hour_prior_delay_rate"] = historical_df[rate_col]

            # Compute coverage statistics
            n_no_hist = int(no_history_mask.sum())
            n_insufficient = int(insufficient_mask.sum())
            n_sufficient = int(sufficient_mask.sum())

            audit_item = {
                "total_records": total_records,
                "no_history_count": n_no_hist,
                "no_history_pct": round(n_no_hist / max(1, total_records) * 100, 2),
                "insufficient_history_count": n_insufficient,
                "insufficient_history_pct": round(n_insufficient / max(1, total_records) * 100, 2),
                "sufficient_history_count": n_sufficient,
                "sufficient_history_pct": round(n_sufficient / max(1, total_records) * 100, 2),
                "min_history_threshold": min_history,
                "fallback_strategy": fallback_strategy,
            }
            coverage_audit[prefix] = audit_item
            if prefix == "prior_airline":
                coverage_audit["carrier"] = coverage_audit["airline"] = audit_item
            elif prefix == "prior_origin":
                coverage_audit["origin"] = coverage_audit["origin_airport"] = audit_item
            elif prefix == "prior_dest":
                coverage_audit["dest"] = coverage_audit["dest_airport"] = audit_item
            elif prefix == "prior_route":
                coverage_audit["route"] = audit_item

        logger.info(
            "Computed Phase 3 historical delay features across %d dimensions (counts, delay counts, rates).",
            len(dimensions),
        )
        hist_notes = {c: f"Phase 3 strictly prior feature: {c}" for c in historical_df.columns}
        return historical_df, hist_notes, coverage_audit
