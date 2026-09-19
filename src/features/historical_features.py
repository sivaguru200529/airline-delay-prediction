"""Time-aware historical feature calculation module.

CRITICAL ANTI-LEAKAGE SPECIFICATION:
For every flight at scheduled departure time T:
    observation_timestamp < T

Guarantees:
    1. A flight never sees its own delay outcome.
    2. A flight never sees future flights.
    3. Concurrent flights (same departure timestamp) are strictly excluded from each other.
    4. First observations and low-history flights use a configurable fallback.
    5. Observation counts are retained alongside rates.
"""

from typing import Dict, List, Optional, Tuple, Union
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
    dates = pd.to_datetime(df[date_col], errors="coerce")
    raw_times = pd.to_numeric(df[dep_time_col], errors="coerce").fillna(0).astype(int)

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
            ts_summary = grp_data.groupby("ts")["target"].agg(["sum", "count"])

            if window_days is None:
                # Expanding strictly prior cumulative sums: shift(1) ensures t < T
                shifted_sum = ts_summary["sum"].cumsum().shift(1).fillna(0.0)
                shifted_count = ts_summary["count"].cumsum().shift(1).fillna(0)
            else:
                # Rolling window strictly prior: sum over [T - window_days, T)
                window_ns = pd.Timedelta(days=window_days).value
                shifted_sum = pd.Series(0.0, index=unique_ts)
                shifted_count = pd.Series(0, index=unique_ts)

                for i, current_t in enumerate(unique_ts):
                    current_t_val = current_t.value
                    window_start = current_t_val - window_ns
                    mask = (unique_ts < current_t) & (unique_ts >= pd.Timestamp(window_start))
                    if np.any(mask):
                        shifted_sum.iloc[i] = ts_summary.loc[mask, "sum"].sum()
                        shifted_count.iloc[i] = ts_summary.loc[mask, "count"].sum()

            # Map results back to all rows with that timestamp in this group
            sum_map = shifted_sum.to_dict()
            count_map = shifted_count.to_dict()

            grp_ts_list = grp_data["ts"].tolist()
            mapped_sums = [sum_map[t] for t in grp_ts_list]
            mapped_counts = [int(count_map[t]) for t in grp_ts_list]

            # Place into original array positions
            pos_indices = [df.index.get_loc(idx) for idx in grp_data["orig_idx"]]
            prior_sums[pos_indices] = mapped_sums
            prior_counts[pos_indices] = mapped_counts

        return pd.Series(prior_sums, index=df.index), pd.Series(prior_counts, index=df.index)

    def _compute_strictly_prior_global_rate(
        self,
        df: pd.DataFrame,
        target_col: str,
        time_col: str,
    ) -> pd.Series:
        """Compute strictly prior global delay rate across all flights prior to timestamp T."""
        temp = pd.DataFrame({
            "target": pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(float),
            "ts": pd.to_datetime(df[time_col]),
            "orig_idx": df.index,
        })

        ts_summary = temp.groupby("ts")["target"].agg(["sum", "count"])
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
        """Generate comprehensive historical delay features with strict anti-leakage boundaries.

        Features generated:
            - historical_origin_delay_rate
            - historical_origin_flight_count
            - historical_destination_delay_rate
            - historical_destination_flight_count
            - historical_airline_delay_rate
            - historical_airline_flight_count
            - historical_route_delay_rate
            - historical_route_flight_count

        Anti-Leakage Guarantees:
            - For flight at time T, uses only records where timestamp < T.
            - Current flight outcome is strictly excluded.
            - Concurrent flight outcomes are strictly excluded.
            - Future flight outcomes are strictly excluded.

        Args:
            df: Input DataFrame containing flight records and delay_target.
            target_col: Binary delay target column name.
            time_col: Departure timestamp column. If None, synthesizes from flight_date and scheduled_dep_time.
            window_days: Optional rolling lookback window in days (default None = expanding).
            min_history: Minimum required prior observations (defaults to config).
            fallback_strategy: 'global_prior' or 'nan' (defaults to config).

        Returns:
            Tuple of (historical_features_df, summary_notes).
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found for historical feature calculation.")

        effective_min_history = min_history if min_history is not None else self.min_history
        effective_fallback = fallback_strategy if fallback_strategy is not None else self.fallback_strategy

        # Ensure departure timestamp exists
        working_df = df.copy()
        if time_col is None or time_col not in working_df.columns:
            time_col = "_scheduled_dep_ts"
            working_df[time_col] = construct_departure_timestamp(working_df)

        logger.info(
            "Computing historical delay features for %d flights (min_history=%d, fallback=%s, window=%s)",
            len(working_df),
            effective_min_history,
            effective_fallback,
            f"{window_days}d" if window_days else "expanding",
        )

        # Compute strictly prior global baseline rate
        global_prior_rates = self._compute_strictly_prior_global_rate(
            working_df, target_col=target_col, time_col=time_col
        )

        # Ensure route exists
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

            # Compute raw group rate where count > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                group_rates = np.where(prior_counts > 0, prior_sums / prior_counts, np.nan)

            # Apply minimum history threshold
            final_rates = np.full(len(working_df), np.nan, dtype=float)
            sufficient_history_mask = prior_counts >= effective_min_history

            # Where count >= min_history, use group historical rate
            final_rates[sufficient_history_mask] = group_rates[sufficient_history_mask]

            # Where count < min_history, apply fallback
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

        logger.info(
            "Generated %d historical feature columns (rates + supporting volume counts).",
            len(historical_features.columns),
        )
        return historical_features, notes
