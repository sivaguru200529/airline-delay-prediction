"""Chronological train/validation/test dataset splitting module.

Enforces strict out-of-time temporal ordering:
    Past (Train) -> Later Period (Validation) -> Future Period (Test)
Guarantees:
    max(train_timestamp) <= min(validation_timestamp)
    max(validation_timestamp) <= min(test_timestamp)

Strictly separates feature matrix X from supervision label y (delay_target)
and purges all post-flight leakage variables.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("model_split")


@dataclass
class ChronologicalSplitResult:
    """Container holding chronological split partitions and metadata."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    split_summary: Dict[str, Any]

    def to_tuple(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
        """Return tuple of (X_train, y_train, X_val, y_val, X_test, y_test, split_summary)."""
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
            self.split_summary,
        )


def split_dataset_chronologically(
    df: pd.DataFrame,
    date_col: str = "flight_date",
    time_col: Optional[str] = "scheduled_dep_time",
    target_col: str = "delay_target",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    leakage_columns: Optional[List[str]] = None,
    drop_target_from_X: bool = True,
    config: Optional[AppConfig] = None,
) -> ChronologicalSplitResult:
    """Split dataset chronologically into train, validation, and test partitions.

    Args:
        df: Input DataFrame containing features and target.
        date_col: Column name for flight date.
        time_col: Optional column name for scheduled departure time (HHMM integer/string).
        target_col: Prediction target column name (e.g. 'delay_target').
        train_pct: Fraction of earliest records allocated to train (e.g. 0.70).
        val_pct: Fraction of intermediate records allocated to validation (e.g. 0.15).
        leakage_columns: Optional list of post-flight leakage columns to purge from X.
        drop_target_from_X: Whether to strictly drop target_col from X_train, X_val, X_test.
        config: Application configuration.

    Returns:
        ChronologicalSplitResult containing feature matrices, target series, and metadata.

    Raises:
        ValueError: If input df is empty, missing target_col, or invalid split fractions.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot split empty dataset.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if train_pct <= 0 or val_pct <= 0 or (train_pct + val_pct) >= 1.0:
        raise ValueError(
            f"Invalid split fractions: train_pct={train_pct}, val_pct={val_pct}. "
            f"Must be positive and satisfy train_pct + val_pct < 1.0."
        )

    cfg = config or get_config()
    leakage_cols = set(leakage_columns or cfg.post_flight_leakage_columns)

    # 1. Build composite sorting timestamp or multi-column sort
    df_clean = df.copy()
    sort_cols = [date_col]
    if time_col and time_col in df_clean.columns:
        sort_cols.append(time_col)

    sorted_df = df_clean.sort_values(sort_cols).reset_index(drop=True)
    total_records = len(sorted_df)

    n_train = int(total_records * train_pct)
    n_val = int(total_records * val_pct)
    n_test = total_records - (n_train + n_val)

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Split produced empty partition with total_records={total_records}: "
            f"train={n_train}, val={n_val}, test={n_test}"
        )

    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_train + n_val:].copy().reset_index(drop=True)

    # 2. Extract targets
    y_train = train_df[target_col].astype(int).copy()
    y_val = val_df[target_col].astype(int).copy()
    y_test = test_df[target_col].astype(int).copy()

    # 3. Purge target and leakage columns from feature matrices X
    cols_to_drop = set()
    if drop_target_from_X:
        cols_to_drop.add(target_col)
    for col in df.columns:
        if col in leakage_cols or any(kw in col.lower() for kw in cfg.forbidden_leakage_keywords):
            cols_to_drop.add(col)

    X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns]).copy()
    X_val = val_df.drop(columns=[c for c in cols_to_drop if c in val_df.columns]).copy()
    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns]).copy()

    # 4. Temporal range validation
    train_dates = (str(train_df[date_col].min()), str(train_df[date_col].max()))
    val_dates = (str(val_df[date_col].min()), str(val_df[date_col].max()))
    test_dates = (str(test_df[date_col].min()), str(test_df[date_col].max()))

    # Chronological integrity check
    if train_dates[1] > val_dates[0]:
        logger.warning(
            "Temporal boundary overlap between Train max (%s) and Val min (%s)",
            train_dates[1], val_dates[0]
        )
    if val_dates[1] > test_dates[0]:
        logger.warning(
            "Temporal boundary overlap between Val max (%s) and Test min (%s)",
            val_dates[1], test_dates[0]
        )

    split_summary = {
        "total_records": total_records,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_pct": round(len(train_df) / total_records * 100, 2),
        "val_pct": round(len(val_df) / total_records * 100, 2),
        "test_pct": round(len(test_df) / total_records * 100, 2),
        "train_date_range": list(train_dates),
        "val_date_range": list(val_dates),
        "test_date_range": list(test_dates),
        "train_delay_rate": round(float(y_train.mean()) * 100, 2),
        "val_delay_rate": round(float(y_val.mean()) * 100, 2),
        "test_delay_rate": round(float(y_test.mean()) * 100, 2),
        "sample_size_limitation": (
            "The current development dataset contains only approximately 481 flights spanning "
            "January 1–10, 2024. This split serves as an engineering verification / smoke test. "
            "Production modeling requires multi-month or multi-year public extracts."
            if total_records < 2000 else "Sufficient sample size for operational modeling."
        ),
    }

    logger.info(
        "Chronological split complete: Train=%d (%s to %s, delay_rate=%.1f%%), "
        "Val=%d (%s to %s, delay_rate=%.1f%%), Test=%d (%s to %s, delay_rate=%.1f%%)",
        len(train_df), train_dates[0], train_dates[1], split_summary["train_delay_rate"],
        len(val_df), val_dates[0], val_dates[1], split_summary["val_delay_rate"],
        len(test_df), test_dates[0], test_dates[1], split_summary["test_delay_rate"],
    )

    return ChronologicalSplitResult(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        split_summary=split_summary,
    )
