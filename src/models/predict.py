"""Inference and probability prediction module for pre-departure flight instances.

Provides reusable prediction functions:
- Supports single-flight dictionary input or batch DataFrame input.
- Validates pre-departure schema and enforces strict anti-leakage guards.
- Predicts continuous delay probability P(Delay >= 15 min | features).
- Applies configurable decision threshold to determine binary predicted_class.
- Maps continuous delay probability to operational risk tiers:
    LOW, MEDIUM, HIGH, VERY HIGH
  via RiskCategoryThresholds from config.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.config import AppConfig, get_config
from src.utils.logger import get_logger

logger = get_logger("model_predict")


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    config: Optional[AppConfig] = None,
) -> Pipeline:
    """Load serialized model pipeline from models/ directory.

    Args:
        model_path: Optional direct path to .joblib file.
        config: Application configuration.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    cfg = config or get_config()
    target_path = Path(model_path) if model_path else (cfg.models_dir / "delay_model.joblib")

    if not target_path.exists():
        raise FileNotFoundError(
            f"Serialized model not found at {target_path}. Run model training first via 'python main.py --phase2b'."
        )

    logger.info("Loading serialized delay prediction model from %s", target_path)
    return joblib.load(target_path)


def predict_delay_probability(
    model: Pipeline,
    features: Union[pd.DataFrame, Dict[str, Any]],
    threshold: Optional[float] = None,
    config: Optional[AppConfig] = None,
) -> Union[Dict[str, Any], pd.DataFrame]:
    """Generate delay probability, binary prediction, and operational risk tier.

    Strictly enforces pre-departure integrity:
        - Rejects or strips any post-flight leakage columns.
        - Strips delay_target if present.

    Args:
        model: Trained scikit-learn Pipeline.
        features: Single record dictionary or batch pandas DataFrame.
        threshold: Classification decision threshold (default: config.default_classification_threshold or 0.50).
        config: Application configuration.

    Returns:
        If input is Dict:
            Dictionary with 'delay_probability', 'predicted_class', 'risk_level', 'threshold'.
        If input is DataFrame:
            DataFrame containing original features + 'delay_probability', 'predicted_class', 'risk_level'.
    """
    cfg = config or get_config()
    decision_threshold = threshold if threshold is not None else cfg.default_classification_threshold

    # Convert single dictionary to single-row DataFrame
    is_dict_input = isinstance(features, dict)
    if is_dict_input:
        df_input = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df_input = features.copy()
    else:
        raise TypeError(f"Expected DataFrame or dict for features, got {type(features).__name__}")

    # Enforce anti-leakage invariants: strip post-flight outcome columns and delay_target
    leakage_cols = set(cfg.post_flight_leakage_columns)
    cols_to_drop = [c for c in df_input.columns if c in leakage_cols or c == "delay_target"]
    if cols_to_drop:
        logger.warning(
            "Purged potential post-flight or target columns from prediction input: %s",
            cols_to_drop
        )
        df_clean = df_input.drop(columns=cols_to_drop)
    else:
        df_clean = df_input

    # Generate continuous delay probabilities
    try:
        probabilities = model.predict_proba(df_clean)[:, 1]
    except Exception as e:
        logger.error("Failed to generate model probabilities: %s", e)
        raise

    predicted_classes = (probabilities >= decision_threshold).astype(int)
    risk_tiers = [cfg.risk_thresholds.classify_risk(float(p)) for p in probabilities]

    if is_dict_input:
        return {
            "delay_probability": round(float(probabilities[0]), 4),
            "predicted_class": int(predicted_classes[0]),
            "risk_level": risk_tiers[0],
            "threshold": round(float(decision_threshold), 3),
        }
    else:
        result_df = df_clean.copy()
        result_df["delay_probability"] = [round(float(p), 4) for p in probabilities]
        result_df["predicted_class"] = predicted_classes
        result_df["risk_level"] = risk_tiers
        return result_df
