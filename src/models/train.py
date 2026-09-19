"""Model training orchestration module (Planned for Phase 2).

TODO (Phase 2):
    - Implement time-based train/test splitting (e.g. historical period -> train, later period -> test).
    - Implement baseline model: Logistic Regression.
    - Implement tree-based models: Random Forest, XGBoost.
    - Implement cross-validation with temporal folds (TimeSeriesSplit).
    - Save serialized models to models/ directory.
"""

from typing import Any, Dict, Optional
import pandas as pd


def train_model(
    train_data: pd.DataFrame,
    model_type: str = "xgboost",
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train delay prediction classification model.
    
    NOTE: Actual training implementation is deferred to Phase 2.
    """
    raise NotImplementedError("Model training pipeline is planned for Phase 2.")
