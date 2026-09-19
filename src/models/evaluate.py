"""Model evaluation and metrics calculation module (Planned for Phase 2).

TODO (Phase 2):
    - Compute multi-metric classification performance:
        * ROC-AUC and PR-AUC (Precision-Recall AUC)
        * Precision, Recall, F1-score (at various decision thresholds)
        * Balanced Accuracy and Confusion Matrix
        * Brier Score (probability calibration)
    - Compute SHAP values for global and local feature importance.
    - Export evaluation figures to reports/figures/.
"""

from typing import Any, Dict
import pandas as pd


def evaluate_model(
    model: Any,
    test_features: pd.DataFrame,
    test_targets: pd.Series,
) -> Dict[str, float]:
    """Evaluate model performance on out-of-time test dataset.

    NOTE: Evaluation pipeline is deferred to Phase 2.
    """
    raise NotImplementedError("Model evaluation pipeline is planned for Phase 2.")
