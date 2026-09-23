"""Machine learning training, prediction, evaluation, and splitting modules (Phase 2B)."""

from src.models.evaluate import (
    calculate_classification_metrics,
    compare_models,
    compute_calibration_curve,
    compute_confusion_matrix,
    evaluate_thresholds,
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_shap_summary,
)
from src.models.predict import load_model, predict_delay_probability
from src.models.split import ChronologicalSplitResult, split_dataset_chronologically
from src.models.train import (
    build_preprocessor,
    determine_feature_groups,
    extract_feature_importances,
    run_training_pipeline,
    train_models,
)

__all__ = [
    "split_dataset_chronologically",
    "ChronologicalSplitResult",
    "calculate_classification_metrics",
    "compute_confusion_matrix",
    "evaluate_thresholds",
    "compute_calibration_curve",
    "compare_models",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_calibration_curves",
    "plot_feature_importance",
    "plot_shap_summary",
    "load_model",
    "predict_delay_probability",
    "determine_feature_groups",
    "build_preprocessor",
    "train_models",
    "extract_feature_importances",
    "run_training_pipeline",
]
