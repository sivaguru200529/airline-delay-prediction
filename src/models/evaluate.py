"""Model evaluation and metrics calculation module.

Provides reusable, model-agnostic functions for:
- Classification performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score)
- Confusion matrix extraction
- Threshold sensitivity analysis
- Probability calibration analysis
- Model comparison tables
- Visualization export (Confusion matrix, ROC, PR, Calibration, Feature importance, SHAP)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger("model_evaluation")


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, List[int]],
    y_prob: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate multi-metric evaluation for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for positive class (delay_target = 1).
        threshold: Classification decision threshold (default: 0.50).

    Returns:
        Dict containing Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score,
        and confusion matrix elements.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # Base counts
    n_samples = len(y_true_arr)
    n_pos = int(np.sum(y_true_arr))
    n_neg = n_samples - n_pos

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    # ROC-AUC & PR-AUC require at least one positive and negative sample
    if n_pos > 0 and n_neg > 0:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            roc_auc = 0.50
        try:
            pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
        except ValueError:
            pr_auc = float(n_pos / n_samples)
    else:
        roc_auc = 0.50
        pr_auc = float(n_pos / max(n_samples, 1))

    # Confusion Matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "n_samples": n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": round(float(threshold), 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def compute_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, int]:
    """Compute confusion matrix components (TN, FP, FN, TP)."""
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def evaluate_thresholds(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Analyze precision, recall, and F1 across candidate decision thresholds.

    Evaluated strictly on validation data to prevent test leakage.

    Args:
        y_true: Ground truth binary target.
        y_prob: Predicted delay probabilities.
        thresholds: List of thresholds to evaluate.

    Returns:
        pd.DataFrame containing sensitivity analysis across thresholds.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    rows = []
    for thresh in thresholds:
        metrics = calculate_classification_metrics(y_true, y_prob, threshold=thresh)
        rows.append({
            "threshold": thresh,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["confusion_matrix"]["tp"],
            "fp": metrics["confusion_matrix"]["fp"],
            "fn": metrics["confusion_matrix"]["fn"],
            "tn": metrics["confusion_matrix"]["tn"],
        })

    return pd.DataFrame(rows)


def compute_calibration_curve(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Compute calibration curve and Brier score.

    Args:
        y_true: Ground truth binary target.
        y_prob: Predicted delay probabilities.
        n_bins: Number of probability discretization bins.

    Returns:
        Dict containing fraction_of_positives, mean_predicted_value, and brier_score.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    # Use uniform strategy for small datasets
    prob_true, prob_pred = calibration_curve(
        y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform"
    )
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    return {
        "fraction_of_positives": [round(float(x), 4) for x in prob_true],
        "mean_predicted_value": [round(float(x), 4) for x in prob_pred],
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
    }


def compare_models(
    models_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Format standardized model comparison table.

    Args:
        models_results: Mapping of model_name -> metrics_dict from calculate_classification_metrics.

    Returns:
        Formatted pd.DataFrame suitable for reporting.
    """
    rows = []
    for model_name, metrics in models_results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1": metrics.get("f1", 0.0),
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "Brier Score": metrics.get("brier_score", 0.0),
        })

    return pd.DataFrame(rows)


# ==============================================================================
# VISUALIZATION EXPORT UTILITIES
# ==============================================================================

def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    model_name: str = "Selected Model",
    save_path: Optional[Path] = None,
) -> None:
    """Export formatted confusion matrix figure."""
    matrix = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    # Annotations
    for (i, j), z in np.ndenumerate(matrix):
        label = f"{z:,}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
        color = "white" if z > (matrix.max() / 2) else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred On-Time (0)", "Pred Delayed (1)"], fontsize=10)
    ax.set_yticklabels(["Actual On-Time (0)", "Actual Delayed (1)"], fontsize=10)
    ax.set_title(f"Confusion Matrix: {model_name}\n(Development Dataset)", pad=15, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved confusion matrix plot to %s", save_path)
    plt.close(fig)


def plot_roc_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    save_path: Optional[Path] = None,
) -> None:
    """Export multi-model ROC Curve figure."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (y_true, y_prob) in curves_dict.items():
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
        else:
            ax.plot([0, 1], [0, 1], "--", label=f"{name} (Single-Class)")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison\n(Out-of-Time Validation)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved ROC curves to %s", save_path)
    plt.close(fig)


def plot_pr_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    save_path: Optional[Path] = None,
) -> None:
    """Export Precision-Recall curve figure."""
    fig, ax = plt.subplots(figsize=(7, 6))

    baseline_rate = None
    for name, (y_true, y_prob) in curves_dict.items():
        if len(np.unique(y_true)) > 1:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
            if baseline_rate is None:
                baseline_rate = float(np.mean(y_true))

    if baseline_rate is not None:
        ax.axhline(baseline_rate, color="k", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_rate:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Delayed Flights Captured)", fontsize=11)
    ax.set_ylabel("Precision (Correctly Flagged Delays)", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curve Comparison\n(Imbalanced Target Evaluation)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved PR curves to %s", save_path)
    plt.close(fig)


def plot_calibration_curves(
    curves_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    save_path: Optional[Path] = None,
) -> None:
    """Export probability calibration curve figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", lw=2, label="Perfectly Calibrated")

    for name, (y_true, y_prob) in curves_dict.items():
        if len(np.unique(y_true)) > 1:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")
            brier = brier_score_loss(y_true, y_prob)
            ax.plot(prob_pred, prob_true, "s-", lw=2, label=f"{name} (Brier = {brier:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Fraction of Delays", fontsize=11)
    ax.set_title("Probability Calibration Reliability Diagram\n(Development Dataset)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved calibration curves to %s", save_path)
    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Tree-Based Model",
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """Export ranked feature importance bar chart using meaningful feature names."""
    df_plot = importance_df.sort_values(by="importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(5, int(top_n * 0.35))))
    bars = ax.barh(df_plot["feature"], df_plot["importance"], color="#2b5c8f", alpha=0.85)
    ax.set_xlabel("Gini / Gain Importance Weight", fontsize=11)
    ax.set_title(f"Top {top_n} Predictive Features: {model_name}\n(Meaningful Transformed Names)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)
    plt.close(fig)


def plot_shap_summary(
    shap_values: np.ndarray,
    features_sample: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> bool:
    """Generate SHAP summary plot using human-readable feature names.

    Args:
        shap_values: SHAP value matrix from TreeExplainer.
        features_sample: DataFrame with named columns corresponding to shap_values.
        save_path: Output file destination.

    Returns:
        True if SHAP plot generated successfully, False otherwise.
    """
    try:
        import shap
        fig = plt.figure(figsize=(9, 6))
        # Handle 3D multiclass shape if returned by some versions
        vals = shap_values
        if isinstance(vals, list) and len(vals) == 2:
            vals = vals[1]
        elif isinstance(vals, np.ndarray) and len(vals.shape) == 3:
            vals = vals[:, :, 1]

        shap.summary_plot(
            vals,
            features_sample,
            max_display=15,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Global Feature Contribution Summary\n(Feature Contribution to Model Output)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            logger.info("Saved SHAP summary plot to %s", save_path)
        plt.close(fig)
        return True
    except Exception as e:
        logger.warning("Could not render SHAP summary plot: %s", e)
        return False
