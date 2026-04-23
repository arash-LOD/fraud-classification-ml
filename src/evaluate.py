"""
evaluate.py
-----------
Model evaluation utilities: metrics, plots, and threshold tuning
for the fraud classification project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute and log a full suite of classification metrics."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"]  = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"]   = average_precision_score(y_true, y_proba)

    logger.info("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="model", save=True):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name}.png")
        plt.savefig(path, dpi=150)
        logger.info(f"Confusion matrix saved to {path}")
    plt.show()
    return cm


def plot_roc_curve(y_true, y_proba, model_name="model", save=True):
    """Plot the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name}.png")
        plt.savefig(path, dpi=150)
        logger.info(f"ROC curve saved to {path}")
    plt.show()
    return auc


def plot_precision_recall_curve(y_true, y_proba, model_name="model", save=True):
    """Plot the Precision-Recall curve (especially useful for imbalanced data)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color="steelblue", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"pr_curve_{model_name}.png")
        plt.savefig(path, dpi=150)
        logger.info(f"PR curve saved to {path}")
    plt.show()
    return pr_auc


def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Sweep decision thresholds and return the one that maximises the chosen metric.
    Supported metrics: 'f1', 'precision', 'recall'.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    if metric == "f1":
        scores = f1_scores
    elif metric == "precision":
        scores = precision[:-1]
    elif metric == "recall":
        scores = recall[:-1]
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from: f1, precision, recall.")

    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold ({metric}): {best_threshold:.4f}  Score: {scores[best_idx]:.4f}")
    return best_threshold


def evaluate_all(y_true, y_pred, y_proba, model_name="model"):
    """Run the full evaluation suite."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    plot_confusion_matrix(y_true, y_pred, model_name=model_name)
    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, model_name=model_name)
        plot_precision_recall_curve(y_true, y_proba, model_name=model_name)
        best_thresh = find_best_threshold(y_true, y_proba)
        metrics["best_threshold"] = best_thresh
    return metrics
