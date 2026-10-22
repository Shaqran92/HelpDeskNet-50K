"""
Model evaluation and metrics computation.

Provides functions for computing classification metrics, calibration
metrics, confusion matrices, and generating evaluation reports.
All functions clearly document the evaluation split and label provenance.
"""

import logging
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

TARGET_CLASSES = ["Billing", "Refund", "Technical", "Account Access", "Shipping", "Other"]


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: Optional[list[str]] = None,
    split_name: str = "unknown",
    label_provenance: str = "proxy",
) -> dict:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : list[str], optional
        Target class names. Derived from data if None.
    split_name : str
        Name of the evaluation split (e.g., 'validation', 'test').
    label_provenance : str
        How the true labels were obtained ('proxy', 'expert', 'simulated').

    Returns
    -------
    dict
        Dictionary of metrics.
    """
    if classes is None:
        classes = sorted(set(y_true.unique()) | set(np.unique(y_pred)))

    # Present classes (some may have no samples)
    present_labels = sorted(set(y_true.unique()) & set(classes))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=present_labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=present_labels, zero_division=0)

    macro_precision = precision_score(
        y_true, y_pred, average="macro", labels=present_labels, zero_division=0
    )
    macro_recall = recall_score(
        y_true, y_pred, average="macro", labels=present_labels, zero_division=0
    )

    # Per-class metrics
    per_class = {}
    for cls in classes:
        cls_mask_true = y_true == cls
        cls_mask_pred = y_pred == cls
        tp = int((cls_mask_true & cls_mask_pred).sum())
        fp = int((~cls_mask_true & cls_mask_pred).sum())
        fn = int((cls_mask_true & ~cls_mask_pred).sum())
        support = int(cls_mask_true.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class": per_class,
        "n_samples": len(y_true),
        "split_name": split_name,
        "label_provenance": label_provenance,
        "classes": classes,
        "provenance_note": (
            f"Evaluated on '{split_name}' split. Labels are '{label_provenance}'. "
            f"Results may differ with independently verified labels."
        ),
    }


def compute_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: Optional[list[str]] = None,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Compute confusion matrix as a labeled DataFrame.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : list[str], optional
        Class labels.
    normalize : str, optional
        'true' (rows), 'pred' (columns), 'all', or None.

    Returns
    -------
    pd.DataFrame
        Confusion matrix.
    """
    if classes is None:
        classes = sorted(set(y_true.unique()) | set(np.unique(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
    return pd.DataFrame(cm, index=classes, columns=classes)


def compute_calibration_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: list[str],
    n_bins: int = 10,
) -> dict:
    """Compute probability calibration metrics.

    Parameters
    ----------
    y_true : pd.Series
        True labels.
    y_proba : np.ndarray
        Predicted probability matrix (n_samples, n_classes).
    classes : list[str]
        Class names corresponding to probability columns.
    n_bins : int
        Number of bins for reliability diagram.

    Returns
    -------
    dict
        Brier score, ECE, per-class calibration data.
    """
    # Overall Brier score (one-vs-all average)
    brier_scores = {}
    calibration_data = {}

    for i, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        cls_proba = y_proba[:, i]

        # Brier score
        brier = brier_score_loss(binary_true, cls_proba)
        brier_scores[cls] = float(brier)

        # Calibration curve
        try:
            fraction_pos, mean_predicted = calibration_curve(
                binary_true, cls_proba, n_bins=n_bins, strategy="uniform"
            )
            calibration_data[cls] = {
                "fraction_positive": fraction_pos.tolist(),
                "mean_predicted": mean_predicted.tolist(),
            }
        except ValueError:
            calibration_data[cls] = {
                "fraction_positive": [],
                "mean_predicted": [],
                "note": "Could not compute calibration curve.",
            }

    # Expected Calibration Error (ECE)
    max_proba = y_proba.max(axis=1)
    y_pred_classes = np.array([classes[i] for i in y_proba.argmax(axis=1)])
    correct = (y_pred_classes == y_true.values).astype(float)

    ece = _compute_ece(correct, max_proba, n_bins)

    # Multi-class log loss
    try:
        ll = log_loss(y_true, y_proba, labels=classes)
    except ValueError:
        ll = None

    return {
        "brier_scores": brier_scores,
        "mean_brier": float(np.mean(list(brier_scores.values()))),
        "ece": float(ece),
        "log_loss": float(ll) if ll is not None else None,
        "calibration_curves": calibration_data,
        "n_bins": n_bins,
        "note": (
            "Calibration metrics computed on validation data. "
            "Model confidence is NOT the probability that the "
            "observed label is wrong."
        ),
    }


def _compute_ece(
    correct: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Parameters
    ----------
    correct : np.ndarray
        Binary array of correct predictions.
    confidence : np.ndarray
        Model confidence values.
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        ECE value.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(correct)

    for i in range(n_bins):
        mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
        if i == 0:
            mask = mask | (confidence == bin_boundaries[i])

        n_bin = mask.sum()
        if n_bin > 0:
            avg_confidence = confidence[mask].mean()
            avg_accuracy = correct[mask].mean()
            ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)

    return ece


def save_evaluation_report(
    metrics: dict,
    output_path: str,
    model_name: str = "unnamed",
) -> str:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary.
    output_path : str
        Output file path.
    model_name : str
        Model identifier.

    Returns
    -------
    str
        Path to saved file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    report = {
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Evaluation report saved to %s", output_path)
    return output_path


def save_predictions(
    ticket_ids: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: list[str],
    output_path: str,
    model_name: str = "unnamed",
    split_name: str = "unknown",
) -> str:
    """Save per-record predictions for analysis.

    Parameters
    ----------
    ticket_ids : pd.Series
        Record identifiers.
    y_true : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Probability estimates.
    classes : list[str]
        Class names.
    output_path : str
        Output CSV path.
    model_name : str
        Model identifier.
    split_name : str
        Split name.

    Returns
    -------
    str
        Path to saved file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.DataFrame({
        "ticket_id": ticket_ids.values,
        "true_label": y_true.values,
        "predicted_label": y_pred,
        "correct": y_true.values == y_pred,
    })

    if y_proba is not None:
        for i, cls in enumerate(classes):
            df[f"prob_{cls}"] = y_proba[:, i]
        df["max_confidence"] = y_proba.max(axis=1)
        df["predicted_by_proba"] = [classes[i] for i in y_proba.argmax(axis=1)]

    df["model_name"] = model_name
    df["split_name"] = split_name
    df.to_csv(output_path, index=False)

    logger.info("Predictions saved to %s (%d records)", output_path, len(df))
    return output_path


def compare_models(
    results: dict[str, dict],
) -> pd.DataFrame:
    """Compare multiple model results in a summary table.

    Parameters
    ----------
    results : dict[str, dict]
        Model name → metrics dict mapping.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {
            "model": model_name,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "macro_precision": metrics.get("macro_precision"),
            "macro_recall": metrics.get("macro_recall"),
            "n_samples": metrics.get("n_samples"),
            "split": metrics.get("split_name"),
            "label_provenance": metrics.get("label_provenance"),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return df
