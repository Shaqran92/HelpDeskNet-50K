"""
Robust training methods for learning with noisy labels.

Implements label filtering, confidence-weighted training, and
noise-robust loss strategies. Documents which labels each method uses.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger(__name__)


def filter_by_label_issues(
    X: pd.Series,
    y: pd.Series,
    label_issue_scores: np.ndarray,
    threshold: float = 0.5,
    min_retain_fraction: float = 0.5,
) -> tuple[pd.Series, pd.Series, dict]:
    """Filter training data by removing suspected label issues.

    Parameters
    ----------
    X : pd.Series
        Training text.
    y : pd.Series
        Training labels.
    label_issue_scores : np.ndarray
        Scores indicating likelihood of label issues (higher = more suspicious).
    threshold : float
        Score threshold above which records are removed.
    min_retain_fraction : float
        Minimum fraction of data to retain (safety guard).

    Returns
    -------
    tuple[pd.Series, pd.Series, dict]
        Filtered X, filtered y, and filtering report.

    Notes
    -----
    WARNING: Filtering can remove difficult but correctly labeled examples.
    Always compare classification performance with and without filtering.
    """
    n_original = len(X)

    # Determine which records to keep
    keep_mask = label_issue_scores <= threshold

    # Safety check: don't remove too much data
    retain_fraction = keep_mask.sum() / n_original
    if retain_fraction < min_retain_fraction:
        # Adjust threshold to keep at least min_retain_fraction
        sorted_scores = np.sort(label_issue_scores)
        adjusted_idx = int(n_original * (1 - min_retain_fraction))
        adjusted_threshold = sorted_scores[min(adjusted_idx, n_original - 1)]
        keep_mask = label_issue_scores <= adjusted_threshold
        logger.warning(
            "Threshold %.3f would remove too much data (%.1f%%). "
            "Adjusted to %.3f to retain at least %.0f%%.",
            threshold, (1 - retain_fraction) * 100,
            adjusted_threshold, min_retain_fraction * 100,
        )

    X_filtered = X[keep_mask].reset_index(drop=True)
    y_filtered = y[keep_mask].reset_index(drop=True)

    report = {
        "n_original": n_original,
        "n_filtered": len(X_filtered),
        "n_removed": n_original - len(X_filtered),
        "removal_fraction": 1 - len(X_filtered) / n_original,
        "threshold_used": float(threshold),
        "retain_fraction": len(X_filtered) / n_original,
        "class_distribution_before": y.value_counts().to_dict(),
        "class_distribution_after": y_filtered.value_counts().to_dict(),
        "warning": (
            "Filtering may remove difficult but correctly labeled examples. "
            "Compare performance with and without filtering."
        ),
    }

    logger.info(
        "Filtered %d / %d records (%.1f%% removed).",
        report["n_removed"], n_original, report["removal_fraction"] * 100,
    )

    return X_filtered, y_filtered, report


def train_with_corrected_labels(
    model,
    X: pd.Series,
    y_observed: pd.Series,
    corrections: pd.DataFrame,
    ticket_id_col: str = "ticket_id",
    corrected_label_col: str = "reviewed_label",
    model_name: str = "unnamed",
) -> tuple:
    """Train model using corrected labels where available.

    Parameters
    ----------
    model : sklearn estimator
        Model to train.
    X : pd.Series
        Training text (indexed by ticket_id or aligned with y_observed).
    y_observed : pd.Series
        Original observed labels.
    corrections : pd.DataFrame
        Review corrections with ticket_id and corrected label.
    ticket_id_col : str
        Column in corrections with ticket IDs.
    corrected_label_col : str
        Column with corrected labels.
    model_name : str
        Name for logging.

    Returns
    -------
    tuple
        (fitted_model, corrected_labels, correction_report)
    """
    y_corrected = y_observed.copy()

    if corrections is not None and len(corrections) > 0:
        correction_map = dict(
            zip(corrections[ticket_id_col], corrections[corrected_label_col])
        )
        n_corrections = 0
        for idx in y_corrected.index:
            ticket_id = idx  # Assumes index is ticket_id
            if ticket_id in correction_map:
                old_label = y_corrected[idx]
                new_label = correction_map[ticket_id]
                if old_label != new_label:
                    y_corrected[idx] = new_label
                    n_corrections += 1
    else:
        n_corrections = 0

    logger.info(
        "Training '%s' with %d label corrections applied.",
        model_name, n_corrections,
    )

    model.fit(X, y_corrected)

    report = {
        "model_name": model_name,
        "n_corrections_applied": n_corrections,
        "n_total_training": len(X),
        "labels_used": "observed + reviewed corrections",
        "note": (
            "Labels that were not reviewed retain their observed "
            "(possibly noisy) values."
        ),
    }

    return model, y_corrected, report


def sample_weight_by_confidence(
    y: pd.Series,
    label_issue_scores: np.ndarray,
    method: str = "inverse",
) -> np.ndarray:
    """Compute sample weights based on label confidence.

    Parameters
    ----------
    y : pd.Series
        Training labels.
    label_issue_scores : np.ndarray
        Label issue scores (higher = more suspicious).
    method : str
        Weighting method:
        - 'inverse': weight = 1 - score
        - 'soft': weight = exp(-score)
        - 'binary': weight = 0 if score > 0.5, else 1

    Returns
    -------
    np.ndarray
        Sample weights.
    """
    scores = np.clip(label_issue_scores, 0, 1)

    if method == "inverse":
        weights = 1.0 - scores
    elif method == "soft":
        weights = np.exp(-scores)
    elif method == "binary":
        weights = (scores <= 0.5).astype(float)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Ensure minimum weight to avoid zero-weight samples
    weights = np.maximum(weights, 0.01)

    return weights


def run_robust_experiment(
    model_builder: callable,
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    label_issue_scores: Optional[np.ndarray] = None,
    corrections: Optional[pd.DataFrame] = None,
    evaluate_fn: Optional[callable] = None,
    model_name: str = "unnamed",
    builder_kwargs: Optional[dict] = None,
) -> dict:
    """Run a suite of robust training experiments.

    Compares:
    1. Standard training on observed labels
    2. Training after filtering suspected issues
    3. Training with confidence weights
    4. Training with corrected labels (if available)

    Parameters
    ----------
    model_builder : callable
        Function that builds an unfitted model.
    X_train, y_train : pd.Series
        Training data with observed labels.
    X_val, y_val : pd.Series
        Validation data.
    label_issue_scores : np.ndarray, optional
        Scores for label-issue filtering/weighting.
    corrections : pd.DataFrame, optional
        Label corrections from review.
    evaluate_fn : callable, optional
        Function(y_true, y_pred) -> dict.
    model_name : str
        Base model name.
    builder_kwargs : dict, optional
        Args for model builder.

    Returns
    -------
    dict
        Results for each training strategy.
    """
    if builder_kwargs is None:
        builder_kwargs = {}

    results = {}

    # 1. Standard training
    logger.info("Running standard training...")
    model_std = model_builder(**builder_kwargs)
    model_std.fit(X_train, y_train)
    y_pred_std = model_std.predict(X_val)
    if evaluate_fn:
        results["standard"] = evaluate_fn(y_val, y_pred_std)
    results["standard_labels_used"] = "observed (possibly noisy)"

    # 2. Filtered training
    if label_issue_scores is not None:
        logger.info("Running filtered training...")
        X_filt, y_filt, filt_report = filter_by_label_issues(
            X_train, y_train, label_issue_scores
        )
        model_filt = model_builder(**builder_kwargs)
        model_filt.fit(X_filt, y_filt)
        y_pred_filt = model_filt.predict(X_val)
        if evaluate_fn:
            results["filtered"] = evaluate_fn(y_val, y_pred_filt)
        results["filtered_report"] = filt_report
        results["filtered_labels_used"] = "observed, filtered by issue score"

    # 3. Confidence-weighted training (for models supporting sample_weight)
    if label_issue_scores is not None:
        logger.info("Running confidence-weighted training...")
        weights = sample_weight_by_confidence(y_train, label_issue_scores)
        model_weighted = model_builder(**builder_kwargs)
        try:
            # Pipeline: need to pass sample_weight to the classifier step
            if hasattr(model_weighted, "named_steps"):
                clf_name = list(model_weighted.named_steps.keys())[-1]
                model_weighted.fit(
                    X_train, y_train,
                    **{f"{clf_name}__sample_weight": weights}
                )
            else:
                model_weighted.fit(X_train, y_train, sample_weight=weights)

            y_pred_weighted = model_weighted.predict(X_val)
            if evaluate_fn:
                results["weighted"] = evaluate_fn(y_val, y_pred_weighted)
            results["weighted_labels_used"] = "observed, confidence-weighted"
        except (TypeError, ValueError) as e:
            logger.warning("Weighted training not supported: %s", e)
            results["weighted_error"] = str(e)

    # 4. Corrected-label training
    if corrections is not None and len(corrections) > 0:
        logger.info("Running corrected-label training...")
        model_corrected = model_builder(**builder_kwargs)
        model_corrected, y_corrected, corr_report = train_with_corrected_labels(
            model_corrected, X_train, y_train, corrections
        )
        y_pred_corr = model_corrected.predict(X_val)
        if evaluate_fn:
            results["corrected"] = evaluate_fn(y_val, y_pred_corr)
        results["corrected_report"] = corr_report
        results["corrected_labels_used"] = "observed + reviewed corrections"

    return results
