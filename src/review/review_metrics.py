"""
Review metrics and triage performance evaluation.

Provides quantitative functions for measuring the effectiveness of
expert review allocations, including Precision@K, review yield,
class coverage, and inter-reviewer agreement (Cohen's kappa).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def compute_review_yield(errors_found: int, budget: int) -> float:
    """Compute the review yield (errors detected per review).

    Parameters
    ----------
    errors_found : int
        Number of true mislabeled records detected.
    budget : int
        Total number of reviews allocated.

    Returns
    -------
    float
        Yield ratio in [0, 1].
    """
    if budget <= 0:
        return 0.0
    return float(errors_found / budget)


def compute_precision_at_k(
    ranked_ids: pd.Series,
    true_error_ids: set,
    k_values: Optional[list[int]] = None,
) -> dict:
    """Compute Precision@K and Average Precision for suspicious label rankings.

    Parameters
    ----------
    ranked_ids : pd.Series
        Record IDs ordered from most to least suspicious.
    true_error_ids : set
        Set of record IDs that are genuinely mislabeled.
    k_values : list[int], optional
        List of cutoffs. Defaults to [100, 200, 500, 1000, 2000].

    Returns
    -------
    dict
        Precision and true positive counts for each K, plus Average Precision.
    """
    if k_values is None:
        k_values = [100, 200, 500, 1000, 2000]

    results = {}
    n_ranked = len(ranked_ids)

    for k in k_values:
        k_actual = min(k, n_ranked)
        if k_actual == 0:
            results[f"precision_at_{k}"] = 0.0
            results[f"true_positives_at_{k}"] = 0
            results[f"k_actual_{k}"] = 0
            continue

        top_k = set(ranked_ids.iloc[:k_actual])
        tp = len(top_k & true_error_ids)
        prec = tp / k_actual

        results[f"precision_at_{k}"] = float(prec)
        results[f"true_positives_at_{k}"] = int(tp)
        results[f"k_actual_{k}"] = int(k_actual)

    # Average precision
    hits = 0
    sum_prec = 0.0
    for i, rid in enumerate(ranked_ids):
        if rid in true_error_ids:
            hits += 1
            sum_prec += hits / (i + 1)

    avg_prec = sum_prec / len(true_error_ids) if len(true_error_ids) > 0 else 0.0
    results["average_precision"] = float(avg_prec)
    results["total_true_errors"] = len(true_error_ids)

    return results


def compute_class_coverage(
    reviewed_df: pd.DataFrame,
    class_col: str = "mapped_label",
    all_classes: Optional[list[str]] = None,
) -> dict:
    """Compute the class representation across reviewed tickets.

    Parameters
    ----------
    reviewed_df : pd.DataFrame
        DataFrame of reviewed tickets.
    class_col : str
        Column containing class labels.
    all_classes : list[str], optional
        List of all expected target classes.

    Returns
    -------
    dict
        Unique classes covered, coverage ratio, and per-class counts.
    """
    counts = reviewed_df[class_col].value_counts().to_dict()
    n_covered = len(counts)

    if all_classes is not None and len(all_classes) > 0:
        coverage_ratio = n_covered / len(all_classes)
        missing_classes = [c for c in all_classes if c not in counts]
    else:
        coverage_ratio = 1.0
        missing_classes = []

    return {
        "n_classes_covered": int(n_covered),
        "coverage_ratio": float(coverage_ratio),
        "per_class_counts": counts,
        "missing_classes": missing_classes,
    }


def compute_inter_reviewer_agreement(
    reviewer_a_labels: pd.Series,
    reviewer_b_labels: pd.Series,
) -> dict:
    """Calculate Cohen's kappa and observed agreement between two reviewers.

    Parameters
    ----------
    reviewer_a_labels : pd.Series
        Labels assigned by Reviewer A.
    reviewer_b_labels : pd.Series
        Labels assigned by Reviewer B.

    Returns
    -------
    dict
        cohens_kappa, observed_agreement, interpretation.
    """
    if len(reviewer_a_labels) != len(reviewer_b_labels):
        raise ValueError("Length mismatch between reviewer label series.")

    n = len(reviewer_a_labels)
    if n == 0:
        raise ValueError("Cannot compute agreement on empty series.")

    observed_agreement = float((reviewer_a_labels.values == reviewer_b_labels.values).sum() / n)
    kappa = float(cohen_kappa_score(reviewer_a_labels, reviewer_b_labels))

    if kappa < 0.2:
        interp = "slight"
    elif kappa < 0.4:
        interp = "fair"
    elif kappa < 0.6:
        interp = "moderate"
    elif kappa < 0.8:
        interp = "substantial"
    else:
        interp = "almost_perfect"

    return {
        "n_comparisons": int(n),
        "observed_agreement": observed_agreement,
        "cohens_kappa": kappa,
        "kappa_interpretation": interp,
    }
