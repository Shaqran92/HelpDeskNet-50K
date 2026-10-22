"""
Statistical analysis utilities.

Provides general-purpose statistical functions for the HelpDeskNet-50K
project, including hypothesis tests, effect sizes, and visualization
helpers.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def chi_square_independence(
    contingency_table: pd.DataFrame,
) -> dict:
    """Perform chi-square test of independence on a contingency table.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Contingency table (e.g., class × tier cross-tabulation).

    Returns
    -------
    dict
        chi2_statistic, p_value, dof, expected_frequencies,
        cramers_v, interpretation.
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Cramér's V effect size
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0

    interpretation = _interpret_cramers_v(cramers_v)

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "expected_frequencies": pd.DataFrame(
            expected,
            index=contingency_table.index,
            columns=contingency_table.columns,
        ),
        "cramers_v": float(cramers_v),
        "effect_size_interpretation": interpretation,
        "n_total": int(n),
    }


def _interpret_cramers_v(v: float) -> str:
    """Interpret Cramér's V effect size."""
    if v < 0.1:
        return "negligible"
    elif v < 0.3:
        return "small"
    elif v < 0.5:
        return "medium"
    else:
        return "large"


def proportion_test(
    successes_1: int,
    n_1: int,
    successes_2: int,
    n_2: int,
    alternative: str = "two-sided",
) -> dict:
    """Two-proportion z-test.

    Parameters
    ----------
    successes_1 : int
        Number of successes in group 1.
    n_1 : int
        Total observations in group 1.
    successes_2 : int
        Number of successes in group 2.
    n_2 : int
        Total observations in group 2.
    alternative : str
        'two-sided', 'larger', or 'smaller'.

    Returns
    -------
    dict
        z_statistic, p_value, prop_1, prop_2, difference,
        pooled_proportion.
    """
    p1 = successes_1 / n_1 if n_1 > 0 else 0.0
    p2 = successes_2 / n_2 if n_2 > 0 else 0.0
    p_pool = (successes_1 + successes_2) / (n_1 + n_2) if (n_1 + n_2) > 0 else 0.0

    if p_pool == 0 or p_pool == 1 or n_1 == 0 or n_2 == 0:
        return {
            "z_statistic": None,
            "p_value": None,
            "prop_1": float(p1),
            "prop_2": float(p2),
            "difference": float(p2 - p1),
            "pooled_proportion": float(p_pool),
            "note": "Cannot compute z-test: degenerate proportions or zero sample size.",
        }

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_1 + 1 / n_2))
    z = (p1 - p2) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "larger":
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == "smaller":
        p_value = stats.norm.cdf(z)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    return {
        "z_statistic": float(z),
        "p_value": float(p_value),
        "prop_1": float(p1),
        "prop_2": float(p2),
        "difference": float(p2 - p1),
        "pooled_proportion": float(p_pool),
        "alternative": alternative,
    }


def bootstrap_metric(
    values: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for an arbitrary metric.

    Parameters
    ----------
    values : np.ndarray
        Data values to bootstrap over.
    metric_fn : callable
        Function that takes an array and returns a scalar.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence_level : float
        Confidence level for the interval.
    seed : int
        Random seed.

    Returns
    -------
    dict
        point_estimate, ci_lower, ci_upper, se, n_bootstrap, seed.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    point_estimate = metric_fn(values)

    bootstrap_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_estimates[i] = metric_fn(sample)

    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    se = float(np.std(bootstrap_estimates))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "confidence_level": confidence_level,
    }


def compute_class_balance_metrics(
    labels: pd.Series,
    classes: Optional[list[str]] = None,
) -> dict:
    """Compute class balance and imbalance metrics.

    Parameters
    ----------
    labels : pd.Series
        Label series.
    classes : list[str], optional
        Explicit class list. If None, derived from data.

    Returns
    -------
    dict
        class_counts, class_proportions, imbalance_ratio,
        entropy, max_entropy, normalized_entropy.
    """
    counts = labels.value_counts()
    if classes is not None:
        # Ensure all classes are represented
        for cls in classes:
            if cls not in counts.index:
                counts[cls] = 0
        counts = counts[classes]

    n = counts.sum()
    proportions = counts / n if n > 0 else counts * 0

    # Imbalance ratio (max / min)
    non_zero = counts[counts > 0]
    imbalance_ratio = (
        float(non_zero.max() / non_zero.min())
        if len(non_zero) > 1 else float("inf")
    )

    # Shannon entropy
    p = proportions[proportions > 0].values
    entropy = float(-np.sum(p * np.log2(p)))
    max_entropy = float(np.log2(len(counts))) if len(counts) > 0 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "class_counts": counts.to_dict(),
        "class_proportions": proportions.to_dict(),
        "n_total": int(n),
        "n_classes": int(len(counts)),
        "imbalance_ratio": imbalance_ratio,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions.

    Parameters
    ----------
    p1, p2 : float
        Proportions to compare.

    Returns
    -------
    float
        Cohen's h value.
    """
    return float(2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))


def interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size magnitude.

    Parameters
    ----------
    h : float
        Cohen's h value.

    Returns
    -------
    str
        Interpretation string.
    """
    abs_h = abs(h)
    if abs_h < 0.2:
        return "small"
    elif abs_h < 0.5:
        return "medium"
    else:
        return "large"


def summarize_results(
    noise_estimate: dict,
    class_noise: pd.DataFrame,
    tier_comparison: Optional[dict] = None,
    transition_matrix: Optional[pd.DataFrame] = None,
) -> dict:
    """Combine analysis results into a summary structure.

    Parameters
    ----------
    noise_estimate : dict
        Overall noise rate estimate.
    class_noise : pd.DataFrame
        Per-class noise rates.
    tier_comparison : dict, optional
        Tier comparison results.
    transition_matrix : pd.DataFrame, optional
        Label transition matrix.

    Returns
    -------
    dict
        Combined summary for reporting and dashboard display.
    """
    summary = {
        "overall_noise": noise_estimate,
        "class_noise": class_noise.to_dict(orient="records"),
        "most_noisy_class": (
            class_noise.loc[class_noise["noise_rate"].idxmax(), "class"]
            if not class_noise["noise_rate"].isna().all()
            else None
        ),
        "least_noisy_class": (
            class_noise.loc[class_noise["noise_rate"].idxmin(), "class"]
            if not class_noise["noise_rate"].isna().all()
            else None
        ),
    }

    if tier_comparison is not None:
        summary["tier_comparison"] = tier_comparison

    if transition_matrix is not None:
        # Find most confused pair (off-diagonal)
        tm_vals = transition_matrix.to_numpy(copy=True)
        np.fill_diagonal(tm_vals, 0)
        max_idx = np.unravel_index(tm_vals.argmax(), tm_vals.shape)
        most_confused = (
            transition_matrix.index[max_idx[0]],
            transition_matrix.columns[max_idx[1]],
            float(tm_vals[max_idx]),
        )
        summary["most_confused_pair"] = {
            "from": most_confused[0],
            "to": most_confused[1],
            "rate": most_confused[2],
        }
        summary["transition_matrix"] = transition_matrix.to_dict()

    summary["provenance_note"] = (
        "All noise-rate estimates use source/proxy labels as reference. "
        "These are not independently verified ground-truth labels. "
        "Results describe the simulation, not necessarily real annotation errors."
    )

    return summary
