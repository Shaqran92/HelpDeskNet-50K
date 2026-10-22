"""
Noise metrics and label-quality analysis.

Provides functions for estimating label-noise rates, computing
confidence intervals, building transition matrices, and comparing
worker tiers. All functions clearly distinguish simulated vs.
independently verified results.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def estimate_noise_rate(
    reference_labels: pd.Series,
    observed_labels: pd.Series,
    confidence_level: float = 0.95,
) -> dict:
    """Estimate overall noise rate with Wilson confidence interval.

    Parameters
    ----------
    reference_labels : pd.Series
        Ground-truth / proxy-reference labels.
    observed_labels : pd.Series
        Observed (possibly noisy) labels.
    confidence_level : float
        Confidence level for the interval (default 0.95).

    Returns
    -------
    dict
        Keys: n_total, n_errors, noise_rate, ci_lower, ci_upper,
        ci_method, confidence_level, label_provenance.
    """
    if len(reference_labels) != len(observed_labels):
        raise ValueError(
            f"Length mismatch: reference={len(reference_labels)}, "
            f"observed={len(observed_labels)}"
        )

    n = len(reference_labels)
    if n == 0:
        raise ValueError("Cannot estimate noise rate from empty arrays.")

    errors = (reference_labels.values != observed_labels.values).sum()
    rate = errors / n

    # Wilson score interval for binomial proportion
    ci_lower, ci_upper = _wilson_interval(errors, n, confidence_level)

    result = {
        "n_total": int(n),
        "n_errors": int(errors),
        "noise_rate": float(rate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_method": "wilson",
        "confidence_level": confidence_level,
        "label_provenance": "proxy_reference (source labels used as reference)",
    }
    logger.info(
        "Noise rate: %.4f [%.4f, %.4f] (n=%d, errors=%d)",
        rate, ci_lower, ci_upper, n, errors,
    )
    return result


def _wilson_interval(
    successes: int, n: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of 'successes' (here: errors found).
    n : int
        Total sample size.
    confidence_level : float
        Confidence level.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound).
    """
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return (max(0.0, centre - margin), min(1.0, centre + margin))


def estimate_class_noise_rates(
    reference_labels: pd.Series,
    observed_labels: pd.Series,
    classes: list[str],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate noise rate per class with confidence intervals.

    Parameters
    ----------
    reference_labels : pd.Series
        Reference labels.
    observed_labels : pd.Series
        Observed labels.
    classes : list[str]
        List of class names.
    confidence_level : float
        Confidence level for intervals.

    Returns
    -------
    pd.DataFrame
        Per-class noise rate estimates with CI.
    """
    rows = []
    for cls in classes:
        mask = reference_labels == cls
        n_class = mask.sum()
        if n_class == 0:
            rows.append({
                "class": cls,
                "n_total": 0,
                "n_errors": 0,
                "noise_rate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            })
            continue

        ref_cls = reference_labels[mask]
        obs_cls = observed_labels[mask]
        errors = (ref_cls.values != obs_cls.values).sum()
        rate = errors / n_class
        ci_lower, ci_upper = _wilson_interval(errors, n_class, confidence_level)

        rows.append({
            "class": cls,
            "n_total": int(n_class),
            "n_errors": int(errors),
            "noise_rate": float(rate),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        })

    return pd.DataFrame(rows)


def build_transition_matrix(
    reference_labels: pd.Series,
    observed_labels: pd.Series,
    classes: list[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """Build label transition (confusion) matrix.

    Rows = reference (true) labels, Columns = observed (noisy) labels.

    Parameters
    ----------
    reference_labels : pd.Series
        Reference labels.
    observed_labels : pd.Series
        Observed labels.
    classes : list[str]
        Ordered class names.
    normalize : bool
        If True, normalize rows to sum to 1.

    Returns
    -------
    pd.DataFrame
        Transition matrix with classes as row/column labels.
    """
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=float)

    for ref, obs in zip(reference_labels, observed_labels):
        if ref in classes and obs in classes:
            matrix.loc[ref, obs] += 1

    if normalize:
        row_sums = matrix.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # avoid division by zero
        matrix = matrix.div(row_sums, axis=0)

    return matrix


def compare_worker_tiers(
    df: pd.DataFrame,
    tier_col: str = "worker_tier",
    reference_col: str = "reference_label",
    observed_col: str = "observed_label",
    confidence_level: float = 0.95,
) -> dict:
    """Compare error rates between worker tiers A and B.

    Computes point estimates, confidence intervals, rate difference,
    effect size (Cohen's h), and a two-proportion z-test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with tier, reference, and observed label columns.
    tier_col : str
        Column indicating worker tier.
    reference_col : str
        Column with reference labels.
    observed_col : str
        Column with observed labels.
    confidence_level : float
        Confidence level for intervals.

    Returns
    -------
    dict
        Comparison statistics.
    """
    tiers = df[tier_col].unique()
    tier_stats = {}

    for tier in sorted(tiers):
        mask = df[tier_col] == tier
        ref = df.loc[mask, reference_col]
        obs = df.loc[mask, observed_col]
        n = len(ref)
        errors = (ref.values != obs.values).sum()
        rate = errors / n if n > 0 else 0.0
        ci_low, ci_high = _wilson_interval(errors, n, confidence_level)
        tier_stats[tier] = {
            "n_total": int(n),
            "n_errors": int(errors),
            "error_rate": float(rate),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
        }

    result = {"tier_stats": tier_stats, "confidence_level": confidence_level}

    # If exactly two tiers, compute comparison statistics
    tier_keys = sorted(tier_stats.keys())
    if len(tier_keys) == 2:
        t_a, t_b = tier_keys[0], tier_keys[1]
        n_a = tier_stats[t_a]["n_total"]
        n_b = tier_stats[t_b]["n_total"]
        e_a = tier_stats[t_a]["n_errors"]
        e_b = tier_stats[t_b]["n_errors"]
        p_a = tier_stats[t_a]["error_rate"]
        p_b = tier_stats[t_b]["error_rate"]

        # Rate difference
        diff = p_b - p_a
        result["rate_difference"] = float(diff)

        # Two-proportion z-test
        if n_a > 0 and n_b > 0:
            p_pooled = (e_a + e_b) / (n_a + n_b)
            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
                z_stat = diff / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                result["z_statistic"] = float(z_stat)
                result["p_value"] = float(p_value)
            else:
                result["z_statistic"] = None
                result["p_value"] = None

            # Cohen's h effect size
            h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
            result["cohens_h"] = float(h)

            # CI for rate difference (Newcombe method approx)
            se_diff = np.sqrt(
                p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b
            )
            z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            result["diff_ci_lower"] = float(diff - z_crit * se_diff)
            result["diff_ci_upper"] = float(diff + z_crit * se_diff)

    result["note"] = (
        "These results are from simulated worker tiers unless "
        "independently verified annotations are available."
    )

    return result


def compute_audit_sample_stats(
    df: pd.DataFrame,
    sample_size: int = 400,
    reference_col: str = "reference_label",
    observed_col: str = "observed_label",
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict:
    """Draw an audit sample and compute noise-rate estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with labels.
    sample_size : int
        Number of records to sample.
    reference_col : str
        Reference label column.
    observed_col : str
        Observed label column.
    seed : int
        Random seed.
    confidence_level : float
        Confidence level.

    Returns
    -------
    dict
        Audit sample statistics including noise rate and CI.
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    actual_size = min(sample_size, n)

    indices = rng.choice(n, size=actual_size, replace=False)
    sample = df.iloc[indices]

    ref = sample[reference_col]
    obs = sample[observed_col]
    errors = (ref.values != obs.values).sum()
    rate = errors / actual_size
    ci_low, ci_high = _wilson_interval(errors, actual_size, confidence_level)

    return {
        "audit_sample_size": int(actual_size),
        "population_size": int(n),
        "n_errors_found": int(errors),
        "estimated_noise_rate": float(rate),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "ci_method": "wilson",
        "confidence_level": confidence_level,
        "seed": seed,
        "representativeness_warning": (
            "This estimate assumes the audit sample is representative "
            "of the full dataset. Stratified or clustered sampling "
            "may be needed for more reliable estimates."
        ),
    }


def noise_rate_by_worker(
    df: pd.DataFrame,
    worker_col: str = "worker_id",
    tier_col: str = "worker_tier",
    reference_col: str = "reference_label",
    observed_col: str = "observed_label",
) -> pd.DataFrame:
    """Compute noise rate per worker.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with worker and label columns.
    worker_col : str
        Worker ID column.
    tier_col : str
        Worker tier column.
    reference_col : str
        Reference label column.
    observed_col : str
        Observed label column.

    Returns
    -------
    pd.DataFrame
        Per-worker noise statistics.
    """
    rows = []
    for worker_id, group in df.groupby(worker_col):
        n = len(group)
        errors = (
            group[reference_col].values != group[observed_col].values
        ).sum()
        rate = errors / n if n > 0 else 0.0
        tier = group[tier_col].iloc[0] if tier_col in group.columns else "unknown"
        ci_low, ci_high = _wilson_interval(errors, n)
        rows.append({
            "worker_id": worker_id,
            "worker_tier": tier,
            "n_annotations": int(n),
            "n_errors": int(errors),
            "error_rate": float(rate),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
        })

    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)
