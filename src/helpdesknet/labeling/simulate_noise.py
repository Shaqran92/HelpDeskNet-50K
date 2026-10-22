"""Label noise simulation module for HelpDeskNet-50K.

Provides controlled, reproducible simulation of annotation errors on customer-support
tickets across three distinct noise mechanisms:
    1. Symmetric (uniform) noise: labels flipped to any alternative class with equal probability.
    2. Class-dependent (systematic) noise: labels flipped according to domain-specific
       confusion pairs (e.g., Billing <-> Refund, Technical <-> Account Access).
    3. Worker-dependent noise: simulated annotator tiers (Tier A high accuracy,
       Tier B crowd/novice) with distinct error rates.

Core principles:
- Ground-truth / proxy-reference labels are preserved in a separate column and NEVER overwritten.
- All noise injection is deterministic and reproducible via numpy.random.RandomState.
- Injected error status (is_injected_error) and worker metadata (worker_id, worker_tier)
  are explicitly tracked.
- Test-set leakage is prevented: noisy datasets clearly distinguish proxy reference
  from observed labels.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard target classes for HelpDeskNet-50K
TARGET_CLASSES: list[str] = [
    "Billing",
    "Refund",
    "Technical",
    "Account Access",
    "Shipping",
    "Other",
]

# Default domain confusion pairs (bidirectional systematic confusion)
DEFAULT_CONFUSION_PAIRS: dict[str, str] = {
    "Billing": "Refund",
    "Refund": "Billing",
    "Technical": "Account Access",
    "Account Access": "Technical",
    "Shipping": "Other",
    "Other": "Shipping",
}


class NoiseSimulator:
    """Simulates controlled annotation noise on ticket labels.

    Attributes:
        classes: List of valid class labels.
        seed: Random seed for deterministic simulation.
    """

    def __init__(
        self,
        classes: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize NoiseSimulator with target classes and random seed.

        Args:
            classes: List of class names. If None, defaults to TARGET_CLASSES.
            seed: Integer random seed (default: 42).

        Raises:
            TypeError: If classes is not a list or seed is not an integer.
            ValueError: If classes is empty or contains duplicate values.
        """
        if classes is None:
            classes = list(TARGET_CLASSES)
        elif not isinstance(classes, (list, tuple)):
            raise TypeError(f"classes must be a list or tuple of strings, got {type(classes).__name__}")

        if not classes:
            raise ValueError("classes list cannot be empty.")

        if len(classes) != len(set(classes)):
            raise ValueError(f"Duplicate classes found: {classes}")

        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

        self.classes = list(classes)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

        logger.info(
            "NoiseSimulator initialized with %d classes (seed=%d)",
            len(self.classes),
            self.seed,
        )

    def _get_rng(self, seed: int | None = None) -> np.random.RandomState:
        """Obtain a numpy RandomState instance.

        Args:
            seed: Optional integer seed override. If None, uses configured seed.

        Returns:
            np.random.RandomState: Deterministic random state.
        """
        if seed is not None:
            return np.random.RandomState(int(seed))
        return np.random.RandomState(self.seed)

    def _validate_input_labels(self, labels: pd.Series) -> None:
        """Validate input label Series before simulation.

        Args:
            labels: pd.Series of class labels.

        Raises:
            TypeError: If labels is not a pandas Series.
            ValueError: If labels is empty or contains unknown/missing classes.
        """
        if not isinstance(labels, pd.Series):
            raise TypeError(f"labels must be a pandas Series, got {type(labels).__name__}")

        if labels.empty:
            raise ValueError("Input labels Series is empty.")

        if labels.isna().any():
            n_nan = int(labels.isna().sum())
            raise ValueError(f"Input labels contain {n_nan} missing (NaN) values.")

        unknown = set(labels.unique()) - set(self.classes)
        if unknown:
            raise ValueError(
                f"Input labels contain unrecognized classes: {sorted(unknown)}. "
                f"Valid classes: {self.classes}"
            )

    def _validate_rate(self, rate: float) -> None:
        """Validate noise rate parameter.

        Args:
            rate: Noise corruption rate.

        Raises:
            TypeError: If rate is not a float or int.
            ValueError: If rate is outside [0.0, 1.0].
        """
        if not isinstance(rate, (float, int, np.floating, np.integer)):
            raise TypeError(f"rate must be a float or int, got {type(rate).__name__}")

        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"rate must be in range [0.0, 1.0], got {rate}")

    def symmetric_noise(
        self,
        labels: pd.Series,
        rate: float,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Inject symmetric (uniform random) noise into a label Series.

        For each example, with probability `rate`, the label is replaced by one
        of the other (K - 1) classes chosen uniformly at random.

        Args:
            labels: Original reference labels as a pd.Series.
            rate: Probability of corrupting each label (0.0 to 1.0).
            seed: Optional random seed override for determinism.

        Returns:
            pd.DataFrame: Contains columns:
                - `observed_label`: Noisy/observed label.
                - `is_injected_error`: Boolean flag indicating whether label was flipped.
                - `reference_label`: Unmodified original reference label.
        """
        self._validate_input_labels(labels)
        self._validate_rate(rate)

        rng = self._get_rng(seed)
        n = len(labels)
        ref_values = labels.to_numpy()

        if rate == 0.0 or len(self.classes) <= 1:
            logger.info("Symmetric noise called with rate=%.4f (0 errors injected)", rate)
            return pd.DataFrame(
                {
                    "observed_label": labels.copy(),
                    "is_injected_error": pd.Series(False, index=labels.index, dtype=bool),
                    "reference_label": labels.copy(),
                },
                index=labels.index,
            )

        # Draw binary error flags
        rand_uniform = rng.rand(n)
        error_mask = rand_uniform < rate

        # Precompute candidate replacement classes for each class
        replacement_candidates: dict[str, list[str]] = {
            c: [other for other in self.classes if other != c]
            for c in self.classes
        }

        observed_values = ref_values.copy()
        n_errors = int(error_mask.sum())

        if n_errors > 0:
            error_indices = np.where(error_mask)[0]
            # Random uniform index among the (K - 1) alternatives
            choice_indices = rng.randint(0, len(self.classes) - 1, size=n_errors)

            for i, idx in enumerate(error_indices):
                curr_class = ref_values[idx]
                candidates = replacement_candidates[curr_class]
                observed_values[idx] = candidates[choice_indices[i]]

        result_df = pd.DataFrame(
            {
                "observed_label": observed_values,
                "is_injected_error": error_mask,
                "reference_label": labels.copy(),
            },
            index=labels.index,
        )

        logger.info(
            "Symmetric noise applied: target_rate=%.4f, actual_rate=%.4f (n=%d, errors=%d)",
            rate,
            n_errors / n,
            n,
            n_errors,
        )
        return result_df

    def class_dependent_noise(
        self,
        labels: pd.Series,
        rate: float,
        confusion_pairs: dict[str, str | list[str]] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Inject class-dependent (systematic) label noise based on confusion pairs.

        For each example with reference label y, with probability `rate`, the label
        is flipped to the configured confused class. If no pair is configured for y,
        the label remains unchanged.

        Args:
            labels: Original reference labels as a pd.Series.
            rate: Corruption probability for classes with defined confusion pairs.
            confusion_pairs: Dict mapping source class -> target class (or list of target
                classes). If None, defaults to DEFAULT_CONFUSION_PAIRS:
                - Billing <-> Refund
                - Technical <-> Account Access
                - Shipping <-> Other
            seed: Optional random seed override for determinism.

        Returns:
            pd.DataFrame: Contains columns:
                - `observed_label`: Noisy/observed label.
                - `is_injected_error`: Boolean flag indicating whether label was flipped.
                - `reference_label`: Unmodified original reference label.

        Raises:
            ValueError: If confusion_pairs contains invalid classes.
        """
        self._validate_input_labels(labels)
        self._validate_rate(rate)

        if confusion_pairs is None:
            pairs = dict(DEFAULT_CONFUSION_PAIRS)
        else:
            pairs = dict(confusion_pairs)

        # Validate confusion pairs
        for src, dest in pairs.items():
            if src not in self.classes:
                raise ValueError(f"Confusion source class '{src}' not in valid classes: {self.classes}")
            dest_list = [dest] if isinstance(dest, str) else list(dest)
            for d in dest_list:
                if d not in self.classes:
                    raise ValueError(f"Confusion target class '{d}' not in valid classes: {self.classes}")

        rng = self._get_rng(seed)
        n = len(labels)
        ref_values = labels.to_numpy()

        if rate == 0.0:
            logger.info("Class-dependent noise called with rate=0.0 (0 errors injected)")
            return pd.DataFrame(
                {
                    "observed_label": labels.copy(),
                    "is_injected_error": pd.Series(False, index=labels.index, dtype=bool),
                    "reference_label": labels.copy(),
                },
                index=labels.index,
            )

        rand_uniform = rng.rand(n)
        error_mask = np.zeros(n, dtype=bool)
        observed_values = ref_values.copy()

        for i in range(n):
            current_label = ref_values[i]
            if current_label in pairs and rand_uniform[i] < rate:
                target = pairs[current_label]
                if isinstance(target, (list, tuple)):
                    chosen = rng.choice(target)
                else:
                    chosen = target

                if chosen != current_label:
                    observed_values[i] = chosen
                    error_mask[i] = True

        result_df = pd.DataFrame(
            {
                "observed_label": observed_values,
                "is_injected_error": error_mask,
                "reference_label": labels.copy(),
            },
            index=labels.index,
        )

        n_errors = int(error_mask.sum())
        logger.info(
            "Class-dependent noise applied: target_rate=%.4f, actual_rate=%.4f (n=%d, errors=%d)",
            rate,
            n_errors / n,
            n,
            n_errors,
        )
        return result_df

    def worker_dependent_noise(
        self,
        labels: pd.Series,
        rate: float,
        n_workers_a: int = 5,
        n_workers_b: int = 5,
        tier_a_multiplier: float = 0.5,
        tier_b_multiplier: float = 1.5,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate annotator-level noise with Tier A and Tier B worker profiles.

        Simulates realistic crowd labeling where annotators have varying error rates:
        - Tier A: Expert annotators with lower error rate (rate * tier_a_multiplier).
        - Tier B: Crowd / novice annotators with higher error rate (rate * tier_b_multiplier).

        Tickets are assigned to simulated workers, each injecting errors according to
        their tier-specific error probability.

        Args:
            labels: Original reference labels as a pd.Series.
            rate: Base error rate (0.0 to 1.0).
            n_workers_a: Number of Tier A annotators (default: 5).
            n_workers_b: Number of Tier B annotators (default: 5).
            tier_a_multiplier: Error rate multiplier for Tier A (default: 0.5).
            tier_b_multiplier: Error rate multiplier for Tier B (default: 1.5).
            seed: Optional random seed override for determinism.

        Returns:
            pd.DataFrame: Contains columns:
                - `observed_label`: Label provided by the assigned worker.
                - `is_injected_error`: Boolean flag indicating whether label was flipped.
                - `reference_label`: Unmodified original reference label.
                - `worker_id`: Identifier of assigned worker (e.g., 'worker_A_01').
                - `worker_tier`: Assigned tier ('Tier A' or 'Tier B').

        Raises:
            ValueError: If worker counts are <= 0 or multipliers are negative.
        """
        self._validate_input_labels(labels)
        self._validate_rate(rate)

        if n_workers_a < 1 or n_workers_b < 1:
            raise ValueError(
                f"Worker counts must be >= 1 (got n_workers_a={n_workers_a}, n_workers_b={n_workers_b})"
            )
        if tier_a_multiplier < 0.0 or tier_b_multiplier < 0.0:
            raise ValueError("Worker tier multipliers must be non-negative.")

        rng = self._get_rng(seed)
        n = len(labels)
        ref_values = labels.to_numpy()

        # Compute tier error rates clamped to [0.0, 1.0]
        tier_a_rate = min(1.0, max(0.0, float(rate * tier_a_multiplier)))
        tier_b_rate = min(1.0, max(0.0, float(rate * tier_b_multiplier)))

        # Define workers and worker metadata
        workers_a = [f"worker_A_{i+1:02d}" for i in range(n_workers_a)]
        workers_b = [f"worker_B_{i+1:02d}" for i in range(n_workers_b)]
        all_workers = workers_a + workers_b
        total_workers = len(all_workers)

        worker_tier_map = {w: "Tier A" for w in workers_a}
        worker_tier_map.update({w: "Tier B" for w in workers_b})

        worker_rate_map = {w: tier_a_rate for w in workers_a}
        worker_rate_map.update({w: tier_b_rate for w in workers_b})

        # Assign tickets to workers uniformly at random
        worker_indices = rng.randint(0, total_workers, size=n)
        assigned_worker_ids = [all_workers[idx] for idx in worker_indices]
        assigned_tiers = [worker_tier_map[w] for w in assigned_worker_ids]
        assigned_rates = np.array([worker_rate_map[w] for w in assigned_worker_ids])

        # Generate error mask
        rand_uniform = rng.rand(n)
        error_mask = rand_uniform < assigned_rates

        # Replacement choices
        replacement_candidates: dict[str, list[str]] = {
            c: [other for other in self.classes if other != c]
            for c in self.classes
        }

        observed_values = ref_values.copy()
        n_errors = int(error_mask.sum())

        if n_errors > 0 and len(self.classes) > 1:
            error_indices = np.where(error_mask)[0]
            choice_indices = rng.randint(0, len(self.classes) - 1, size=n_errors)

            for i, idx in enumerate(error_indices):
                curr_class = ref_values[idx]
                candidates = replacement_candidates[curr_class]
                observed_values[idx] = candidates[choice_indices[i]]

        result_df = pd.DataFrame(
            {
                "observed_label": observed_values,
                "is_injected_error": error_mask,
                "reference_label": labels.copy(),
                "worker_id": assigned_worker_ids,
                "worker_tier": assigned_tiers,
            },
            index=labels.index,
        )

        logger.info(
            "Worker-dependent noise applied: base_rate=%.4f (Tier A: %.4f, Tier B: %.4f), "
            "actual_rate=%.4f (errors=%d/%d)",
            rate,
            tier_a_rate,
            tier_b_rate,
            n_errors / n,
            n_errors,
            n,
        )
        return result_df

    def apply_noise(
        self,
        labels: pd.Series,
        mechanism: str,
        rate: float,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Dispatch noise injection to the specified mechanism.

        Args:
            labels: Original reference labels as a pd.Series.
            mechanism: One of 'symmetric', 'class_dependent', 'worker_dependent'.
            rate: Noise corruption rate (0.0 to 1.0).
            **kwargs: Additional mechanism-specific arguments:
                - for 'class_dependent': confusion_pairs, seed
                - for 'worker_dependent': n_workers_a, n_workers_b, tier_a_multiplier,
                  tier_b_multiplier, seed
                - for 'symmetric': seed

        Returns:
            pd.DataFrame: Simulation results containing observed_label,
                is_injected_error, reference_label, and optional worker metadata.

        Raises:
            ValueError: If mechanism is unrecognized.
        """
        mech_clean = str(mechanism).strip().lower()

        if mech_clean == "symmetric":
            return self.symmetric_noise(labels=labels, rate=rate, **kwargs)
        elif mech_clean == "class_dependent":
            return self.class_dependent_noise(labels=labels, rate=rate, **kwargs)
        elif mech_clean == "worker_dependent":
            return self.worker_dependent_noise(labels=labels, rate=rate, **kwargs)
        else:
            raise ValueError(
                f"Unsupported noise mechanism '{mechanism}'. "
                f"Supported mechanisms: ['symmetric', 'class_dependent', 'worker_dependent']"
            )


def build_transition_matrix(
    reference_labels: pd.Series,
    observed_labels: pd.Series,
    classes: list[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a label transition (confusion) matrix from reference and observed labels.

    Rows correspond to reference (clean) labels, columns correspond to observed
    (noisy) labels:
        T_ij = P(observed = j | reference = i)

    Args:
        reference_labels: Reference / ground-truth labels.
        observed_labels: Observed / simulated noisy labels.
        classes: Ordered list of class labels. If None, defaults to TARGET_CLASSES.
        normalize: If True, normalizes each row to sum to 1.0 (probabilities).
            If False, returns raw transition counts.

    Returns:
        pd.DataFrame: K x K matrix indexed by reference_label and columned by observed_label.

    Raises:
        TypeError: If inputs are not pandas Series.
        ValueError: If lengths do not match or classes list is empty.
    """
    if not isinstance(reference_labels, pd.Series):
        reference_labels = pd.Series(reference_labels)
    if not isinstance(observed_labels, pd.Series):
        observed_labels = pd.Series(observed_labels)

    if len(reference_labels) != len(observed_labels):
        raise ValueError(
            f"Length mismatch: reference={len(reference_labels)}, "
            f"observed={len(observed_labels)}"
        )

    if classes is None:
        classes = list(TARGET_CLASSES)
    else:
        classes = list(classes)

    if not classes:
        raise ValueError("classes list cannot be empty.")

    matrix = pd.DataFrame(0.0, index=classes, columns=classes)
    matrix.index.name = "reference_label"
    matrix.columns.name = "observed_label"

    for ref, obs in zip(reference_labels, observed_labels):
        if ref in classes and obs in classes:
            matrix.loc[ref, obs] += 1.0

    if normalize:
        row_sums = matrix.sum(axis=1)
        # Avoid division by zero for unrepresented classes
        row_sums = row_sums.replace(0.0, 1.0)
        matrix = matrix.div(row_sums, axis=0)

    return matrix


def validate_noise_simulation(
    result: pd.DataFrame,
    expected_rate: float | None = None,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Validate the integrity and statistical properties of a noise simulation output.

    Verifies:
    1. Schema contains `reference_label`, `observed_label`, and `is_injected_error`.
    2. No null or NaN values in critical columns.
    3. Error consistency: whenever is_injected_error is True, observed != reference;
       whenever is_injected_error is False, observed == reference.
    4. Empirical error rate is within `tolerance` of `expected_rate` (if provided).
    5. Tier error rates if worker columns are present.

    Args:
        result: Simulation DataFrame to validate.
        expected_rate: Optional expected base noise rate to verify against.
        tolerance: Maximum acceptable absolute difference between empirical and
            expected error rate (default: 0.05).

    Returns:
        dict[str, Any]: Validation report with validation status, metrics, and checks.

    Raises:
        TypeError: If result is not a pandas DataFrame.
        ValueError: If required columns are completely missing.
    """
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"result must be a pandas DataFrame, got {type(result).__name__}")

    required_cols = {"reference_label", "observed_label", "is_injected_error"}
    missing = required_cols - set(result.columns)
    if missing:
        raise ValueError(f"Simulation result missing required columns: {sorted(missing)}")

    n_total = len(result)
    checks_passed: dict[str, bool] = {}
    warnings: list[str] = []

    if n_total == 0:
        return {
            "is_valid": False,
            "total_records": 0,
            "injected_errors_count": 0,
            "empirical_error_rate": 0.0,
            "per_class_error_rate": {},
            "per_class_counts": {},
            "worker_tier_error_rates": None,
            "checks_passed": {"non_empty": False},
            "warnings": ["DataFrame is empty."],
        }

    # 1. No nulls check
    no_nulls = not result[list(required_cols)].isna().any().any()
    checks_passed["no_nulls"] = bool(no_nulls)
    if not no_nulls:
        warnings.append("Null values detected in label simulation columns.")

    # 2. Flag consistency check
    is_err = result["is_injected_error"].astype(bool)
    ref = result["reference_label"].astype(str)
    obs = result["observed_label"].astype(str)

    mismatch = ref != obs
    flag_consistent = bool((is_err == mismatch).all())
    checks_passed["flag_consistency"] = flag_consistent
    if not flag_consistent:
        inconsistent_count = int((is_err != mismatch).sum())
        warnings.append(f"Flag inconsistency: {inconsistent_count} records have mismatch != is_injected_error.")

    # 3. Empirical rates
    n_errors = int(is_err.sum())
    empirical_rate = round(float(n_errors / n_total), 4)

    # 4. Per-class error rates
    per_class_rates: dict[str, float] = {}
    per_class_counts: dict[str, int] = {}
    for cls, group in result.groupby("reference_label"):
        cls_str = str(cls)
        n_cls = len(group)
        n_cls_err = int(group["is_injected_error"].sum())
        per_class_counts[cls_str] = n_cls
        per_class_rates[cls_str] = round(n_cls_err / n_cls, 4) if n_cls > 0 else 0.0

    # 5. Worker tier rates (if present)
    tier_rates: dict[str, float] | None = None
    if "worker_tier" in result.columns:
        tier_rates = {}
        for tier, group in result.groupby("worker_tier"):
            tier_str = str(tier)
            t_total = len(group)
            t_err = int(group["is_injected_error"].sum())
            tier_rates[tier_str] = round(t_err / t_total, 4) if t_total > 0 else 0.0

    # 6. Expected rate validation
    if expected_rate is not None:
        rate_diff = abs(empirical_rate - expected_rate)
        rate_ok = bool(rate_diff <= tolerance)
        checks_passed["expected_rate_within_tolerance"] = rate_ok
        if not rate_ok:
            warnings.append(
                f"Empirical rate ({empirical_rate:.4f}) deviates from expected ({expected_rate:.4f}) "
                f"by {rate_diff:.4f} > tolerance ({tolerance:.4f})."
            )

    is_valid = bool(all(checks_passed.values()))

    report: dict[str, Any] = {
        "is_valid": is_valid,
        "total_records": n_total,
        "injected_errors_count": n_errors,
        "empirical_error_rate": empirical_rate,
        "per_class_error_rate": per_class_rates,
        "per_class_counts": per_class_counts,
        "worker_tier_error_rates": tier_rates,
        "checks_passed": checks_passed,
        "warnings": warnings,
    }

    logger.info(
        "Validation complete: is_valid=%s, empirical_rate=%.4f (n=%d, errors=%d)",
        is_valid,
        empirical_rate,
        n_total,
        n_errors,
    )
    return report
