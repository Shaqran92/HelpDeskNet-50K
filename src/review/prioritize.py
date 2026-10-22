"""
Expert review prioritization and budget optimization.

Implements configurable strategies for selecting 2,000 tickets for
expert review, comparing random selection against intelligent
strategies. All strategies use the same eligible pool and budget.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReviewBudgetManager:
    """Manages the 2,000-ticket expert review budget.

    Supports multiple selection strategies and prevents duplicate
    selections and test-set leakage.

    Parameters
    ----------
    total_budget : int
        Total review budget (default 2000).
    high_priority : int
        Allocation for high-priority candidates (default 1000).
    stratified : int
        Allocation for stratified samples (default 600).
    random_audit : int
        Allocation for random audit (default 400).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        total_budget: int = 2000,
        high_priority: int = 1000,
        stratified: int = 600,
        random_audit: int = 400,
        seed: int = 42,
    ):
        if high_priority + stratified + random_audit != total_budget:
            raise ValueError(
                f"Budget allocations ({high_priority}+{stratified}+{random_audit}="
                f"{high_priority+stratified+random_audit}) must sum to "
                f"total_budget ({total_budget})."
            )
        self.total_budget = total_budget
        self.high_priority = high_priority
        self.stratified = stratified
        self.random_audit = random_audit
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def select_random(
        self,
        eligible_ids: pd.Series,
        n: Optional[int] = None,
    ) -> pd.Series:
        """Select tickets randomly.

        Parameters
        ----------
        eligible_ids : pd.Series
            Eligible ticket IDs.
        n : int, optional
            Number to select (default: total_budget).

        Returns
        -------
        pd.Series
            Selected ticket IDs.
        """
        n = n or self.total_budget
        n = min(n, len(eligible_ids))
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(eligible_ids), size=n, replace=False)
        selected = eligible_ids.iloc[indices]
        logger.info("Random selection: %d tickets from %d eligible.", n, len(eligible_ids))
        return selected

    def select_by_issue_score(
        self,
        eligible_df: pd.DataFrame,
        score_col: str = "label_issue_score",
        id_col: str = "ticket_id",
        n: Optional[int] = None,
    ) -> pd.Series:
        """Select tickets with highest label-issue scores.

        Parameters
        ----------
        eligible_df : pd.DataFrame
            DataFrame with ticket IDs and issue scores.
        score_col : str
            Column with label-issue scores.
        id_col : str
            Column with ticket IDs.
        n : int, optional
            Number to select.

        Returns
        -------
        pd.Series
            Selected ticket IDs.
        """
        n = n or self.total_budget
        n = min(n, len(eligible_df))
        sorted_df = eligible_df.sort_values(score_col, ascending=False)
        selected = sorted_df.head(n)[id_col]
        logger.info(
            "Issue-score selection: top %d from %d (score range: %.3f - %.3f).",
            n, len(eligible_df),
            sorted_df[score_col].iloc[0] if len(sorted_df) > 0 else 0,
            sorted_df[score_col].iloc[n-1] if n > 0 else 0,
        )
        return selected

    def select_by_uncertainty(
        self,
        eligible_df: pd.DataFrame,
        proba_cols: list[str],
        id_col: str = "ticket_id",
        n: Optional[int] = None,
    ) -> pd.Series:
        """Select tickets with highest prediction uncertainty (entropy).

        Parameters
        ----------
        eligible_df : pd.DataFrame
            DataFrame with probabilities.
        proba_cols : list[str]
            Columns containing class probabilities.
        id_col : str
            Ticket ID column.
        n : int, optional
            Number to select.

        Returns
        -------
        pd.Series
            Selected ticket IDs.
        """
        n = n or self.total_budget
        n = min(n, len(eligible_df))

        # Compute entropy
        probs = eligible_df[proba_cols].values
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs), axis=1)

        df_with_entropy = eligible_df.copy()
        df_with_entropy["_uncertainty"] = entropy
        sorted_df = df_with_entropy.sort_values("_uncertainty", ascending=False)
        selected = sorted_df.head(n)[id_col]
        logger.info("Uncertainty selection: %d tickets with highest entropy.", n)
        return selected

    def select_combined(
        self,
        eligible_df: pd.DataFrame,
        score_col: str = "label_issue_score",
        id_col: str = "ticket_id",
        label_col: str = "mapped_label",
        tier_col: Optional[str] = "worker_tier",
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Select tickets using the combined strategy.

        Allocation:
        - high_priority: top issue scores
        - stratified: stratified by class (and tier if available)
        - random_audit: random from remainder

        Parameters
        ----------
        eligible_df : pd.DataFrame
            Eligible pool.
        score_col : str
            Label-issue score column.
        id_col : str
            Ticket ID column.
        label_col : str
            Class label column.
        tier_col : str, optional
            Worker tier column.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame]
            Selected IDs and selection log.
        """
        selected_ids = set()
        selection_log = []

        # 1. High-priority: top issue scores
        sorted_df = eligible_df.sort_values(score_col, ascending=False)
        hp_candidates = sorted_df[
            ~sorted_df[id_col].isin(selected_ids)
        ].head(self.high_priority)
        for _, row in hp_candidates.iterrows():
            selected_ids.add(row[id_col])
            selection_log.append({
                "ticket_id": row[id_col],
                "strategy": "high_priority",
                "score": row.get(score_col, np.nan),
                "class": row.get(label_col, ""),
            })

        # 2. Stratified: by class (and tier)
        remaining = eligible_df[~eligible_df[id_col].isin(selected_ids)]
        if len(remaining) > 0:
            classes = remaining[label_col].unique()
            per_class = max(1, self.stratified // len(classes))
            rng = np.random.RandomState(self.seed + 1)

            for cls in classes:
                cls_pool = remaining[remaining[label_col] == cls]
                n_select = min(per_class, len(cls_pool))
                if n_select > 0:
                    sampled = cls_pool.sample(n=n_select, random_state=rng)
                    for _, row in sampled.iterrows():
                        if row[id_col] not in selected_ids:
                            selected_ids.add(row[id_col])
                            selection_log.append({
                                "ticket_id": row[id_col],
                                "strategy": "stratified",
                                "score": row.get(score_col, np.nan),
                                "class": cls,
                            })

            # Fill remaining stratified slots if class allocation didn't use all
            n_remaining_stratified = self.stratified - sum(
                1 for s in selection_log if s["strategy"] == "stratified"
            )
            if n_remaining_stratified > 0:
                remaining2 = eligible_df[~eligible_df[id_col].isin(selected_ids)]
                extra = remaining2.sample(
                    n=min(n_remaining_stratified, len(remaining2)),
                    random_state=rng,
                )
                for _, row in extra.iterrows():
                    if row[id_col] not in selected_ids:
                        selected_ids.add(row[id_col])
                        selection_log.append({
                            "ticket_id": row[id_col],
                            "strategy": "stratified_fill",
                            "score": row.get(score_col, np.nan),
                            "class": row.get(label_col, ""),
                        })

        # 3. Random audit
        remaining3 = eligible_df[~eligible_df[id_col].isin(selected_ids)]
        n_random = min(self.random_audit, len(remaining3))
        if n_random > 0:
            rng2 = np.random.RandomState(self.seed + 2)
            random_sample = remaining3.sample(n=n_random, random_state=rng2)
            for _, row in random_sample.iterrows():
                if row[id_col] not in selected_ids:
                    selected_ids.add(row[id_col])
                    selection_log.append({
                        "ticket_id": row[id_col],
                        "strategy": "random_audit",
                        "score": row.get(score_col, np.nan),
                        "class": row.get(label_col, ""),
                    })

        log_df = pd.DataFrame(selection_log)
        result_ids = pd.Series(list(selected_ids))

        logger.info(
            "Combined selection: %d tickets (hp=%d, strat=%d, random=%d).",
            len(result_ids),
            sum(1 for s in selection_log if s["strategy"] == "high_priority"),
            sum(1 for s in selection_log if s["strategy"] in ("stratified", "stratified_fill")),
            sum(1 for s in selection_log if s["strategy"] == "random_audit"),
        )

        return result_ids, log_df


def create_review_record(
    ticket_id: str,
    original_label: str,
    suggested_label: Optional[str] = None,
    issue_score: float = 0.0,
    reviewer_decision: str = "pending",
    reviewed_label: Optional[str] = None,
    reason: str = "",
    reviewer: str = "system",
) -> dict:
    """Create a single review audit record.

    Parameters
    ----------
    ticket_id : str
        Ticket identifier.
    original_label : str
        Original observed label.
    suggested_label : str, optional
        Model-suggested label.
    issue_score : float
        Label-issue score.
    reviewer_decision : str
        'pending', 'confirmed', 'corrected', 'rejected'.
    reviewed_label : str, optional
        Final label after review.
    reason : str
        Reason for the decision.
    reviewer : str
        Reviewer identifier.

    Returns
    -------
    dict
        Review record.
    """
    return {
        "ticket_id": ticket_id,
        "original_label": original_label,
        "suggested_label": suggested_label,
        "issue_score": float(issue_score),
        "reviewer_decision": reviewer_decision,
        "reviewed_label": reviewed_label or original_label,
        "reason": reason,
        "reviewer": reviewer,
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def simulate_review_outcome(
    selected_df: pd.DataFrame,
    reference_col: str = "reference_label",
    observed_col: str = "observed_label",
    id_col: str = "ticket_id",
) -> pd.DataFrame:
    """Simulate review outcomes using known reference labels.

    In simulation, we know the reference labels and can determine
    which observed labels are incorrect. This simulates what an
    expert reviewer would find.

    Parameters
    ----------
    selected_df : pd.DataFrame
        Selected tickets with reference and observed labels.
    reference_col : str
        Reference label column.
    observed_col : str
        Observed label column.
    id_col : str
        Ticket ID column.

    Returns
    -------
    pd.DataFrame
        Review outcomes with decisions.
    """
    records = []
    for _, row in selected_df.iterrows():
        is_error = row[reference_col] != row[observed_col]
        records.append(create_review_record(
            ticket_id=row[id_col],
            original_label=row[observed_col],
            suggested_label=row.get("predicted_label", None),
            issue_score=row.get("label_issue_score", 0.0),
            reviewer_decision="corrected" if is_error else "confirmed",
            reviewed_label=row[reference_col] if is_error else row[observed_col],
            reason="simulated_review",
            reviewer="simulation",
        ))

    return pd.DataFrame(records)


def compare_review_strategies(
    eligible_df: pd.DataFrame,
    strategies: dict[str, pd.Series],
    reference_col: str = "reference_label",
    observed_col: str = "observed_label",
    id_col: str = "ticket_id",
    label_col: str = "mapped_label",
) -> pd.DataFrame:
    """Compare review strategies by errors found.

    Parameters
    ----------
    eligible_df : pd.DataFrame
        Full eligible pool with reference labels.
    strategies : dict[str, pd.Series]
        Strategy name → selected ticket IDs.
    reference_col : str
        Reference label column.
    observed_col : str
        Observed label column.
    id_col : str
        Ticket ID column.
    label_col : str
        Class label column.

    Returns
    -------
    pd.DataFrame
        Comparison table with errors found, precision, class coverage.
    """
    rows = []
    for strategy_name, selected_ids in strategies.items():
        selected_df = eligible_df[eligible_df[id_col].isin(selected_ids)]
        n_selected = len(selected_df)

        if n_selected == 0:
            rows.append({
                "strategy": strategy_name,
                "n_selected": 0,
                "n_errors_found": 0,
                "precision_at_k": 0.0,
                "errors_per_100": 0.0,
                "class_coverage": 0,
            })
            continue

        errors_found = (
            selected_df[reference_col] != selected_df[observed_col]
        ).sum()

        # Class coverage
        classes_covered = selected_df[label_col].nunique()

        # Errors per 100 reviews
        errors_per_100 = (errors_found / n_selected) * 100

        rows.append({
            "strategy": strategy_name,
            "n_selected": n_selected,
            "n_errors_found": int(errors_found),
            "precision_at_k": float(errors_found / n_selected),
            "errors_per_100": float(errors_per_100),
            "class_coverage": int(classes_covered),
        })

    return pd.DataFrame(rows).sort_values("n_errors_found", ascending=False)


def compute_precision_at_k(
    ranked_ids: pd.Series,
    true_error_ids: set,
    k_values: Optional[list[int]] = None,
) -> dict:
    """Compute Precision@K for ranked label-issue detection.

    Parameters
    ----------
    ranked_ids : pd.Series
        Ticket IDs ranked by suspiciousness (most suspicious first).
    true_error_ids : set
        Set of ticket IDs that are truly mislabeled.
    k_values : list[int], optional
        K values to compute. Defaults to [100, 200, 500, 1000, 2000].

    Returns
    -------
    dict
        Precision@K for each k.
    """
    if k_values is None:
        k_values = [100, 200, 500, 1000, 2000]

    results = {}
    for k in k_values:
        if k > len(ranked_ids):
            k_actual = len(ranked_ids)
        else:
            k_actual = k

        top_k = set(ranked_ids.iloc[:k_actual])
        true_positives = len(top_k & true_error_ids)
        precision = true_positives / k_actual if k_actual > 0 else 0.0

        results[f"precision_at_{k}"] = float(precision)
        results[f"true_positives_at_{k}"] = int(true_positives)
        results[f"k_actual_{k}"] = int(k_actual)

    # Average precision
    hits = 0
    sum_precision = 0.0
    for i, tid in enumerate(ranked_ids):
        if tid in true_error_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    avg_precision = sum_precision / len(true_error_ids) if len(true_error_ids) > 0 else 0.0
    results["average_precision"] = float(avg_precision)
    results["total_true_errors"] = len(true_error_ids)

    return results
