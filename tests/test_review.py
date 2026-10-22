"""Tests for review prioritization module."""

import numpy as np
import pandas as pd
import pytest

from src.review.prioritize import (
    ReviewBudgetManager,
    create_review_record,
    simulate_review_outcome,
    compare_review_strategies,
    compute_precision_at_k,
)


@pytest.fixture
def eligible_pool():
    """Create a sample eligible pool for testing."""
    np.random.seed(42)
    n = 5000
    classes = ["Billing", "Refund", "Technical", "Account Access", "Shipping", "Other"]
    df = pd.DataFrame({
        "ticket_id": [f"T{i:05d}" for i in range(n)],
        "mapped_label": np.random.choice(classes, n),
        "observed_label": np.random.choice(classes, n),
        "reference_label": np.random.choice(classes, n),
        "label_issue_score": np.random.uniform(0, 1, n),
        "worker_tier": np.random.choice(["A", "B"], n),
    })
    # Ensure some known errors
    error_mask = df["reference_label"] != df["observed_label"]
    df["is_error"] = error_mask
    return df


class TestBudgetManager:
    """Tests for ReviewBudgetManager."""

    def test_budget_validation(self):
        """Budget allocations must sum to total."""
        with pytest.raises(ValueError, match="must sum"):
            ReviewBudgetManager(total_budget=2000, high_priority=1000,
                              stratified=600, random_audit=500)

    def test_valid_budget(self):
        mgr = ReviewBudgetManager()
        assert mgr.total_budget == 2000
        assert mgr.high_priority + mgr.stratified + mgr.random_audit == 2000

    def test_random_selection_size(self, eligible_pool):
        mgr = ReviewBudgetManager()
        selected = mgr.select_random(eligible_pool["ticket_id"])
        assert len(selected) == 2000

    def test_random_no_duplicates(self, eligible_pool):
        mgr = ReviewBudgetManager()
        selected = mgr.select_random(eligible_pool["ticket_id"])
        assert len(selected) == len(selected.unique())

    def test_random_deterministic(self, eligible_pool):
        mgr = ReviewBudgetManager(seed=42)
        s1 = mgr.select_random(eligible_pool["ticket_id"])
        mgr2 = ReviewBudgetManager(seed=42)
        s2 = mgr2.select_random(eligible_pool["ticket_id"])
        assert set(s1) == set(s2)

    def test_issue_score_selection(self, eligible_pool):
        mgr = ReviewBudgetManager()
        selected = mgr.select_by_issue_score(eligible_pool)
        assert len(selected) == 2000
        # Check they are the highest scores
        top_scores = eligible_pool.nlargest(2000, "label_issue_score")["ticket_id"]
        assert set(selected.values) == set(top_scores.values)

    def test_combined_selection_size(self, eligible_pool):
        mgr = ReviewBudgetManager()
        selected_ids, log = mgr.select_combined(eligible_pool, label_col="mapped_label")
        assert len(selected_ids) <= 2000
        assert len(log) > 0

    def test_combined_no_duplicates(self, eligible_pool):
        mgr = ReviewBudgetManager()
        selected_ids, log = mgr.select_combined(eligible_pool, label_col="mapped_label")
        assert len(selected_ids) == len(set(selected_ids))


class TestReviewRecord:
    """Tests for review record creation."""

    def test_create_record(self):
        record = create_review_record(
            ticket_id="T001",
            original_label="Billing",
            suggested_label="Refund",
            issue_score=0.85,
        )
        assert record["ticket_id"] == "T001"
        assert record["original_label"] == "Billing"
        assert record["suggested_label"] == "Refund"
        assert record["reviewer_decision"] == "pending"

    def test_default_reviewed_label(self):
        record = create_review_record(
            ticket_id="T001",
            original_label="Billing",
        )
        assert record["reviewed_label"] == "Billing"


class TestSimulateReview:
    """Tests for simulated review outcomes."""

    def test_simulation_identifies_errors(self):
        df = pd.DataFrame({
            "ticket_id": ["T1", "T2", "T3"],
            "reference_label": ["A", "B", "C"],
            "observed_label": ["A", "C", "C"],  # T2 is wrong
        })
        outcomes = simulate_review_outcome(df)
        assert len(outcomes) == 3
        # T2 should be corrected
        t2 = outcomes[outcomes["ticket_id"] == "T2"].iloc[0]
        assert t2["reviewer_decision"] == "corrected"
        assert t2["reviewed_label"] == "B"
        # T1 and T3 should be confirmed
        t1 = outcomes[outcomes["ticket_id"] == "T1"].iloc[0]
        assert t1["reviewer_decision"] == "confirmed"


class TestCompareStrategies:
    """Tests for strategy comparison."""

    def test_comparison_output(self, eligible_pool):
        mgr = ReviewBudgetManager(seed=42)
        random_ids = mgr.select_random(eligible_pool["ticket_id"])
        issue_ids = mgr.select_by_issue_score(eligible_pool)

        strategies = {
            "random": random_ids,
            "issue_score": issue_ids,
        }
        result = compare_review_strategies(eligible_pool, strategies)
        assert len(result) == 2
        assert "n_errors_found" in result.columns
        assert "precision_at_k" in result.columns


class TestPrecisionAtK:
    """Tests for Precision@K computation."""

    def test_perfect_ranking(self):
        # All errors are ranked first
        ranked = pd.Series(["E1", "E2", "E3", "C1", "C2"])
        errors = {"E1", "E2", "E3"}
        result = compute_precision_at_k(ranked, errors, k_values=[3, 5])
        assert result["precision_at_3"] == 1.0

    def test_worst_ranking(self):
        # All errors are ranked last
        ranked = pd.Series(["C1", "C2", "E1", "E2", "E3"])
        errors = {"E1", "E2", "E3"}
        result = compute_precision_at_k(ranked, errors, k_values=[2])
        assert result["precision_at_2"] == 0.0

    def test_average_precision(self):
        ranked = pd.Series(["E1", "C1", "E2", "C2", "E3"])
        errors = {"E1", "E2", "E3"}
        result = compute_precision_at_k(ranked, errors, k_values=[5])
        assert result["average_precision"] > 0
        assert result["total_true_errors"] == 3
