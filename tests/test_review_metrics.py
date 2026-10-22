"""Unit tests for review_metrics module."""

import numpy as np
import pandas as pd
import pytest

from helpdesknet.review.review_metrics import (
    compute_review_yield,
    compute_precision_at_k,
    compute_class_coverage,
    compute_inter_reviewer_agreement,
)


class TestReviewYield:
    """Tests for review yield calculation."""

    def test_positive_yield(self):
        y = compute_review_yield(500, 2000)
        assert y == 0.25

    def test_zero_budget(self):
        assert compute_review_yield(0, 0) == 0.0

    def test_full_yield(self):
        assert compute_review_yield(2000, 2000) == 1.0


class TestPrecisionAtK:
    """Tests for Precision@K calculations."""

    def test_precision_at_k(self):
        ranked = pd.Series(["A", "B", "C", "D", "E"])
        true_errors = {"A", "B"}
        res = compute_precision_at_k(ranked, true_errors, k_values=[2, 4])
        assert res["precision_at_2"] == 1.0
        assert res["precision_at_4"] == 0.5
        assert res["average_precision"] > 0.0


class TestClassCoverage:
    """Tests for class coverage analysis."""

    def test_class_coverage(self):
        df = pd.DataFrame({"mapped_label": ["Technical", "Billing", "Refund"]})
        all_classes = ["Technical", "Billing", "Refund", "Account Access", "Shipping", "Other"]
        cov = compute_class_coverage(df, "mapped_label", all_classes)
        assert cov["n_classes_covered"] == 3
        assert cov["coverage_ratio"] == 0.5
        assert "Shipping" in cov["missing_classes"]


class TestReviewerAgreement:
    """Tests for inter-reviewer agreement and Cohen's kappa."""

    def test_perfect_agreement(self):
        labels = pd.Series(["A", "B", "C", "A", "B"] * 20)
        res = compute_inter_reviewer_agreement(labels, labels)
        assert res["observed_agreement"] == 1.0
        assert res["cohens_kappa"] == 1.0
        assert res["kappa_interpretation"] == "almost_perfect"

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_inter_reviewer_agreement(pd.Series(["A"]), pd.Series(["A", "B"]))
