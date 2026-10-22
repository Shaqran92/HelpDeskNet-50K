"""Tests for noise metrics and statistics modules."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.noise_metrics import (
    estimate_noise_rate,
    estimate_class_noise_rates,
    build_transition_matrix,
    compare_worker_tiers,
    compute_audit_sample_stats,
    noise_rate_by_worker,
    _wilson_interval,
)
from src.analysis.statistics import (
    chi_square_independence,
    proportion_test,
    bootstrap_metric,
    compute_class_balance_metrics,
    cohens_h,
)


class TestWilsonInterval:
    """Tests for Wilson confidence interval."""

    def test_zero_successes(self):
        lo, hi = _wilson_interval(0, 100)
        assert lo == 0.0 or lo > 0.0  # Lower bound >= 0
        assert hi > 0.0
        assert hi < 1.0

    def test_all_successes(self):
        lo, hi = _wilson_interval(100, 100)
        assert lo > 0.0
        assert hi == 1.0 or hi < 1.0

    def test_half(self):
        lo, hi = _wilson_interval(50, 100)
        assert lo < 0.5
        assert hi > 0.5
        assert lo > 0.3
        assert hi < 0.7

    def test_zero_n(self):
        lo, hi = _wilson_interval(0, 0)
        assert lo == 0.0
        assert hi == 0.0


class TestEstimateNoiseRate:
    """Tests for noise rate estimation."""

    def test_no_noise(self):
        ref = pd.Series(["A", "B", "C"] * 100)
        obs = pd.Series(["A", "B", "C"] * 100)
        result = estimate_noise_rate(ref, obs)
        assert result["noise_rate"] == 0.0
        assert result["n_errors"] == 0
        assert result["n_total"] == 300

    def test_known_noise(self):
        ref = pd.Series(["A"] * 100)
        obs = pd.Series(["A"] * 90 + ["B"] * 10)
        result = estimate_noise_rate(ref, obs)
        assert result["noise_rate"] == 0.1
        assert result["n_errors"] == 10
        assert result["ci_lower"] < 0.1
        assert result["ci_upper"] > 0.1

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            estimate_noise_rate(pd.Series(["A"]), pd.Series(["A", "B"]))

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            estimate_noise_rate(pd.Series([], dtype=str), pd.Series([], dtype=str))


class TestClassNoiseRates:
    """Tests for per-class noise rate estimation."""

    def test_two_classes(self):
        ref = pd.Series(["A"] * 50 + ["B"] * 50)
        obs = pd.Series(["A"] * 45 + ["B"] * 5 + ["B"] * 48 + ["A"] * 2)
        result = estimate_class_noise_rates(ref, obs, ["A", "B"])
        assert len(result) == 2
        assert result.loc[result["class"] == "A", "noise_rate"].values[0] == 0.1
        assert result.loc[result["class"] == "B", "noise_rate"].values[0] == 0.04

    def test_missing_class(self):
        ref = pd.Series(["A"] * 100)
        obs = pd.Series(["A"] * 100)
        result = estimate_class_noise_rates(ref, obs, ["A", "B"])
        assert len(result) == 2
        b_row = result[result["class"] == "B"]
        assert b_row["n_total"].values[0] == 0


class TestTransitionMatrix:
    """Tests for transition matrix building."""

    def test_identity(self):
        labels = pd.Series(["A", "B", "C"] * 10)
        tm = build_transition_matrix(labels, labels, ["A", "B", "C"])
        # Diagonal should be 1.0
        for cls in ["A", "B", "C"]:
            assert abs(tm.loc[cls, cls] - 1.0) < 1e-10

    def test_shape(self):
        ref = pd.Series(["A", "B"])
        obs = pd.Series(["B", "A"])
        tm = build_transition_matrix(ref, obs, ["A", "B"])
        assert tm.shape == (2, 2)

    def test_all_wrong(self):
        ref = pd.Series(["A"] * 10)
        obs = pd.Series(["B"] * 10)
        tm = build_transition_matrix(ref, obs, ["A", "B"])
        assert tm.loc["A", "B"] == 1.0
        assert tm.loc["A", "A"] == 0.0


class TestWorkerTierComparison:
    """Tests for worker tier comparison."""

    def test_different_tiers(self):
        df = pd.DataFrame({
            "worker_tier": ["A"] * 100 + ["B"] * 100,
            "reference_label": ["X"] * 200,
            "observed_label": ["X"] * 95 + ["Y"] * 5 + ["X"] * 80 + ["Y"] * 20,
        })
        result = compare_worker_tiers(df)
        assert "tier_stats" in result
        assert result["rate_difference"] > 0
        assert result["p_value"] < 0.05


class TestAuditSample:
    """Tests for audit sampling."""

    def test_deterministic(self):
        df = pd.DataFrame({
            "reference_label": ["A"] * 1000,
            "observed_label": ["A"] * 900 + ["B"] * 100,
        })
        r1 = compute_audit_sample_stats(df, sample_size=200, seed=42)
        r2 = compute_audit_sample_stats(df, sample_size=200, seed=42)
        assert r1["estimated_noise_rate"] == r2["estimated_noise_rate"]

    def test_sample_size_limit(self):
        df = pd.DataFrame({
            "reference_label": ["A"] * 50,
            "observed_label": ["A"] * 50,
        })
        result = compute_audit_sample_stats(df, sample_size=400, seed=42)
        assert result["audit_sample_size"] == 50


class TestClassBalance:
    """Tests for class balance metrics."""

    def test_balanced(self):
        labels = pd.Series(["A", "B", "C"] * 100)
        result = compute_class_balance_metrics(labels)
        assert result["imbalance_ratio"] == 1.0
        assert abs(result["normalized_entropy"] - 1.0) < 0.01

    def test_imbalanced(self):
        labels = pd.Series(["A"] * 900 + ["B"] * 100)
        result = compute_class_balance_metrics(labels)
        assert result["imbalance_ratio"] == 9.0


class TestProportionTest:
    """Tests for two-proportion z-test."""

    def test_equal_proportions(self):
        result = proportion_test(50, 100, 50, 100)
        assert abs(result["z_statistic"]) < 0.01
        assert result["p_value"] > 0.9

    def test_different_proportions(self):
        result = proportion_test(10, 100, 50, 100)
        assert result["p_value"] < 0.01


class TestCohensH:
    """Tests for Cohen's h effect size."""

    def test_equal(self):
        assert abs(cohens_h(0.5, 0.5)) < 1e-10

    def test_different(self):
        assert cohens_h(0.1, 0.9) > 0
