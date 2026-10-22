"""Comprehensive unit tests for the labeling and noise simulation modules.

Tests:
    - src/labeling/map_categories.py
    - src/labeling/simulate_noise.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.labeling.map_categories import (
    QUEUE_TO_TARGET_MAP,
    TAG_BASED_OVERRIDES,
    TARGET_CLASSES,
    apply_label_mapping,
    get_mapping_report,
    save_mapping_documentation,
)
from src.labeling.simulate_noise import (
    DEFAULT_CONFUSION_PAIRS,
    NoiseSimulator,
    build_transition_matrix,
    validate_noise_simulation,
)


# ==============================================================================
# Tests for map_categories.py
# ==============================================================================


class TestMapCategories:
    """Test suite for category mapping and tag overrides."""

    def test_constants_structure(self) -> None:
        """Verify taxonomy constants conform to specifications."""
        assert isinstance(TARGET_CLASSES, list)
        assert len(TARGET_CLASSES) == 6
        assert set(TARGET_CLASSES) == {
            "Billing",
            "Refund",
            "Technical",
            "Account Access",
            "Shipping",
            "Other",
        }

        # Check all values in QUEUE_TO_TARGET_MAP are in TARGET_CLASSES
        for queue, target in QUEUE_TO_TARGET_MAP.items():
            assert isinstance(queue, str)
            assert target in TARGET_CLASSES

        # Check all keys in TAG_BASED_OVERRIDES are in TARGET_CLASSES
        for target, kws in TAG_BASED_OVERRIDES.items():
            assert target in TARGET_CLASSES
            assert isinstance(kws, list)
            assert len(kws) > 0

    def test_apply_mapping_direct_queue(self) -> None:
        """Test standard direct mapping from queue to target class."""
        df = pd.DataFrame(
            {
                "queue": [
                    "Technical Support",
                    "Billing and Payments",
                    "Returns and Exchanges",
                    "Customer Service",
                    "General Inquiry",
                ],
                "tag_1": [None, None, None, None, None],
                "tag_2": [None, None, None, None, None],
            }
        )

        res = apply_label_mapping(df)

        assert list(res["mapped_label"]) == [
            "Technical",
            "Billing",
            "Refund",
            "Other",
            "Other",
        ]
        assert list(res["mapping_source"]) == ["queue_direct"] * 5
        # Specific queues should be high confidence, broad Other should be medium
        assert list(res["mapping_confidence"]) == [
            "high",
            "high",
            "high",
            "medium",
            "medium",
        ]

    def test_apply_mapping_tag_overrides(self) -> None:
        """Test tag-based overrides reclassifying tickets."""
        df = pd.DataFrame(
            {
                "queue": [
                    "Technical Support",     # Should be reclassified to Account Access
                    "Customer Service",      # Should be reclassified to Shipping
                    "Customer Service",      # Should be reclassified to Refund
                    "Sales and Pre-Sales",   # Should be reclassified to Billing
                ],
                "tag_1": ["Account", "Shipping", "Refund", "Billing"],
                "tag_2": ["Network", "Disruption", "Feedback", "Product"],
            }
        )

        res = apply_label_mapping(df)

        assert list(res["mapped_label"]) == [
            "Account Access",
            "Shipping",
            "Refund",
            "Billing",
        ]
        assert list(res["mapping_source"]) == ["tag_override"] * 4
        # Overriding specific queue (Technical) gives medium confidence,
        # overriding broad queue (Customer Service, Sales) gives high confidence
        assert list(res["mapping_confidence"]) == [
            "medium",
            "high",
            "high",
            "high",
        ]

    def test_conditional_overrides_when_already_matching(self) -> None:
        """Test that Billing/Refund tags do not trigger override when queue is already Billing/Refund."""
        df = pd.DataFrame(
            {
                "queue": [
                    "Billing and Payments",
                    "Returns and Exchanges",
                ],
                "tag_1": ["Payment", "Return"],
                "tag_2": ["Invoice", "Refund"],
            }
        )

        res = apply_label_mapping(df)

        assert list(res["mapped_label"]) == ["Billing", "Refund"]
        # Since queue already was Billing/Refund, it is queue_direct confirmed
        assert list(res["mapping_source"]) == ["queue_direct", "queue_direct"]
        assert list(res["mapping_confidence"]) == ["high", "high"]

    def test_ambiguous_conflicting_tags(self) -> None:
        """Test that tickets with conflicting tag overrides are flagged as ambiguous."""
        df = pd.DataFrame(
            {
                "queue": ["Billing and Payments", "Customer Service"],
                "tag_1": ["Billing", "Account"],
                "tag_2": ["Refund", "Shipping"],
            }
        )

        res = apply_label_mapping(df)

        assert list(res["mapping_source"]) == ["ambiguous", "ambiguous"]
        assert list(res["mapping_confidence"]) == ["low", "low"]
        # Mapped label retains base queue mapping if known
        assert res.loc[0, "mapped_label"] == "Billing"
        assert res.loc[1, "mapped_label"] == "Other"

    def test_unknown_or_missing_queue(self) -> None:
        """Test handling of missing or unrecognized queue categories."""
        df = pd.DataFrame(
            {
                "queue": [None, "Unrecognized Queue Name"],
                "tag_1": [None, None],
                "tag_2": [None, None],
            }
        )

        res = apply_label_mapping(df)

        assert list(res["mapped_label"]) == ["Other", "Other"]
        assert list(res["mapping_source"]) == ["ambiguous", "ambiguous"]
        assert list(res["mapping_confidence"]) == ["low", "low"]

    def test_immutability_and_empty_df(self) -> None:
        """Test input DataFrame is not mutated and empty DataFrames are handled."""
        original_df = pd.DataFrame({"queue": ["Technical Support"]})
        df_copy = original_df.copy()

        result = apply_label_mapping(original_df)
        pd.testing.assert_frame_equal(original_df, df_copy)
        assert "mapped_label" in result.columns

        # Empty df
        empty_df = pd.DataFrame(columns=["queue", "tag_1"])
        res_empty = apply_label_mapping(empty_df)
        assert res_empty.empty
        assert "mapped_label" in res_empty.columns

    def test_invalid_arguments(self) -> None:
        """Test error handling on invalid input types and missing columns."""
        with pytest.raises(TypeError):
            apply_label_mapping([{"queue": "Tech"}])  # type: ignore

        with pytest.raises(ValueError, match="not found in DataFrame"):
            apply_label_mapping(pd.DataFrame({"wrong_col": [1, 2]}), queue_col="queue")

    def test_get_mapping_report(self) -> None:
        """Test mapping report statistics computation."""
        df = pd.DataFrame(
            {
                "queue": ["Technical Support", "Billing and Payments", "Customer Service"],
                "tag_1": ["Account", "Billing", None],
                "tag_2": [None, None, None],
            }
        )
        mapped = apply_label_mapping(df)
        report = get_mapping_report(mapped)

        assert report["total_records"] == 3
        assert "mapped_label_counts" in report
        assert "mapping_source_counts" in report
        assert "ambiguity_rate" in report
        assert "high_confidence_rate" in report
        assert "queue_to_mapped" in report

        # Test error if required columns missing
        with pytest.raises(ValueError, match="missing required mapping columns"):
            get_mapping_report(df)

    def test_save_mapping_documentation(self, tmp_path: Path) -> None:
        """Test documentation generation and file output."""
        doc_path = tmp_path / "docs" / "label_mapping.md"
        saved = save_mapping_documentation(doc_path)

        assert saved.exists()
        text = saved.read_text(encoding="utf-8")
        assert "# HelpDeskNet-50K Category Mapping" in text
        assert "QUEUE_TO_TARGET_MAP" in text or "Queue-to-Target" in text
        assert "Account Access" in text


# ==============================================================================
# Tests for simulate_noise.py
# ==============================================================================


class TestSimulateNoise:
    """Test suite for label noise simulation mechanisms."""

    @pytest.fixture
    def sample_labels(self) -> pd.Series:
        """Fixture returning a balanced sample Series of target classes."""
        return pd.Series(TARGET_CLASSES * 200, name="ground_truth_proxy")

    def test_simulator_initialization(self) -> None:
        """Test NoiseSimulator constructor and input validations."""
        sim = NoiseSimulator(seed=123)
        assert sim.classes == TARGET_CLASSES
        assert sim.seed == 123

        # Invalid classes
        with pytest.raises(ValueError, match="cannot be empty"):
            NoiseSimulator(classes=[])

        with pytest.raises(ValueError, match="Duplicate classes"):
            NoiseSimulator(classes=["A", "A", "B"])

        with pytest.raises(TypeError):
            NoiseSimulator(seed="invalid")  # type: ignore

    def test_symmetric_noise_zero_and_full_rate(self, sample_labels: pd.Series) -> None:
        """Test symmetric noise at 0% and 100% boundary rates."""
        sim = NoiseSimulator(seed=42)

        # Rate 0.0 -> no errors
        res_zero = sim.symmetric_noise(sample_labels, rate=0.0)
        assert (res_zero["is_injected_error"] == False).all()
        assert (res_zero["observed_label"] == sample_labels).all()
        assert (res_zero["reference_label"] == sample_labels).all()

        # Rate 1.0 -> 100% errors, every label flipped
        res_full = sim.symmetric_noise(sample_labels, rate=1.0)
        assert (res_full["is_injected_error"] == True).all()
        assert (res_full["observed_label"] != sample_labels).all()
        assert (res_full["reference_label"] == sample_labels).all()
        # All observed labels must still be valid classes
        assert set(res_full["observed_label"]).issubset(set(TARGET_CLASSES))

    def test_symmetric_noise_intermediate_rate_and_determinism(self, sample_labels: pd.Series) -> None:
        """Test symmetric noise rate accuracy and deterministic repeatability."""
        sim1 = NoiseSimulator(seed=42)
        sim2 = NoiseSimulator(seed=42)

        res1 = sim1.symmetric_noise(sample_labels, rate=0.20)
        res2 = sim2.symmetric_noise(sample_labels, rate=0.20)

        # Exact determinism
        pd.testing.assert_frame_equal(res1, res2)

        # Empirical rate within statistical expectation (n=1200)
        empirical_rate = res1["is_injected_error"].mean()
        assert 0.16 <= empirical_rate <= 0.24

        # Non-destruction of input Series
        labels_copy = sample_labels.copy()
        sim1.symmetric_noise(sample_labels, rate=0.50)
        pd.testing.assert_series_equal(sample_labels, labels_copy)

    def test_class_dependent_noise_default_pairs(self, sample_labels: pd.Series) -> None:
        """Test class-dependent noise flips to expected confusion pairs."""
        sim = NoiseSimulator(seed=42)
        res = sim.class_dependent_noise(sample_labels, rate=1.0)

        assert (res["is_injected_error"] == True).all()
        for ref, obs in zip(res["reference_label"], res["observed_label"]):
            assert obs == DEFAULT_CONFUSION_PAIRS[ref]

        # Partial rate test
        res_partial = sim.class_dependent_noise(sample_labels, rate=0.15)
        emp_rate = res_partial["is_injected_error"].mean()
        assert 0.11 <= emp_rate <= 0.19

    def test_worker_dependent_noise_tiers(self, sample_labels: pd.Series) -> None:
        """Test worker-dependent noise differentiates Tier A and Tier B error rates."""
        sim = NoiseSimulator(seed=42)
        base_rate = 0.20
        res = sim.worker_dependent_noise(
            sample_labels,
            rate=base_rate,
            n_workers_a=5,
            n_workers_b=5,
            tier_a_multiplier=0.5,
            tier_b_multiplier=1.5,
        )

        assert "worker_id" in res.columns
        assert "worker_tier" in res.columns
        assert set(res["worker_tier"]) == {"Tier A", "Tier B"}

        tier_a_errors = res[res["worker_tier"] == "Tier A"]["is_injected_error"].mean()
        tier_b_errors = res[res["worker_tier"] == "Tier B"]["is_injected_error"].mean()

        # Tier A target: 0.20 * 0.5 = 0.10
        # Tier B target: 0.20 * 1.5 = 0.30
        assert 0.06 <= tier_a_errors <= 0.14
        assert 0.24 <= tier_b_errors <= 0.36
        assert tier_b_errors > tier_a_errors

    def test_apply_noise_dispatcher(self, sample_labels: pd.Series) -> None:
        """Test dispatching mechanism through apply_noise."""
        sim = NoiseSimulator(seed=42)

        res_sym = sim.apply_noise(sample_labels, mechanism="symmetric", rate=0.10)
        assert len(res_sym) == len(sample_labels)

        res_cls = sim.apply_noise(sample_labels, mechanism="class_dependent", rate=0.10)
        assert len(res_cls) == len(sample_labels)

        res_wrk = sim.apply_noise(sample_labels, mechanism="worker_dependent", rate=0.10)
        assert "worker_tier" in res_wrk.columns

        with pytest.raises(ValueError, match="Unsupported noise mechanism"):
            sim.apply_noise(sample_labels, mechanism="invalid_mechanism", rate=0.10)

    def test_build_transition_matrix(self) -> None:
        """Test building normalized and unnormalized transition matrices."""
        ref = pd.Series(["Billing", "Billing", "Technical", "Refund"])
        obs = pd.Series(["Billing", "Refund", "Technical", "Refund"])

        # Normalized
        mat_norm = build_transition_matrix(ref, obs, classes=TARGET_CLASSES, normalize=True)
        assert mat_norm.shape == (6, 6)
        # Billing row: 1 Billing, 1 Refund -> 0.5 each
        assert mat_norm.loc["Billing", "Billing"] == 0.5
        assert mat_norm.loc["Billing", "Refund"] == 0.5
        # Technical row: 1 Technical -> 1.0
        assert mat_norm.loc["Technical", "Technical"] == 1.0

        # Unnormalized counts
        mat_counts = build_transition_matrix(ref, obs, classes=TARGET_CLASSES, normalize=False)
        assert mat_counts.loc["Billing", "Billing"] == 1.0
        assert mat_counts.loc["Billing", "Refund"] == 1.0

    def test_validate_noise_simulation(self, sample_labels: pd.Series) -> None:
        """Test validation suite for simulation outputs."""
        sim = NoiseSimulator(seed=42)
        res = sim.worker_dependent_noise(sample_labels, rate=0.20)

        # Valid run check
        val = validate_noise_simulation(res, expected_rate=0.20, tolerance=0.06)
        assert val["is_valid"] is True
        assert val["checks_passed"]["no_nulls"] is True
        assert val["checks_passed"]["flag_consistency"] is True
        assert val["checks_passed"]["expected_rate_within_tolerance"] is True
        assert val["worker_tier_error_rates"] is not None

        # Intentionally corrupt flags to test detection
        corrupt_res = res.copy()
        corrupt_res.loc[0, "is_injected_error"] = not corrupt_res.loc[0, "is_injected_error"]
        corrupt_val = validate_noise_simulation(corrupt_res)
        assert corrupt_val["is_valid"] is False
        assert corrupt_val["checks_passed"]["flag_consistency"] is False
