"""Unit and integration tests for data validation, audit, and quality checks in HelpDeskNet-50K."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_raw_dataset
from src.data.validate_data import (
    compute_class_frequencies,
    detect_duplicates,
    detect_near_duplicates,
    generate_data_audit_report,
    validate_labels,
    validate_no_empty_text,
    validate_required_columns,
)

RAW_DATASET_PATH = Path("d:/HacktoberFest/data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv")


class TestValidateRequiredColumns:
    """Tests for validate_required_columns."""

    def test_all_columns_present(self):
        """Verify empty list when all columns exist."""
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
        missing = validate_required_columns(df, ["col_a", "col_b"])
        assert missing == []

    def test_missing_columns_detected(self):
        """Verify list of missing columns returned correctly."""
        df = pd.DataFrame({"col_a": [1]})
        missing = validate_required_columns(df, ["col_a", "missing_x", "missing_y"])
        assert missing == ["missing_x", "missing_y"]

    def test_raw_dataset_has_all_expected_columns(self):
        """Verify raw dataset contains all 16 specified columns."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        expected_cols = [
            "subject",
            "body",
            "answer",
            "type",
            "queue",
            "priority",
            "language",
            "version",
            "tag_1",
            "tag_2",
            "tag_3",
            "tag_4",
            "tag_5",
            "tag_6",
            "tag_7",
            "tag_8",
        ]
        missing = validate_required_columns(df, expected_cols)
        assert missing == []


class TestValidateNoEmptyText:
    """Tests for validate_no_empty_text."""

    def test_detects_null_and_whitespace_text(self):
        """Verify identification of NaN, empty string, and whitespace-only rows."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "body": ["Valid ticket text", None, "   ", "", "Another valid ticket"],
            }
        )
        empty_rows = validate_no_empty_text(df, text_col="body")
        assert len(empty_rows) == 3
        assert empty_rows["id"].tolist() == [2, 3, 4]

    def test_raw_dataset_has_zero_empty_bodies(self):
        """Verify raw dataset fact: 0 missing bodies."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        empty_bodies = validate_no_empty_text(df, text_col="body")
        assert len(empty_bodies) == 0


class TestValidateLabels:
    """Tests for validate_labels."""

    def test_detects_invalid_and_missing_labels(self):
        """Verify invalid or NaN labels are captured."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "queue": ["IT Support", "Technical Support", "Unknown Category", None, "Billing and Payments"],
            }
        )
        valid_labels = ["IT Support", "Technical Support", "Billing and Payments"]
        invalid_rows = validate_labels(df, label_col="queue", valid_labels=valid_labels)

        assert len(invalid_rows) == 2
        assert invalid_rows["id"].tolist() == [3, 4]

    def test_raw_dataset_queues_all_valid(self):
        """Verify all queue values in raw dataset match the 10 known queues."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        known_queues = [
            "Technical Support",
            "Product Support",
            "Customer Service",
            "IT Support",
            "Billing and Payments",
            "Returns and Exchanges",
            "Service Outages and Maintenance",
            "Sales and Pre-Sales",
            "Human Resources",
            "General Inquiry",
        ]
        invalid_rows = validate_labels(df, label_col="queue", valid_labels=known_queues)
        assert len(invalid_rows) == 0


class TestDetectDuplicates:
    """Tests for detect_duplicates."""

    def test_detects_exact_duplicates(self):
        """Verify exact duplicate bodies are detected."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "body": ["Repeated body", "Unique body", "Repeated body", "Another unique"],
            }
        )
        duplicates = detect_duplicates(df, text_col="body")
        assert len(duplicates) == 2
        assert duplicates["id"].tolist() == [1, 3]

    def test_raw_dataset_has_zero_duplicate_bodies(self):
        """Verify raw dataset fact: no duplicate bodies."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        dups = detect_duplicates(df, text_col="body")
        assert len(dups) == 0


class TestDetectNearDuplicates:
    """Tests for detect_near_duplicates."""

    def test_detects_near_duplicate_pairs(self):
        """Verify detection of near-duplicate templates."""
        t1 = (
            "Dear Customer Support, I am reaching out to inquire about the billing options and payment "
            "plans available for your platform. Please provide plan details and pricing."
        )
        t2 = (
            "Dear Customer Support, I am reaching out to inquire about the billing options and payment "
            "plans available for your platform. Please provide plan details with pricing."
        )
        t3 = (
            "Hello, my computer screen is completely blank and will not turn on even when plugged in."
        )

        df = pd.DataFrame({"id": ["A", "B", "C"], "body": [t1, t2, t3]})
        near_dups = detect_near_duplicates(df, text_col="body", threshold=0.90)

        assert len(near_dups) == 1
        assert near_dups.iloc[0]["similarity"] >= 0.90
        assert near_dups.iloc[0]["index_1"] == 0
        assert near_dups.iloc[0]["index_2"] == 1

    def test_empty_dataframe_returns_empty_result(self):
        """Verify graceful handling of empty DataFrame."""
        df = pd.DataFrame(columns=["body"])
        result = detect_near_duplicates(df, text_col="body")
        assert len(result) == 0
        assert "similarity" in result.columns


class TestComputeClassFrequencies:
    """Tests for compute_class_frequencies."""

    def test_computes_exact_counts_and_percentages(self):
        """Verify class counts and percentages match known distribution."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        freq_df = compute_class_frequencies(df, label_col="queue")

        assert len(freq_df) == 10
        assert freq_df.iloc[0]["queue"] == "Technical Support"
        assert freq_df.iloc[0]["count"] == 8362
        assert abs(freq_df["percentage"].sum() - 100.0) < 0.1

        # Check all queue counts match facts
        expected_counts = {
            "Technical Support": 8362,
            "Product Support": 5252,
            "Customer Service": 4268,
            "IT Support": 3433,
            "Billing and Payments": 2788,
            "Returns and Exchanges": 1437,
            "Service Outages and Maintenance": 1148,
            "Sales and Pre-Sales": 918,
            "Human Resources": 576,
            "General Inquiry": 405,
        }
        for _, row in freq_df.iterrows():
            assert row["count"] == expected_counts[row["queue"]]


class TestGenerateDataAuditReport:
    """Tests for generate_data_audit_report."""

    def test_generate_and_save_report(self):
        """Verify comprehensive report creation and JSON persistence."""
        df = pd.DataFrame(
            {
                "subject": ["Sub 1", None],
                "body": ["Body 1 with several words", "Body 2 is longer with more text"],
                "queue": ["IT Support", "Technical Support"],
                "language": ["en", "en"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "audit_report.json"
            report = generate_data_audit_report(df, output_path=out_file)

            assert report["dataset_summary"]["total_rows"] == 2
            assert report["dataset_summary"]["total_columns"] == 4
            assert report["missingness"]["missing_counts"]["subject"] == 1
            assert report["missingness"]["missing_counts"]["body"] == 0
            assert report["distributions"]["languages"]["en"] == 2

            # Verify saved file exists and can be loaded
            assert out_file.exists()
            with open(out_file, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["dataset_summary"]["total_rows"] == 2
