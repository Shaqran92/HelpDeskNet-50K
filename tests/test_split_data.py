"""Unit and integration tests for data splitting, split integrity, and manifest persistence in HelpDeskNet-50K."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import create_ticket_id, filter_by_language, load_raw_dataset
from src.data.split_data import (
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
    validate_split_integrity,
)

RAW_DATASET_PATH = Path("d:/HacktoberFest/data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv")


class TestCreateStratifiedSplit:
    """Tests for create_stratified_split function."""

    def test_stratified_split_proportions(self):
        """Verify train/val/test splits match requested 80/10/10 proportions."""
        # Create a synthetic dataset with balanced classes
        records = []
        for c in ["Billing", "Technical", "Account", "Shipping"]:
            for i in range(100):
                records.append({"ticket_id": f"{c}_{i}", "body": f"text for {c} {i}", "queue": c})
        df = pd.DataFrame(records)

        train, val, test = create_stratified_split(
            df,
            label_col="queue",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            seed=42,
        )

        assert len(train) == 320
        assert len(val) == 40
        assert len(test) == 40
        assert len(train) + len(val) + len(test) == 400

        # Verify class proportions across splits
        for c in ["Billing", "Technical", "Account", "Shipping"]:
            assert (train["queue"] == c).sum() == 80
            assert (val["queue"] == c).sum() == 10
            assert (test["queue"] == c).sum() == 10

    def test_stratified_split_reproducibility(self):
        """Verify fixed seed creates deterministic, identical splits."""
        records = []
        for c in ["A", "B"]:
            for i in range(50):
                records.append({"ticket_id": f"{c}_{i}", "queue": c})
        df = pd.DataFrame(records)

        t1, v1, te1 = create_stratified_split(df, label_col="queue", seed=42)
        t2, v2, te2 = create_stratified_split(df, label_col="queue", seed=42)
        t3, v3, te3 = create_stratified_split(df, label_col="queue", seed=99)

        assert t1["ticket_id"].tolist() == t2["ticket_id"].tolist()
        assert v1["ticket_id"].tolist() == v2["ticket_id"].tolist()
        assert te1["ticket_id"].tolist() == te2["ticket_id"].tolist()

        assert t1["ticket_id"].tolist() != t3["ticket_id"].tolist()

    def test_invalid_proportions_raise_valueerror(self):
        """Verify ValueError when proportions do not sum to 1.0."""
        df = pd.DataFrame({"queue": ["A", "B"], "ticket_id": [1, 2]})
        with pytest.raises(ValueError, match="Split sizes must sum to 1.0"):
            create_stratified_split(df, label_col="queue", train_size=0.7, val_size=0.1, test_size=0.1)

    def test_rare_class_raises_valueerror(self):
        """Verify error when a class has fewer than 2 samples."""
        df = pd.DataFrame({"queue": ["Class_A", "Class_A", "Singleton"], "ticket_id": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2 samples per class"):
            create_stratified_split(df, label_col="queue", seed=42)


class TestValidateSplitIntegrity:
    """Tests for validate_split_integrity function."""

    def test_integrity_passes_when_disjoint(self):
        """Verify integrity passes when partitions have zero overlapping IDs."""
        train = pd.DataFrame({"ticket_id": ["T1", "T2", "T3"], "queue": ["A", "A", "B"]})
        val = pd.DataFrame({"ticket_id": ["T4"], "queue": ["B"]})
        test = pd.DataFrame({"ticket_id": ["T5"], "queue": ["A"]})

        report = validate_split_integrity(train, val, test, id_col="ticket_id")
        assert report["integrity_passed"] is True
        assert report["overlap_detected"] is False
        assert report["total_records"] == 5
        assert report["overlaps"]["train_val"] == 0
        assert report["overlaps"]["train_test"] == 0
        assert report["overlaps"]["val_test"] == 0

    def test_integrity_fails_when_overlap_injected(self):
        """Verify integrity check flags data leakage if an ID is present in two splits."""
        train = pd.DataFrame({"ticket_id": ["T1", "T2", "LEAKED_ID"], "queue": ["A", "A", "B"]})
        val = pd.DataFrame({"ticket_id": ["T4"], "queue": ["B"]})
        test = pd.DataFrame({"ticket_id": ["LEAKED_ID", "T5"], "queue": ["A", "B"]})

        report = validate_split_integrity(train, val, test, id_col="ticket_id")
        assert report["integrity_passed"] is False
        assert report["overlap_detected"] is True
        assert report["overlaps"]["train_test"] == 1


class TestSaveAndLoadSplitManifest:
    """Tests for save_split_manifest and load_split_manifest functions."""

    def test_save_and_load_manifest_roundtrip(self):
        """Verify saving and loading manifests preserves exact IDs and splits."""
        train = pd.DataFrame({"ticket_id": ["T1", "T2"], "body": ["b1", "b2"], "queue": ["Q1", "Q2"]})
        val = pd.DataFrame({"ticket_id": ["T3"], "body": ["b3"], "queue": ["Q1"]})
        test = pd.DataFrame({"ticket_id": ["T4"], "body": ["b4"], "queue": ["Q2"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            saved_paths = save_split_manifest(train, val, test, output_dir=out_dir)

            assert "split_manifest" in saved_paths
            assert Path(saved_paths["split_manifest"]).exists()
            assert Path(saved_paths["train_csv"]).exists()

            # Load manifest
            loaded = load_split_manifest(saved_paths["split_manifest"])
            assert loaded["total_records"] == 4
            assert loaded["counts"]["train"] == 2
            assert loaded["counts"]["val"] == 1
            assert loaded["counts"]["test"] == 1
            assert loaded["splits"]["train"] == ["T1", "T2"]
            assert loaded["splits"]["val"] == ["T3"]
            assert loaded["splits"]["test"] == ["T4"]

            # Load from directory
            loaded_dir = load_split_manifest(out_dir)
            assert loaded_dir["total_records"] == 4

    def test_load_non_existent_manifest_raises_error(self):
        """Verify FileNotFoundError for missing manifest path."""
        with pytest.raises(FileNotFoundError):
            load_split_manifest("d:/HacktoberFest/data/does_not_exist/manifest.csv")


class TestFullDataPipelineIntegration:
    """End-to-end integration test of loading, filtering, ID assignment, and splitting."""

    def test_full_pipeline_run(self):
        """Execute complete pipeline on the raw dataset: load -> filter -> ID -> split -> audit."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        # 1. Load raw dataset
        raw_df = load_raw_dataset(RAW_DATASET_PATH)
        assert len(raw_df) == 28587

        # 2. Filter English records
        en_df = filter_by_language(raw_df, language="en")
        assert len(en_df) == 16338

        # 3. Add deterministic ticket_id
        ticket_df = create_ticket_id(en_df, prefix="TICKET")
        assert ticket_df.columns[0] == "ticket_id"
        assert ticket_df.iloc[0]["ticket_id"] == "TICKET-000001"
        assert ticket_df.iloc[-1]["ticket_id"] == "TICKET-016338"

        # 4. Stratified split (80/10/10) on queue
        train, val, test = create_stratified_split(
            ticket_df,
            label_col="queue",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            seed=42,
        )

        assert len(train) == 13070
        assert len(val) == 1634
        assert len(test) == 1634
        assert len(train) + len(val) + len(test) == 16338

        # 5. Integrity verification
        integrity_report = validate_split_integrity(train, val, test, id_col="ticket_id")
        assert integrity_report["integrity_passed"] is True
        assert integrity_report["overlap_detected"] is False

        # 6. Save manifests to temporary directory and verify roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_split_manifest(train, val, test, output_dir=tmpdir)
            manifest_info = load_split_manifest(saved_paths["split_manifest"])
            assert manifest_info["total_records"] == 16338
            assert manifest_info["counts"]["train"] == 13070
            assert manifest_info["counts"]["val"] == 1634
            assert manifest_info["counts"]["test"] == 1634
