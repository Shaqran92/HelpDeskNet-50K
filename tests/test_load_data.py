"""Unit and integration tests for data loading and schema inspection in HelpDeskNet-50K."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import (
    create_ticket_id,
    filter_by_language,
    inspect_schema,
    load_multi_file_corpus,
    load_raw_dataset,
    select_records,
)

RAW_DATASET_PATH = Path("d:/HacktoberFest/data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
RAW_DATASET_20K_PATH = Path("d:/HacktoberFest/data/raw/dataset-tickets-multi-lang-4-20k.csv")



class TestLoadRawDataset:
    """Tests for load_raw_dataset function."""

    def test_load_existing_raw_dataset(self):
        """Verify that the official raw dataset loads successfully with correct dimensions."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 28587
        assert len(df.columns) == 16
        assert "body" in df.columns
        assert "language" in df.columns
        assert "queue" in df.columns

    def test_load_non_existent_file_raises_filenotfound(self):
        """Verify FileNotFoundError is raised for non-existent file path."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_raw_dataset("d:/HacktoberFest/data/raw/does_not_exist.csv")

    def test_load_invalid_type_raises_typeerror(self):
        """Verify TypeError is raised if filepath is not a string or Path."""
        with pytest.raises(TypeError, match="filepath must be a str or Path"):
            load_raw_dataset(12345)  # type: ignore

    def test_load_empty_file_raises_valueerror(self):
        """Verify ValueError is raised if file is empty."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="empty"):
                load_raw_dataset(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestInspectSchema:
    """Tests for inspect_schema function."""

    def test_inspect_schema_raw_dataset(self):
        """Verify schema inspection matches known facts about the raw dataset."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        schema = inspect_schema(df)

        assert schema["row_count"] == 28587
        assert schema["column_count"] == 16
        assert schema["missing_counts"]["subject"] == 3838
        assert schema["missing_counts"]["body"] == 0
        assert schema["missing_counts"]["answer"] == 7
        assert schema["duplicate_rows_count"] == 0
        assert schema["duplicate_bodies_count"] == 0
        assert schema["memory_usage_mb"] > 0

    def test_inspect_schema_invalid_type_raises_typeerror(self):
        """Verify TypeError when non-DataFrame is passed."""
        with pytest.raises(TypeError, match="Expected pd.DataFrame"):
            inspect_schema([1, 2, 3])  # type: ignore


class TestFilterByLanguage:
    """Tests for filter_by_language function."""

    def test_filter_en_language(self):
        """Verify filtering by English yields exactly 16,338 records."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        en_df = filter_by_language(df, language="en")
        assert len(en_df) == 16338
        assert (en_df["language"] == "en").all()
        # Verify index was reset
        assert list(en_df.index) == list(range(16338))

    def test_filter_de_language(self):
        """Verify filtering by German yields exactly 12,249 records."""
        if not RAW_DATASET_PATH.exists():
            pytest.skip(f"Dataset file not found at {RAW_DATASET_PATH}")

        df = load_raw_dataset(RAW_DATASET_PATH)
        de_df = filter_by_language(df, language="de")
        assert len(de_df) == 12249
        assert (de_df["language"] == "de").all()

    def test_filter_case_insensitive(self):
        """Verify language filtering is case-insensitive."""
        sample_df = pd.DataFrame({"language": ["en", "EN", "De", "es"], "body": ["a", "b", "c", "d"]})
        result = filter_by_language(sample_df, "EN")
        assert len(result) == 2

    def test_filter_missing_language_column_raises_keyerror(self):
        """Verify KeyError when language column is absent."""
        sample_df = pd.DataFrame({"body": ["a", "b"]})
        with pytest.raises(KeyError, match="Required column 'language' not found"):
            filter_by_language(sample_df, "en")


class TestSelectRecords:
    """Tests for select_records deterministic sampling."""

    def test_select_records_fewer_than_max(self):
        """When total records <= max_records, all records are returned without dropping."""
        sample_df = pd.DataFrame({"body": [f"text {i}" for i in range(100)]})
        result = select_records(sample_df, max_records=50000)
        assert len(result) == 100

    def test_select_records_more_than_max(self):
        """When total records > max_records, exact max_records are selected deterministically."""
        sample_df = pd.DataFrame({"body": [f"text {i}" for i in range(500)]})
        res1 = select_records(sample_df, max_records=100, seed=42)
        res2 = select_records(sample_df, max_records=100, seed=42)
        res3 = select_records(sample_df, max_records=100, seed=999)

        assert len(res1) == 100
        assert len(res2) == 100
        # Determinism check
        assert res1["body"].tolist() == res2["body"].tolist()
        # Seed variation check
        assert res1["body"].tolist() != res3["body"].tolist()

    def test_select_records_invalid_args_raises_error(self):
        """Verify argument validation."""
        sample_df = pd.DataFrame({"body": ["a", "b"]})
        with pytest.raises(ValueError, match="max_records must be a positive integer"):
            select_records(sample_df, max_records=0)
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            select_records(sample_df, max_records=10, seed=-1)


class TestCreateTicketId:
    """Tests for create_ticket_id function."""

    def test_create_ticket_id_adds_column_at_start(self):
        """Verify deterministic IDs are added as the first column."""
        sample_df = pd.DataFrame({"body": ["issue 1", "issue 2", "issue 3"], "queue": ["IT", "HR", "Sales"]})
        res = create_ticket_id(sample_df, prefix="TICKET")

        assert "ticket_id" in res.columns
        assert res.columns[0] == "ticket_id"
        assert res["ticket_id"].tolist() == ["TICKET-000001", "TICKET-000002", "TICKET-000003"]
        assert len(res) == 3
        # Original DataFrame not mutated
        assert "ticket_id" not in sample_df.columns

    def test_create_ticket_id_uniqueness(self):
        """Verify all generated ticket IDs are unique."""
        sample_df = pd.DataFrame({"body": [f"ticket {i}" for i in range(1000)]})
        res = create_ticket_id(sample_df)
        assert res["ticket_id"].nunique() == 1000


class TestLoadMultiFileCorpus:
    """Tests for load_multi_file_corpus function."""

    def test_load_multi_file_empty_list_raises(self):
        """Empty list of filepaths raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            load_multi_file_corpus([])

    def test_load_multi_file_invalid_type_raises(self):
        """Non-list filepaths raises TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            load_multi_file_corpus("some/path.csv")  # type: ignore

    def test_load_multi_file_combines_and_deduplicates(self):
        """Verify combining mock datasets and deduplicating on body."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f1:
            f1.write("body,queue,language\nTicket 1,IT,en\nTicket 2,HR,en\n")
            p1 = Path(f1.name)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f2:
            f2.write("body,queue,language\nTicket 2,HR,en\nTicket 3,Sales,en\n")
            p2 = Path(f2.name)

        try:
            combined = load_multi_file_corpus([p1, p2], deduplicate_on="body")
            assert len(combined) == 3
            assert set(combined["body"]) == {"Ticket 1", "Ticket 2", "Ticket 3"}

            combined_no_dedup = load_multi_file_corpus([p1, p2], deduplicate_on=None)
            assert len(combined_no_dedup) == 4
        finally:
            p1.unlink(missing_ok=True)
            p2.unlink(missing_ok=True)

    def test_load_raw_slices_if_present(self):
        """Integration test on actual raw slices if available."""
        if not RAW_DATASET_PATH.exists() or not RAW_DATASET_20K_PATH.exists():
            pytest.skip("Raw files not present")

        combined = load_multi_file_corpus([RAW_DATASET_PATH, RAW_DATASET_20K_PATH], deduplicate_on="body")
        # Slice 1 (28587) + Slice 2 (20000) - 8399 overlap = 40186
        assert len(combined) == 40186
        assert "body" in combined.columns
        assert "queue" in combined.columns

