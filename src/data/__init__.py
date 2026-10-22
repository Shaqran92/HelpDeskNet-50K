"""Data ingestion, validation, and splitting module for HelpDeskNet-50K.

Provides functionality to load raw customer-support datasets, validate schema consistency,
filter languages, construct reproducible ticket identifiers, audit missing values and duplicates,
and construct leakage-free, stratified splits for training, validation, and untouched testing.
"""

from __future__ import annotations

from src.data.load_data import (
    create_ticket_id,
    filter_by_language,
    inspect_schema,
    load_raw_dataset,
    select_records,
)
from src.data.split_data import (
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
    validate_split_integrity,
)
from src.data.validate_data import (
    compute_class_frequencies,
    detect_duplicates,
    detect_near_duplicates,
    generate_data_audit_report,
    validate_labels,
    validate_no_empty_text,
    validate_required_columns,
)

__all__ = [
    # Data loading
    "load_raw_dataset",
    "inspect_schema",
    "filter_by_language",
    "select_records",
    "create_ticket_id",
    # Data validation
    "validate_required_columns",
    "validate_no_empty_text",
    "validate_labels",
    "detect_duplicates",
    "detect_near_duplicates",
    "generate_data_audit_report",
    "compute_class_frequencies",
    # Data splitting
    "create_stratified_split",
    "validate_split_integrity",
    "save_split_manifest",
    "load_split_manifest",
]
