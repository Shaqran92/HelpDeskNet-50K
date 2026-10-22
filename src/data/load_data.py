"""Data loading and ingestion module for HelpDeskNet-50K.

Provides functionality for loading raw customer-support datasets, inspecting schema,
filtering by language, deterministically sampling records up to a target size,
and generating reproducible ticket identifiers.

Core principles:
- Deterministic execution with fixed seeds.
- Clear separation between dataset record identifiers and customer identities
  (ticket_ids track records through the pipeline, never claiming to represent real users).
- Never treat source labels as ground truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_dataset(filepath: str | Path) -> pd.DataFrame:
    """Load a raw customer-support ticket dataset from a CSV file.

    Validates that the file exists and is accessible before reading into a pandas
    DataFrame. Logs row and column counts upon successful ingestion.

    Args:
        filepath: Path to the raw CSV dataset file.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        TypeError: If filepath is not a string or Path object.
        FileNotFoundError: If the specified file does not exist or is a directory.
        ValueError: If the file is empty or cannot be parsed as a CSV.
        RuntimeError: If an unexpected error occurs during file reading.
    """
    if not isinstance(filepath, (str, Path)):
        raise TypeError(f"filepath must be a str or Path, got {type(filepath).__name__}")

    path_obj = Path(filepath)

    if not path_obj.exists():
        logger.error("Dataset file not found: %s", path_obj.resolve())
        raise FileNotFoundError(f"Dataset file not found: {path_obj.resolve()}")

    if not path_obj.is_file():
        logger.error("Path is not a regular file: %s", path_obj.resolve())
        raise FileNotFoundError(f"Path is not a regular file: {path_obj.resolve()}")

    logger.info("Loading raw dataset from %s", path_obj.resolve())

    try:
        df = pd.read_csv(path_obj)
    except pd.errors.EmptyDataError as exc:
        logger.error("File is empty: %s", path_obj)
        raise ValueError(f"Dataset file is empty: {path_obj}") from exc
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse CSV file %s: %s", path_obj, exc)
        raise ValueError(f"Failed to parse CSV file: {path_obj}") from exc
    except Exception as exc:
        logger.error("Unexpected error loading dataset from %s: %s", path_obj, exc)
        raise RuntimeError(f"Unexpected error loading dataset from {path_obj}: {exc}") from exc

    if df.empty:
        logger.warning("Dataset at %s was loaded but contains 0 rows.", path_obj)

    logger.info(
        "Successfully loaded raw dataset: %d rows, %d columns from %s",
        len(df),
        len(df.columns),
        path_obj.name,
    )
    return df


def inspect_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect the schema, missing values, duplicates, and memory usage of a DataFrame.

    Generates a structured dictionary containing column names, data types, row/column
    counts, null value counts, null percentages, and duplicate counts.

    Args:
        df: Input pandas DataFrame to inspect.

    Returns:
        dict[str, Any]: Dictionary containing schema inspection details:
            - 'row_count': Total number of rows
            - 'column_count': Total number of columns
            - 'columns': List of column names
            - 'dtypes': Mapping of column name to data type string
            - 'missing_counts': Mapping of column name to null count
            - 'missing_percentages': Mapping of column name to null percentage (0-100)
            - 'duplicate_rows_count': Total duplicate rows across all columns
            - 'duplicate_bodies_count': Duplicate count in 'body' column (if present)
            - 'memory_usage_mb': Total memory consumption in megabytes

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    row_count = int(len(df))
    column_count = int(len(df.columns))
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_percentages = {
        col: round(float((df[col].isna().sum() / row_count) * 100), 2) if row_count > 0 else 0.0
        for col in df.columns
    }
    duplicate_rows_count = int(df.duplicated().sum())
    duplicate_bodies_count = (
        int(df["body"].duplicated().sum()) if "body" in df.columns else None
    )

    try:
        memory_usage_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
    except Exception:
        memory_usage_mb = round(float(df.memory_usage().sum() / (1024 * 1024)), 2)

    logger.info(
        "Schema inspection: %d rows, %d columns, %d duplicate rows, memory: %.2f MB",
        row_count,
        column_count,
        duplicate_rows_count,
        memory_usage_mb,
    )

    return {
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_rows_count": duplicate_rows_count,
        "duplicate_bodies_count": duplicate_bodies_count,
        "memory_usage_mb": memory_usage_mb,
    }


def filter_by_language(df: pd.DataFrame, language: str = "en") -> pd.DataFrame:
    """Filter dataset records to a specified language.

    Performs case-insensitive matching against the 'language' column.
    Returns a fresh DataFrame copy with a clean RangeIndex.

    Args:
        df: Input pandas DataFrame containing a 'language' column.
        language: Language code to filter by (default: 'en').

    Returns:
        pd.DataFrame: Filtered DataFrame containing only records in the specified language.

    Raises:
        TypeError: If df is not a DataFrame or language is not a string.
        ValueError: If language is an empty or whitespace string.
        KeyError: If the 'language' column is missing from the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if language is None or (isinstance(language, str) and language.strip().lower() in ("all", "*", "none")):
        logger.info("Language filter disabled ('%s'): retaining all %d records.", language, len(df))
        return df.copy().reset_index(drop=True)
    if not isinstance(language, str):
        raise TypeError(f"language must be a string, got {type(language).__name__}")
    if not language.strip():
        raise ValueError("language parameter cannot be empty or whitespace")

    if "language" not in df.columns:
        logger.error("Required column 'language' not found in DataFrame columns: %s", list(df.columns))
        raise KeyError(
            f"Required column 'language' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    target_lang = language.strip().lower()
    mask = df["language"].astype(str).str.strip().str.lower() == target_lang
    filtered_df = df.loc[mask].copy().reset_index(drop=True)

    initial_count = len(df)
    filtered_count = len(filtered_df)
    retained_pct = (filtered_count / initial_count * 100) if initial_count > 0 else 0.0

    logger.info(
        "Filtered dataset by language='%s': %d/%d records retained (%.2f%%)",
        language,
        filtered_count,
        initial_count,
        retained_pct,
    )

    if filtered_count == 0 and initial_count > 0:
        available_languages = df["language"].dropna().unique().tolist()
        logger.warning(
            "Filter for language='%s' returned 0 records. Available languages in dataset: %s",
            language,
            available_languages,
        )

    return filtered_df


def select_records(
    df: pd.DataFrame,
    max_records: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Select up to max_records with documented deterministic sampling logic.

    Selection logic:
    1. If the DataFrame contains fewer than or equal to max_records rows, all records
       are preserved, and a warning is logged indicating that the available count is
       below the target threshold.
    2. If the DataFrame contains more than max_records rows, a deterministic uniform
       random sample of exactly max_records is selected using the fixed seed.
    3. The resulting DataFrame index is reset to a clean RangeIndex.

    Args:
        df: Input pandas DataFrame.
        max_records: Maximum number of records to select (default: 50,000).
        seed: Random seed for deterministic sampling (default: 42).

    Returns:
        pd.DataFrame: Selected subset of records.

    Raises:
        TypeError: If df is not a DataFrame, or max_records/seed are not integers.
        ValueError: If max_records <= 0 or seed < 0.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(max_records, int):
        raise TypeError(f"max_records must be an integer, got {type(max_records).__name__}")
    if max_records <= 0:
        raise ValueError(f"max_records must be a positive integer, got {max_records}")
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")

    total_available = len(df)

    if total_available <= max_records:
        logger.warning(
            "Requested max_records=%d, but dataset only contains %d records. "
            "Selecting all %d available records without sampling.",
            max_records,
            total_available,
            total_available,
        )
        return df.copy().reset_index(drop=True)

    logger.info(
        "Sampling %d records from %d available records using seed=%d",
        max_records,
        total_available,
        seed,
    )
    sampled_df = df.sample(n=max_records, random_state=seed).reset_index(drop=True)
    return sampled_df


def create_ticket_id(
    df: pd.DataFrame,
    prefix: str = "TICKET",
    id_col: str = "ticket_id",
) -> pd.DataFrame:
    """Add a deterministic ticket_id column to the DataFrame.

    IMPORTANT GROUND TRUTH & PROVENANCE NOTICE:
    Generated ticket identifiers (e.g. 'TICKET-000001') are deterministic synthetic
    keys used solely for tracking records through the preprocessing, validation,
    splitting, and auditing pipelines.
    - These IDs DO NOT represent customer IDs, account numbers, or unique real users.
    - Generating ticket IDs does NOT establish ground truth.
    - Source labels must remain segregated from reference or observed labels.

    Args:
        df: Input pandas DataFrame.
        prefix: Prefix string for generated IDs (default: 'TICKET').
        id_col: Name of the ID column to create (default: 'ticket_id').

    Returns:
        pd.DataFrame: DataFrame copy with the deterministic ID column inserted as
            the first column.

    Raises:
        TypeError: If df is not a DataFrame or prefix/id_col are not strings.
        ValueError: If prefix or id_col are empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string")
    if not isinstance(id_col, str) or not id_col.strip():
        raise ValueError("id_col must be a non-empty string")

    df_out = df.copy()

    if id_col in df_out.columns:
        logger.warning(
            "Column '%s' already exists in DataFrame and will be overwritten with deterministic IDs.",
            id_col,
        )

    # Format sequential 1-based index with zero padding to 6 digits
    generated_ids = [f"{prefix}-{i + 1:06d}" for i in range(len(df_out))]
    df_out[id_col] = generated_ids

    # Place the ticket_id column at the first position
    columns_order = [id_col] + [col for col in df_out.columns if col != id_col]
    df_out = df_out[columns_order]

    logger.info(
        "Added deterministic ticket ID column '%s' to %d records (range: %s to %s)",
        id_col,
        len(df_out),
        generated_ids[0] if generated_ids else "N/A",
        generated_ids[-1] if generated_ids else "N/A",
    )
    return df_out


def load_multi_file_corpus(
    filepaths: list[str | Path],
    deduplicate_on: str | list[str] | None = "body",
    drop_na_text: bool = True,
    text_col: str = "body",
) -> pd.DataFrame:
    """Load and combine multiple raw customer-support datasets into a unified DataFrame.

    Validates all files, concatenates them, logs individual and combined statistics,
    optionally removes rows with null/empty text, and deduplicates based on the specified
    column(s) (preserving the first occurrence).

    Args:
        filepaths: List of file paths to raw CSV files.
        deduplicate_on: Column name or list of column names on which to deduplicate.
            Pass None to skip deduplication. Defaults to 'body'.
        drop_na_text: Whether to drop rows where text_col is null or empty whitespace.
            Defaults to True.
        text_col: The primary text column to inspect for nulls. Defaults to 'body'.

    Returns:
        pd.DataFrame: Combined, cleaned, and optionally deduplicated DataFrame.

    Raises:
        TypeError: If filepaths is not a list.
        ValueError: If filepaths is empty.
        FileNotFoundError: If any file in filepaths does not exist.
    """
    if not isinstance(filepaths, list):
        raise TypeError(f"filepaths must be a list of paths, got {type(filepaths).__name__}")
    if not filepaths:
        raise ValueError("filepaths list cannot be empty")

    dfs = []
    total_raw_rows = 0
    for fp in filepaths:
        df_part = load_raw_dataset(fp)
        dfs.append(df_part)
        total_raw_rows += len(df_part)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info("Combined %d files: %d total rows before cleaning/deduplication", len(filepaths), total_raw_rows)

    if drop_na_text and text_col in combined_df.columns:
        valid_mask = combined_df[text_col].notna() & (combined_df[text_col].astype(str).str.strip() != "")
        n_dropped = len(combined_df) - int(valid_mask.sum())
        if n_dropped > 0:
            logger.info("Dropped %d rows with null or whitespace-only '%s'", n_dropped, text_col)
            combined_df = combined_df.loc[valid_mask].reset_index(drop=True)

    if deduplicate_on is not None:
        if isinstance(deduplicate_on, str) and deduplicate_on in combined_df.columns:
            n_before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=[deduplicate_on], keep="first").reset_index(drop=True)
            logger.info(
                "Deduplicated on '%s': removed %d duplicates, %d unique records retained",
                deduplicate_on,
                n_before - len(combined_df),
                len(combined_df),
            )
        elif isinstance(deduplicate_on, list) and all(c in combined_df.columns for c in deduplicate_on):
            n_before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=deduplicate_on, keep="first").reset_index(drop=True)
            logger.info(
                "Deduplicated on %s: removed %d duplicates, %d unique records retained",
                deduplicate_on,
                n_before - len(combined_df),
                len(combined_df),
            )

    return combined_df


