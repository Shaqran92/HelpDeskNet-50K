"""Data validation, audit, and quality verification module for HelpDeskNet-50K.

Provides functionality for checking column completeness, validating text integrity,
verifying label values, detecting exact and near duplicates, computing class
frequencies, and exporting comprehensive data audit reports.

Core principles:
- Transparent and reproducible data validation.
- Separation of source labels, reference labels, and noisy labels.
- Never assume source categories or labels represent verified ground truth.
- Robust handling of edge cases and missing values.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
) -> list[str]:
    """Validate that all required columns are present in the DataFrame.

    Args:
        df: Input pandas DataFrame to validate.
        required_cols: List of column names expected to be present.

    Returns:
        list[str]: List of column names that are missing from the DataFrame.
            Returns an empty list if all required columns are present.

    Raises:
        TypeError: If df is not a DataFrame or required_cols is not a list.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(required_cols, (list, tuple, set)):
        raise TypeError(f"required_cols must be a list/sequence of strings, got {type(required_cols).__name__}")

    current_cols = set(df.columns)
    missing = [col for col in required_cols if col not in current_cols]

    if missing:
        logger.warning(
            "DataFrame is missing %d required column(s): %s (available: %s)",
            len(missing),
            missing,
            list(df.columns),
        )
    else:
        logger.info(
            "All %d required columns are present in the DataFrame.",
            len(required_cols),
        )

    return missing


def validate_no_empty_text(
    df: pd.DataFrame,
    text_col: str = "body",
) -> pd.DataFrame:
    """Identify rows where the specified text column contains null, empty, or whitespace-only text.

    Args:
        df: Input pandas DataFrame.
        text_col: Name of the text column to validate (default: 'body').

    Returns:
        pd.DataFrame: A subset DataFrame of rows containing null, empty, or whitespace-only text.

    Raises:
        TypeError: If df is not a DataFrame or text_col is not a string.
        KeyError: If text_col is not present in df.columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(text_col, str):
        raise TypeError(f"text_col must be a string, got {type(text_col).__name__}")
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    series = df[text_col]
    is_null = series.isna()
    is_empty_or_whitespace = series.astype(str).str.strip() == ""
    empty_mask = is_null | is_empty_or_whitespace

    empty_rows = df[empty_mask].copy()

    total_rows = len(df)
    empty_count = len(empty_rows)
    empty_pct = (empty_count / total_rows * 100) if total_rows > 0 else 0.0

    if empty_count > 0:
        logger.warning(
            "Found %d rows (%.2f%%) with null or empty text in column '%s'.",
            empty_count,
            empty_pct,
            text_col,
        )
    else:
        logger.info("Validation passed: No null or empty text found in column '%s'.", text_col)

    return empty_rows


def validate_labels(
    df: pd.DataFrame,
    label_col: str,
    valid_labels: list[str],
) -> pd.DataFrame:
    """Identify rows where the label column contains values not in the allowed valid_labels list.

    NOTE: Source labels provided by raw datasets must not be treated as ground truth.
    This validation ensures category values adhere to a known dictionary/schema.

    Args:
        df: Input pandas DataFrame.
        label_col: Name of the label column to validate.
        valid_labels: List of allowed valid label strings.

    Returns:
        pd.DataFrame: A subset DataFrame of rows with missing, null, or invalid labels.

    Raises:
        TypeError: If df is not a DataFrame, label_col is not a string, or valid_labels is not a sequence.
        KeyError: If label_col is not present in df.columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(label_col, str):
        raise TypeError(f"label_col must be a string, got {type(label_col).__name__}")
    if not isinstance(valid_labels, (list, tuple, set)):
        raise TypeError(f"valid_labels must be a list/sequence of strings, got {type(valid_labels).__name__}")
    if label_col not in df.columns:
        raise KeyError(
            f"Column '{label_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    valid_set = set(valid_labels)
    series = df[label_col]
    invalid_mask = series.isna() | (~series.isin(valid_set))

    invalid_rows = df[invalid_mask].copy()

    total_rows = len(df)
    invalid_count = len(invalid_rows)
    invalid_pct = (invalid_count / total_rows * 100) if total_rows > 0 else 0.0

    if invalid_count > 0:
        encountered_invalid = series[invalid_mask].dropna().unique().tolist()
        logger.warning(
            "Found %d rows (%.2f%%) with invalid labels in column '%s'. "
            "Invalid values encountered: %s",
            invalid_count,
            invalid_pct,
            label_col,
            encountered_invalid[:10],
        )
    else:
        logger.info(
            "Validation passed: All labels in column '%s' match the valid label set.",
            label_col,
        )

    return invalid_rows


def detect_duplicates(
    df: pd.DataFrame,
    text_col: str = "body",
) -> pd.DataFrame:
    """Identify all rows that have duplicate text in the specified column.

    Returns all occurrences of duplicated text values (including original and copies)
    to facilitate investigation and deduplication audits.

    Args:
        df: Input pandas DataFrame.
        text_col: Name of the text column to inspect for duplicates (default: 'body').

    Returns:
        pd.DataFrame: DataFrame containing all duplicate rows, sorted by text_col.

    Raises:
        TypeError: If df is not a DataFrame or text_col is not a string.
        KeyError: If text_col is not present in df.columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(text_col, str):
        raise TypeError(f"text_col must be a string, got {type(text_col).__name__}")
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    dup_mask = df.duplicated(subset=[text_col], keep=False)
    duplicate_rows = df[dup_mask].sort_values(by=text_col).copy()

    total_rows = len(df)
    dup_count = len(duplicate_rows)
    unique_dup_texts = duplicate_rows[text_col].nunique()

    logger.info(
        "Duplicate check on column '%s': %d duplicate rows found across %d unique text strings (total rows: %d)",
        text_col,
        dup_count,
        unique_dup_texts,
        total_rows,
    )

    return duplicate_rows


def detect_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "body",
    threshold: float = 0.95,
    n_chars: int | None = None,
) -> pd.DataFrame:
    """Detect near-duplicate ticket pairs using a prefix-blocking heuristic and similarity verification.

    Heuristic mechanism:
    1. Customer support tickets often share boilerplate templates or greetings. To avoid
       an O(N^2) pairwise comparison on tens of thousands of tickets, records are first
       grouped by their first N characters (prefix blocking).
    2. Within each prefix group containing 2 or more records, pairwise text similarity
       is evaluated using difflib.SequenceMatcher.
    3. Pairs meeting or exceeding the similarity threshold are returned.

    Args:
        df: Input pandas DataFrame.
        text_col: Column containing ticket body text (default: 'body').
        threshold: Text similarity threshold between 0.0 and 1.0 (default: 0.95).
            If threshold > 1.0, it is interpreted as the exact character prefix length N.
        n_chars: Optional explicit prefix length N for candidate blocking. If None,
            derived from threshold (e.g. max(10, int(100 * threshold))).

    Returns:
        pd.DataFrame: DataFrame of near-duplicate pairs with columns:
            - 'id_1': Ticket identifier or index for first record
            - 'id_2': Ticket identifier or index for second record
            - 'index_1': Original row index for first record
            - 'index_2': Original row index for second record
            - 'similarity': Float similarity score between 0.0 and 1.0
            - 'text_1': Truncated body text of first record
            - 'text_2': Truncated body text of second record

    Raises:
        TypeError: If df is not a DataFrame, text_col is not a string, or threshold is invalid.
        KeyError: If text_col is not present in df.columns.
        ValueError: If threshold is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(text_col, str):
        raise TypeError(f"text_col must be a string, got {type(text_col).__name__}")
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a float or int, got {type(threshold).__name__}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    columns = ["id_1", "id_2", "index_1", "index_2", "similarity", "text_1", "text_2"]
    if df.empty or len(df) < 2:
        logger.info("DataFrame has fewer than 2 rows; returning empty near-duplicates DataFrame.")
        return pd.DataFrame(columns=columns)

    # Determine prefix blocking length
    if n_chars is not None:
        prefix_len = max(5, int(n_chars))
    elif threshold > 1.0:
        prefix_len = int(threshold)
    else:
        prefix_len = max(10, int(100 * threshold))

    sim_cutoff = float(threshold) if threshold <= 1.0 else 0.0

    has_ticket_id = "ticket_id" in df.columns
    clean_series = df[text_col].fillna("").astype(str).str.strip()
    prefixes = clean_series.str[:prefix_len]

    # Find prefixes that occur more than once
    prefix_counts = prefixes.value_counts()
    dup_prefixes = prefix_counts[prefix_counts > 1].index

    pairs: list[dict[str, Any]] = []

    for prefix in dup_prefixes:
        # Skip empty prefixes
        if not prefix:
            continue
        group_indices = df[prefixes == prefix].index.tolist()
        group_len = len(group_indices)

        for i in range(group_len):
            idx1 = group_indices[i]
            text1 = clean_series.loc[idx1]

            for j in range(i + 1, group_len):
                idx2 = group_indices[j]
                text2 = clean_series.loc[idx2]

                sim = SequenceMatcher(None, text1, text2).ratio()
                if sim >= sim_cutoff:
                    id_1 = str(df.loc[idx1, "ticket_id"]) if has_ticket_id else str(idx1)
                    id_2 = str(df.loc[idx2, "ticket_id"]) if has_ticket_id else str(idx2)

                    pairs.append(
                        {
                            "id_1": id_1,
                            "id_2": id_2,
                            "index_1": idx1,
                            "index_2": idx2,
                            "similarity": round(float(sim), 4),
                            "text_1": text1[:200],
                            "text_2": text2[:200],
                        }
                    )

    result_df = pd.DataFrame(pairs, columns=columns)
    logger.info(
        "Near-duplicate detection (prefix_len=%d, cutoff=%.2f): identified %d candidate pairs.",
        prefix_len,
        sim_cutoff,
        len(result_df),
    )
    return result_df


def compute_class_frequencies(
    df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Compute class frequency distributions, counts, and percentages for a label column.

    Args:
        df: Input pandas DataFrame.
        label_col: Name of the label/category column to analyze.

    Returns:
        pd.DataFrame: Frequency table with columns:
            - label_col: Category name (or '<MISSING>' for nulls)
            - 'count': Number of records in category
            - 'percentage': Percentage of total records (0.0 - 100.0)

    Raises:
        TypeError: If df is not a DataFrame or label_col is not a string.
        KeyError: If label_col is not in df.columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(label_col, str):
        raise TypeError(f"label_col must be a string, got {type(label_col).__name__}")
    if label_col not in df.columns:
        raise KeyError(
            f"Column '{label_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    series = df[label_col].fillna("<MISSING>")
    counts = series.value_counts(dropna=False)
    percentages = series.value_counts(normalize=True, dropna=False) * 100

    freq_df = pd.DataFrame(
        {
            label_col: counts.index.astype(str),
            "count": counts.values,
            "percentage": percentages.round(4).values,
        }
    )

    logger.info(
        "Computed class frequencies for column '%s': %d distinct classes across %d rows.",
        label_col,
        len(freq_df),
        len(df),
    )
    return freq_df


def generate_data_audit_report(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive data audit report and optionally save it to a JSON file.

    The report encapsulates row and column counts, data types, missing value analysis,
    duplicate counts, language distributions, category/queue frequencies, and text
    length statistics.

    Args:
        df: Input pandas DataFrame to audit.
        output_path: Optional file path to write the JSON report to.

    Returns:
        dict[str, Any]: Complete audit report dictionary.

    Raises:
        TypeError: If df is not a DataFrame.
        IOError: If output_path cannot be written.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    total_rows = int(len(df))
    total_cols = int(len(df.columns))

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_percentages = {
        col: round(float((df[col].isna().sum() / total_rows) * 100), 2) if total_rows > 0 else 0.0
        for col in df.columns
    }

    # Text length metrics if 'body' exists
    text_stats: dict[str, Any] = {}
    if "body" in df.columns:
        body_lens = df["body"].dropna().astype(str).str.len()
        body_words = df["body"].dropna().astype(str).str.split().str.len()
        if not body_lens.empty:
            text_stats = {
                "char_length_min": int(body_lens.min()),
                "char_length_max": int(body_lens.max()),
                "char_length_mean": round(float(body_lens.mean()), 2),
                "char_length_median": round(float(body_lens.median()), 2),
                "char_length_std": round(float(body_lens.std()), 2),
                "word_count_min": int(body_words.min()),
                "word_count_max": int(body_words.max()),
                "word_count_mean": round(float(body_words.mean()), 2),
                "word_count_median": round(float(body_words.median()), 2),
                "word_count_std": round(float(body_words.std()), 2),
            }

    # Categorical distributions
    languages = (
        {str(k): int(v) for k, v in df["language"].value_counts(dropna=False).items()}
        if "language" in df.columns
        else {}
    )
    queue_dist = (
        {str(k): int(v) for k, v in df["queue"].value_counts(dropna=False).items()}
        if "queue" in df.columns
        else {}
    )
    priority_dist = (
        {str(k): int(v) for k, v in df["priority"].value_counts(dropna=False).items()}
        if "priority" in df.columns
        else {}
    )
    type_dist = (
        {str(k): int(v) for k, v in df["type"].value_counts(dropna=False).items()}
        if "type" in df.columns
        else {}
    )

    report: dict[str, Any] = {
        "report_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline_stage": "Phase 2 - Data Acquisition & Audit",
            "dataset_name": "HelpDeskNet-50K",
        },
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2)
            if total_rows > 0
            else 0.0,
        },
        "missingness": {
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages,
            "columns_with_missing_values": [col for col, count in missing_counts.items() if count > 0],
        },
        "duplicates": {
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "duplicate_bodies": int(df["body"].duplicated().sum()) if "body" in df.columns else None,
            "duplicate_subjects": int(df["subject"].duplicated().sum()) if "subject" in df.columns else None,
        },
        "distributions": {
            "languages": languages,
            "queues": queue_dist,
            "priorities": priority_dist,
            "types": type_dist,
        },
        "text_statistics": text_stats,
        "integrity_checks": {
            "has_empty_bodies": int(df["body"].isna().sum()) > 0 if "body" in df.columns else False,
            "no_duplicate_bodies": bool(df["body"].duplicated().sum() == 0) if "body" in df.columns else True,
            "missing_subjects_count": int(df["subject"].isna().sum()) if "subject" in df.columns else 0,
        },
    }

    if output_path is not None:
        out_file = Path(output_path)
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Comprehensive data audit report saved successfully to %s", out_file.resolve())
        except Exception as exc:
            logger.error("Failed to write data audit report to %s: %s", out_file, exc)
            raise IOError(f"Failed to write data audit report to {out_file}: {exc}") from exc

    return report
