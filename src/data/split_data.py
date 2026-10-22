"""Data splitting and split integrity validation module for HelpDeskNet-50K.

Provides functionality for creating leakage-free, stratified train/val/test splits,
verifying split integrity (ensuring disjoint partitions with zero record leakage),
and saving/loading split manifests for reproducible experiments.

Core principles:
- Fixed random seeds for strict experimental reproducibility.
- Zero test-set leakage: the test partition is strictly held out and untouched.
- Deterministic manifests enable tracking records and reproducing exact splits
  across all downstream modeling and error analysis stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    rare_class_handling: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits from a DataFrame.

    Uses sklearn.model_selection.train_test_split in a two-stage hierarchical process:
    1. First split separates the untouched test set from the train+val pool based on test_size.
    2. Second split partitions the remaining pool into train and validation sets,
       maintaining exact proportional stratification across all three subsets.

    Args:
        df: Input pandas DataFrame.
        label_col: Name of the categorical column to stratify on.
        train_size: Proportion of records allocated to training (default: 0.8).
        val_size: Proportion of records allocated to validation (default: 0.1).
        test_size: Proportion of records allocated to untouched test (default: 0.1).
        seed: Random seed for reproducible splitting (default: 42).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of (train_df, val_df, test_df),
            each with reset RangeIndex.

    Raises:
        TypeError: If df is not a DataFrame, label_col is not a string, or split sizes are not floats/ints.
        ValueError: If split proportions do not sum to 1.0, or if sizes are non-positive,
            or if label_col contains missing values or classes with fewer than 2 instances.
        KeyError: If label_col is not in df.columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")
    if not isinstance(label_col, str):
        raise TypeError(f"label_col must be a string, got {type(label_col).__name__}")
    if label_col not in df.columns:
        raise KeyError(
            f"Stratification column '{label_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")

    for name, size in [("train_size", train_size), ("val_size", val_size), ("test_size", test_size)]:
        if not isinstance(size, (int, float)):
            raise TypeError(f"{name} must be a float or int, got {type(size).__name__}")
        if size <= 0:
            raise ValueError(f"{name} must be strictly positive, got {size}")

    total_proportions = train_size + val_size + test_size
    if abs(total_proportions - 1.0) > 1e-5:
        raise ValueError(
            f"Split sizes must sum to 1.0, got train={train_size}, val={val_size}, "
            f"test={test_size} (sum={total_proportions:.6f})"
        )

    # Check for null labels
    null_labels_count = df[label_col].isna().sum()
    if null_labels_count > 0:
        raise ValueError(
            f"Stratification column '{label_col}' contains {null_labels_count} null/missing values. "
            "Stratified splitting requires clean, non-null labels."
        )

    # Check class sample counts
    class_counts = df[label_col].value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2 and rare_class_handling == "error":
        rare_classes = class_counts[class_counts < 2].index.tolist()
        raise ValueError(
            f"Stratification requires at least 2 samples per class. "
            f"The following classes have fewer than 2 samples: {rare_classes}"
        )

    # If rare_class_handling == "train", classes with count < 3 cannot be split across 3 partitions
    if rare_class_handling == "train":
        rare_classes = class_counts[class_counts < 3].index.tolist()
        if rare_classes:
            logger.warning(
                "Classes with fewer than 3 samples detected: %s. "
                "Allocating these rare records directly to the training partition to avoid stratification failure.",
                rare_classes,
            )
            df_rare = df[df[label_col].isin(rare_classes)]
            df_stratifiable = df[~df[label_col].isin(rare_classes)]
        else:
            df_rare = pd.DataFrame(columns=df.columns)
            df_stratifiable = df
    else:
        df_rare = pd.DataFrame(columns=df.columns)
        df_stratifiable = df

    total_rows = len(df_stratifiable)
    logger.info(
        "Creating stratified split for %d records on column '%s' with proportions: "
        "train=%.2f, val=%.2f, test=%.2f (seed=%d)",
        total_rows,
        label_col,
        train_size,
        val_size,
        test_size,
        seed,
    )

    # Stage 1: Partition off untouched test set
    train_val_df, test_df = train_test_split(
        df_stratifiable,
        test_size=test_size,
        random_state=seed,
        stratify=df_stratifiable[label_col],
    )

    # Stage 2: Partition train_val into train and validation sets
    relative_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df[label_col],
    )

    if not df_rare.empty:
        train_df = pd.concat([train_df, df_rare], ignore_index=True)

    # Reset indices
    train_out = train_df.reset_index(drop=True)
    val_out = val_df.reset_index(drop=True)
    test_out = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split completed successfully: train=%d (%.2f%%), val=%d (%.2f%%), test=%d (%.2f%%)",
        len(train_out),
        (len(train_out) / len(df)) * 100,
        len(val_out),
        (len(val_out) / len(df)) * 100,
        len(test_out),
        (len(test_out) / len(df)) * 100,
    )

    return train_out, val_out, test_out


def validate_split_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = "ticket_id",
) -> dict[str, Any]:
    """Validate that split partitions have zero overlap (leakage) and report partition statistics.

    Checks:
    1. Zero intersection among train, validation, and test ticket identifiers.
    2. Sum of partition sizes matches total distinct record count.
    3. Category label distribution across splits to ensure consistency.

    Args:
        train: Training partition DataFrame.
        val: Validation partition DataFrame.
        test: Test partition DataFrame.
        id_col: Column name identifying distinct records (default: 'ticket_id').

    Returns:
        dict[str, Any]: Detailed split integrity report dictionary:
            - 'integrity_passed': True if no leakage detected and sizes align
            - 'total_records': Combined record count
            - 'split_sizes': Mapping of split name to count
            - 'split_proportions': Mapping of split name to proportion
            - 'id_col': Identifier column used
            - 'overlap_detected': Boolean indicating if any ID is shared across splits
            - 'overlaps': Counts of overlapping IDs between partition pairs
            - 'label_distributions': Category distributions per split (if label columns present)

    Raises:
        TypeError: If train, val, or test are not DataFrames, or id_col is not a string.
        KeyError: If id_col is missing from any of the partition DataFrames.
    """
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        if not isinstance(split_df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame for '{name}', got {type(split_df).__name__}")
        if id_col not in split_df.columns:
            raise KeyError(
                f"Identifier column '{id_col}' not found in '{name}' partition. "
                f"Available columns: {list(split_df.columns)}"
            )

    train_ids = set(train[id_col])
    val_ids = set(val[id_col])
    test_ids = set(test[id_col])

    train_val_overlap = len(train_ids & val_ids)
    train_test_overlap = len(train_ids & test_ids)
    val_test_overlap = len(val_ids & test_ids)

    has_overlap = (train_val_overlap > 0) or (train_test_overlap > 0) or (val_test_overlap > 0)

    total_records = len(train) + len(val) + len(test)
    distinct_ids = len(train_ids | val_ids | test_ids)
    sizes_match = total_records == distinct_ids

    integrity_passed = (not has_overlap) and sizes_match

    split_sizes = {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
    }
    split_proportions = {
        "train": round(len(train) / total_records, 4) if total_records > 0 else 0.0,
        "val": round(len(val) / total_records, 4) if total_records > 0 else 0.0,
        "test": round(len(test) / total_records, 4) if total_records > 0 else 0.0,
    }

    # Inspect label distributions if a candidate label column exists
    candidate_label_cols = ["queue", "source_label", "mapped_label", "category"]
    label_col_found = None
    for cand in candidate_label_cols:
        if cand in train.columns and cand in val.columns and cand in test.columns:
            label_col_found = cand
            break

    label_distributions: dict[str, dict[str, int]] = {}
    if label_col_found:
        label_distributions = {
            "train": {str(k): int(v) for k, v in train[label_col_found].value_counts().items()},
            "val": {str(k): int(v) for k, v in val[label_col_found].value_counts().items()},
            "test": {str(k): int(v) for k, v in test[label_col_found].value_counts().items()},
        }

    report: dict[str, Any] = {
        "integrity_passed": integrity_passed,
        "total_records": total_records,
        "distinct_ids": distinct_ids,
        "split_sizes": split_sizes,
        "split_proportions": split_proportions,
        "id_col": id_col,
        "overlap_detected": has_overlap,
        "overlaps": {
            "train_val": train_val_overlap,
            "train_test": train_test_overlap,
            "val_test": val_test_overlap,
        },
        "label_column_checked": label_col_found,
        "label_distributions": label_distributions,
    }

    if has_overlap:
        logger.error(
            "CRITICAL DATA LEAKAGE: Overlapping IDs detected across splits! "
            "train-val: %d, train-test: %d, val-test: %d",
            train_val_overlap,
            train_test_overlap,
            val_test_overlap,
        )
    else:
        logger.info(
            "Split integrity check PASSED: Zero overlap across train (%d), val (%d), test (%d).",
            len(train),
            len(val),
            len(test),
        )

    return report


def save_split_manifest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
    id_col: str = "ticket_id",
) -> dict[str, str]:
    """Save split manifests and dataset partition files as CSVs.

    Outputs saved to output_dir:
    1. 'split_manifest.csv': Master manifest mapping ticket_id to split name ('train', 'val', 'test').
    2. 'train_manifest.csv': List of ticket IDs in train partition.
    3. 'val_manifest.csv': List of ticket IDs in val partition.
    4. 'test_manifest.csv': List of ticket IDs in test partition.
    5. 'train.csv', 'val.csv', 'test.csv': Full record partitions for modeling.

    Args:
        train: Training partition DataFrame.
        val: Validation partition DataFrame.
        test: Test partition DataFrame.
        output_dir: Target directory path to save manifest files.
        id_col: Name of the ticket ID column (default: 'ticket_id').

    Returns:
        dict[str, str]: Mapping of manifest file keys to absolute string filepaths.

    Raises:
        TypeError: If train, val, or test are not DataFrames.
        KeyError: If id_col is missing from any partition DataFrame.
        IOError: If files cannot be written to output_dir.
    """
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        if not isinstance(split_df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame for '{name}', got {type(split_df).__name__}")
        if id_col not in split_df.columns:
            raise KeyError(
                f"Column '{id_col}' not found in '{name}' partition. "
                f"Available columns: {list(split_df.columns)}"
            )

    out_path = Path(output_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("Failed to create manifest output directory %s: %exc", out_path, exc)
        raise IOError(f"Could not create directory {out_path}: {exc}") from exc

    # Prepare master manifest
    train_manifest = train[[id_col]].copy()
    train_manifest["split"] = "train"

    val_manifest = val[[id_col]].copy()
    val_manifest["split"] = "val"

    test_manifest = test[[id_col]].copy()
    test_manifest["split"] = "test"

    split_manifest = pd.concat([train_manifest, val_manifest, test_manifest], ignore_index=True)

    saved_files: dict[str, str] = {}
    try:
        # Save manifest files
        manifest_file = out_path / "split_manifest.csv"
        split_manifest.to_csv(manifest_file, index=False)
        saved_files["split_manifest"] = str(manifest_file.resolve())

        train_manifest_file = out_path / "train_manifest.csv"
        train_manifest.to_csv(train_manifest_file, index=False)
        saved_files["train_manifest"] = str(train_manifest_file.resolve())

        val_manifest_file = out_path / "val_manifest.csv"
        val_manifest.to_csv(val_manifest_file, index=False)
        saved_files["val_manifest"] = str(val_manifest_file.resolve())

        test_manifest_file = out_path / "test_manifest.csv"
        test_manifest.to_csv(test_manifest_file, index=False)
        saved_files["test_manifest"] = str(test_manifest_file.resolve())

        # Save full data partitions
        train_file = out_path / "train.csv"
        train.to_csv(train_file, index=False)
        saved_files["train_csv"] = str(train_file.resolve())

        val_file = out_path / "val.csv"
        val.to_csv(val_file, index=False)
        saved_files["val_csv"] = str(val_file.resolve())

        test_file = out_path / "test.csv"
        test.to_csv(test_file, index=False)
        saved_files["test_csv"] = str(test_file.resolve())

    except Exception as exc:
        logger.error("Error writing split manifests to %s: %s", out_path, exc)
        raise IOError(f"Failed to write split manifests to {out_path}: {exc}") from exc

    logger.info(
        "Saved split manifests and partition files to %s (master: %d records)",
        out_path.resolve(),
        len(split_manifest),
    )
    return saved_files


def load_split_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Load a split manifest CSV file or manifest directory.

    Args:
        manifest_path: Path to 'split_manifest.csv' or directory containing it.

    Returns:
        dict[str, Any]: Dictionary containing:
            - 'manifest_df': Complete pd.DataFrame of the manifest
            - 'splits': Mapping of split name ('train', 'val', 'test') to list of ticket IDs
            - 'counts': Mapping of split name to integer record count
            - 'total_records': Total count of records across all splits

    Raises:
        TypeError: If manifest_path is not a str or Path.
        FileNotFoundError: If manifest_path does not exist.
        ValueError: If manifest CSV does not contain expected columns ('split' and an ID column).
    """
    if not isinstance(manifest_path, (str, Path)):
        raise TypeError(f"manifest_path must be a str or Path, got {type(manifest_path).__name__}")

    path_obj = Path(manifest_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Split manifest path not found: {path_obj.resolve()}")

    if path_obj.is_dir():
        candidate_file = path_obj / "split_manifest.csv"
        if candidate_file.exists():
            path_obj = candidate_file
        else:
            raise FileNotFoundError(
                f"Directory {path_obj.resolve()} does not contain 'split_manifest.csv'."
            )

    logger.info("Loading split manifest from %s", path_obj.resolve())
    manifest_df = pd.read_csv(path_obj)

    if "split" not in manifest_df.columns:
        raise ValueError(
            f"Manifest file {path_obj} does not contain required column 'split'. "
            f"Available columns: {list(manifest_df.columns)}"
        )

    # Determine ID column (e.g. 'ticket_id' or first non-'split' column)
    id_candidates = [col for col in manifest_df.columns if col != "split"]
    if not id_candidates:
        raise ValueError(f"Manifest file {path_obj} contains only 'split' column, no ID column found.")
    id_col = "ticket_id" if "ticket_id" in id_candidates else id_candidates[0]

    splits: dict[str, list[str]] = {}
    counts: dict[str, int] = {}

    for split_name in ["train", "val", "test"]:
        split_mask = manifest_df["split"] == split_name
        split_ids = manifest_df.loc[split_mask, id_col].astype(str).tolist()
        splits[split_name] = split_ids
        counts[split_name] = len(split_ids)

    logger.info(
        "Loaded split manifest: %d total records (train=%d, val=%d, test=%d)",
        len(manifest_df),
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
    )

    return {
        "manifest_df": manifest_df,
        "id_col": id_col,
        "splits": splits,
        "counts": counts,
        "total_records": len(manifest_df),
    }
