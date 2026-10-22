"""Category mapping and label harmonization module for HelpDeskNet-50K.

Maps raw customer-support queue categories and domain-specific tags into six
standardized target classes:
    - Billing
    - Refund
    - Technical
    - Account Access
    - Shipping
    - Other

Core principles:
- Raw source labels are preserved; heuristic mappings are never treated as ground truth.
- Explicit tracking of mapping provenance ('queue_direct', 'tag_override', 'ambiguous')
  and mapping confidence ('high', 'medium', 'low').
- Deterministic and non-destructive: input DataFrames are never mutated in place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Six target classes for HelpDeskNet-50K
TARGET_CLASSES: list[str] = [
    "Billing",
    "Refund",
    "Technical",
    "Account Access",
    "Shipping",
    "Other",
]

# Primary mapping: source ticket queue -> target class
QUEUE_TO_TARGET_MAP: dict[str, str] = {
    # 10 Standard Support Queues
    "Billing and Payments": "Billing",
    "Returns and Exchanges": "Refund",
    "Technical Support": "Technical",
    "IT Support": "Technical",
    "Product Support": "Technical",
    "Service Outages and Maintenance": "Technical",
    "Customer Service": "Other",
    "Sales and Pre-Sales": "Other",
    "Human Resources": "Other",
    "General Inquiry": "Other",
    # Vertical Domain Support Queues (Multi-Dataset Harmonization)
    "IT & Technology/Hardware Support": "Technical",
    "IT & Technology/Software Development": "Technical",
    "IT & Technology/Network Infrastructure": "Technical",
    "IT & Technology/Security Operations": "Technical",
    "Finance/Investments": "Billing",
    "Finance/Personal Finance": "Billing",
    "Shopping/E-commerce": "Billing",
    "Travel & Transportation/Air Travel": "Shipping",
    "Travel & Transportation/Land Travel": "Shipping",
    "Autos & Vehicles/Maintenance": "Technical",
    "Jobs & Education/Online Courses": "Account Access",
}

# Tag-based override rules: keywords that reclassify a ticket if present in tag columns
TAG_BASED_OVERRIDES: dict[str, list[str]] = {
    "Account Access": ["Account", "Login", "Password"],
    "Shipping": ["Shipping", "Delivery", "Logistics"],
    "Refund": ["Refund", "Return"],
    "Billing": ["Billing", "Payment", "Invoice"],
}


def _match_tag_classes(tags: list[str]) -> set[str]:
    """Identify all target classes matched by the provided tags based on TAG_BASED_OVERRIDES.

    Args:
        tags: List of cleaned string tags associated with the ticket.

    Returns:
        set[str]: Set of all target classes matched by any tag.
    """
    matches: set[str] = set()

    for target_class, keywords in TAG_BASED_OVERRIDES.items():
        for tag in tags:
            tag_lower = tag.lower()
            if any(kw.lower() in tag_lower for kw in keywords):
                matches.add(target_class)
                break

    return matches


def apply_label_mapping(
    df: pd.DataFrame,
    queue_col: str = "queue",
    tag_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply queue mapping and tag-based override rules to harmonize ticket categories.

    Creates three new columns on a copy of the input DataFrame:
    - `mapped_label`: Final harmonized target class (one of TARGET_CLASSES).
    - `mapping_source`: Provenance of the mapping ('queue_direct', 'tag_override', 'ambiguous').
    - `mapping_confidence`: Confidence level ('high', 'medium', 'low').

    Confidence assignment logic:
    - 'high':
        * Specific queue mapped directly with no conflicting tags
          ('Billing and Payments', 'Returns and Exchanges', 'Technical Support', etc.).
        * Direct queue mapping confirmed by matching tag.
        * Reclassification of generic/uninformative queues ('Other') to a specific
          class via clear tag override.
    - 'medium':
        * Direct mapping of broad/catch-all queues ('Customer Service', 'General Inquiry',
          'Sales and Pre-Sales', 'Human Resources') to 'Other'.
        * Tag-based reclassification that overrides a specific queue
          (e.g., Technical Support reclassified to Account Access).
    - 'low':
        * Conflicting tag overrides (ambiguous multi-intent signals).
        * Unknown or missing queue value falling back to 'Other'.

    Args:
        df: Input DataFrame containing customer-support tickets.
        queue_col: Name of the column containing raw queue categories (default: 'queue').
        tag_cols: List of column names containing ticket tags. If None, defaults to
            ['tag_1', 'tag_2'].

    Returns:
        pd.DataFrame: A new DataFrame with original columns plus `mapped_label`,
            `mapping_source`, and `mapping_confidence`.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If df is empty or queue_col is not present in df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        logger.warning("apply_label_mapping called with an empty DataFrame.")
        result_df = df.copy()
        result_df["mapped_label"] = pd.Series(dtype=object)
        result_df["mapping_source"] = pd.Series(dtype=object)
        result_df["mapping_confidence"] = pd.Series(dtype=object)
        return result_df

    if queue_col not in df.columns:
        raise ValueError(
            f"Queue column '{queue_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if tag_cols is None:
        tag_cols = ["tag_1", "tag_2"]

    # Identify available tag columns
    available_tag_cols = [c for c in tag_cols if c in df.columns]
    if not available_tag_cols:
        logger.warning(
            "None of the specified tag columns %s exist in DataFrame. "
            "Tag overrides will be skipped.",
            tag_cols,
        )

    logger.info(
        "Applying label mapping to %d records (queue_col='%s', tag_cols=%s)",
        len(df),
        queue_col,
        available_tag_cols,
    )

    mapped_labels: list[str] = []
    mapping_sources: list[str] = []
    mapping_confidences: list[str] = []

    for _, row in df.iterrows():
        queue_raw = row.get(queue_col)
        queue_str = str(queue_raw).strip() if pd.notna(queue_raw) else None

        base_class = QUEUE_TO_TARGET_MAP.get(queue_str) if queue_str else None

        # Extract non-empty tag strings
        tags: list[str] = []
        for col in available_tag_cols:
            val = row.get(col)
            if pd.notna(val):
                cleaned = str(val).strip()
                if cleaned:
                    tags.append(cleaned)

        # Identify all target classes suggested by tags
        matched_tag_classes = _match_tag_classes(tags)

        # Decision logic
        if len(matched_tag_classes) > 1:
            # Conflicting tag signals: multiple distinct classes suggested by tags
            final_label = base_class if base_class is not None else "Other"
            source = "ambiguous"
            confidence = "low"
        elif len(matched_tag_classes) == 1:
            suggested_class = next(iter(matched_tag_classes))
            if suggested_class == base_class:
                # Tag confirms base queue category (e.g. Billing queue with Billing tag)
                final_label = base_class
                source = "queue_direct"
                confidence = "high"
            else:
                # True tag override reclassifying the ticket
                final_label = suggested_class
                source = "tag_override"
                # High confidence if enriching broad 'Other' queue, medium if overriding specific queue
                confidence = "high" if (base_class == "Other" or base_class is None) else "medium"
        else:
            # No tag override matched
            if base_class is not None:
                final_label = base_class
                source = "queue_direct"
                # High confidence for specific queues, medium for broad catch-all queues
                confidence = "medium" if base_class == "Other" else "high"
            else:
                # Missing or unrecognized queue
                final_label = "Other"
                source = "ambiguous"
                confidence = "low"

        mapped_labels.append(final_label)
        mapping_sources.append(source)
        mapping_confidences.append(confidence)

    result_df = df.copy()
    result_df["mapped_label"] = mapped_labels
    result_df["mapping_source"] = mapping_sources
    result_df["mapping_confidence"] = mapping_confidences

    logger.info(
        "Label mapping complete. Class distribution: %s",
        dict(pd.Series(mapped_labels).value_counts()),
    )
    return result_df


def get_mapping_report(df: pd.DataFrame) -> dict[str, Any]:
    """Compute detailed mapping statistics and diagnostics from a mapped DataFrame.

    Args:
        df: DataFrame containing `mapped_label`, `mapping_source`, and
            `mapping_confidence` columns (as produced by `apply_label_mapping`).

    Returns:
        dict[str, Any]: Summary dictionary containing:
            - total_records: int
            - mapped_label_counts: dict[str, int]
            - mapped_label_percentages: dict[str, float]
            - mapping_source_counts: dict[str, int]
            - mapping_source_percentages: dict[str, float]
            - mapping_confidence_counts: dict[str, int]
            - mapping_confidence_percentages: dict[str, float]
            - ambiguity_rate: float
            - high_confidence_rate: float
            - queue_to_mapped: dict[str, dict[str, int]] (if queue column present)

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If required mapping columns are missing from df.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    required_cols = {"mapped_label", "mapping_source", "mapping_confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame missing required mapping columns: {sorted(missing)}. "
            "Run apply_label_mapping() first."
        )

    n_total = len(df)
    if n_total == 0:
        logger.warning("get_mapping_report called on empty DataFrame.")
        return {
            "total_records": 0,
            "mapped_label_counts": {},
            "mapped_label_percentages": {},
            "mapping_source_counts": {},
            "mapping_source_percentages": {},
            "mapping_confidence_counts": {},
            "mapping_confidence_percentages": {},
            "ambiguity_rate": 0.0,
            "high_confidence_rate": 0.0,
            "queue_to_mapped": {},
        }

    label_counts = df["mapped_label"].value_counts().to_dict()
    label_pcts = {k: round(v / n_total * 100, 2) for k, v in label_counts.items()}

    source_counts = df["mapping_source"].value_counts().to_dict()
    source_pcts = {k: round(v / n_total * 100, 2) for k, v in source_counts.items()}

    conf_counts = df["mapping_confidence"].value_counts().to_dict()
    conf_pcts = {k: round(v / n_total * 100, 2) for k, v in conf_counts.items()}

    ambiguous_count = source_counts.get("ambiguous", 0)
    ambiguity_rate = round(ambiguous_count / n_total, 4)

    high_conf_count = conf_counts.get("high", 0)
    high_conf_rate = round(high_conf_count / n_total, 4)

    report: dict[str, Any] = {
        "total_records": int(n_total),
        "mapped_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "mapped_label_percentages": label_pcts,
        "mapping_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "mapping_source_percentages": source_pcts,
        "mapping_confidence_counts": {str(k): int(v) for k, v in conf_counts.items()},
        "mapping_confidence_percentages": conf_pcts,
        "ambiguity_rate": ambiguity_rate,
        "high_confidence_rate": high_conf_rate,
    }

    if "queue" in df.columns:
        crosstab = pd.crosstab(df["queue"], df["mapped_label"])
        report["queue_to_mapped"] = {
            str(q): {str(c): int(cnt) for c, cnt in row.items()}
            for q, row in crosstab.to_dict(orient="index").items()
        }

    logger.info(
        "Mapping report generated: %d records, ambiguity_rate=%.4f, high_conf_rate=%.4f",
        n_total,
        ambiguity_rate,
        high_conf_rate,
    )
    return report


def save_mapping_documentation(output_path: str | Path) -> Path:
    """Save the label mapping taxonomy, override rules, and guidelines as a Markdown file.

    Args:
        output_path: Filepath where the Markdown documentation will be written.

    Returns:
        Path: Resolved Path to the generated documentation file.

    Raises:
        TypeError: If output_path is not a str or Path.
        IOError: If writing to the file fails.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError(f"output_path must be a str or Path, got {type(output_path).__name__}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# HelpDeskNet-50K Category Mapping and Label Taxonomy

## 1. Overview and Core Principles

HelpDeskNet-50K studies label noise, annotation disagreement, and robust classification
on customer-support ticket datasets.

**Core Research Principle:**
A heuristic label mapping or source queue value is **not** ground truth. Raw source
categories, heuristic mappings, proxy reference labels, and noisy labels are maintained
in strictly separate columns to prevent label leakage and misleading evaluations.

---

## 2. Target Classes

The taxonomy standardizes tickets into six mutually exclusive classes:

| Target Class | Description | Example Topics |
| :--- | :--- | :--- |
| `Billing` | Invoices, charges, payment failures, subscription pricing | Payment gateways, credit card errors, receipt requests |
| `Refund` | Return requests, reimbursement, order cancellations, refunds | Return merchandise, money-back requests, fee reversals |
| `Technical` | Software bugs, hardware faults, system outages, crashes | Error codes, crashes, connectivity loss, slow performance |
| `Account Access` | Authentication, login problems, password reset, permissions | 2FA issues, locked accounts, credential resets |
| `Shipping` | Delivery delays, parcel tracking, logistics, shipment loss | Carrier delays, lost packages, incorrect delivery address |
| `Other` | General inquiries, HR, marketing, unclassified customer feedback | Presales questions, career inquiries, feedback surveys |

---

## 3. Queue-to-Target Direct Mapping

The initial category is determined by the ticket's originating queue:

| Raw Source Queue | Target Class | Baseline Confidence |
| :--- | :--- | :--- |
| `Billing and Payments` | `Billing` | High |
| `Returns and Exchanges` | `Refund` | High |
| `Technical Support` | `Technical` | High |
| `IT Support` | `Technical` | High |
| `Product Support` | `Technical` | High |
| `Service Outages and Maintenance` | `Technical` | High |
| `Customer Service` | `Other` | Medium |
| `Sales and Pre-Sales` | `Other` | Medium |
| `Human Resources` | `Other` | Medium |
| `General Inquiry` | `Other` | Medium |

---

## 4. Tag-Based Reclassification Rules

Tickets in general queues often describe specific issues that can be recovered via tags:

| Target Class | Trigger Keywords (in `tag_1` or `tag_2`) | Condition / Exception |
| :--- | :--- | :--- |
| `Account Access` | `Account`, `Login`, `Password` | Always overrides |
| `Shipping` | `Shipping`, `Delivery`, `Logistics` | Always overrides |
| `Refund` | `Refund`, `Return` | Only if queue is not already `Refund` |
| `Billing` | `Billing`, `Payment`, `Invoice` | Only if queue is not already `Billing` |

---

## 5. Provenance and Confidence Tracking

Every mapped record includes:

1. `mapped_label`: The assigned target class (one of `TARGET_CLASSES`).
2. `mapping_source`:
   - `queue_direct`: Assigned directly via queue mapping without tag intervention.
   - `tag_override`: Reclassified based on tags.
   - `ambiguous`: Multiple conflicting tag signals or unresolvable queue.
3. `mapping_confidence`:
   - `high`: Specific queue confirmed or generic queue enriched with unambiguous tag.
   - `medium`: Specific queue overridden by tag or generic queue mapped to `Other`.
   - `low`: Conflicting tag signals or unrecognized queue.

---

## 6. Reproducibility

All mappings are deterministic and pure functions of ticket metadata. Seeds and versioned
rules ensure 100% reproducibility across experiments.
"""

    try:
        out_file.write_text(content, encoding="utf-8")
        logger.info("Mapping documentation saved successfully to %s", out_file.resolve())
    except Exception as e:
        logger.error("Failed to write mapping documentation to %s: %s", out_file, e)
        raise IOError(f"Could not write documentation to {out_file}: {e}") from e

    return out_file
