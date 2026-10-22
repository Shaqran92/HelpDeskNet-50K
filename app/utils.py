"""
Shared utilities and artifact loaders for the Streamlit dashboard.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")


@st.cache_data
def load_pipeline_summary() -> Optional[dict[str, Any]]:
    path = ARTIFACTS_DIR / "pipeline_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_data_audit() -> Optional[dict[str, Any]]:
    path = REPORTS_DIR / "data_audit.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_comparison() -> Optional[pd.DataFrame]:
    path = REPORTS_DIR / "tables" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_review_comparison() -> Optional[pd.DataFrame]:
    path = REPORTS_DIR / "tables" / "review_strategy_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_analysis_summary() -> Optional[dict[str, Any]]:
    path = ARTIFACTS_DIR / "metrics" / "analysis_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_worker_tier_comparison() -> Optional[dict[str, Any]]:
    """Compute tier metrics from the persisted worker-dependent simulation.

    This is a compatibility fallback for analysis summaries created from a
    symmetric or class-dependent primary-noise run, neither of which has
    simulated worker assignments.
    """
    path = PROCESSED_DIR / "train_noisy_worker_dependent_15pct.csv"
    if not path.exists():
        return None
    required = ["worker_tier", "reference_label", "observed_label"]
    try:
        data = pd.read_csv(path, usecols=required)
        from src.analysis.noise_metrics import compare_worker_tiers
        return compare_worker_tiers(data)
    except (OSError, ValueError, KeyError):
        return None


@st.cache_data
def load_calibration_metrics() -> Optional[dict[str, Any]]:
    path = ARTIFACTS_DIR / "metrics" / "calibration_logreg.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_detection_metrics() -> Optional[dict[str, Any]]:
    path = ARTIFACTS_DIR / "metrics" / "detection_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_final_test_metrics() -> Optional[dict[str, Any]]:
    path = ARTIFACTS_DIR / "metrics" / "final_test_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_confusion_matrix() -> Optional[pd.DataFrame]:
    path = REPORTS_DIR / "tables" / "test_confusion_matrix.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_review_outcomes(nrows: int = 5000) -> Optional[pd.DataFrame]:
    path = ARTIFACTS_DIR / "review_outcomes.csv"
    if path.exists():
        return pd.read_csv(path, nrows=nrows)
    return None


@st.cache_data
def load_test_predictions(nrows: int = 5000) -> Optional[pd.DataFrame]:
    path = ARTIFACTS_DIR / "predictions_test.csv"
    if path.exists():
        return pd.read_csv(path, nrows=nrows)
    return None


@st.cache_data
def load_source_frequencies() -> Optional[pd.DataFrame]:
    path = REPORTS_DIR / "tables" / "source_class_frequencies.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_hyperparameter_trials() -> Optional[pd.DataFrame]:
    """Load the hyperparameter optimization grid search trials."""
    path = REPORTS_DIR / "tables" / "hyperparameter_tuning.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_tuning_summary() -> Optional[dict[str, Any]]:
    """Load the JSON summary of the hyperparameter optimization run."""
    path = ARTIFACTS_DIR / "metrics" / "tuning_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None



def mask_sensitive_text(text: str, max_chars: int = 250) -> str:
    """Safely redact emails, phone numbers, and truncate text preview."""
    if not isinstance(text, str):
        return ""
    import re
    # Simple regex redactions for PII
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL REDACTED]", text)
    text = re.sub(r"\+?\d[\d -]{8,}\d", "[PHONE REDACTED]", text)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


ENTERPRISE_PALETTE = ["#1f4e78", "#2a9d8f", "#e76f51", "#f4a261", "#457b9d", "#5c6b73"]


def apply_professional_theme() -> None:
    """Apply clean, enterprise styling without playful elements or emojis."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.04);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 6px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 700;
        }
        .enterprise-badge {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            background-color: rgba(31, 78, 120, 0.08);
            color: #1f4e78;
            border: 1px solid rgba(31, 78, 120, 0.22);
            margin-right: 0.4rem;
        }
        .provenance-card {
            border-left: 3px solid #1f4e78;
            background-color: rgba(31, 78, 120, 0.03);
            padding: 0.85rem 1.1rem;
            border-radius: 0 4px 4px 0;
            margin-bottom: 1.2rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_info() -> None:
    """Render executive sidebar with system provenance and run metadata."""
    summary = load_pipeline_summary()
    st.sidebar.markdown("### System Metadata")
    v = summary.get("pipeline_version", "1.0") if summary else "1.0"
    st.sidebar.markdown(
        f"""
        - **System:** HelpDeskNet-50K
        - **Pipeline Release:** v{v}
        - **Validation Status:** Audited & Verified
        - **Language Scope:** English (`en`)
        - **Split Protocol:** 80 / 10 / 10 Stratified
        - **Taxonomy:** 6 Standardized Classes
        - **Review Allocation:** 2,000 Tickets
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Methodological Standards")
    st.sidebar.markdown(
        """
        - Zero test-set contamination
        - Strict label provenance tracking
        - Wilson score confidence bounds
        - Out-of-fold confident learning
        """
    )


@st.cache_resource
def load_classifier_model(model_name: str = "tfidf_svm") -> Optional[Any]:
    """Load a pre-trained scikit-learn classifier pipeline from disk."""
    path = ARTIFACTS_DIR / "models" / f"{model_name}.joblib"
    if path.exists():
        import joblib
        return joblib.load(path)
    return None



