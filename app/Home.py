"""
HelpDeskNet-50K: Executive Overview and Dashboard Entrypoint.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

try:
    from app.utils import (
        load_pipeline_summary,
        load_model_comparison,
        load_review_comparison,
        load_final_test_metrics,
        apply_professional_theme,
        render_sidebar_info,
    )
except (ModuleNotFoundError, ImportError):
    from utils import (
        load_pipeline_summary,
        load_model_comparison,
        load_review_comparison,
        load_final_test_metrics,
        apply_professional_theme,
        render_sidebar_info,
    )

st.set_page_config(
    page_title="HelpDeskNet-50K | Executive Overview",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_professional_theme()
render_sidebar_info()

st.title("HelpDeskNet-50K: Robust Classification & Review Optimization")
st.markdown(
    """
    <span class="enterprise-badge">Research Prototype</span>
    <span class="enterprise-badge">Cleanlab Confident Learning</span>
    <span class="enterprise-badge">Zero Test Contamination</span>
    """,
    unsafe_allow_html=True,
)

summary = load_pipeline_summary()
test_metrics = load_final_test_metrics()
models_df = load_model_comparison()
review_df = load_review_comparison()

st.markdown(
    """
    <div class="provenance-card">
        <strong>Scientific Provenance & Audit Notice:</strong> Unverified source labels from the candidate corpus
        are treated strictly as <em>proxy reference labels</em>. Injected errors and worker profiles are
        <em>statistically simulated</em> under deterministic experimental seeds. Algorithmic issue scores serve as
        triage prioritization signals rather than definitive ground-truth verdicts.
    </div>
    """,
    unsafe_allow_html=True,
)

if summary is None:
    st.error("Pipeline artifacts not found. Execute `python -m src.pipeline` to generate experiment outputs.")
    st.stop()

st.subheader("Key Performance Indicators")

splits = summary.get("split_sizes", {}) if summary else {}
train_n = splits.get("train", 40000)
val_n = splits.get("val", 5000)
test_n = splits.get("test", 5000)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Audited Records",
        value=f"{summary.get('dataset_records', 50000):,}",
        help="Usable records selected deterministically across multi-dataset raw corpus without synthetic text generation.",
    )

with col2:
    st.metric(
        label="Split Protocol",
        value="80 / 10 / 10",
        help=f"Train ({train_n:,}), Validation ({val_n:,}), Test ({test_n:,}) with zero record overlap.",
    )

with col3:
    st.metric(
        label="Simulated Noise Rate",
        value="15.0%",
        help="Parametric noise setting evaluated across training folds.",
    )

with col4:
    best_model = models_df.iloc[0] if models_df is not None else {}
    st.metric(
        label="Top Model (Val Macro-F1)",
        value=f"{best_model.get('macro_f1', 0):.4f}",
        delta=f"{best_model.get('model', 'N/A')}",
        help="Linear SVM with TF-IDF features evaluated on the untouched validation set.",
    )

with col5:
    if review_df is not None and len(review_df) > 0:
        top_strat = review_df.iloc[0]
        rand_strat = review_df[review_df["strategy"] == "random"].iloc[0] if "random" in review_df["strategy"].values else {}
        delta_val = top_strat["errors_per_100"] - rand_strat.get("errors_per_100", 0)
        st.metric(
            label="Review Yield",
            value=f"{top_strat['errors_per_100']:.1f} / 100",
            delta=f"+{delta_val:.1f} vs Random",
            help="Defective labels identified per 100 tickets reviewed under a 2,000-ticket budget.",
        )

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Methodological Architecture")
    st.markdown("""
    1. **Data Ingestion & Hygiene Validation:**
       - Systematic audit of raw ticket corpora with SHA-256 integrity verification.
       - Language filtering to English (`en`) subset (16,338 records) with deterministic non-identifying keys.
       - Strict detection of missingness, whitespace-only content, and candidate duplicate text.

    2. **Taxonomy Harmonization:**
       - Documented mapping from 10 raw customer queues into 6 standardized target categories (`Billing`, `Refund`, `Technical`, `Account Access`, `Shipping`, `Other`).
       - Multi-attribute tag-based reclassification for cross-intent tickets.

    3. **Leakage-Free Partitioning:**
       - Stratified split maintaining class proportions across Train, Validation, and Final Test partitions.
       - Uncompromising barrier preventing validation/test records from entering training, feature fitting, or review ranking.

    4. **Confident Learning & Noise Simulation:**
       - Controlled parametric error injection across symmetric, class-dependent, and multi-tier worker profiles.
       - 5-fold Out-Of-Fold (OOF) cross-validated probability generation paired with Cleanlab noise scoring.

    5. **Expert Review Budget Optimization:**
       - Evaluation of 2,000-ticket review allocations across Random, Confident Learning, and Stratified triage strategies.
    """)

with col_right:
    st.subheader("Classifier Benchmark Summary")
    if models_df is not None:
        display_models = models_df[["model", "accuracy", "macro_f1", "weighted_f1", "split"]].copy()
        display_models.columns = ["Architecture", "Accuracy", "Macro-F1", "Weighted-F1", "Split"]
        st.dataframe(display_models.style.format({
            "Accuracy": "{:.4f}",
            "Macro-F1": "{:.4f}",
            "Weighted-F1": "{:.4f}",
        }), use_container_width=True)

    st.subheader("Review Strategy Yield (2,000 Budget)")
    if review_df is not None:
        display_review = review_df[["strategy", "n_errors_found", "precision_at_k", "errors_per_100", "class_coverage"]].copy()
        display_review.columns = ["Strategy", "Errors Found", "Precision@2K", "Errors / 100", "Classes"]
        st.dataframe(display_review.style.format({
            "Precision@2K": "{:.4f}",
            "Errors / 100": "{:.2f}",
        }), use_container_width=True)

    st.subheader("Audit Efficiency & Resource Impact")
    st.markdown(
        """
        - **Simulation only:** Review-yield metrics estimate recovery of known injected errors, not verified source-label defects.
        - **Human-review requirement:** Operational time and cost savings must be measured after an authorized review study.
        """
    )

st.markdown("---")
st.caption(
    f"HelpDeskNet-50K Architecture • Build: {summary.get('pipeline_version', '1.0')} • "
    f"Generated: {summary.get('timestamp', 'N/A')} • Runtime: Python 3.14, scikit-learn 1.9, Cleanlab 2.9"
)
