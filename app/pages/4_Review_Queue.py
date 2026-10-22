"""
Page 4: Expert Review Queue and Budget Optimization.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from app.utils import (
        load_review_comparison,
        load_detection_metrics,
        load_review_outcomes,
        mask_sensitive_text,
        apply_professional_theme,
        render_sidebar_info,
        ENTERPRISE_PALETTE,
    )
except (ModuleNotFoundError, ImportError):
    from utils import (
        load_review_comparison,
        load_detection_metrics,
        load_review_outcomes,
        mask_sensitive_text,
        apply_professional_theme,
        render_sidebar_info,
        ENTERPRISE_PALETTE,
    )

st.set_page_config(page_title="Review Queue | HelpDeskNet-50K", layout="wide")

apply_professional_theme()
render_sidebar_info()

st.title("Simulated Review Queue & Budget Optimization")
st.markdown(
    """
    <span class="enterprise-badge">Human-in-the-Loop Triage</span>
    <span class="enterprise-badge">2,000 Ticket Budget</span>
    <span class="enterprise-badge">Confident Learning</span>
    <span class="enterprise-badge">Dynamic Simulator</span>
    """,
    unsafe_allow_html=True,
)
st.markdown("Prioritizing a future human-review workflow using synthetic corruption and multi-strategy triage.")

st.markdown(
    """
    <div class="provenance-card">
        <strong>Triage Signal Principle:</strong> A high label-issue score indicates algorithmic suspicion based on
        out-of-fold probability discrepancy. It does <em>not</em> prove the original label is false. Expert human
        review validates or rejects suggestions.
    </div>
    """,
    unsafe_allow_html=True,
)

rev_df = load_review_comparison()
det_metrics = load_detection_metrics()
outcomes_df = load_review_outcomes(nrows=2000)

if rev_df is None:
    st.error("Review budget experiment results not found. Execute the pipeline first.")
    st.stop()

st.subheader("Strategy Comparison on Fixed 2,000 Review Budget")
col1, col2, col3 = st.columns(3)

with col1:
    best_strat = rev_df.iloc[0]
    st.metric(
        "Injected Errors Found (Simulation)",
        f"{best_strat['n_errors_found']} / 2,000",
        delta=f"{best_strat['errors_per_100']:.1f}% Yield",
    )

with col2:
    rand_strat = rev_df[rev_df["strategy"] == "random"].iloc[0] if "random" in rev_df["strategy"].values else {}
    st.metric(
        "Random Injected Errors (Simulation)",
        f"{rand_strat.get('n_errors_found', 0)} / 2,000",
        delta=f"{rand_strat.get('errors_per_100', 0):.1f}% Yield",
    )

with col3:
    gain = best_strat["n_errors_found"] - rand_strat.get("n_errors_found", 0)
    pct_boost = (gain / rand_strat.get("n_errors_found", 1)) * 100
    st.metric(
        "Intelligent Review Gain",
        f"+{gain} Errors Caught",
        delta=f"+{pct_boost:.1f}% Efficiency",
    )

fig_eff = px.bar(
    rev_df,
    x="strategy",
    y="errors_per_100",
    color="strategy",
    text="errors_per_100",
    title="Review Efficiency: Confirmed Errors Identified Per 100 Reviews",
    labels={"errors_per_100": "Errors Caught per 100 Reviews", "strategy": "Triage Strategy"},
    color_discrete_sequence=[ENTERPRISE_PALETTE[0], ENTERPRISE_PALETTE[1], ENTERPRISE_PALETTE[2]],
)
fig_eff.update_layout(
    height=340,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_eff, use_container_width=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Review Audit Queue (Top Priority)",
    "Precision@K & Detection Curve",
    "Audit ROI & Capacity Simulator",
])

with tab1:
    st.subheader("Simulated Review Queue")
    st.caption("Reviewer decisions shown below are simulation outcomes, not human-review records.")

    if outcomes_df is not None:
        f1, f2, f3 = st.columns(3)
        with f1:
            decision_filter = st.multiselect(
                "Filter by Reviewer Decision",
                options=outcomes_df["reviewer_decision"].unique().tolist(),
                default=outcomes_df["reviewer_decision"].unique().tolist(),
            )
        with f2:
            class_filter = st.multiselect(
                "Filter by Original Label",
                options=outcomes_df["original_label"].unique().tolist(),
                default=outcomes_df["original_label"].unique().tolist(),
            )
        with f3:
            search_query = st.text_input("Search Ticket ID", "")

        score_col1, score_col2 = st.columns([2, 1])
        with score_col1:
            min_score = st.slider("Minimum Algorithmic Issue Score", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
        with score_col2:
            st.metric("Queue Depth", f"{len(outcomes_df):,} Candidates")

        filtered = outcomes_df[
            outcomes_df["reviewer_decision"].isin(decision_filter)
            & outcomes_df["original_label"].isin(class_filter)
            & (outcomes_df["issue_score"] >= min_score)
        ]
        if search_query:
            filtered = filtered[filtered["ticket_id"].str.contains(search_query, case=False, na=False)]

        st.caption(f"Displaying {len(filtered):,} records meeting active triage filters.")

        display_cols = ["ticket_id", "original_label", "suggested_label", "issue_score", "reviewer_decision", "reviewed_label", "reviewer"]
        st.dataframe(filtered[display_cols].head(500), use_container_width=True)

        csv_filtered = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Export Filtered Review Queue ({len(filtered):,} Records)",
            data=csv_filtered,
            file_name="helpdesknet50k_filtered_review_queue.csv",
            mime="text/csv",
        )
    else:
        st.info("Review queue records not available.")

with tab2:
    st.subheader("Cleanlab Detection Precision@K Curve")
    if det_metrics:
        k_points = [100, 200, 500, 1000, 2000]
        precisions = [det_metrics.get(f"precision_at_{k}", 0) * 100 for k in k_points]
        tp_counts = [det_metrics.get(f"true_positives_at_{k}", 0) for k in k_points]

        curve_df = pd.DataFrame({
            "Review Budget (K)": k_points,
            "Precision@K (%)": precisions,
            "True Errors Identified": tp_counts,
        })

        fig_curve = px.line(
            curve_df,
            x="Review Budget (K)",
            y="Precision@K (%)",
            markers=True,
            title=f"Precision@K Decay Curve (Average Precision: {det_metrics.get('average_precision', 0):.4f})",
            color_discrete_sequence=[ENTERPRISE_PALETTE[0]],
        )
        fig_curve.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        display_curve = curve_df.copy()
        display_curve.columns = ["Audit Budget Threshold (K)", "Precision@K", "Total Errors Caught"]
        st.dataframe(display_curve.style.format({"Precision@K": "{:.2f}%"}), use_container_width=True)

with tab3:
    st.subheader("Interactive Audit Capacity & Resource Simulator")
    st.markdown(
        """
        Dynamically simulate human auditor capacity, error yields, and operational cost savings across
        customizable review budgets.
        """
    )

    sim_k = st.slider("Simulate Human Review Allocation Budget (Tickets)", min_value=100, max_value=4000, value=2000, step=100)
    auditor_hourly_rate = st.number_input("Auditor Hourly Compensation ($/hour)", min_value=20, max_value=200, value=50, step=5)
    review_minutes_per_ticket = st.number_input("Average Review Time per Ticket (Minutes)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

    base_err_rate = 6009 / 40000.0  # Empirical 15.02% noise rate on 40,000 train
    rand_errors = int(sim_k * base_err_rate)
    
    # Model precision curve interpolation
    if sim_k <= 100:
        prec = 1.00
    elif sim_k <= 500:
        prec = 0.99
    elif sim_k <= 1000:
        prec = 0.975
    elif sim_k <= 2000:
        prec = 0.946
    else:
        prec = max(0.65, 0.946 - ((sim_k - 2000) / 2000.0) * 0.25)

    cleanlab_errors = min(6009, int(sim_k * prec))
    gain_errors = cleanlab_errors - rand_errors

    hours_spent = (sim_k * review_minutes_per_ticket) / 60.0
    cost_total = hours_spent * auditor_hourly_rate
    cost_per_error_cleanlab = cost_total / cleanlab_errors if cleanlab_errors > 0 else 0
    cost_per_error_rand = cost_total / rand_errors if rand_errors > 0 else 0

    # Equivalent random budget required to discover cleanlab_errors
    equiv_rand_budget = int(cleanlab_errors / base_err_rate)
    equiv_hours = (equiv_rand_budget * review_minutes_per_ticket) / 60.0
    hours_saved = max(0.0, equiv_hours - hours_spent)

    sim1, sim2, sim3, sim4 = st.columns(4)
    with sim1:
        st.metric("Cleanlab Errors Caught", f"{cleanlab_errors:,}", delta=f"{cleanlab_errors/sim_k*100:.1f}% Yield")
    with sim2:
        st.metric("Random Baseline Errors", f"{rand_errors:,}", delta=f"{rand_errors/sim_k*100:.1f}% Yield")
    with sim3:
        st.metric("Net Error Discovery Boost", f"+{gain_errors:,} Errors", delta=f"+{gain_errors/rand_errors*100:+.1f}% Efficiency")
    with sim4:
        st.metric("Human Auditor Hours Saved", f"{hours_saved:.1f} Hours", delta=f"${hours_saved * auditor_hourly_rate:,.0f} Cost Savings")

    st.markdown(
        f"""
        <div class="provenance-card">
            <strong>Operational Value Summary at {sim_k:,} Ticket Budget:</strong><br>
            • To catch <strong>{cleanlab_errors:,} corrupted labels</strong> under unassisted random audit would require reviewing 
            <strong>{equiv_rand_budget:,} tickets</strong> ({equiv_hours:.1f} hours).<br>
            • Confident Learning triage delivers the identical result in <strong>{sim_k:,} reviews</strong> ({hours_spent:.1f} hours), 
            freeing <strong>{hours_saved:.1f} expert hours</strong> and reducing defect remediation cost from 
            <strong>${cost_per_error_rand:.2f}</strong> to <strong>${cost_per_error_cleanlab:.2f} per defect</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
