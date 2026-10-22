"""
Page 2: Label Quality, Disagreement, and Noise Analysis.
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
import plotly.graph_objects as go

try:
    from app.utils import (
        load_analysis_summary,
        load_worker_tier_comparison,
        apply_professional_theme,
        render_sidebar_info,
        ENTERPRISE_PALETTE,
    )
except (ModuleNotFoundError, ImportError):
    from utils import (
        load_analysis_summary,
        load_worker_tier_comparison,
        apply_professional_theme,
        render_sidebar_info,
        ENTERPRISE_PALETTE,
    )

st.set_page_config(page_title="Label Quality | HelpDeskNet-50K", layout="wide")

apply_professional_theme()
render_sidebar_info()

st.title("Label Quality & Noise Simulation Analysis")
st.markdown(
    """
    <span class="enterprise-badge">Wilson Confidence Intervals</span>
    <span class="enterprise-badge">Annotator Tier Divergence</span>
    <span class="enterprise-badge">Transition Matrix</span>
    """,
    unsafe_allow_html=True,
)
st.markdown("Quantifying label corruption, transition probabilities, and statistical worker tier differentials.")

st.markdown(
    """
    <div class="provenance-card">
        <strong>Simulation Context:</strong> The error metrics presented here derive from controlled noise simulation
        experiments across configured worker profiles (Tier A: 0.5x error multiplier, Tier B: 1.5x error multiplier).
        Unverified dataset source categories serve strictly as proxy references.
    </div>
    """,
    unsafe_allow_html=True,
)

summary = load_analysis_summary()

if summary is None:
    st.error("Analysis summary metrics not found. Execute the pipeline first.")
    st.stop()

overall = summary.get("overall_noise", {})
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sample Audited", f"{overall.get('n_total', 0):,}")
with col2:
    st.metric("Injected Corruption Count", f"{overall.get('n_errors', 0):,}")
with col3:
    rate = overall.get("noise_rate", 0) * 100
    st.metric("Empirical Noise Rate", f"{rate:.2f}%")
with col4:
    ci_l = overall.get("ci_lower", 0) * 100
    ci_u = overall.get("ci_upper", 0) * 100
    st.metric("Wilson 95% Confidence Interval", f"[{ci_l:.2f}%, {ci_u:.2f}%]")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Per-Class Noise Rates",
    "Worker Tier Comparison",
    "Label Transition Matrix",
    "Statistical Methodology",
])

with tab1:
    st.subheader("Class-Level Noise Susceptibility")
    class_noise = pd.DataFrame(summary.get("class_noise", []))
    if not class_noise.empty:
        class_noise["noise_pct"] = class_noise["noise_rate"] * 100
        class_noise["ci_lower_pct"] = class_noise["ci_lower"] * 100
        class_noise["ci_upper_pct"] = class_noise["ci_upper"] * 100

        fig_cn = go.Figure()
        fig_cn.add_trace(go.Bar(
            x=class_noise["class"],
            y=class_noise["noise_pct"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=class_noise["ci_upper_pct"] - class_noise["noise_pct"],
                arrayminus=class_noise["noise_pct"] - class_noise["ci_lower_pct"],
            ),
            marker_color=ENTERPRISE_PALETTE[0],
            text=class_noise["noise_pct"].apply(lambda v: f"{v:.2f}%"),
            textposition="auto",
        ))
        fig_cn.update_layout(
            title="Class Noise Rates with Wilson 95% Confidence Intervals",
            yaxis_title="Empirical Noise Rate (%)",
            xaxis_title="Target Category",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_cn, use_container_width=True)

        display_cn = class_noise[["class", "n_total", "n_errors", "noise_pct", "ci_lower_pct", "ci_upper_pct"]].copy()
        display_cn.columns = ["Target Class", "Total Samples", "Errors", "Noise Rate", "CI Lower (95%)", "CI Upper (95%)"]
        st.dataframe(display_cn.style.format({
            "Noise Rate": "{:.2f}%",
            "CI Lower (95%)": "{:.2f}%",
            "CI Upper (95%)": "{:.2f}%",
        }), use_container_width=True)

with tab2:
    st.subheader("Simulated Annotator Tiers: Tier A vs. Tier B")
    tier_info = summary.get("tier_comparison") or load_worker_tier_comparison()
    if tier_info:
        tier_stats = tier_info.get("tier_stats", {})
        stats_a = tier_stats.get("Tier A", tier_stats.get("A", {}))
        stats_b = tier_stats.get("Tier B", tier_stats.get("B", {}))

        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            st.metric("Tier A Error Rate", f"{stats_a.get('error_rate', 0)*100:.2f}%", help="Expert worker profile (0.5x error multiplier)")
        with tc2:
            st.metric("Tier B Error Rate", f"{stats_b.get('error_rate', 0)*100:.2f}%", help="Crowd worker profile (1.5x error multiplier)")
        with tc3:
            diff = tier_info.get("rate_difference", 0) * 100
            st.metric("Rate Difference (B - A)", f"+{diff:.2f}%")
        with tc4:
            p_val = tier_info.get("p_value", 0)
            st.metric("Two-Proportion Z-Test", "p < 1e-10" if p_val and p_val < 1e-4 else f"p = {p_val:.4f}")

        st.markdown(f"""
        **Hypothesis Testing & Effect Size Breakdown:**
        - **Cohen's h Effect Size:** `{tier_info.get('cohens_h', 0):.4f}` (Reflects significant divergence between annotator tiers)
        - **Z-Statistic:** `{tier_info.get('z_statistic', 0):.2f}`
        - **Difference 95% Confidence Interval:** `[{tier_info.get('diff_ci_lower', 0)*100:.2f}%, {tier_info.get('diff_ci_upper', 0)*100:.2f}%]`
        """)
    else:
        st.info("Worker-tier data is unavailable. Run a worker-dependent noise simulation to create it.")

with tab3:
    st.subheader(r"Class Transition Matrix $T_{ij} = P(Y_{obs} = j \mid Y_{ref} = i)$")
    tm_data = summary.get("transition_matrix")
    if tm_data:
        tm_df = pd.DataFrame(tm_data)
        fig_heat = px.imshow(
            tm_df,
            labels=dict(x="Observed (Noisy) Label", y="Reference (True) Label", color="Transition Prob"),
            x=tm_df.columns,
            y=tm_df.index,
            color_continuous_scale="Blues",
            text_auto=".3f",
            aspect="auto",
        )
        fig_heat.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        most_confused = summary.get("most_confused_pair")
        if most_confused:
            st.markdown(
                f"""
                <div class="provenance-card">
                    <strong>Highest Off-Diagonal Confusion:</strong> Transition from <code>{most_confused.get('from')}</code>
                    to <code>{most_confused.get('to')}</code> occurred at rate <code>{most_confused.get('rate', 0):.4f}</code>.
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab4:
    st.subheader("Mathematical Formulations")
    st.markdown(r"""
    #### 1. Wilson Score Interval for Binomial Proportions
    Rather than relying on the standard normal approximation (Wald interval), which degrades near extreme proportions or small sample sizes, we compute asymmetric Wilson score intervals:

    $$
    w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
    $$

    where $z = 1.95996$ for the 95% confidence level.

    #### 2. Cohen's $h$ for Comparing Two Proportions
    To quantify the effect size between annotator tier error rates $p_1$ (Tier A) and $p_2$ (Tier B), we use the arcsine transformation:

    $$
    h = 2 \arcsin(\sqrt{p_1}) - 2 \arcsin(\sqrt{p_2})
    $$

    - Small effect: $h \approx 0.20$
    - Medium effect: $h \approx 0.50$
    - Large effect: $h \ge 0.80$

    #### 3. Transition Matrix Formulation
    Under class-conditional noise, transition probability $T_{ij}$ represents the probability of observing label $j$ given true reference class $i$:

    $$
    T_{ij} = P(\tilde{Y} = j \mid Y^* = i) \quad \text{such that} \quad \sum_{j} T_{ij} = 1
    $$
    """)
