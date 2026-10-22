"""Regression coverage for worker-tier simulation labels used by the dashboard."""

import pandas as pd

from src.analysis.noise_metrics import compare_worker_tiers


def test_worker_tier_comparison_preserves_simulator_tier_names():
    data = pd.DataFrame({
        "worker_tier": ["Tier A"] * 4 + ["Tier B"] * 4,
        "reference_label": ["Billing"] * 8,
        "observed_label": ["Billing", "Billing", "Billing", "Refund", "Billing", "Refund", "Refund", "Refund"],
    })

    result = compare_worker_tiers(data)

    assert set(result["tier_stats"]) == {"Tier A", "Tier B"}
    assert result["tier_stats"]["Tier B"]["error_rate"] > result["tier_stats"]["Tier A"]["error_rate"]
