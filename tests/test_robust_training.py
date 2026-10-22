"""Regression tests for robust-learning experiment orchestration."""

import numpy as np
import pandas as pd

from src.models.baseline import build_tfidf_logreg
from src.models.robust_training import run_robust_experiment


def test_robust_experiment_reports_standard_filtered_and_weighted_results():
    texts = pd.Series(["invoice payment", "billing charge", "server outage", "app crash"] * 4)
    labels = pd.Series(["Billing", "Billing", "Technical", "Technical"] * 4)
    result = run_robust_experiment(
        build_tfidf_logreg,
        texts,
        labels,
        pd.Series(["invoice charge", "system outage"]),
        pd.Series(["Billing", "Technical"]),
        label_issue_scores=np.full(len(texts), 0.1),
        evaluate_fn=lambda truth, prediction: {"accuracy": float((truth == prediction).mean())},
    )

    assert "standard" in result
    assert "filtered" in result
    assert "weighted" in result
