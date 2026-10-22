"""Tests for hyperparameter tuning module."""

import json
import numpy as np
import pandas as pd
import pytest

from src.models.tune import (
    evaluate_trial_predictions,
    run_hyperparameter_search,
    save_tuning_artifacts,
    TARGET_CLASSES,
)


@pytest.fixture
def sample_split_data():
    """Create a minimal train/val dataset for tuning tests."""
    classes = TARGET_CLASSES
    texts_train = [
        f"billing problem {i}" if i % 6 == 0 else
        f"refund my payment {i}" if i % 6 == 1 else
        f"technical crash bug {i}" if i % 6 == 2 else
        f"account password login {i}" if i % 6 == 3 else
        f"shipping parcel package {i}" if i % 6 == 4 else
        f"general inquiry question {i}"
        for i in range(120)
    ]
    labels_train = [classes[i % 6] for i in range(120)]

    texts_val = [
        f"billing issue {i}" if i % 6 == 0 else
        f"refund money {i}" if i % 6 == 1 else
        f"technical software failure {i}" if i % 6 == 2 else
        f"account signin reset {i}" if i % 6 == 3 else
        f"shipping track order {i}" if i % 6 == 4 else
        f"other question inquiry {i}"
        for i in range(60)
    ]
    labels_val = [classes[i % 6] for i in range(60)]

    return (
        pd.Series(texts_train),
        pd.Series(labels_train),
        pd.Series(texts_val),
        pd.Series(labels_val),
    )


def test_evaluate_trial_predictions():
    y_true = np.array(["Billing", "Refund", "Technical", "Account Access"])
    y_pred = np.array(["Billing", "Refund", "Technical", "Shipping"])

    metrics = evaluate_trial_predictions(y_true, y_pred, classes=TARGET_CLASSES)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["accuracy"] == 0.75


def test_run_hyperparameter_search_minimal(sample_split_data):
    X_tr, y_tr, X_val, y_val = sample_split_data
    results = run_hyperparameter_search(
        X_tr, y_tr, X_val, y_val, classes=TARGET_CLASSES, random_state=42
    )

    assert "trials_df" in results
    assert "best_trial" in results
    assert "total_trials" in results
    assert isinstance(results["trials_df"], pd.DataFrame)
    assert len(results["trials_df"]) > 0
    assert "macro_f1" in results["best_trial"]
    assert results["best_trial"]["macro_f1"] >= 0.0


def test_save_tuning_artifacts(tmp_path, sample_split_data):
    X_tr, y_tr, X_val, y_val = sample_split_data
    results = run_hyperparameter_search(
        X_tr, y_tr, X_val, y_val, classes=TARGET_CLASSES, random_state=42
    )

    rep_dir = tmp_path / "reports"
    art_dir = tmp_path / "artifacts"
    save_tuning_artifacts(results, reports_dir=rep_dir, artifacts_dir=art_dir)

    csv_path = rep_dir / "tables" / "hyperparameter_tuning.csv"
    json_path = art_dir / "metrics" / "tuning_summary.json"

    assert csv_path.exists()
    assert json_path.exists()

    df_loaded = pd.read_csv(csv_path)
    assert len(df_loaded) == results["total_trials"]

    with open(json_path, "r", encoding="utf-8") as f:
        summary_loaded = json.load(f)
    assert "best_trial" in summary_loaded
    assert "top_5_trials" in summary_loaded
