"""Tests for baseline models and evaluation modules."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import (
    build_dummy_classifier,
    build_tfidf_logreg,
    build_tfidf_svm,
    build_tuned_svm,
    build_tuned_logreg,
    get_model_registry,
    train_model,
    predict,
    predict_proba,
    save_model,
    load_model,
    TARGET_CLASSES,
)
from src.models.evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compare_models,
    _compute_ece,
)


# Fixtures
@pytest.fixture
def sample_text_data():
    """Create sample text data for testing."""
    texts = [
        "billing issue with my account",
        "I need a refund for order",
        "technical problem with software",
        "cannot access my account login",
        "shipping delay on package",
        "general inquiry about service",
        "payment failed on checkout",
        "return item and get refund",
        "error message when running app",
        "password reset not working",
    ] * 50  # 500 samples
    labels = (
        ["Billing", "Refund", "Technical", "Account Access", "Shipping",
         "Other", "Billing", "Refund", "Technical", "Account Access"] * 50
    )
    return pd.Series(texts), pd.Series(labels)


class TestModelBuilders:
    """Tests for model builder functions."""

    def test_dummy_classifier(self):
        model = build_dummy_classifier()
        assert model is not None
        assert model.strategy == "most_frequent"

    def test_tfidf_logreg(self):
        pipeline = build_tfidf_logreg()
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_tfidf_svm(self):
        pipeline = build_tfidf_svm()
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps

    def test_tuned_svm(self):
        pipeline = build_tuned_svm()
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps
        assert pipeline.named_steps["clf"].C == 1.0

    def test_tuned_logreg(self):
        pipeline = build_tuned_logreg()
        assert "tfidf" in pipeline.named_steps
        assert "clf" in pipeline.named_steps
        assert pipeline.named_steps["clf"].C == 8.0

    def test_model_registry(self):
        registry = get_model_registry()
        assert "dummy" in registry
        assert "tfidf_logreg" in registry
        assert "tfidf_svm" in registry
        assert "tfidf_svm_tuned" in registry
        assert "tfidf_logreg_tuned" in registry


class TestModelTraining:
    """Tests for model training and prediction."""

    def test_dummy_train_predict(self, sample_text_data):
        X, y = sample_text_data
        model = build_dummy_classifier()
        model = train_model(model, X, y, "test_dummy")
        preds = predict(model, X, "test_dummy")
        assert len(preds) == len(X)

    def test_logreg_train_predict(self, sample_text_data):
        X, y = sample_text_data
        model = build_tfidf_logreg()
        model = train_model(model, X, y, "test_logreg")
        preds = predict(model, X, "test_logreg")
        assert len(preds) == len(X)
        # Should predict valid classes
        assert all(p in TARGET_CLASSES for p in preds)

    def test_logreg_predict_proba(self, sample_text_data):
        X, y = sample_text_data
        model = build_tfidf_logreg()
        model = train_model(model, X, y, "test_logreg")
        proba = predict_proba(model, X, "test_logreg")
        assert proba is not None
        assert proba.shape[0] == len(X)
        # Probabilities should sum to ~1
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_svm_train_predict(self, sample_text_data):
        X, y = sample_text_data
        model = build_tfidf_svm()
        model = train_model(model, X, y, "test_svm")
        preds = predict(model, X, "test_svm")
        assert len(preds) == len(X)


class TestModelSaveLoad:
    """Tests for model serialization."""

    def test_save_load(self, sample_text_data, tmp_path):
        X, y = sample_text_data
        model = build_tfidf_logreg()
        model = train_model(model, X, y, "test")
        path = save_model(model, "test_model", str(tmp_path), metadata={"test": True})
        loaded = load_model(path)
        preds_original = predict(model, X[:10], "original")
        preds_loaded = predict(loaded, X[:10], "loaded")
        assert np.array_equal(preds_original, preds_loaded)


class TestClassificationMetrics:
    """Tests for classification metrics computation."""

    def test_perfect_predictions(self):
        y_true = pd.Series(["A", "B", "C"] * 10)
        y_pred = np.array(["A", "B", "C"] * 10)
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_all_wrong(self):
        y_true = pd.Series(["A"] * 30)
        y_pred = np.array(["B"] * 30)
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_metadata_fields(self):
        y_true = pd.Series(["A", "B"] * 10)
        y_pred = np.array(["A", "B"] * 10)
        metrics = compute_classification_metrics(
            y_true, y_pred, split_name="test", label_provenance="proxy"
        )
        assert metrics["split_name"] == "test"
        assert metrics["label_provenance"] == "proxy"
        assert metrics["n_samples"] == 20


class TestConfusionMatrix:
    """Tests for confusion matrix."""

    def test_shape(self):
        y_true = pd.Series(["A", "B", "C"] * 10)
        y_pred = np.array(["A", "B", "C"] * 10)
        cm = compute_confusion_matrix(y_true, y_pred, ["A", "B", "C"])
        assert cm.shape == (3, 3)

    def test_diagonal(self):
        y_true = pd.Series(["A", "B"] * 10)
        y_pred = np.array(["A", "B"] * 10)
        cm = compute_confusion_matrix(y_true, y_pred, ["A", "B"])
        assert cm.loc["A", "A"] == 10
        assert cm.loc["B", "B"] == 10
        assert cm.loc["A", "B"] == 0


class TestModelComparison:
    """Tests for model comparison."""

    def test_comparison_table(self):
        results = {
            "model_a": {"accuracy": 0.9, "macro_f1": 0.85, "n_samples": 100, "split_name": "val"},
            "model_b": {"accuracy": 0.8, "macro_f1": 0.75, "n_samples": 100, "split_name": "val"},
        }
        df = compare_models(results)
        assert len(df) == 2
        assert df.iloc[0]["model"] == "model_a"  # Higher macro_f1


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration(self):
        # Perfect calibration: confidence matches accuracy
        correct = np.ones(100)
        confidence = np.ones(100) * 0.99
        ece = _compute_ece(correct, confidence)
        assert ece < 0.05  # Should be near zero

    def test_poor_calibration(self):
        correct = np.zeros(100)
        confidence = np.ones(100) * 0.99
        ece = _compute_ece(correct, confidence)
        assert ece > 0.5  # Should be high
