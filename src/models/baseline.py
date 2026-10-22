"""
Baseline classifiers for customer-support ticket classification.

Implements dummy, TF-IDF + LogisticRegression, and TF-IDF + LinearSVM
baselines. All models fit vectorizers only on training data and support
out-of-fold probability estimation for label-issue detection.
"""

import logging
import os
import json
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Target classes in canonical order
TARGET_CLASSES = ["Billing", "Refund", "Technical", "Account Access", "Shipping", "Other"]


def build_dummy_classifier(strategy: str = "most_frequent") -> DummyClassifier:
    """Create a dummy (majority-class) classifier.

    Parameters
    ----------
    strategy : str
        Strategy for the dummy classifier ('most_frequent', 'stratified', etc.).

    Returns
    -------
    DummyClassifier
        Unfitted dummy classifier.
    """
    return DummyClassifier(strategy=strategy)


def build_tfidf_logreg(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    max_features : int
        Max vocabulary size for TF-IDF.
    ngram_range : tuple
        N-gram range for TF-IDF.
    C : float
        Regularization parameter.
    class_weight : str, optional
        Class weight strategy ('balanced' or None).
    random_state : int
        Random seed.

    Returns
    -------
    Pipeline
        Unfitted sklearn pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def build_tfidf_svm(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a TF-IDF + Linear SVM pipeline with calibration.

    LinearSVC does not natively produce probabilities, so we wrap it
    with CalibratedClassifierCV for probability estimation.

    Parameters
    ----------
    max_features : int
        Max vocabulary size.
    ngram_range : tuple
        N-gram range.
    C : float
        Regularization.
    class_weight : str, optional
        Class weight strategy.
    random_state : int
        Random seed.

    Returns
    -------
    Pipeline
        Unfitted pipeline with calibrated SVM.
    """
    base_svm = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=5000,
        random_state=random_state,
        dual="auto",
    )
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", calibrated_svm),
    ])


def build_tuned_svm(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a high-accuracy TF-IDF + LinearSVC pipeline with tuned parameters."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=4000,
            random_state=random_state,
            dual="auto",
        )),
    ])


def build_tuned_logreg(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    C: float = 8.0,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a high-accuracy TF-IDF + Logistic Regression pipeline with tuned parameters."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )),
    ])


def get_model_registry() -> dict:
    """Return a registry of available model builders.

    Returns
    -------
    dict
        Model name → builder function mapping.
    """
    return {
        "dummy": build_dummy_classifier,
        "tfidf_logreg": build_tfidf_logreg,
        "tfidf_svm": build_tfidf_svm,
        "tfidf_svm_tuned": build_tuned_svm,
        "tfidf_logreg_tuned": build_tuned_logreg,
        "tfidf_logreg_weighted": lambda **kw: build_tfidf_logreg(
            class_weight="balanced", **kw
        ),
        "tfidf_svm_weighted": lambda **kw: build_tfidf_svm(
            class_weight="balanced", **kw
        ),
    }


def train_model(
    model,
    X_train: pd.Series,
    y_train: pd.Series,
    model_name: str = "unnamed",
) -> object:
    """Train a model on the given data.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted model or pipeline.
    X_train : pd.Series
        Training text.
    y_train : pd.Series
        Training labels.
    model_name : str
        Name for logging.

    Returns
    -------
    object
        Fitted model.
    """
    logger.info("Training model '%s' on %d examples...", model_name, len(X_train))
    start = datetime.now()

    if isinstance(model, DummyClassifier):
        # Dummy doesn't use text features
        model.fit(np.zeros((len(X_train), 1)), y_train)
    else:
        model.fit(X_train, y_train)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info("Model '%s' trained in %.1f seconds.", model_name, elapsed)
    return model


def predict(
    model,
    X: pd.Series,
    model_name: str = "unnamed",
) -> np.ndarray:
    """Generate predictions from a fitted model.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    X : pd.Series
        Text data.
    model_name : str
        Name for logging.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    if isinstance(model, DummyClassifier):
        return model.predict(np.zeros((len(X), 1)))
    return model.predict(X)


def predict_proba(
    model,
    X: pd.Series,
    model_name: str = "unnamed",
) -> Optional[np.ndarray]:
    """Generate probability estimates from a fitted model.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    X : pd.Series
        Text data.
    model_name : str
        Name for logging.

    Returns
    -------
    np.ndarray or None
        Predicted probabilities, or None if not supported.
    """
    if isinstance(model, DummyClassifier):
        return model.predict_proba(np.zeros((len(X), 1)))

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    logger.warning("Model '%s' does not support predict_proba.", model_name)
    return None


def generate_oof_probabilities(
    model_builder: callable,
    X: pd.Series,
    y: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
    model_name: str = "unnamed",
    builder_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Generate out-of-fold probability estimates for training data.

    This avoids using in-sample predictions for label-issue detection.

    Parameters
    ----------
    model_builder : callable
        Function that returns an unfitted model.
    X : pd.Series
        Training text.
    y : pd.Series
        Training labels.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for fold splitting.
    model_name : str
        Name for logging.
    builder_kwargs : dict, optional
        Keyword arguments for the model builder.

    Returns
    -------
    np.ndarray
        Out-of-fold probability matrix (n_samples, n_classes).
    """
    logger.info(
        "Generating %d-fold OOF probabilities for '%s' on %d examples...",
        n_folds, model_name, len(X),
    )

    if builder_kwargs is None:
        builder_kwargs = {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    classes = sorted(y.unique())
    n_classes = len(classes)
    oof_probs = np.zeros((len(X), n_classes))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info("  Fold %d/%d...", fold_idx + 1, n_folds)
        model = model_builder(**builder_kwargs)

        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val = X.iloc[val_idx]

        if isinstance(model, DummyClassifier):
            model.fit(np.zeros((len(X_fold_train), 1)), y_fold_train)
            probs = model.predict_proba(np.zeros((len(X_fold_val), 1)))
        else:
            model.fit(X_fold_train, y_fold_train)
            probs = model.predict_proba(X_fold_val)

        # Map probabilities to canonical class order
        model_classes = list(model.classes_)
        for i, cls in enumerate(classes):
            if cls in model_classes:
                cls_idx = model_classes.index(cls)
                oof_probs[val_idx, i] = probs[:, cls_idx]

    logger.info("OOF probability generation complete for '%s'.", model_name)
    return oof_probs


def save_model(
    model,
    model_name: str,
    output_dir: str,
    metadata: Optional[dict] = None,
) -> str:
    """Save a fitted model and its metadata.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    model_name : str
        Model identifier.
    output_dir : str
        Directory to save to.
    metadata : dict, optional
        Additional metadata to save.

    Returns
    -------
    str
        Path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    if metadata is not None:
        meta_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        # Convert non-serializable types
        clean_meta = _make_json_serializable(metadata)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(clean_meta, f, indent=2, default=str)
        logger.info("Metadata saved to %s", meta_path)

    return model_path


def load_model(model_path: str) -> object:
    """Load a saved model.

    Parameters
    ----------
    model_path : str
        Path to saved model file.

    Returns
    -------
    object
        Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
