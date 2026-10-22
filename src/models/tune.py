"""
Hyperparameter tuning and model optimization module for HelpDeskNet-50K.

Evaluates regularization schedules, class weighting strategies, and n-gram representations
using strictly held-out validation sets to prevent test-set data leakage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

TARGET_CLASSES = ["Billing", "Refund", "Technical", "Account Access", "Shipping", "Other"]


def evaluate_trial_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    classes: list[str] = TARGET_CLASSES,
) -> dict[str, Any]:
    """Compute comprehensive classification performance metrics for a tuning trial."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class = {}
    for cls in classes:
        if cls in report:
            per_class[cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class,
    }


def run_hyperparameter_search(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    classes: list[str] = TARGET_CLASSES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute systematic hyperparameter grid optimization across SVM and Logistic Regression.

    Evaluates:
    - Regularization strength C for LinearSVC: [0.2, 0.5, 1.0, 2.0]
    - Regularization strength C for LogisticRegression: [0.5, 1.0, 2.0, 5.0, 8.0]
    - Class weighting: [None, 'balanced']
    - TF-IDF feature representations (sublinear TF, n-gram ranges (1, 1) and (1, 2))

    Returns:
        dict containing 'trials_df', 'best_model', 'best_params', 'best_macro_f1', 'summary'.
    """
    logger.info("Initiating hyperparameter search across %d train and %d val samples...", len(X_train), len(X_val))

    # Fit canonical TF-IDF representation
    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_tr_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)

    trials: list[dict[str, Any]] = []

    # 1. LinearSVC Grid
    svc_c_values = [0.2, 0.5, 1.0, 2.0]
    for C in svc_c_values:
        for cw in [None, "balanced"]:
            t0 = time.time()
            clf = LinearSVC(
                C=C,
                class_weight=cw,
                max_iter=4000,
                random_state=random_state,
                dual="auto",
            )
            clf.fit(X_tr_vec, y_train)
            fit_time = time.time() - t0

            preds = clf.predict(X_val_vec)
            metrics = evaluate_trial_predictions(y_val, preds, classes=classes)

            trial_record = {
                "trial_id": f"svm_C{C}_cw{cw}",
                "model_family": "LinearSVC",
                "C": C,
                "class_weight": str(cw),
                "fit_time_seconds": round(fit_time, 2),
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "macro_precision": round(metrics["macro_precision"], 4),
                "macro_recall": round(metrics["macro_recall"], 4),
            }
            trials.append(trial_record)
            logger.info("Trial %s: Accuracy=%.4f, Macro-F1=%.4f (%.2fs)", trial_record["trial_id"], metrics["accuracy"], metrics["macro_f1"], fit_time)

    # 2. Logistic Regression Grid
    logreg_c_values = [0.5, 1.0, 2.0, 5.0, 8.0]
    for C in logreg_c_values:
        for cw in [None, "balanced"]:
            t0 = time.time()
            clf = LogisticRegression(
                C=C,
                class_weight=cw,
                max_iter=400,
                tol=1e-3,
                random_state=random_state,
                solver="lbfgs",
            )
            clf.fit(X_tr_vec, y_train)
            fit_time = time.time() - t0

            preds = clf.predict(X_val_vec)
            metrics = evaluate_trial_predictions(y_val, preds, classes=classes)

            trial_record = {
                "trial_id": f"logreg_C{C}_cw{cw}",
                "model_family": "LogisticRegression",
                "C": C,
                "class_weight": str(cw),
                "fit_time_seconds": round(fit_time, 2),
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "macro_precision": round(metrics["macro_precision"], 4),
                "macro_recall": round(metrics["macro_recall"], 4),
            }
            trials.append(trial_record)
            logger.info("Trial %s: Accuracy=%.4f, Macro-F1=%.4f (%.2fs)", trial_record["trial_id"], metrics["accuracy"], metrics["macro_f1"], fit_time)

    trials_df = pd.DataFrame(trials).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best_trial = trials_df.iloc[0].to_dict()

    logger.info("Hyperparameter search complete. Top performer: %s with Macro-F1=%.4f", best_trial["trial_id"], best_trial["macro_f1"])

    return {
        "trials_df": trials_df,
        "best_trial": best_trial,
        "total_trials": len(trials),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_tuning_artifacts(
    tuning_results: dict[str, Any],
    reports_dir: str | Path = "reports",
    artifacts_dir: str | Path = "artifacts",
) -> None:
    """Save hyperparameter tuning tables and JSON summaries."""
    tables_dir = Path(reports_dir) / "tables"
    metrics_dir = Path(artifacts_dir) / "metrics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    trials_df = tuning_results["trials_df"]
    trials_csv_path = tables_dir / "hyperparameter_tuning.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info("Saved tuning table to %s", trials_csv_path)

    summary = {
        "best_trial": tuning_results["best_trial"],
        "total_trials": tuning_results["total_trials"],
        "timestamp": tuning_results["timestamp"],
        "top_5_trials": trials_df.head(5).to_dict(orient="records"),
    }
    summary_path = metrics_dir / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved tuning summary JSON to %s", summary_path)
