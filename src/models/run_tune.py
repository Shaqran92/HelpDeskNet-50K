"""
Run systematic hyperparameter optimization for HelpDeskNet-50K models.

Optimizes regularization parameters and class weighting on the validation split,
evaluates out-of-sample improvements, updates model comparison artifacts,
and persists tuned model pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.tune import (
    run_hyperparameter_search,
    save_tuning_artifacts,
    evaluate_trial_predictions,
    TARGET_CLASSES,
)
from src.models.evaluate import compute_classification_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("tune_runner")


def main():
    logger.info("=== HelpDeskNet-50K Hyperparameter Optimization Runner ===")
    
    # 1. Load Datasets
    data_dir = ROOT_DIR / "data" / "processed"
    train_path = data_dir / "train_noisy_symmetric_15pct.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not val_path.exists():
        logger.error("Required processed datasets not found. Ensure pipeline has run.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    logger.info("Loaded datasets: %d train, %d val, %d test samples",
                len(train_df), len(val_df), len(test_df) if test_df is not None else 0)

    X_train = train_df["body"].fillna("")
    y_train = train_df["observed_label"]
    X_val = val_df["body"].fillna("")
    y_val = val_df["mapped_label"]

    # 2. Run Systematic Search
    logger.info("Starting hyperparameter search across SVM and Logistic Regression...")
    t0 = time.time()
    search_results = run_hyperparameter_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        classes=TARGET_CLASSES,
        random_state=42,
    )
    search_duration = time.time() - t0
    logger.info("Hyperparameter search finished in %.2f seconds.", search_duration)

    # 3. Save Tuning Artifacts
    save_tuning_artifacts(
        search_results,
        reports_dir=ROOT_DIR / "reports",
        artifacts_dir=ROOT_DIR / "artifacts",
    )

    trials_df = search_results["trials_df"]
    print("\n" + "=" * 75)
    print("TOP 5 HYPERPARAMETER CONFIGURATIONS")
    print("=" * 75)
    print(trials_df[["trial_id", "model_family", "C", "class_weight", "accuracy", "macro_f1", "weighted_f1", "fit_time_seconds"]].head(5).to_string(index=False))
    print("=" * 75 + "\n")

    # 4. Identify Best SVM and Best LogReg
    best_svm_trial = trials_df[trials_df["model_family"] == "LinearSVC"].iloc[0]
    best_lr_trial = trials_df[trials_df["model_family"] == "LogisticRegression"].iloc[0]

    logger.info("Optimal SVM configuration: C=%s, class_weight=%s -> Macro-F1=%.4f, Acc=%.4f",
                best_svm_trial["C"], best_svm_trial["class_weight"],
                best_svm_trial["macro_f1"], best_svm_trial["accuracy"])
    logger.info("Optimal LogReg configuration: C=%s, class_weight=%s -> Macro-F1=%.4f, Acc=%.4f",
                best_lr_trial["C"], best_lr_trial["class_weight"],
                best_lr_trial["macro_f1"], best_lr_trial["accuracy"])

    # 5. Build and Persist Full End-to-End Tuned Pipelines
    models_dir = ROOT_DIR / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Best Tuned SVM Pipeline
    svm_cw = None if best_svm_trial["class_weight"] == "None" else "balanced"
    tuned_svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", LinearSVC(
            C=float(best_svm_trial["C"]),
            class_weight=svm_cw,
            max_iter=4000,
            random_state=42,
            dual="auto",
        )),
    ])
    logger.info("Retraining tuned SVM pipeline on full noisy training corpus...")
    tuned_svm_pipe.fit(X_train, y_train)
    joblib.dump(tuned_svm_pipe, models_dir / "tfidf_svm_tuned.joblib")
    # Also save as tfidf_svm.joblib so live inference console immediately serves it
    joblib.dump(tuned_svm_pipe, models_dir / "tfidf_svm.joblib")
    logger.info("Saved tuned SVM pipeline to artifacts/models/tfidf_svm.joblib")

    # Best Tuned Logistic Regression Pipeline
    lr_cw = None if best_lr_trial["class_weight"] == "None" else "balanced"
    tuned_lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )),
        ("clf", LogisticRegression(
            C=float(best_lr_trial["C"]),
            class_weight=lr_cw,
            max_iter=400,
            tol=1e-3,
            random_state=42,
            solver="lbfgs",
        )),
    ])
    logger.info("Retraining tuned Logistic Regression pipeline on full noisy training corpus...")
    tuned_lr_pipe.fit(X_train, y_train)
    joblib.dump(tuned_lr_pipe, models_dir / "tfidf_logreg_tuned.joblib")
    joblib.dump(tuned_lr_pipe, models_dir / "tfidf_logreg.joblib")
    logger.info("Saved tuned Logistic Regression pipeline to artifacts/models/tfidf_logreg.joblib")

    # 6. Evaluate on Validation and Test Sets
    val_preds_svm = tuned_svm_pipe.predict(X_val)
    val_preds_lr = tuned_lr_pipe.predict(X_val)

    metrics_svm_val = compute_classification_metrics(
        y_val, val_preds_svm, TARGET_CLASSES, split_name="validation", label_provenance="proxy"
    )
    metrics_lr_val = compute_classification_metrics(
        y_val, val_preds_lr, TARGET_CLASSES, split_name="validation", label_provenance="proxy"
    )

    test_metrics_svm = None
    if test_df is not None:
        X_test = test_df["body"].fillna("")
        y_test = test_df["mapped_label"]
        test_preds_svm = tuned_svm_pipe.predict(X_test)
        test_metrics_svm = compute_classification_metrics(
            y_test, test_preds_svm, TARGET_CLASSES, split_name="test", label_provenance="proxy"
        )
        logger.info("Tuned SVM on Test Set -> Accuracy: %.4f, Macro-F1: %.4f",
                    test_metrics_svm["accuracy"], test_metrics_svm["macro_f1"])

    # 7. Update Model Comparison Table (Validation)
    comp_path = ROOT_DIR / "reports" / "tables" / "model_comparison.csv"
    existing_comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

    comparison_records = [
        {
            "model": "tfidf_svm (tuned)",
            "accuracy": metrics_svm_val["accuracy"],
            "macro_f1": metrics_svm_val["macro_f1"],
            "weighted_f1": metrics_svm_val["weighted_f1"],
            "macro_precision": metrics_svm_val["macro_precision"],
            "macro_recall": metrics_svm_val["macro_recall"],
            "n_samples": len(val_df),
            "split": "validation",
            "label_provenance": "proxy",
        },
        {
            "model": "tfidf_logreg (tuned)",
            "accuracy": metrics_lr_val["accuracy"],
            "macro_f1": metrics_lr_val["macro_f1"],
            "weighted_f1": metrics_lr_val["weighted_f1"],
            "macro_precision": metrics_lr_val["macro_precision"],
            "macro_recall": metrics_lr_val["macro_recall"],
            "n_samples": len(val_df),
            "split": "validation",
            "label_provenance": "proxy",
        },
        {
            "model": "tfidf_svm (baseline)",
            "accuracy": 0.7590,
            "macro_f1": 0.5566,
            "weighted_f1": 0.7402,
            "macro_precision": 0.8968,
            "macro_recall": 0.4891,
            "n_samples": len(val_df),
            "split": "validation",
            "label_provenance": "proxy",
        },
        {
            "model": "tfidf_logreg (baseline)",
            "accuracy": 0.7400,
            "macro_f1": 0.5032,
            "weighted_f1": 0.7175,
            "macro_precision": 0.8953,
            "macro_recall": 0.4391,
            "n_samples": len(val_df),
            "split": "validation",
            "label_provenance": "proxy",
        },
        {
            "model": "dummy",
            "accuracy": 0.4962,
            "macro_f1": 0.1105,
            "weighted_f1": 0.3291,
            "macro_precision": 0.0827,
            "macro_recall": 0.1667,
            "n_samples": len(val_df),
            "split": "validation",
            "label_provenance": "proxy",
        },
    ]

    new_comp_df = pd.DataFrame(comparison_records)
    new_comp_df.to_csv(comp_path, index=False)
    # Also save to artifacts/metrics/
    metrics_dir = ROOT_DIR / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    new_comp_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
    logger.info("Saved updated model comparison to %s and %s", comp_path, metrics_dir / "model_comparison.csv")

    # 8. Update pipeline_summary.json
    summary_path = ROOT_DIR / "artifacts" / "pipeline_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["model_comparison"] = comparison_records
        if test_metrics_svm:
            summary["test_metrics"] = {
                "accuracy": round(test_metrics_svm["accuracy"], 4),
                "macro_f1": round(test_metrics_svm["macro_f1"], 4),
                "model": "tfidf_svm (tuned)",
                "note": "Evaluated on strictly held-out untouched test split (5,000 samples)",
            }
        summary["hyperparameter_tuning"] = {
            "total_trials": len(trials_df),
            "best_svm": {
                "C": float(best_svm_trial["C"]),
                "class_weight": str(best_svm_trial["class_weight"]),
                "accuracy": float(best_svm_trial["accuracy"]),
                "macro_f1": float(best_svm_trial["macro_f1"]),
            },
            "best_logreg": {
                "C": float(best_lr_trial["C"]),
                "class_weight": str(best_lr_trial["class_weight"]),
                "accuracy": float(best_lr_trial["accuracy"]),
                "macro_f1": float(best_lr_trial["macro_f1"]),
            },
            "accuracy_gain_vs_baseline": round(float(best_svm_trial["accuracy"]) - 0.7590, 4),
            "macro_f1_gain_vs_baseline": round(float(best_svm_trial["macro_f1"]) - 0.5566, 4),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Updated pipeline summary with hyperparameter tuning metrics.")

    print("\n" + "=" * 75)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 75)
    print(f"LinearSVM Tuned:  Accuracy = {metrics_svm_val['accuracy']:.4f} (baseline: 0.7590, +{(metrics_svm_val['accuracy'] - 0.7590)*100:+.2f}%)")
    print(f"                  Macro-F1 = {metrics_svm_val['macro_f1']:.4f} (baseline: 0.5566, +{(metrics_svm_val['macro_f1'] - 0.5566)*100:+.2f}%)")
    print(f"LogReg Tuned:     Accuracy = {metrics_lr_val['accuracy']:.4f} (baseline: 0.7400, +{(metrics_lr_val['accuracy'] - 0.7400)*100:+.2f}%)")
    print(f"                  Macro-F1 = {metrics_lr_val['macro_f1']:.4f} (baseline: 0.5032, +{(metrics_lr_val['macro_f1'] - 0.5032)*100:+.2f}%)")
    if test_metrics_svm:
        print(f"Test Split Eval:  Accuracy = {test_metrics_svm['accuracy']:.4f}, Macro-F1 = {test_metrics_svm['macro_f1']:.4f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
