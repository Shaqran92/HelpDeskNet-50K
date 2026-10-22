"""
HelpDeskNet-50K: End-to-end pipeline runner.

Orchestrates the complete workflow from data loading through model
training, label-issue detection, and review-budget optimization.
Run with: python -m src.pipeline --config configs/experiment.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.reproducibility import (
    set_global_seeds,
    get_package_versions,
    compute_file_checksum,
)
from src.data.load_data import (
    load_raw_dataset,
    load_multi_file_corpus,
    inspect_schema,
    filter_by_language,
    select_records,
    create_ticket_id,
)
from src.data.validate_data import (
    validate_required_columns,
    validate_no_empty_text,
    generate_data_audit_report,
    compute_class_frequencies,
)
from src.data.split_data import (
    create_stratified_split,
    validate_split_integrity,
    save_split_manifest,
)
from src.labeling.map_categories import apply_label_mapping, get_mapping_report
from src.labeling.simulate_noise import NoiseSimulator, build_transition_matrix
from src.analysis.noise_metrics import (
    estimate_noise_rate,
    estimate_class_noise_rates,
    compare_worker_tiers,
    compute_audit_sample_stats,
)
from src.analysis.statistics import compute_class_balance_metrics, summarize_results
from src.models.baseline import (
    build_dummy_classifier,
    build_tfidf_logreg,
    build_tfidf_svm,
    train_model,
    predict,
    predict_proba,
    generate_oof_probabilities,
    save_model,
)
from src.models.evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_calibration_metrics,
    save_evaluation_report,
    save_predictions,
    compare_models,
    TARGET_CLASSES,
)
from src.models.robust_training import run_robust_experiment
from src.review.prioritize import (
    ReviewBudgetManager,
    simulate_review_outcome,
    compare_review_strategies,
    compute_precision_at_k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger("pipeline")


def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_best_validation_model(
    model_results: dict[str, dict], trained_models: dict[str, object]
) -> tuple[str, object]:
    """Select the final-test candidate using validation macro-F1 only.

    Keeping this choice in one small helper makes it explicit that the final
    test set cannot influence model selection.
    """
    if not model_results or not trained_models:
        raise ValueError("model_results and trained_models must not be empty")
    best_name = max(model_results, key=lambda name: model_results[name]["macro_f1"])
    return best_name, trained_models[best_name]


def run_pipeline(
    data_config_path: str = "configs/data.yaml",
    exp_config_path: str = "configs/experiment.yaml",
):
    """Run the complete HelpDeskNet-50K pipeline.

    Parameters
    ----------
    data_config_path : str
        Path to data configuration.
    exp_config_path : str
        Path to experiment configuration.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("HelpDeskNet-50K Pipeline Starting")
    logger.info("=" * 60)

    # Load configs
    data_cfg = load_config(data_config_path)
    exp_cfg = load_config(exp_config_path)
    seed = exp_cfg.get("experiment", {}).get("random_seed", 42)
    set_global_seeds(seed)

    # Create output directories
    processed_dir = data_cfg["dataset"].get("processed_dir",
                    data_cfg.get("output", {}).get("processed_dir", "data/processed"))
    artifacts_dir = data_cfg["dataset"].get("artifacts_dir",
                    data_cfg.get("output", {}).get("artifacts_dir", "artifacts"))
    metrics_dir = os.path.join(artifacts_dir, "metrics")
    models_dir = os.path.join(artifacts_dir, "models")
    reports_dir = "reports"

    for d in [processed_dir, artifacts_dir, metrics_dir, models_dir,
              reports_dir, os.path.join(reports_dir, "figures"),
              os.path.join(reports_dir, "tables")]:
        os.makedirs(d, exist_ok=True)

    # Save package versions
    versions = get_package_versions()
    with open(os.path.join(artifacts_dir, "package_versions.json"), "w") as f:
        json.dump(versions, f, indent=2)
    logger.info("Package versions saved.")

    # ========================================================
    # PHASE 1: Data Loading and Validation
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 1: Data Loading and Validation")
    logger.info("=" * 40)

    raw_files = data_cfg["dataset"].get("raw_files")
    text_col = data_cfg["dataset"]["text_column"]
    label_col = data_cfg["dataset"]["source_label_column"]

    if raw_files and isinstance(raw_files, list):
        logger.info("Multi-file dataset configuration detected (%d files).", len(raw_files))
        for rf in raw_files:
            if not os.path.exists(rf):
                logger.error("Raw data file not found: %s", rf)
                return
            cs = compute_file_checksum(rf)
            logger.info("Raw file %s (SHA-256): %s", os.path.basename(rf), cs)
        df_raw = load_multi_file_corpus(raw_files, deduplicate_on=text_col, drop_na_text=True)
    else:
        raw_file = data_cfg["dataset"]["raw_file"]
        if not os.path.exists(raw_file):
            logger.error("Raw data file not found: %s", raw_file)
            logger.error("Please place the dataset file in the expected location.")
            return
        checksum = compute_file_checksum(raw_file)
        logger.info("Raw file checksum (SHA-256): %s", checksum)
        df_raw = load_raw_dataset(raw_file)

    schema = inspect_schema(df_raw)
    logger.info("Raw dataset: %d rows, %d columns", schema["row_count"], len(schema["columns"]))

    # Validate required columns
    text_col = data_cfg["dataset"]["text_column"]
    label_col = data_cfg["dataset"]["source_label_column"]
    required_cols = [text_col, label_col]
    missing = validate_required_columns(df_raw, required_cols)
    if missing:
        logger.error("Missing required columns: %s", missing)
        return

    # Filter by language
    lang = data_cfg["dataset"].get("language_filter", "en")
    if lang:
        df = filter_by_language(df_raw, language=lang)
        logger.info("Filtered to '%s': %d rows", lang, len(df))
    else:
        df = df_raw.copy()

    # Select records
    max_records = data_cfg["dataset"].get("max_records", 50000)
    df = select_records(df, max_records=max_records, seed=seed)
    df = create_ticket_id(df)
    logger.info("Selected %d records for processing.", len(df))

    # Validate text
    empty_text = validate_no_empty_text(df, text_col)
    if len(empty_text) > 0:
        logger.warning("Found %d rows with empty text. Removing.", len(empty_text))
        df = df[df[text_col].notna() & (df[text_col].str.strip() != "")]

    # Generate audit report
    audit = generate_data_audit_report(df, os.path.join(reports_dir, "data_audit.json"))
    logger.info("Data audit report saved.")

    # Class frequencies
    freq = compute_class_frequencies(df, label_col)
    freq.to_csv(os.path.join(reports_dir, "tables", "source_class_frequencies.csv"), index=False)
    logger.info("Source class frequencies:\n%s", freq.to_string())

    # ========================================================
    # PHASE 2: Label Mapping
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 2: Label Mapping")
    logger.info("=" * 40)

    tag_cols = data_cfg["dataset"].get("tag_columns", ["tag_1", "tag_2"])
    df = apply_label_mapping(df, queue_col=label_col, tag_cols=tag_cols)
    mapping_report = get_mapping_report(df)
    logger.info("Label mapping complete. Distribution:")
    for cls, count in mapping_report.get("mapped_label_distribution", {}).items():
        logger.info("  %s: %d", cls, count)

    # Save mapped data
    df.to_csv(os.path.join(processed_dir, "mapped_dataset.csv"), index=False)

    # ========================================================
    # PHASE 3: Train/Val/Test Split
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 3: Train/Val/Test Split")
    logger.info("=" * 40)

    split_cfg = data_cfg.get("splitting", {})
    train_df, val_df, test_df = create_stratified_split(
        df,
        label_col="mapped_label",
        train_size=split_cfg.get("train_size", 0.8),
        val_size=split_cfg.get("val_size", 0.1),
        test_size=split_cfg.get("test_size", 0.1),
        seed=split_cfg.get("random_seed", seed),
        rare_class_handling="train",
    )

    integrity = validate_split_integrity(train_df, val_df, test_df)
    logger.info("Split sizes: train=%d, val=%d, test=%d",
                len(train_df), len(val_df), len(test_df))
    logger.info("Split integrity: %s", integrity.get("status", "unknown"))

    save_split_manifest(train_df, val_df, test_df, processed_dir)

    # Save split DataFrames
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # ========================================================
    # PHASE 4: Noise Simulation
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 4: Noise Simulation")
    logger.info("=" * 40)

    noise_cfg = exp_cfg.get("noise_simulation", {})
    if noise_cfg.get("enabled", True):
        simulator = NoiseSimulator(classes=TARGET_CLASSES, seed=seed)
        noise_results = {}

        for mechanism_cfg in noise_cfg.get("mechanisms", []):
            mechanism = mechanism_cfg["name"]
            for rate in mechanism_cfg.get("rates", [0.15]):
                exp_name = f"{mechanism}_{int(rate*100)}pct"
                logger.info("Simulating %s noise at %.0f%%...", mechanism, rate * 100)

                result = simulator.apply_noise(
                    train_df["mapped_label"],
                    mechanism=mechanism,
                    rate=rate,
                )

                # Merge results into training data
                train_noisy = train_df.copy()
                for col in result.columns:
                    train_noisy[col] = result[col].values

                # Save noisy dataset
                noisy_path = os.path.join(processed_dir, f"train_noisy_{exp_name}.csv")
                train_noisy.to_csv(noisy_path, index=False)
                noise_results[exp_name] = train_noisy
                logger.info("  Saved: %s (actual noise: %.2f%%)",
                            noisy_path,
                            (result["is_injected_error"].sum() / len(result)) * 100)

        # Use 15% symmetric noise as the primary experimental setting
        primary_noise_key = "symmetric_15pct"
        if primary_noise_key in noise_results:
            train_noisy = noise_results[primary_noise_key]
        else:
            # Fallback to first available
            first_key = list(noise_results.keys())[0] if noise_results else None
            if first_key:
                train_noisy = noise_results[first_key]
                primary_noise_key = first_key
            else:
                train_noisy = train_df.copy()
                train_noisy["observed_label"] = train_noisy["mapped_label"]
                train_noisy["reference_label"] = train_noisy["mapped_label"]
                train_noisy["is_injected_error"] = False
    else:
        train_noisy = train_df.copy()
        train_noisy["observed_label"] = train_noisy["mapped_label"]
        train_noisy["reference_label"] = train_noisy["mapped_label"]
        train_noisy["is_injected_error"] = False
        primary_noise_key = "no_noise"

    # ========================================================
    # PHASE 5: Statistical Analysis
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 5: Statistical Analysis")
    logger.info("=" * 40)

    noise_est = estimate_noise_rate(
        train_noisy["reference_label"],
        train_noisy["observed_label"],
    )
    logger.info("Overall noise rate: %.4f [%.4f, %.4f]",
                noise_est["noise_rate"], noise_est["ci_lower"], noise_est["ci_upper"])

    class_noise = estimate_class_noise_rates(
        train_noisy["reference_label"],
        train_noisy["observed_label"],
        TARGET_CLASSES,
    )
    logger.info("Per-class noise rates:\n%s", class_noise.to_string())

    tm = build_transition_matrix(
        train_noisy["reference_label"],
        train_noisy["observed_label"],
        TARGET_CLASSES,
    )

    # Tier analysis belongs to the worker-dependent simulation, not necessarily
    # the primary training-noise run (which is commonly symmetric).
    tier_comp = None
    worker_tier_key = "worker_dependent_15pct"
    worker_tier_df = noise_results.get(worker_tier_key) if noise_cfg.get("enabled", True) else None
    if worker_tier_df is not None and "worker_tier" in worker_tier_df.columns:
        tier_comp = compare_worker_tiers(worker_tier_df)
        logger.info("Worker-tier comparison computed from %s.", worker_tier_key)
    elif "worker_tier" in train_noisy.columns:
        tier_comp = compare_worker_tiers(train_noisy)

    analysis_summary = summarize_results(noise_est, class_noise, tier_comp, tm)

    # Save analysis results
    with open(os.path.join(metrics_dir, "analysis_summary.json"), "w") as f:
        json.dump(analysis_summary, f, indent=2, default=str)

    # ========================================================
    # PHASE 6: Baseline Models
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 6: Baseline Models")
    logger.info("=" * 40)

    # Use observed (noisy) labels for training
    X_train = train_noisy[text_col]
    y_train = train_noisy["observed_label"]
    X_val = val_df[text_col]
    y_val = val_df["mapped_label"]

    all_model_results = {}

    # 1. Dummy classifier
    logger.info("Training dummy classifier...")
    dummy = build_dummy_classifier()
    dummy = train_model(dummy, X_train, y_train, "dummy")
    y_pred_dummy = predict(dummy, X_val, "dummy")
    metrics_dummy = compute_classification_metrics(
        y_val, y_pred_dummy, TARGET_CLASSES,
        split_name="validation", label_provenance="proxy",
    )
    all_model_results["dummy"] = metrics_dummy
    save_model(dummy, "dummy", models_dir, metadata=metrics_dummy)

    # 2. TF-IDF + LogReg
    logger.info("Training TF-IDF + Logistic Regression...")
    logreg = build_tfidf_logreg()
    logreg = train_model(logreg, X_train, y_train, "tfidf_logreg")
    y_pred_lr = predict(logreg, X_val, "tfidf_logreg")
    y_proba_lr = predict_proba(logreg, X_val, "tfidf_logreg")
    metrics_lr = compute_classification_metrics(
        y_val, y_pred_lr, TARGET_CLASSES,
        split_name="validation", label_provenance="proxy",
    )
    all_model_results["tfidf_logreg"] = metrics_lr
    save_model(logreg, "tfidf_logreg", models_dir, metadata=metrics_lr)

    # 3. TF-IDF + SVM
    logger.info("Training TF-IDF + Linear SVM...")
    svm = build_tfidf_svm()
    svm = train_model(svm, X_train, y_train, "tfidf_svm")
    y_pred_svm = predict(svm, X_val, "tfidf_svm")
    y_proba_svm = predict_proba(svm, X_val, "tfidf_svm")
    metrics_svm = compute_classification_metrics(
        y_val, y_pred_svm, TARGET_CLASSES,
        split_name="validation", label_provenance="proxy",
    )
    all_model_results["tfidf_svm"] = metrics_svm
    save_model(svm, "tfidf_svm", models_dir, metadata=metrics_svm)

    trained_models = {"dummy": dummy, "tfidf_logreg": logreg, "tfidf_svm": svm}
    selected_model_name, selected_model = select_best_validation_model(
        all_model_results, trained_models
    )
    logger.info("Selected '%s' for final testing using validation macro-F1.", selected_model_name)

    # Save predictions
    if y_proba_lr is not None:
        save_predictions(
            val_df["ticket_id"], y_val, y_pred_lr, y_proba_lr,
            TARGET_CLASSES,
            os.path.join(artifacts_dir, "predictions_val_logreg.csv"),
            "tfidf_logreg", "validation",
        )

    # Model comparison
    comparison = compare_models(all_model_results)
    comparison.to_csv(os.path.join(reports_dir, "tables", "model_comparison.csv"), index=False)
    logger.info("Model comparison:\n%s", comparison.to_string())

    # Calibration (for LogReg which has probabilities)
    if y_proba_lr is not None:
        cal_metrics = compute_calibration_metrics(y_val, y_proba_lr, TARGET_CLASSES)
        with open(os.path.join(metrics_dir, "calibration_logreg.json"), "w") as f:
            json.dump(cal_metrics, f, indent=2, default=str)
        logger.info("Calibration ECE: %.4f, Mean Brier: %.4f",
                     cal_metrics["ece"], cal_metrics["mean_brier"])

    # ========================================================
    # PHASE 7: Label-Issue Detection (OOF + Cleanlab)
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 7: Label-Issue Detection")
    logger.info("=" * 40)

    # Generate OOF probabilities
    n_folds = exp_cfg.get("experiment", {}).get("n_folds", 5)
    oof_probs = generate_oof_probabilities(
        build_tfidf_logreg, X_train, y_train,
        n_folds=n_folds, seed=seed, model_name="tfidf_logreg_oof",
    )

    # Save OOF probabilities
    np.save(os.path.join(artifacts_dir, "oof_probabilities.npy"), oof_probs)

    # Cleanlab integration
    try:
        from cleanlab.rank import get_label_quality_scores
        train_classes = sorted(y_train.unique())
        class_to_idx = {c: i for i, c in enumerate(train_classes)}
        int_labels = np.array([class_to_idx[l] for l in y_train.values])
        quality_scores = get_label_quality_scores(
            labels=int_labels,
            pred_probs=oof_probs,
        )
        # Invert so higher = more suspicious
        label_issue_scores = 1.0 - quality_scores
        logger.info("Cleanlab label-issue scores computed successfully.")
    except ImportError:
        logger.warning("Cleanlab not available. Using entropy-based scoring.")
        # Fallback: use prediction entropy as issue score
        probs_clipped = np.clip(oof_probs, 1e-10, 1.0)
        label_issue_scores = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        max_entropy = np.log(len(TARGET_CLASSES))
        label_issue_scores = label_issue_scores / max_entropy
    except Exception as e:
        logger.error("Cleanlab error: %s. Using entropy fallback.", e)
        probs_clipped = np.clip(oof_probs, 1e-10, 1.0)
        label_issue_scores = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
        max_entropy = np.log(len(TARGET_CLASSES))
        label_issue_scores = label_issue_scores / max_entropy

    # Add scores to training data
    train_noisy["label_issue_score"] = label_issue_scores

    # Evaluate against known injected errors
    true_errors = set(
        train_noisy[train_noisy["is_injected_error"] == True]["ticket_id"]
    )
    ranked_ids = train_noisy.sort_values("label_issue_score", ascending=False)["ticket_id"]
    detection_metrics = compute_precision_at_k(ranked_ids, true_errors)
    logger.info("Label-issue detection metrics:")
    for k, v in detection_metrics.items():
        logger.info("  %s: %s", k, v)

    with open(os.path.join(metrics_dir, "detection_metrics.json"), "w") as f:
        json.dump(detection_metrics, f, indent=2, default=str)

    # ========================================================
    # PHASE 8: Robust-learning comparison (validation only)
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 8: Robust-learning comparison")
    logger.info("=" * 40)

    def _robust_evaluate(y_true, y_pred):
        return compute_classification_metrics(
            y_true, y_pred, TARGET_CLASSES,
            split_name="validation",
            label_provenance="proxy (source labels, not independently verified)",
        )

    robust_results = run_robust_experiment(
        build_tfidf_logreg, X_train, y_train, X_val, y_val,
        label_issue_scores=label_issue_scores, evaluate_fn=_robust_evaluate,
        model_name="tfidf_logreg",
    )
    robust_rows = []
    for strategy in ("standard", "filtered", "weighted", "corrected"):
        metrics = robust_results.get(strategy)
        if metrics is not None:
            robust_rows.append({
                "strategy": strategy,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "label_provenance": metrics["label_provenance"],
            })
    pd.DataFrame(robust_rows).to_csv(
        os.path.join(reports_dir, "tables", "robust_learning_comparison.csv"), index=False
    )
    with open(os.path.join(metrics_dir, "robust_learning.json"), "w") as f:
        json.dump(robust_results, f, indent=2, default=str)

    # ========================================================
    # PHASE 9: Review Budget Optimization
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 8: Review Budget Optimization")
    logger.info("=" * 40)

    budget_cfg = exp_cfg.get("review_budget", {})
    budget_mgr = ReviewBudgetManager(
        total_budget=budget_cfg.get("total", 2000),
        high_priority=budget_cfg.get("high_priority", 1000),
        stratified=budget_cfg.get("stratified", 600),
        random_audit=budget_cfg.get("random_audit", 400),
        seed=seed,
    )

    # Eligible pool = training set only (never test)
    eligible = train_noisy.copy()

    # Strategy 1: Random
    random_ids = budget_mgr.select_random(eligible["ticket_id"])

    # Strategy 2: Issue score
    issue_ids = budget_mgr.select_by_issue_score(eligible)

    # Strategy 3: Combined
    combined_ids, combined_log = budget_mgr.select_combined(
        eligible,
        score_col="label_issue_score",
        label_col="observed_label",
    )

    # Compare strategies
    strategies = {
        "random": random_ids,
        "issue_score": issue_ids,
        "combined": combined_ids,
    }

    strategy_comparison = compare_review_strategies(
        eligible, strategies,
        reference_col="reference_label",
        observed_col="observed_label",
    )
    logger.info("Review strategy comparison:\n%s", strategy_comparison.to_string())

    strategy_comparison.to_csv(
        os.path.join(reports_dir, "tables", "review_strategy_comparison.csv"),
        index=False,
    )

    # Simulate review outcomes for best strategy
    best_strategy = strategy_comparison.iloc[0]["strategy"]
    best_ids = strategies[best_strategy]
    selected_for_review = eligible[eligible["ticket_id"].isin(best_ids)]
    review_outcomes = simulate_review_outcome(selected_for_review)
    review_outcomes.to_csv(
        os.path.join(artifacts_dir, "review_outcomes.csv"),
        index=False,
    )

    # Combined log
    combined_log.to_csv(
        os.path.join(artifacts_dir, "review_selection_log.csv"),
        index=False,
    )

    # ========================================================
    # PHASE 9: Final Test Evaluation
    # ========================================================
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 9: Final Test Evaluation")
    logger.info("=" * 40)

    X_test = test_df[text_col]
    y_test = test_df["mapped_label"]

    # Evaluate only the validation-selected model on the final test set.
    y_pred_test = predict(selected_model, X_test, selected_model_name)
    y_proba_test = predict_proba(selected_model, X_test, selected_model_name)

    test_metrics = compute_classification_metrics(
        y_test, y_pred_test, TARGET_CLASSES,
        split_name="final_test",
        label_provenance="proxy (source labels, not independently verified)",
    )
    logger.info("Final test metrics:")
    logger.info("  Accuracy: %.4f", test_metrics["accuracy"])
    logger.info("  Macro-F1: %.4f", test_metrics["macro_f1"])

    save_evaluation_report(
        test_metrics,
        os.path.join(metrics_dir, "final_test_metrics.json"),
        selected_model_name,
    )

    # Confusion matrix
    cm = compute_confusion_matrix(y_test, y_pred_test, TARGET_CLASSES)
    cm.to_csv(os.path.join(reports_dir, "tables", "test_confusion_matrix.csv"))

    if y_proba_test is not None:
        save_predictions(
            test_df["ticket_id"], y_test, y_pred_test, y_proba_test,
            TARGET_CLASSES,
            os.path.join(artifacts_dir, "predictions_test.csv"),
            selected_model_name, "final_test",
        )

    # ========================================================
    # Summary
    # ========================================================
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("=" * 60)
    logger.info("Key results:")
    logger.info("  Dataset: %d records (%s language)", len(df), lang)
    logger.info("  Split: train=%d, val=%d, test=%d",
                len(train_df), len(val_df), len(test_df))
    logger.info("  Primary noise: %s", primary_noise_key)
    logger.info("  Best model (val): %s (macro-F1=%.4f)",
                comparison.iloc[0]["model"], comparison.iloc[0]["macro_f1"])
    logger.info("  Test macro-F1: %.4f", test_metrics["macro_f1"])
    logger.info("  Best review strategy: %s (%d errors found)",
                best_strategy,
                int(strategy_comparison.iloc[0]["n_errors_found"]))

    # Save final summary
    summary = {
        "pipeline_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "dataset_records": len(df),
        "language_filter": lang,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "primary_noise": primary_noise_key,
        "model_comparison": comparison.to_dict(orient="records"),
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        },
        "review_comparison": strategy_comparison.to_dict(orient="records"),
        "detection_metrics": detection_metrics,
        "label_provenance": "proxy (source labels used as reference)",
    }

    with open(os.path.join(artifacts_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Summary saved to %s", os.path.join(artifacts_dir, "pipeline_summary.json"))


def main() -> None:
    """Run the command-line pipeline entry point."""
    parser = argparse.ArgumentParser(description="HelpDeskNet-50K Pipeline")
    parser.add_argument(
        "--data-config", default="configs/data.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--exp-config", default="configs/experiment.yaml",
        help="Path to experiment configuration file",
    )
    args = parser.parse_args()
    run_pipeline(args.data_config, args.exp_config)


if __name__ == "__main__":
    main()
