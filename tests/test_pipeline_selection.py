"""Regression tests for final-model selection in the pipeline."""

import pytest

from src.pipeline import select_best_validation_model


def test_final_model_is_selected_by_validation_macro_f1() -> None:
    models = {"logreg": object(), "svm": object()}
    results = {
        "logreg": {"macro_f1": 0.51},
        "svm": {"macro_f1": 0.62},
    }

    name, model = select_best_validation_model(results, models)

    assert name == "svm"
    assert model is models["svm"]


def test_final_model_selection_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        select_best_validation_model({}, {})
