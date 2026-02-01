"""Evaluation module for model performance assessment."""

from ced_ml.evaluation.holdout import (
    HoldoutResult,
    compute_holdout_metrics,
    compute_top_risk_capture,
    evaluate_holdout,
    extract_holdout_data,
    load_holdout_indices,
    load_model_artifact,
    save_holdout_predictions,
)
from ced_ml.evaluation.predict import (
    export_predictions,
    generate_predictions,
    generate_predictions_with_adjustment,
    predict_on_holdout,
    predict_on_test,
    predict_on_validation,
)
from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter

__all__ = [
    "OutputDirectories",
    "ResultsWriter",
    "generate_predictions",
    "generate_predictions_with_adjustment",
    "export_predictions",
    "predict_on_validation",
    "predict_on_test",
    "predict_on_holdout",
    "evaluate_holdout",
    "HoldoutResult",
    "load_holdout_indices",
    "load_model_artifact",
    "extract_holdout_data",
    "compute_holdout_metrics",
    "compute_top_risk_capture",
    "save_holdout_predictions",
]
