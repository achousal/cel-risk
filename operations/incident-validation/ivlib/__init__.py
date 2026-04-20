"""
Incident validation library (ivlib).

Modular building blocks for the incident-validation experiment:
bootstrap feature selection, 3x4 strategy/weight CV factorial,
OOF calibration, and final refit + locked-test evaluation.

Thin orchestration lives in ../scripts/run_incident_validation.py.
"""

from .data import load_and_split
from .strategies import get_training_indices, STRATEGIES, STRATEGY_TAGS
from .weights import compute_class_weight, class_weight_to_sample_weight, WEIGHT_SCHEMES
from .features import (
    run_feature_selection,
    derive_per_strategy_panels,
    derive_core_panel,
    jaccard_overlap,
)
from .modelspec import ModelSpec, get_model_spec, VALID_MODELS, MODEL_OUTPUT_DIRS
from .cvloop import run_cv, tune_and_evaluate_fold
from .calibration import fit_winner_calibrator, compute_calibration_metrics
from .finalize import final_refit_and_test, run_exhaustive_post, select_winner
from .io import save_features, load_features, save_results
from .reporting import build_report

__all__ = [
    "load_and_split",
    "get_training_indices", "STRATEGIES", "STRATEGY_TAGS",
    "compute_class_weight", "class_weight_to_sample_weight", "WEIGHT_SCHEMES",
    "run_feature_selection",
    "derive_per_strategy_panels", "derive_core_panel", "jaccard_overlap",
    "ModelSpec", "get_model_spec", "VALID_MODELS", "MODEL_OUTPUT_DIRS",
    "run_cv", "tune_and_evaluate_fold",
    "fit_winner_calibrator", "compute_calibration_metrics",
    "final_refit_and_test", "run_exhaustive_post", "select_winner",
    "save_features", "load_features", "save_results",
    "build_report",
]
