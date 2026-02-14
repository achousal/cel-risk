"""
Models package for CeliacRiskML.

This package contains model-related functionality including:
- Calibration metrics and wrappers
- Prevalence adjustment for risk calibration
- Model registry and hyperparameter distributions
- Nested CV training orchestration
"""

from .calibration import (
    CalibrationMetrics,
    OOFCalibratedModel,
    OOFCalibrator,
    apply_oof_calibrator,
    calib_intercept_metric,
    calib_slope_metric,
    calibration_intercept_slope,
    expected_calibration_error,
    fit_oof_calibrator,
    get_calibrated_cv_param_name,
    get_calibrated_estimator_param_name,
    maybe_calibrate_estimator,
)
from .hyperparams import (
    get_param_distributions,
)
from .nested_cv import (
    NestedCVResult,
    get_model_n_iter,
    oof_predictions_with_nested_cv,
)
from .prevalence import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
)
from .prevalence import (
    PrevalenceAdjustedModel as PrevalenceModel,
)
from .prevalence import (
    adjust_probabilities_for_prevalence as adjust_prevalence,
)
from .registry import (
    build_linear_svm_calibrated,
    build_logistic_regression,
    build_models,
    build_random_forest,
    build_xgboost,
    compute_scale_pos_weight_from_y,
    make_logspace,
    parse_class_weight_options,
)
from .training import (
    oof_predictions_with_nested_cv as _oof_predictions_with_nested_cv_compat,
)

__all__ = [
    # Calibration
    "CalibrationMetrics",
    "calibration_intercept_slope",
    "calib_intercept_metric",
    "calib_slope_metric",
    "expected_calibration_error",
    "adjust_probabilities_for_prevalence",
    "PrevalenceAdjustedModel",
    "get_calibrated_estimator_param_name",
    "get_calibrated_cv_param_name",
    "maybe_calibrate_estimator",
    "OOFCalibrator",
    "OOFCalibratedModel",
    "fit_oof_calibrator",
    "apply_oof_calibrator",
    # Prevalence (aliases)
    "adjust_prevalence",
    "PrevalenceModel",
    # Registry
    "build_models",
    "build_logistic_regression",
    "build_linear_svm_calibrated",
    "build_random_forest",
    "build_xgboost",
    "make_logspace",
    "parse_class_weight_options",
    "compute_scale_pos_weight_from_y",
    # Hyperparams
    "get_param_distributions",
    # Nested CV (training)
    "NestedCVResult",
    "get_model_n_iter",
    "oof_predictions_with_nested_cv",
]
