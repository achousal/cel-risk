"""
Nested cross-validation orchestration for model training.

This module provides backward compatibility imports from the nested_cv module.
All nested CV functionality has been moved to ced_ml.models.nested_cv.

Provides:
- Out-of-fold (OOF) prediction generation with repeated stratified CV
- Nested hyperparameter tuning (outer CV + inner GridSearchCV/RandomizedSearchCV)
- Feature selection tracking across CV folds
- Protein selection extraction from fitted models
- Optional post-hoc calibration
"""

# Re-export all functions from nested_cv for backward compatibility
# Including private functions that may be used by other modules
from .nested_cv import (
    _DEFAULT_N_ITER,
    NestedCVResult,
    _apply_per_fold_calibration,
    _build_hyperparameter_search,
    _convert_numpy_types,
    _extract_from_kbest_transformed,
    _extract_from_model_coefficients,
    _extract_from_rf_permutation,
    _extract_selected_proteins_from_fold,
    _get_model_feature_count,
    _get_search_n_jobs,
    _resolve_rfecv_step,
    _scoring_to_direction,
    get_model_n_iter,
    oof_predictions_with_nested_cv,
)

__all__ = [
    "NestedCVResult",
    "get_model_n_iter",
    "oof_predictions_with_nested_cv",
    # Private functions and constants re-exported for backward compatibility
    "_DEFAULT_N_ITER",
    "_apply_per_fold_calibration",
    "_build_hyperparameter_search",
    "_convert_numpy_types",
    "_extract_from_kbest_transformed",
    "_extract_from_model_coefficients",
    "_extract_from_rf_permutation",
    "_extract_selected_proteins_from_fold",
    "_get_model_feature_count",
    "_get_search_n_jobs",
    "_resolve_rfecv_step",
    "_scoring_to_direction",
]
