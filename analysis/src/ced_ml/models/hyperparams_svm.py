"""
Linear SVM hyperparameter search space definitions.

Provides:
- Parameter distributions for RandomizedSearchCV
- Optuna parameter distributions with native range specifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hyperparams_common import (
    _make_logspace,
    _parse_class_weight_options,
)

if TYPE_CHECKING:
    from ..config import TrainingConfig


def _get_svm_params(
    config: TrainingConfig, randomize: bool, rng: np.random.Generator | None
) -> dict[str, list]:
    """Linear SVM hyperparameters (wrapped in CalibratedClassifierCV)."""
    # C values
    C_grid = _make_logspace(
        config.svm.C_min,
        config.svm.C_max,
        config.svm.C_points,
        rng=rng if randomize else None,
    )

    # Class weights
    class_weight_options = _parse_class_weight_options(config.svm.class_weight_options)

    # Parameter prefix depends on sklearn version
    # Newer: estimator__C, older: base_estimator__C
    # Use estimator__ (modern sklearn)
    params = {"clf__estimator__C": C_grid}
    if class_weight_options:
        params["clf__estimator__class_weight"] = class_weight_options

    return params


def _get_svm_params_optuna(config: TrainingConfig) -> dict[str, dict]:
    """Build Optuna specs for Linear SVM."""
    params = {}

    # C parameter - log scale
    if config.svm.optuna_C is not None:
        c_low, c_high = config.svm.optuna_C
    else:
        c_low, c_high = config.svm.C_min, config.svm.C_max
    params["clf__estimator__C"] = {
        "type": "float",
        "low": c_low,
        "high": c_high,
        "log": True,
    }

    # Class weights (categorical)
    class_weight_options = _parse_class_weight_options(config.svm.class_weight_options)
    if class_weight_options and len(class_weight_options) > 1:
        params["clf__estimator__class_weight"] = {
            "type": "categorical",
            "choices": class_weight_options,
        }

    return params
