"""
Hyperparameter search space definitions for all models.

This module serves as a facade that re-exports hyperparameter utilities
from model-specific modules:
- hyperparams_common: Shared utilities (grid generation, parsing, randomization)
- hyperparams_lr: Logistic Regression hyperparameters
- hyperparams_svm: Linear SVM hyperparameters
- hyperparams_rf: Random Forest hyperparameters
- hyperparams_xgb: XGBoost hyperparameters

Provides:
- Parameter distributions for RandomizedSearchCV
- Grid randomization for sensitivity analysis
- Model-specific tuning ranges
- Optuna parameter distributions
- RFE tune spaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..data.schema import ModelName

# Re-export common utilities
from .hyperparams_common import (
    DEFAULT_OPTUNA_RANGES,
    RFE_TUNE_SPACES,
    _is_log_spaced,
    _make_logspace,
    _parse_class_weight_options,
    _randomize_float_list,
    _randomize_int_list,
    _to_optuna_spec,
    get_rfe_tune_space,
    get_rfe_tune_spaces_from_training_config,
    resolve_class_weight,
    resolve_class_weights_in_params,
)

# Import model-specific functions (not re-exported as public API)
from .hyperparams_lr import _get_lr_params, _get_lr_params_optuna
from .hyperparams_rf import _get_rf_params, _get_rf_params_optuna
from .hyperparams_svm import _get_svm_params, _get_svm_params_optuna
from .hyperparams_xgb import _get_xgb_params, _get_xgb_params_optuna

if TYPE_CHECKING:
    from ..config import TrainingConfig


def get_param_distributions(
    model_name: str,
    config: TrainingConfig,
    xgb_spw: float | None = None,
    grid_rng: np.random.Generator | None = None,
) -> dict[str, list]:
    """
    Get parameter distribution for RandomizedSearchCV.

    Args:
        model_name: Model identifier (RF, XGBoost, LR_EN, LR_L1, LinSVM_cal)
        config: TrainingConfiguration object
        xgb_spw: XGBoost scale_pos_weight (optional override)
        grid_rng: Optional RNG for grid randomization (sensitivity analysis)

    Returns:
        Dictionary mapping parameter names to value lists
        Empty dict if model has no hyperparameters to tune
    """
    param_dists = {}
    randomize = grid_rng is not None

    # Feature selection parameters (if applicable)
    strategy = config.features.feature_selection_strategy
    if strategy == "multi_stage":
        k_grid = config.features.k_grid
        if not k_grid:
            raise ValueError("feature_selection_strategy='multi_stage' requires features.k_grid")

        # Always use 'sel' step name
        param_dists["sel__selector__k"] = k_grid

    # Model-specific parameters
    if model_name in (ModelName.LR_EN, ModelName.LR_L1):
        param_dists.update(_get_lr_params(config, randomize, grid_rng, model_name=model_name))

    elif model_name == ModelName.LinSVM_cal:
        param_dists.update(_get_svm_params(config, randomize, grid_rng))

    elif model_name == ModelName.RF:
        param_dists.update(_get_rf_params(config, randomize, grid_rng))

    elif model_name == ModelName.XGBoost:
        param_dists.update(_get_xgb_params(config, xgb_spw, randomize, grid_rng))

    return param_dists


def get_param_distributions_optuna(
    model_name: str,
    config: TrainingConfig,
    xgb_spw: float | None = None,
) -> dict[str, dict]:
    """
    Build Optuna parameter distributions with native range specifications.

    This function builds Optuna suggest specs with:
    - Wider search ranges than grid-based approaches
    - Proper log-scale sampling for parameters spanning orders of magnitude
    - Support for config-specified Optuna ranges (optuna_* fields)

    Optuna spec format:
    - int: {"type": "int", "low": min, "high": max, "log": bool}
    - float: {"type": "float", "low": min, "high": max, "log": bool}
    - categorical: {"type": "categorical", "choices": [list]}

    Args:
        model_name: Model identifier (RF, XGBoost, LR_EN, LR_L1, LinSVM_cal)
        config: TrainingConfiguration object
        xgb_spw: XGBoost scale_pos_weight (optional override)

    Returns:
        Dictionary mapping parameter names to Optuna suggest specs
    """
    optuna_dists = {}

    # Feature selection parameters (if applicable)
    strategy = config.features.feature_selection_strategy
    if strategy == "multi_stage":
        k_grid = config.features.k_grid
        if k_grid:
            # Use the k_grid as categorical for feature selection
            optuna_dists["sel__selector__k"] = {"type": "categorical", "choices": k_grid}

    # Model-specific parameters with native Optuna ranges
    if model_name in ("LR_EN", "LR_L1"):
        optuna_dists.update(_get_lr_params_optuna(config, model_name=model_name))

    elif model_name == "LinSVM_cal":
        optuna_dists.update(_get_svm_params_optuna(config))

    elif model_name == "RF":
        optuna_dists.update(_get_rf_params_optuna(config))

    elif model_name == "XGBoost":
        optuna_dists.update(_get_xgb_params_optuna(config, xgb_spw))

    return optuna_dists


# Public API exports
__all__ = [
    # Main distribution functions
    "get_param_distributions",
    "get_param_distributions_optuna",
    # RFE tune spaces
    "get_rfe_tune_space",
    "get_rfe_tune_spaces_from_training_config",
    "RFE_TUNE_SPACES",
    # Optuna defaults
    "DEFAULT_OPTUNA_RANGES",
    # Utility functions (used in tests)
    "_make_logspace",
    "_parse_class_weight_options",
    "_randomize_int_list",
    "_randomize_float_list",
    "_to_optuna_spec",
    "_is_log_spaced",
    # Class weight resolution
    "resolve_class_weight",
    "resolve_class_weights_in_params",
]
