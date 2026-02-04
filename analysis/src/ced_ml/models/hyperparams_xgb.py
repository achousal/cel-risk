"""
XGBoost hyperparameter search space definitions.

Provides:
- Parameter distributions for RandomizedSearchCV
- Optuna parameter distributions with native range specifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hyperparams_common import (
    DEFAULT_OPTUNA_RANGES,
    _randomize_float_list,
    _randomize_int_list,
)

if TYPE_CHECKING:
    from ..config import TrainingConfig


def _get_xgb_params(
    config: TrainingConfig,
    xgb_spw: float | None,
    randomize: bool,
    rng: np.random.Generator | None,
) -> dict[str, list]:
    """XGBoost hyperparameters."""
    n_estimators_grid = config.xgboost.n_estimators_grid.copy()
    max_depth_grid = config.xgboost.max_depth_grid.copy()
    learning_rate_grid = config.xgboost.learning_rate_grid.copy()
    subsample_grid = config.xgboost.subsample_grid.copy()
    colsample_grid = config.xgboost.colsample_bytree_grid.copy()

    # Regularization parameters
    min_child_weight_grid = config.xgboost.min_child_weight_grid.copy()
    gamma_grid = config.xgboost.gamma_grid.copy()
    reg_alpha_grid = config.xgboost.reg_alpha_grid.copy()
    reg_lambda_grid = config.xgboost.reg_lambda_grid.copy()

    # Scale pos weight grid
    if xgb_spw is not None:
        # Use fold-specific value +/- 20%
        spw_grid = [xgb_spw * 0.8, xgb_spw, xgb_spw * 1.2]
    else:
        spw_grid = config.xgboost.scale_pos_weight_grid.copy()

    if randomize and rng:
        n_estimators_grid = _randomize_int_list(n_estimators_grid, rng, min_val=1)
        max_depth_grid = _randomize_int_list(max_depth_grid, rng, min_val=1, unique=True)
        learning_rate_grid = _randomize_float_list(
            learning_rate_grid, rng, min_val=1e-4, log_scale=True
        )
        subsample_grid = _randomize_float_list(subsample_grid, rng, min_val=0.1, max_val=1.0)
        colsample_grid = _randomize_float_list(colsample_grid, rng, min_val=0.1, max_val=1.0)
        spw_grid = _randomize_float_list(spw_grid, rng, min_val=1e-3)
        # Regularization params
        min_child_weight_grid = _randomize_int_list(min_child_weight_grid, rng, min_val=1)
        gamma_grid = _randomize_float_list(gamma_grid, rng, min_val=0.0)
        reg_alpha_grid = _randomize_float_list(reg_alpha_grid, rng, min_val=0.0, log_scale=True)
        reg_lambda_grid = _randomize_float_list(reg_lambda_grid, rng, min_val=0.1, log_scale=True)

    return {
        "clf__n_estimators": n_estimators_grid,
        "clf__max_depth": max_depth_grid,
        "clf__learning_rate": learning_rate_grid,
        "clf__subsample": subsample_grid,
        "clf__colsample_bytree": colsample_grid,
        "clf__scale_pos_weight": spw_grid,
        # Regularization parameters
        "clf__min_child_weight": min_child_weight_grid,
        "clf__gamma": gamma_grid,
        "clf__reg_alpha": reg_alpha_grid,
        "clf__reg_lambda": reg_lambda_grid,
    }


def _get_xgb_params_optuna(config: TrainingConfig, xgb_spw: float | None = None) -> dict[str, dict]:
    """Build Optuna specs for XGBoost with wider ranges and log-scale sampling."""
    defaults = DEFAULT_OPTUNA_RANGES["XGBoost"]
    params = {}

    # n_estimators
    if config.xgboost.optuna_n_estimators is not None:
        low, high = config.xgboost.optuna_n_estimators
    else:
        grid = config.xgboost.n_estimators_grid
        low = max(50, min(grid) - 50) if grid else defaults["n_estimators"]["low"]
        high = max(grid) + 200 if grid else defaults["n_estimators"]["high"]
    params["clf__n_estimators"] = {
        "type": "int",
        "low": low,
        "high": high,
        "log": False,
    }

    # max_depth
    if config.xgboost.optuna_max_depth is not None:
        low, high = config.xgboost.optuna_max_depth
    else:
        grid = config.xgboost.max_depth_grid
        low = max(2, min(grid) - 1) if grid else defaults["max_depth"]["low"]
        high = max(grid) + 2 if grid else defaults["max_depth"]["high"]
    params["clf__max_depth"] = {"type": "int", "low": low, "high": high, "log": False}

    # learning_rate - LOG SCALE (critical for XGBoost tuning)
    if config.xgboost.optuna_learning_rate is not None:
        low, high = config.xgboost.optuna_learning_rate
    else:
        # Use wider range than grid for better exploration
        low = defaults["learning_rate"]["low"]
        high = defaults["learning_rate"]["high"]
    params["clf__learning_rate"] = {
        "type": "float",
        "low": low,
        "high": high,
        "log": True,
    }

    # min_child_weight - LOG SCALE
    if config.xgboost.optuna_min_child_weight is not None:
        low, high = config.xgboost.optuna_min_child_weight
    else:
        low = defaults["min_child_weight"]["low"]
        high = defaults["min_child_weight"]["high"]
    params["clf__min_child_weight"] = {
        "type": "float",
        "low": low,
        "high": high,
        "log": True,
    }

    # gamma
    if config.xgboost.optuna_gamma is not None:
        low, high = config.xgboost.optuna_gamma
    else:
        low = defaults["gamma"]["low"]
        high = defaults["gamma"]["high"]
    params["clf__gamma"] = {"type": "float", "low": low, "high": high, "log": False}

    # subsample
    if config.xgboost.optuna_subsample is not None:
        low, high = config.xgboost.optuna_subsample
    else:
        low = defaults["subsample"]["low"]
        high = defaults["subsample"]["high"]
    params["clf__subsample"] = {"type": "float", "low": low, "high": high, "log": False}

    # colsample_bytree
    if config.xgboost.optuna_colsample_bytree is not None:
        low, high = config.xgboost.optuna_colsample_bytree
    else:
        low = defaults["colsample_bytree"]["low"]
        high = defaults["colsample_bytree"]["high"]
    params["clf__colsample_bytree"] = {
        "type": "float",
        "low": low,
        "high": high,
        "log": False,
    }

    # reg_alpha (L1 regularization) - LOG SCALE
    if config.xgboost.optuna_reg_alpha is not None:
        low, high = config.xgboost.optuna_reg_alpha
    else:
        low = defaults["reg_alpha"]["low"]
        high = defaults["reg_alpha"]["high"]
    params["clf__reg_alpha"] = {"type": "float", "low": low, "high": high, "log": True}

    # reg_lambda (L2 regularization) - LOG SCALE
    if config.xgboost.optuna_reg_lambda is not None:
        low, high = config.xgboost.optuna_reg_lambda
    else:
        low = defaults["reg_lambda"]["low"]
        high = defaults["reg_lambda"]["high"]
    params["clf__reg_lambda"] = {"type": "float", "low": low, "high": high, "log": True}

    # scale_pos_weight
    if xgb_spw is not None:
        # Use fold-specific value +/- 30% as range
        params["clf__scale_pos_weight"] = {
            "type": "float",
            "low": xgb_spw * 0.7,
            "high": xgb_spw * 1.3,
            "log": False,
        }
    else:
        grid = config.xgboost.scale_pos_weight_grid
        if grid:
            params["clf__scale_pos_weight"] = {
                "type": "float",
                "low": min(grid),
                "high": max(grid) * 1.5,
                "log": False,
            }

    return params
