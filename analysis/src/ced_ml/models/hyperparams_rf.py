"""
Random Forest hyperparameter search space definitions.

Provides:
- Parameter distributions for RandomizedSearchCV
- Optuna parameter distributions with native range specifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hyperparams_common import (
    DEFAULT_OPTUNA_RANGES,
    _parse_class_weight_options,
    _randomize_float_list,
    _randomize_int_list,
)

if TYPE_CHECKING:
    from ..config import TrainingConfig


def _get_rf_params(
    config: TrainingConfig, randomize: bool, rng: np.random.Generator | None
) -> dict[str, list]:
    """Random Forest hyperparameters."""
    n_estimators_grid = config.rf.n_estimators_grid.copy()
    max_depth_grid = config.rf.max_depth_grid.copy()
    min_samples_split_grid = config.rf.min_samples_split_grid.copy()
    min_samples_leaf_grid = config.rf.min_samples_leaf_grid.copy()
    max_features_grid = config.rf.max_features_grid.copy()

    if randomize and rng:
        n_estimators_grid = _randomize_int_list(n_estimators_grid, rng, min_val=10)
        max_depth_grid = _randomize_int_list(max_depth_grid, rng, min_val=1, unique=True)
        min_samples_split_grid = _randomize_int_list(min_samples_split_grid, rng, min_val=2)
        min_samples_leaf_grid = _randomize_int_list(min_samples_leaf_grid, rng, min_val=1)
        max_features_grid = _randomize_float_list(max_features_grid, rng, min_val=0.1, max_val=1.0)

    params = {
        "clf__n_estimators": n_estimators_grid,
        "clf__max_depth": max_depth_grid,
        "clf__min_samples_split": min_samples_split_grid,
        "clf__min_samples_leaf": min_samples_leaf_grid,
        "clf__max_features": max_features_grid,
    }

    # Class weights
    class_weight_options = _parse_class_weight_options(config.rf.class_weight_options)
    if class_weight_options:
        params["clf__class_weight"] = class_weight_options

    return params


def _get_rf_params_optuna(config: TrainingConfig) -> dict[str, dict]:
    """Build Optuna specs for Random Forest with wider ranges."""
    defaults = DEFAULT_OPTUNA_RANGES["RF"]
    params = {}

    # n_estimators
    if config.rf.optuna_n_estimators is not None:
        low, high = config.rf.optuna_n_estimators
    else:
        grid = config.rf.n_estimators_grid
        low = min(grid) if grid else defaults["n_estimators"]["low"]
        high = max(grid) if grid else defaults["n_estimators"]["high"]
        # Widen the range slightly beyond grid bounds
        low = max(50, low - 50)
        high = high + 100
    params["clf__n_estimators"] = {
        "type": "int",
        "low": low,
        "high": high,
        "log": False,
    }

    # max_depth - handle None in grid (unlimited depth)
    if config.rf.optuna_max_depth is not None:
        low, high = config.rf.optuna_max_depth
        params["clf__max_depth"] = {
            "type": "int",
            "low": low,
            "high": high,
            "log": False,
        }
    else:
        # Check if grid contains only numeric values
        grid = [v for v in config.rf.max_depth_grid if v is not None]
        if grid:
            low = min(grid)
            high = max(grid) + 10  # Widen range
            params["clf__max_depth"] = {
                "type": "int",
                "low": low,
                "high": high,
                "log": False,
            }
        else:
            # All None or empty - use categorical with None
            params["clf__max_depth"] = {
                "type": "categorical",
                "choices": config.rf.max_depth_grid or [None, 10, 20, 30],
            }

    # min_samples_split
    if config.rf.optuna_min_samples_split is not None:
        low, high = config.rf.optuna_min_samples_split
    else:
        grid = config.rf.min_samples_split_grid
        low = min(grid) if grid else defaults["min_samples_split"]["low"]
        high = max(grid) + 10 if grid else defaults["min_samples_split"]["high"]
    params["clf__min_samples_split"] = {
        "type": "int",
        "low": low,
        "high": high,
        "log": False,
    }

    # min_samples_leaf
    if config.rf.optuna_min_samples_leaf is not None:
        low, high = config.rf.optuna_min_samples_leaf
    else:
        grid = config.rf.min_samples_leaf_grid
        low = min(grid) if grid else defaults["min_samples_leaf"]["low"]
        high = max(grid) + 6 if grid else defaults["min_samples_leaf"]["high"]
    params["clf__min_samples_leaf"] = {
        "type": "int",
        "low": low,
        "high": high,
        "log": False,
    }

    # max_features - handle mixed string/float grid
    if config.rf.optuna_max_features is not None:
        low, high = config.rf.optuna_max_features
        params["clf__max_features"] = {
            "type": "float",
            "low": low,
            "high": high,
            "log": False,
        }
    else:
        # Check if grid has only numeric values
        grid = config.rf.max_features_grid
        numeric_vals = [v for v in grid if isinstance(v, int | float)]
        if numeric_vals and len(numeric_vals) == len(grid):
            # All numeric - use float range
            low = min(numeric_vals)
            high = min(1.0, max(numeric_vals) + 0.2)  # Cap at 1.0
            params["clf__max_features"] = {
                "type": "float",
                "low": low,
                "high": high,
                "log": False,
            }
        else:
            # Mixed or string values - use categorical
            params["clf__max_features"] = {"type": "categorical", "choices": grid}

    # Class weights (categorical)
    class_weight_options = _parse_class_weight_options(config.rf.class_weight_options)
    if class_weight_options and len(class_weight_options) > 1:
        params["clf__class_weight"] = {
            "type": "categorical",
            "choices": class_weight_options,
        }

    return params
