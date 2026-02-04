"""
Logistic Regression hyperparameter search space definitions.

Provides:
- Parameter distributions for RandomizedSearchCV
- Optuna parameter distributions with native range specifications
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .hyperparams_common import (
    DEFAULT_OPTUNA_RANGES,
    _make_logspace,
    _parse_class_weight_options,
    _randomize_float_list,
)

if TYPE_CHECKING:
    from ..config import TrainingConfig


def _get_lr_params(
    config: TrainingConfig,
    randomize: bool,
    rng: np.random.Generator | None,
    model_name: str = "LR_EN",
) -> dict[str, list]:
    """Logistic Regression hyperparameters.

    Args:
        config: TrainingConfiguration object
        randomize: Whether to perturb grids for sensitivity analysis
        rng: Random number generator for perturbation
        model_name: Model identifier (LR_EN or LR_L1)

    Returns:
        Dictionary mapping parameter names to value lists
    """
    from ..data.schema import ModelName

    # C values (inverse regularization strength)
    C_grid = _make_logspace(
        config.lr.C_min,
        config.lr.C_max,
        config.lr.C_points,
        rng=rng if randomize else None,
    )

    params = {"clf__C": C_grid}

    # l1_ratio for ElasticNet (LR_EN only, not LR_L1)
    if model_name == ModelName.LR_EN:
        l1_grid = config.lr.l1_ratio.copy()
        if randomize and rng:
            l1_grid = _randomize_float_list(l1_grid, rng, min_val=0.0, max_val=1.0)
        params["clf__l1_ratio"] = l1_grid

    # Class weights
    class_weight_options = _parse_class_weight_options(config.lr.class_weight_options)
    if class_weight_options:
        params["clf__class_weight"] = class_weight_options

    return params


def _get_lr_params_optuna(config: TrainingConfig, model_name: str = "LR_EN") -> dict[str, dict]:
    """Build Optuna specs for Logistic Regression."""
    defaults = DEFAULT_OPTUNA_RANGES["LR"]
    params = {}

    # C parameter (inverse regularization strength) - log scale
    if config.lr.optuna_C is not None:
        c_low, c_high = config.lr.optuna_C
    else:
        c_low, c_high = config.lr.C_min, config.lr.C_max
    params["clf__C"] = {"type": "float", "low": c_low, "high": c_high, "log": True}

    # l1_ratio for ElasticNet (LR_EN only)
    if model_name == "LR_EN":
        if config.lr.optuna_l1_ratio is not None:
            l1_low, l1_high = config.lr.optuna_l1_ratio
        else:
            # Derive from list or use default range
            l1_values = config.lr.l1_ratio
            l1_low = min(l1_values) if l1_values else defaults["l1_ratio"]["low"]
            l1_high = max(l1_values) if l1_values else defaults["l1_ratio"]["high"]
        params["clf__l1_ratio"] = {
            "type": "float",
            "low": l1_low,
            "high": l1_high,
            "log": False,
        }

    # Class weights (categorical)
    class_weight_options = _parse_class_weight_options(config.lr.class_weight_options)
    if class_weight_options and len(class_weight_options) > 1:
        params["clf__class_weight"] = {
            "type": "categorical",
            "choices": class_weight_options,
        }

    return params
