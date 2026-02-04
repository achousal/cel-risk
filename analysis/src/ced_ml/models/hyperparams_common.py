"""
Shared utilities for hyperparameter search space definitions.

Provides:
- Log-space grid generation
- Class weight parsing
- Grid randomization for sensitivity analysis
- Optuna log-spacing detection
- Default Optuna ranges for all models
- RFE tune spaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ============================================================================
# Grid Generation Utilities
# ============================================================================


def _make_logspace(
    min_val: float,
    max_val: float,
    n_points: int,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """
    Create log-spaced grid.

    Args:
        min_val: Minimum value (e.g. 1e-4)
        max_val: Maximum value (e.g. 1e4)
        n_points: Number of points
        rng: Optional RNG for perturbation

    Returns:
        List of float values
    """
    if n_points < 1:
        return []

    if n_points == 1:
        return [float(np.sqrt(min_val * max_val))]  # Geometric mean

    # Standard log-spaced grid
    grid = np.logspace(np.log10(min_val), np.log10(max_val), num=n_points).tolist()

    # Optional perturbation
    if rng:
        grid = [float(v * rng.uniform(0.8, 1.2)) for v in grid]
        grid = [max(min_val, min(max_val, v)) for v in grid]

    return grid


def _parse_class_weight_options(options_str: str) -> list:
    """
    Parse class_weight options string.

    Format: "balanced,{0:1,1:5},{0:1,1:10}"

    Returns:
        List of class_weight values (None, 'balanced', or dict)
    """
    if not options_str or options_str.strip() == "":
        return [None]

    options = []

    # Split carefully to avoid breaking {k:v,k:v} dicts
    parts = []
    current = []
    in_dict = False

    for char in options_str + ",":  # Add trailing comma to flush last part
        if char == "{":
            in_dict = True
            current.append(char)
        elif char == "}":
            in_dict = False
            current.append(char)
        elif char == "," and not in_dict:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(char)

    # Parse each part
    for opt in parts:
        opt = opt.strip()
        if opt == "":
            continue
        elif opt == "None":
            options.append(None)
        elif opt == "balanced":
            options.append("balanced")
        elif opt.startswith("{"):
            # Parse dict: {0:1,1:5}
            try:
                weight_dict = {}
                opt = opt.strip("{}")
                for pair in opt.split(","):
                    k, v = pair.split(":")
                    weight_dict[int(k.strip())] = float(v.strip())
                options.append(weight_dict)
            except Exception as e:
                # Log parse failure at debug level - non-critical
                import logging

                logging.getLogger(__name__).debug(
                    "Failed to parse class_weight dict '%s': %s", opt, e
                )

    return options if options else [None]


# ============================================================================
# Grid Randomization Utilities
# ============================================================================


def _randomize_int_list(
    values: list[int],
    rng: np.random.Generator,
    min_val: int = 1,
    unique: bool = False,
) -> list[int]:
    """
    Perturb integer grid values for sensitivity analysis.

    Args:
        values: Original grid values (may contain None)
        rng: Random number generator
        min_val: Minimum allowed value
        unique: If True, ensure all values are unique

    Returns:
        Perturbed grid values (None values preserved as-is)
    """
    if not values:
        return []

    perturbed = []
    for v in values:
        if v is None:
            # Preserve None values (e.g., for RF max_depth=None)
            perturbed.append(None)
        else:
            # Perturb by +/- 20%
            delta = max(1, int(v * 0.2))
            new_val = v + rng.integers(-delta, delta + 1)
            new_val = max(min_val, new_val)
            perturbed.append(new_val)

    if unique:
        # Separate None from numeric values for uniqueness
        none_values = [v for v in perturbed if v is None]
        numeric_values = sorted({v for v in perturbed if v is not None})
        perturbed = none_values + numeric_values

    return perturbed


def _randomize_float_list(
    values: list[str | float],
    rng: np.random.Generator,
    min_val: float = 0.0,
    max_val: float = np.inf,
    log_scale: bool = False,
) -> list[str | float]:
    """
    Perturb float grid values for sensitivity analysis.

    Non-numeric values (e.g., "sqrt", "log2") are passed through unchanged.

    Args:
        values: Original grid values
        rng: Random number generator
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        log_scale: If True, perturb in log space

    Returns:
        Perturbed grid values
    """
    if not values:
        return []

    perturbed = []
    for v in values:
        # Skip non-numeric values (e.g., "sqrt", "log2" for max_features)
        if not isinstance(v, int | float):
            perturbed.append(v)
            continue

        if log_scale and v > 0:
            # Perturb in log space
            log_v = np.log10(v)
            log_delta = 0.2  # +/- 0.2 in log space
            new_log_v = log_v + rng.uniform(-log_delta, log_delta)
            new_val = 10**new_log_v
        else:
            # Perturb by +/- 20%
            new_val = v * rng.uniform(0.8, 1.2)

        new_val = max(min_val, min(max_val, new_val))
        perturbed.append(float(new_val))

    return perturbed


# ============================================================================
# Optuna Utilities
# ============================================================================


def _to_optuna_spec(values: list) -> dict | None:
    """
    Convert a single sklearn parameter grid to Optuna spec.

    This is a fallback converter for parameters not handled by
    the model-specific functions above.

    Args:
        values: List of possible values

    Returns:
        Optuna suggest spec dict, or None if conversion fails
    """
    if not values:
        return None

    # Handle single value
    if len(values) == 1:
        return {"type": "categorical", "choices": values}

    # Check if all values are integers
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return {
            "type": "int",
            "low": min(values),
            "high": max(values),
            "log": _is_log_spaced(values),
        }

    # Check if all values are numeric (int or float, but not all int)
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return {
            "type": "float",
            "low": float(min(values)),
            "high": float(max(values)),
            "log": _is_log_spaced(values),
        }

    # Default to categorical
    return {"type": "categorical", "choices": values}


def _is_log_spaced(values: list) -> bool:
    """
    Heuristically detect if values are log-spaced.

    Uses ratio consistency: if consecutive ratios are similar,
    values are likely log-spaced.

    Args:
        values: List of numeric values

    Returns:
        True if values appear to be log-spaced
    """
    if len(values) < 3:
        return False

    # Filter to positive values only
    positive = [v for v in values if isinstance(v, int | float) and v > 0]
    if len(positive) < 3:
        return False

    sorted_vals = sorted(positive)

    # Compute consecutive ratios
    ratios = []
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i] > 0:
            ratios.append(sorted_vals[i + 1] / sorted_vals[i])

    if not ratios:
        return False

    # Check if ratios are relatively consistent (log-spaced characteristic)
    mean_ratio = np.mean(ratios)
    if mean_ratio <= 1.5:
        # Ratios too close to 1 = linear spacing
        return False

    # Check variance in ratios
    ratio_std = np.std(ratios)
    return ratio_std / mean_ratio < 0.5  # Low relative variance = log-spaced


# ============================================================================
# Default Optuna Ranges
# ============================================================================

# Default Optuna ranges with wider search spaces and proper log-scale sampling
# These are used when Optuna-specific config fields are not set

DEFAULT_OPTUNA_RANGES = {
    "XGBoost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
        "max_depth": {"type": "int", "low": 2, "high": 12, "log": False},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "min_child_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "gamma": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "RF": {
        "n_estimators": {"type": "int", "low": 50, "high": 500, "log": False},
        "max_depth": {"type": "int", "low": 3, "high": 20, "log": False},
        "min_samples_split": {"type": "int", "low": 2, "high": 20, "log": False},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10, "log": False},
        "max_features": {"type": "float", "low": 0.1, "high": 1.0, "log": False},
    },
    "LR": {
        "C": {"type": "float", "low": 1e-5, "high": 100.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
    },
    "SVM": {
        "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
    },
}


# ============================================================================
# Reduced Search Spaces for RFE Per-k Tuning
# ============================================================================

# Compact search spaces for quick per-panel-size hyperparameter re-tuning.
# These focus on the parameters most sensitive to dimensionality changes,
# keeping n_estimators fixed for tree models to reduce search cost.

RFE_TUNE_SPACES: dict[str, dict[str, dict]] = {
    # C capped at 1.0 for linear models to prevent overfitting at high feature counts
    # (was 100.0; caused inverted Pareto curves where fewer features -> higher AUROC)
    "LR_EN": {
        "clf__C": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
        "clf__l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
    },
    "LR_L1": {
        "clf__C": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
    },
    "LinSVM_cal": {
        "clf__estimator__C": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    },
    "RF": {
        "clf__max_depth": {"type": "int", "low": 3, "high": 20, "log": False},
        "clf__min_samples_leaf": {"type": "int", "low": 1, "high": 10, "log": False},
    },
    "XGBoost": {
        "clf__learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
        "clf__max_depth": {"type": "int", "low": 2, "high": 12, "log": False},
        "clf__reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
}


def get_rfe_tune_space(
    model_name: str,
    config_overrides: dict[str, dict[str, dict]] | None = None,
) -> dict[str, dict]:
    """Return Optuna search space for RFE per-k hyperparameter tuning.

    If ``config_overrides`` contains an entry for *model_name*, that entry
    is used verbatim (full replacement).  Otherwise falls back to the
    hardcoded ``RFE_TUNE_SPACES`` defaults.

    Args:
        model_name: Model identifier (e.g., "LR_EN", "RF", "XGBoost").
        config_overrides: Optional mapping ``{model_name: {param: spec}}``
            loaded from ``optimize_panel.yaml`` ``rfe_tune_spaces`` section.

    Returns:
        Dictionary mapping parameter names (with clf__ prefix) to Optuna
        suggest specs: {"type": str, "low": num, "high": num, "log": bool}.

    Raises:
        ValueError: If model_name is not recognized in either source.
    """
    if config_overrides and model_name in config_overrides:
        return dict(config_overrides[model_name])

    if model_name not in RFE_TUNE_SPACES:
        raise ValueError(
            f"No RFE tune space defined for model '{model_name}'. "
            f"Known models: {list(RFE_TUNE_SPACES.keys())}"
        )
    return dict(RFE_TUNE_SPACES[model_name])
