"""
Hyperparameter search space definitions for all models.

Provides:
- Parameter distributions for RandomizedSearchCV
- Grid randomization for sensitivity analysis
- Model-specific tuning ranges
"""

import numpy as np

from ..config import TrainingConfig
from ..data.schema import ModelName


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
    if strategy == "hybrid_stability":
        k_grid = config.features.k_grid
        if not k_grid:
            raise ValueError(
                "feature_selection_strategy='hybrid_stability' requires features.k_grid"
            )

        # Always use 'sel' step name
        param_dists["sel__k"] = k_grid

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
            except Exception:
                pass

    return options if options else [None]


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
# Optuna Parameter Distribution Conversion
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
    if strategy == "hybrid_stability":
        k_grid = config.features.k_grid
        if k_grid:
            # Use the k_grid as categorical for feature selection
            optuna_dists["sel__k"] = {"type": "categorical", "choices": k_grid}

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
# Reduced Search Spaces for RFE Per-k Tuning
# ============================================================================

# Compact search spaces for quick per-panel-size hyperparameter re-tuning.
# These focus on the parameters most sensitive to dimensionality changes,
# keeping n_estimators fixed for tree models to reduce search cost.

RFE_TUNE_SPACES: dict[str, dict[str, dict]] = {
    "LR_EN": {
        "clf__C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "clf__l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
    },
    "LR_L1": {
        "clf__C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
    },
    "LinSVM_cal": {
        "clf__estimator__C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
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


def get_rfe_tune_space(model_name: str) -> dict[str, dict]:
    """Return reduced Optuna search space for RFE per-k hyperparameter tuning.

    These spaces focus on parameters most sensitive to panel dimensionality,
    keeping ensemble size (n_estimators) fixed for tree-based models.

    Args:
        model_name: Model identifier (e.g., "LR_EN", "RF", "XGBoost").

    Returns:
        Dictionary mapping parameter names (with clf__ prefix) to Optuna
        suggest specs: {"type": str, "low": num, "high": num, "log": bool}.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in RFE_TUNE_SPACES:
        raise ValueError(
            f"No RFE tune space defined for model '{model_name}'. "
            f"Known models: {list(RFE_TUNE_SPACES.keys())}"
        )
    return dict(RFE_TUNE_SPACES[model_name])
