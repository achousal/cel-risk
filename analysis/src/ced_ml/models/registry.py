"""Model registry and hyperparameter grid definitions.

This module provides:
- Model instantiation (RF, XGBoost, LinSVM, LogisticRegression)
- Hyperparameter grid generation for RandomizedSearchCV

References:
- XGBoost tree_method controls CPU vs GPU acceleration
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ..config.schema import TrainingConfig
from ..data.schema import ModelName

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    logger.debug("XGBoost not available; XGBoost models will raise ImportError if requested.")
    XGBOOST_AVAILABLE = False
    XGBClassifier = None  # type: ignore


# ----------------------------
# Parameter grid utilities
# ----------------------------
T = TypeVar("T")


def _parse_list(s: str, cast_fn: Callable[[str], T], type_name: str) -> list[T]:
    """Parse comma-separated values with specified type casting.

    Args:
        s: Comma-separated string to parse.
        cast_fn: Function to cast each token (e.g., float, int).
        type_name: Type name for debug logging.

    Returns:
        List of parsed values; invalid tokens are skipped with debug logging.
    """
    if not s:
        return []
    out: list[T] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(cast_fn(tok))
        except (ValueError, TypeError):
            logger.debug(f"Failed to parse '{tok}' as {type_name}; skipping token.")
            continue
    return out


def _parse_float_list(s: str) -> list[float]:
    """Parse comma-separated float values."""
    return _parse_list(s, float, "float")


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integer values."""
    return _parse_list(s, int, "int")


def _require_int_list(s: str, name: str) -> list[int]:
    """Parse and validate non-empty integer list."""
    values = _parse_int_list(s)
    if not values:
        raise ValueError(f"{name} must be a non-empty comma-separated list (e.g. '200,500').")
    return values


def _parse_none_int_float_list(s: str) -> list:
    """Parse list with mixed types: None, int, float, or strings like 'sqrt'."""
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        lo = tok.lower()
        if lo in ("none", "null"):
            out.append(None)
            continue
        if lo == "sqrt":
            out.append("sqrt")
            continue
        # try int then float
        try:
            if re.match(r"^[+-]?\d+$", tok):
                out.append(int(tok))
                continue
        except (ValueError, TypeError):
            logger.debug(f"Failed to parse '{tok}' as int via regex; will try float.")
        try:
            out.append(float(tok))
            continue
        except (ValueError, TypeError):
            logger.debug(f"Failed to parse '{tok}' as float; keeping as string.")
            out.append(tok)
    return out


def _coerce_int_or_none_list(vals: list[Any], *, name: str) -> list[Any]:
    """Coerce values to int or None (sklearn max_depth parameter).

    Accepts:
    - int
    - None
    - float with integer value (e.g., 10.0 -> 10)

    Raises:
        ValueError: For non-integer floats or invalid strings
    """
    out: list[Any] = []
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if isinstance(v, bool):
            raise ValueError(f"{name}: invalid boolean value {v}")
        if isinstance(v, int):
            out.append(v)
            continue
        if isinstance(v, float):
            if float(v).is_integer():
                out.append(int(v))
                continue
            raise ValueError(f"{name}: expected int or None, got non-integer float {v}")
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("none", "null"):
                out.append(None)
                continue
            try:
                out.append(int(vv))
                continue
            except (ValueError, TypeError):
                logger.debug(f"{name}: failed to parse '{v}' as int; will try float.")
            try:
                fv = float(vv)
                if float(fv).is_integer():
                    out.append(int(fv))
                    continue
            except (ValueError, TypeError):
                logger.debug(f"{name}: failed to parse '{v}' as float.")
                pass
            raise ValueError(f"{name}: expected int or None, got '{v}'")
        raise ValueError(f"{name}: expected int or None, got {type(v).__name__}={v}")
    return out


def _coerce_min_samples_leaf_list(
    vals: list[Any], *, name: str = "rf_min_samples_leaf_grid"
) -> list[Any]:
    """Coerce min_samples_leaf grid to sklearn-compatible types.

    sklearn accepts:
    - int >= 1
    - float in (0, 1.0) (fraction of samples)

    Also coerces whole-number floats (e.g., 5.0 -> 5) for CLI robustness.
    """
    out: list[Any] = []
    for v in vals:
        if isinstance(v, bool):
            raise ValueError(f"{name}: invalid boolean value {v}")
        if isinstance(v, int):
            if v < 1:
                raise ValueError(f"{name}: int must be >= 1, got {v}")
            out.append(v)
            continue
        if isinstance(v, float):
            if 0.0 < v < 1.0:
                out.append(float(v))
                continue
            if float(v).is_integer():
                iv = int(v)
                if iv < 1:
                    raise ValueError(f"{name}: int must be >= 1, got {iv}")
                out.append(iv)
                continue
            raise ValueError(f"{name}: float must be in (0,1) or an integer value, got {v}")
        if isinstance(v, str):
            vv = v.strip().lower()
            try:
                iv = int(vv)
                if iv < 1:
                    raise ValueError(f"{name}: int must be >= 1, got {iv}")
                out.append(iv)
                continue
            except (ValueError, TypeError):
                logger.debug(f"{name}: failed to parse '{v}' as int; will try float.")
            try:
                fv = float(vv)
                if 0.0 < fv < 1.0:
                    out.append(float(fv))
                    continue
                if float(fv).is_integer():
                    iv = int(fv)
                    if iv < 1:
                        raise ValueError(f"{name}: int must be >= 1, got {iv}")
                    out.append(iv)
                    continue
            except (ValueError, TypeError):
                logger.debug(f"{name}: failed to parse '{v}' as float.")
                pass
            raise ValueError(f"{name}: could not parse '{v}' as int>=1 or float in (0,1)")
        raise ValueError(f"{name}: unsupported type {type(v).__name__}={v}")
    return out


def parse_class_weight_options(s: str) -> list:
    """Parse class_weight options.

    Accepted tokens:
        - ``none`` / ``null`` → unweighted (``None``)
        - ``balanced`` → sklearn balanced reweighting
        - ``log``  → {0: 1, 1: log(n_neg/n_pos)}, resolved per fold at fit time
        - ``sqrt`` → {0: 1, 1: sqrt(n_neg/n_pos)}, resolved per fold at fit time

    ``log``/``sqrt`` flow through as string tokens; they are replaced with a
    concrete weight dict inside the Optuna loop by
    :func:`ced_ml.models.hyperparams_common.resolve_class_weights_in_params`
    using each training fold's labels.

    Examples:
        "none,balanced" -> [None, "balanced"]
        "balanced" -> ["balanced"]
        "log"      -> ["log"]
        "none,log" -> [None, "log"]
        ""         -> [None, "balanced"] (default)
    """
    if not s:
        return [None, "balanced"]
    toks = [t.strip().lower() for t in str(s).split(",") if t.strip()]
    out = []
    for t in toks:
        if t in ("none", "null"):
            out.append(None)
        elif t == "balanced":
            out.append("balanced")
        elif t in ("log", "sqrt"):
            out.append(t)
    # fallback if user passes invalid input
    if not out:
        return [None, "balanced"]
    # dedupe while preserving order
    seen = set()
    out2 = []
    for v in out:
        key = str(v)
        if key in seen:
            continue
        seen.add(key)
        out2.append(v)
    return out2


def make_logspace(
    minv: float, maxv: float, points: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate log-spaced values for regularization parameters.

    Args:
        minv: Minimum value (e.g., 1e-3)
        maxv: Maximum value (e.g., 1e3)
        points: Number of grid points
        rng: Optional RNG for randomized grids

    Returns:
        Array of log-spaced values
    """
    minv = float(minv)
    maxv = float(maxv)
    points = int(points)
    if points < 2:
        points = 2
    if minv <= 0 or maxv <= 0:
        return np.logspace(-3, 3, 13)
    a = np.log10(minv)
    b = np.log10(maxv)
    if rng is not None:
        samples = rng.uniform(a, b, size=points)
        return np.power(10.0, samples)
    return np.logspace(a, b, points)


def compute_scale_pos_weight_from_y(y: np.ndarray) -> float:
    """Compute XGBoost scale_pos_weight from class distribution.

    Args:
        y: Binary labels (0/1).

    Returns:
        scale_pos_weight value (ratio of negatives to positives, >= 1.0).
        Returns 1.0 if no positive samples found (with warning).
    """
    import logging

    logger = logging.getLogger(__name__)

    y = np.asarray(y).astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if n_pos == 0:
        logger.warning("[xgb] No positive samples in training fold; using scale_pos_weight=1.0")
        return 1.0

    spw = float(n_neg) / float(n_pos)
    return max(1.0, spw)


# ----------------------------
# Model builders
# ----------------------------
def build_logistic_regression(
    solver: str = "saga",
    C: float = 1.0,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 42,
    l1_ratio: float = 0.5,
) -> LogisticRegression:
    """Build Logistic Regression estimator.

    Args:
        solver: Optimization algorithm
        C: Inverse regularization strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        l1_ratio: ElasticNet mixing (0=L2, 1=L1)

    Returns:
        Configured LogisticRegression estimator
    """
    return LogisticRegression(
        penalty="elasticnet",
        solver=solver,
        C=C,
        l1_ratio=l1_ratio,
        max_iter=int(max_iter),
        tol=float(tol),
        random_state=int(random_state),
    )


def build_linear_svm_calibrated(
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 2000,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Build calibrated LinearSVC estimator.

    LinearSVC + CalibratedClassifierCV provides probability estimates.

    Args:
        C: Inverse regularization strength
        penalty: Regularization type. 'l2' (default) or 'l1'. L1 forces dual=False
            and produces sparse coefficients, which is useful for feature selection.
        max_iter: Maximum iterations
        calibration_method: 'sigmoid' or 'isotonic'
        calibration_cv: CV folds for calibration
        random_state: Random seed

    Returns:
        CalibratedClassifierCV wrapping LinearSVC
    """
    if penalty not in ("l1", "l2"):
        raise ValueError(f"penalty must be 'l1' or 'l2', got {penalty!r}")
    # L1 requires dual=False; L2 supports dual=True (sklearn default, slightly faster)
    dual = penalty == "l2"
    base_svm = LinearSVC(
        C=C,
        penalty=penalty,
        dual=dual,
        class_weight=None,
        random_state=int(random_state),
        max_iter=int(max_iter),
    )
    return CalibratedClassifierCV(base_svm, method=str(calibration_method), cv=int(calibration_cv))


def build_random_forest(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 5,
    min_samples_split: int = 2,
    max_features: str = "sqrt",
    max_samples: float | None = None,
    bootstrap: bool = True,
    random_state: int = 42,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    """Build Random Forest classifier.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_leaf: Minimum samples per leaf
        min_samples_split: Minimum samples to split
        max_features: Features per split ('sqrt', int, or float)
        max_samples: Samples per tree (None = all)
        bootstrap: Whether to use bootstrap sampling
        random_state: Random seed
        n_jobs: Parallel jobs

    Returns:
        Configured RandomForestClassifier
    """
    rf_kwargs = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": bool(bootstrap),
        "random_state": int(random_state),
        "n_jobs": int(max(1, n_jobs)),
    }

    if max_samples is not None:
        try:
            v = float(max_samples)
            rf_kwargs["max_samples"] = int(v) if v.is_integer() else float(v)
        except (ValueError, TypeError):
            logger.warning(
                f"Failed to parse max_samples={max_samples}; omitting from RandomForestClassifier kwargs.",
                exc_info=True,
            )

    return RandomForestClassifier(**rf_kwargs)


def build_xgboost(
    n_estimators: int = 1000,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    tree_method: str = "hist",
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBClassifier:
    """Build XGBoost classifier.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        subsample: Row sampling fraction
        colsample_bytree: Column sampling fraction
        scale_pos_weight: Balancing of positive/negative weights
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        min_child_weight: Minimum sum of instance weight
        gamma: Minimum loss reduction
        tree_method: 'hist', 'gpu_hist', etc.
        random_state: Random seed
        n_jobs: Parallel jobs (1 for GPU)

    Returns:
        Configured XGBClassifier

    Raises:
        ImportError: If XGBoost not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")

    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        scale_pos_weight=float(scale_pos_weight),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        min_child_weight=int(min_child_weight),
        gamma=float(gamma),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=tree_method,
        random_state=int(random_state),
        n_jobs=int(max(1, n_jobs)) if tree_method != "gpu_hist" else 1,
    )


# ============================================================================
# Central Model Registry
# ============================================================================
#
# Adding a new model to the library:
#   1. Add the enum value in ``ced_ml.data.schema.ModelName`` and a display
#      string in ``MODEL_DISPLAY_NAMES``.
#   2. Write a builder callable with signature ``(config, random_state, n_jobs)
#      -> estimator``.
#   3. Register a ``ModelSpec`` below with builder, color, hyperparam family,
#      and capability flags.
#   4. Add an entry in ``features/model_selector.py:_SELECTOR_ESTIMATORS`` if
#      the model is eligible for feature selection, and in
#      ``models/hyperparams_common.py:RFE_TUNE_SPACES`` if it supports RFE.
#
# Everything else (calibration skip logic, linear-model branches, plotting
# color, weight-key lookup, parameter-distribution dispatch) is driven
# automatically off ``MODEL_REGISTRY`` via the helper functions at the bottom
# of this section.


# sklearn CalibratedClassifierCV accepts only "isotonic" or "sigmoid".
# Map internal calibration-method names onto those two values.
_SKLEARN_CAL_MAP: dict[str, str] = {
    "isotonic": "isotonic",
    "logistic_full": "sigmoid",
    "logistic_intercept": "sigmoid",
    "beta": "sigmoid",
}


@dataclass(frozen=True)
class ModelSpec:
    """Canonical metadata for a model registered in the library.

    Every library branch that switches on model identity should consult
    ``MODEL_REGISTRY`` through the helpers below rather than hardcoding a
    tuple or list of names.

    Attributes:
        name: The enum value identifying this model.
        display_name: Human-facing label (plot legends, reports).
        color: Canonical plot color (hex).
        builder: Factory with signature ``(config, random_state, n_jobs) -> estimator``.
        is_linear: True for models whose predictions are a linear function of
            transformed features (importance extracted from coefficients).
        is_already_calibrated: True if the builder returns an estimator that
            already exposes well-calibrated probabilities. Calibration-strategy
            passes skip wrapping these models to avoid double-calibration.
        weight_key: Key in ``TrainingConfig`` under which class_weight options
            live (e.g. "lr", "svm", "rf", "xgboost"). Used by
            ``ced_ml.recipes.config_gen`` to resolve per-model weighting.
        hyperparam_family: Dispatch tag consumed by parameter-distribution
            builders ("lr", "svm", "rf", "xgb").
    """

    name: ModelName
    display_name: str
    color: str
    builder: Callable[[TrainingConfig, int, int], Any]
    is_linear: bool = False
    is_already_calibrated: bool = False
    weight_key: str = ""
    hyperparam_family: str = ""


def _build_lr_en(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    return build_logistic_regression(
        solver=config.lr.solver,
        C=1.0,
        max_iter=config.lr.max_iter,
        tol=1e-4,
        random_state=random_state,
        l1_ratio=0.5,
    )


def _build_lr_l1(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    return build_logistic_regression(
        solver=config.lr.solver,
        C=1.0,
        max_iter=config.lr.max_iter,
        tol=1e-4,
        random_state=random_state,
        l1_ratio=1.0,
    )


def _build_linsvm_cal_with_penalty(
    config: TrainingConfig, random_state: int, n_jobs: int, *, penalty: str
) -> Any:
    sklearn_cal_method = _SKLEARN_CAL_MAP.get(config.calibration.method, "sigmoid")
    return build_linear_svm_calibrated(
        C=1.0,
        penalty=penalty,
        max_iter=config.svm.max_iter,
        calibration_method=sklearn_cal_method,
        calibration_cv=config.calibration.cv,
        random_state=random_state,
    )


def _build_linsvm_l2_cal(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    return _build_linsvm_cal_with_penalty(config, random_state, n_jobs, penalty="l2")


def _build_linsvm_l1_cal(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    return _build_linsvm_cal_with_penalty(config, random_state, n_jobs, penalty="l1")


def _build_rf(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    n_est = config.rf.n_estimators_grid[0] if config.rf.n_estimators_grid else 100
    return build_random_forest(
        n_estimators=n_est,
        random_state=random_state,
        n_jobs=int(max(1, n_jobs)),
    )


def _build_xgboost(config: TrainingConfig, random_state: int, n_jobs: int) -> Any:
    n_est = config.xgboost.n_estimators_grid[0] if config.xgboost.n_estimators_grid else 100
    max_d = config.xgboost.max_depth_grid[0] if config.xgboost.max_depth_grid else 5
    lr = config.xgboost.learning_rate_grid[0] if config.xgboost.learning_rate_grid else 0.05
    sub = config.xgboost.subsample_grid[0] if config.xgboost.subsample_grid else 0.8
    col = config.xgboost.colsample_bytree_grid[0] if config.xgboost.colsample_bytree_grid else 0.8
    return build_xgboost(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=lr,
        subsample=sub,
        colsample_bytree=col,
        scale_pos_weight=1.0,  # placeholder; recomputed downstream
        reg_alpha=(config.xgboost.reg_alpha_grid[0] if config.xgboost.reg_alpha_grid else 0.0),
        reg_lambda=(config.xgboost.reg_lambda_grid[0] if config.xgboost.reg_lambda_grid else 1.0),
        min_child_weight=(
            config.xgboost.min_child_weight_grid[0] if config.xgboost.min_child_weight_grid else 1
        ),
        gamma=config.xgboost.gamma_grid[0] if config.xgboost.gamma_grid else 0.0,
        tree_method=config.xgboost.tree_method,
        random_state=random_state,
        n_jobs=(int(max(1, n_jobs)) if config.xgboost.tree_method != "gpu_hist" else 1),
    )


MODEL_REGISTRY: dict[ModelName, ModelSpec] = {
    ModelName.LR_EN: ModelSpec(
        name=ModelName.LR_EN,
        display_name="Logistic Regression (ElasticNet)",
        color="#264653",  # dark teal
        builder=_build_lr_en,
        is_linear=True,
        weight_key="lr",
        hyperparam_family="lr",
    ),
    ModelName.LR_L1: ModelSpec(
        name=ModelName.LR_L1,
        display_name="Logistic Regression (L1)",
        color="#287A76",  # teal
        builder=_build_lr_l1,
        is_linear=True,
        weight_key="lr",
        hyperparam_family="lr",
    ),
    ModelName.LinSVM_cal: ModelSpec(
        name=ModelName.LinSVM_cal,
        display_name="Linear SVM (L2, calibrated)",
        color="#e9c46a",  # gold
        builder=_build_linsvm_l2_cal,
        is_linear=True,
        is_already_calibrated=True,
        weight_key="svm",
        hyperparam_family="svm",
    ),
    ModelName.LinSVM_L1_cal: ModelSpec(
        name=ModelName.LinSVM_L1_cal,
        display_name="Linear SVM (L1, calibrated)",
        color="#b58700",  # darker gold
        builder=_build_linsvm_l1_cal,
        is_linear=True,
        is_already_calibrated=True,
        weight_key="svm",
        hyperparam_family="svm",
    ),
    ModelName.RF: ModelSpec(
        name=ModelName.RF,
        display_name="Random Forest",
        color="#2a9d8f",  # teal
        builder=_build_rf,
        weight_key="rf",
        hyperparam_family="rf",
    ),
    ModelName.XGBoost: ModelSpec(
        name=ModelName.XGBoost,
        display_name="XGBoost",
        color="#f4a261",  # orange
        builder=_build_xgboost,
        weight_key="xgboost",
        hyperparam_family="xgb",
    ),
}


def get_model_spec(model_name: Any) -> ModelSpec:
    """Look up the ``ModelSpec`` for ``model_name``.

    Raises:
        ValueError: If ``model_name`` is not a valid ``ModelName`` value or
            is not registered in ``MODEL_REGISTRY``.
    """
    try:
        key = model_name if isinstance(model_name, ModelName) else ModelName(model_name)
    except ValueError as e:
        raise ValueError(
            f"Unknown model: {model_name!r}. Valid values: {sorted(m.value for m in MODEL_REGISTRY)}"
        ) from e
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {key.value!r} is defined in ModelName but not registered in MODEL_REGISTRY. "
            f"Registered: {sorted(m.value for m in MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]


def is_registered_model(model_name: Any) -> bool:
    """True if ``model_name`` resolves to an entry in ``MODEL_REGISTRY``."""
    try:
        get_model_spec(model_name)
    except ValueError:
        return False
    return True


def is_already_calibrated(model_name: Any) -> bool:
    """True if the model's builder yields a pre-calibrated estimator.

    Calibration strategies consult this to skip double-calibration.
    Unknown models return False (conservative default).
    """
    try:
        return get_model_spec(model_name).is_already_calibrated
    except ValueError:
        return False


def is_linear_model(model_name: Any) -> bool:
    """True if the model is linear in its transformed features.

    Feature-importance, RFE coefficient extraction, and SHAP surrogate paths
    consult this instead of hardcoding a tuple of names. Unknown models
    return False.
    """
    try:
        return get_model_spec(model_name).is_linear
    except ValueError:
        return False


def get_hyperparam_family(model_name: Any) -> str:
    """Return the hyperparam-dispatch family ("lr", "svm", "rf", "xgb").

    Empty string for unknown models.
    """
    try:
        return get_model_spec(model_name).hyperparam_family
    except ValueError:
        return ""


def get_registered_model_names() -> list[str]:
    """Return the string names of every model in ``MODEL_REGISTRY``.

    Canonical replacement for hardcoded lists like
    ``["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"]``.
    """
    return [m.value for m in MODEL_REGISTRY]


def build_models(
    model_name: str,
    config: TrainingConfig,
    random_state: int = 42,
    n_jobs: int = 1,
) -> object:
    """Build a single model estimator via ``MODEL_REGISTRY`` dispatch.

    Args:
        model_name: Registered model identifier (see ``MODEL_REGISTRY``).
        config: Training configuration.
        random_state: Random seed.
        n_jobs: CPU cores for parallelizable models (RF/XGBoost).

    Returns:
        sklearn-compatible estimator.

    Raises:
        ValueError: If ``model_name`` is not in ``MODEL_REGISTRY``.
        ImportError: If XGBoost is requested but not installed.
    """
    spec = get_model_spec(model_name)
    return spec.builder(config, random_state, n_jobs)


# ----------------------------
# Hyperparameter grids (moved to hyperparams.py)
# ----------------------------
