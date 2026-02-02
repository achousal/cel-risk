"""Model registry and hyperparameter grid definitions.

This module provides:
- Model instantiation (RF, XGBoost, LinSVM, LogisticRegression)
- Hyperparameter grid generation for RandomizedSearchCV
- sklearn version compatibility handling

References:
- scikit-learn 1.8+ deprecates penalty= in LogisticRegression (use l1_ratio=)
- XGBoost tree_method controls CPU vs GPU acceleration
"""

import logging
import re
from typing import Any

import numpy as np
import sklearn
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
# sklearn version compatibility
# ----------------------------
def _sklearn_version_tuple(ver: str) -> tuple[int, int, int]:
    """Parse sklearn version string (robust to rc/dev suffixes)."""
    nums = re.findall(r"\d+", ver)
    nums = (nums + ["0", "0", "0"])[:3]
    return (int(nums[0]), int(nums[1]), int(nums[2]))


SKLEARN_VER = _sklearn_version_tuple(getattr(sklearn, "__version__", "0.0.0"))


# ----------------------------
# Parameter grid utilities
# ----------------------------
def _parse_float_list(s: str) -> list[float]:
    """Parse comma-separated float values."""
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            logger.debug(f"Failed to parse '{tok}' as float; skipping token.")
            continue
    return out


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integer values."""
    if not s:
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            logger.debug(f"Failed to parse '{x}' as int; skipping token.")
            continue
    return out


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
        except Exception:
            logger.debug(f"Failed to parse '{tok}' as int via regex; will try float.")
        try:
            out.append(float(tok))
            continue
        except Exception:
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
            except Exception:
                logger.debug(f"{name}: failed to parse '{v}' as int; will try float.")
            try:
                fv = float(vv)
                if float(fv).is_integer():
                    out.append(int(fv))
                    continue
            except Exception:
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
            except Exception:
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
            except Exception:
                logger.debug(f"{name}: failed to parse '{v}' as float.")
                pass
            raise ValueError(f"{name}: could not parse '{v}' as int>=1 or float in (0,1)")
        raise ValueError(f"{name}: unsupported type {type(v).__name__}={v}")
    return out


def parse_class_weight_options(s: str) -> list:
    """Parse class_weight options.

    Examples:
        "none,balanced" -> [None, "balanced"]
        "balanced" -> ["balanced"]
        "" -> [None, "balanced"] (default)
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
    penalty: str = "elasticnet",
) -> LogisticRegression:
    """Build Logistic Regression estimator (sklearn 1.8+ compatible).

    Args:
        solver: Optimization algorithm
        C: Inverse regularization strength
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed
        l1_ratio: ElasticNet mixing (0=L2, 1=L1)
        penalty: Penalty type (ignored in sklearn >=1.8)

    Returns:
        Configured LogisticRegression estimator
    """
    lr_common = {
        "solver": solver,
        "C": C,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": int(random_state),
    }

    # sklearn >=1.8 deprecates penalty=, uses l1_ratio
    if SKLEARN_VER >= (1, 8, 0):
        return LogisticRegression(l1_ratio=l1_ratio, **lr_common)
    else:
        return LogisticRegression(penalty=penalty, l1_ratio=l1_ratio, **lr_common)


def build_linear_svm_calibrated(
    C: float = 1.0,
    max_iter: int = 2000,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Build calibrated LinearSVC estimator.

    LinearSVC + CalibratedClassifierCV provides probability estimates.

    Args:
        C: Inverse regularization strength
        max_iter: Maximum iterations
        calibration_method: 'sigmoid' or 'isotonic'
        calibration_cv: CV folds for calibration
        random_state: Random seed

    Returns:
        CalibratedClassifierCV wrapping LinearSVC
    """
    base_svm = LinearSVC(
        C=C, class_weight=None, random_state=int(random_state), max_iter=int(max_iter)
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
        except Exception:
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


def build_models(
    model_name: str,
    config: TrainingConfig,
    random_state: int = 42,
    n_jobs: int = 1,
) -> object:
    """Build a single model estimator.

    Args:
        model_name: Model identifier ('LR_EN', 'LR_L1', 'LinSVM_cal', 'RF', 'XGBoost')
        config: Training configuration
        random_state: Random seed
        n_jobs: CPU cores for RF/XGBoost

    Returns:
        sklearn-compatible estimator

    Raises:
        ValueError: If model_name is unknown
        ImportError: If XGBoost requested but not installed
    """
    if model_name == ModelName.LR_EN:
        return build_logistic_regression(
            solver=config.lr.solver,
            C=1.0,
            max_iter=config.lr.max_iter,
            tol=1e-4,  # LRConfig doesn't have tol field
            random_state=random_state,
            l1_ratio=0.5,
            penalty="elasticnet",
        )

    elif model_name == ModelName.LR_L1:
        return build_logistic_regression(
            solver=config.lr.solver,
            C=1.0,
            max_iter=config.lr.max_iter,
            tol=1e-4,  # LRConfig doesn't have tol field
            random_state=random_state,
            l1_ratio=1.0,
            penalty="l1",
        )

    elif model_name == ModelName.LinSVM_cal:
        return build_linear_svm_calibrated(
            C=1.0,
            max_iter=config.svm.max_iter,
            calibration_method=config.calibration.method,
            calibration_cv=config.calibration.cv,
            random_state=random_state,
        )

    elif model_name == ModelName.RF:
        # Get first value from n_estimators_grid list for default model
        n_est = config.rf.n_estimators_grid[0] if config.rf.n_estimators_grid else 100
        return build_random_forest(
            n_estimators=n_est,
            random_state=random_state,
            n_jobs=int(max(1, n_jobs)),
        )

    elif model_name == ModelName.XGBoost:
        # Get first values from grid lists for default model
        n_est = config.xgboost.n_estimators_grid[0] if config.xgboost.n_estimators_grid else 100
        max_d = config.xgboost.max_depth_grid[0] if config.xgboost.max_depth_grid else 5
        lr = config.xgboost.learning_rate_grid[0] if config.xgboost.learning_rate_grid else 0.05
        sub = config.xgboost.subsample_grid[0] if config.xgboost.subsample_grid else 0.8
        col = (
            config.xgboost.colsample_bytree_grid[0] if config.xgboost.colsample_bytree_grid else 0.8
        )
        spw = 1.0  # Default, will be computed later
        return build_xgboost(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=lr,
            subsample=sub,
            colsample_bytree=col,
            scale_pos_weight=spw,
            reg_alpha=(config.xgboost.reg_alpha_grid[0] if config.xgboost.reg_alpha_grid else 0.0),
            reg_lambda=(
                config.xgboost.reg_lambda_grid[0] if config.xgboost.reg_lambda_grid else 1.0
            ),
            min_child_weight=(
                config.xgboost.min_child_weight_grid[0]
                if config.xgboost.min_child_weight_grid
                else 1
            ),
            gamma=config.xgboost.gamma_grid[0] if config.xgboost.gamma_grid else 0.0,
            tree_method=config.xgboost.tree_method,
            random_state=random_state,
            n_jobs=(int(max(1, n_jobs)) if config.xgboost.tree_method != "gpu_hist" else 1),
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ----------------------------
# Hyperparameter grids (moved to hyperparams.py)
# ----------------------------
