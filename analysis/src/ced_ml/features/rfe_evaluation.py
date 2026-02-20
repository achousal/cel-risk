"""RFE panel evaluation, metrics, and size computation utilities.

Contains functions for computing evaluation sizes, panel recommendations,
bootstrap confidence intervals, and per-size metric evaluation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ced_ml.metrics.discrimination import (
    alpha_sensitivity_at_specificity,
    auroc,
    compute_brier_score,
    prauc,
)

logger = logging.getLogger(__name__)


def compute_eval_sizes(
    max_size: int,
    min_size: int,
    strategy: str = "geometric",
) -> list[int]:
    """Compute panel sizes to evaluate based on elimination strategy.

    Args:
        max_size: Starting panel size.
        min_size: Smallest panel size to evaluate.
        strategy: One of:
            - "geometric": Evaluate at powers of 2 (100, 50, 25, 12, 6, ...). [DEFAULT]
            - "fine": More granular than geometric, evaluates intermediate points.
              Adds quarter-steps between geometric levels (e.g., 100, 75, 50, 37, 25, 18, 12, ...).
            - "linear": Evaluate every size from max to min (slowest, most data).

    Returns:
        List of panel sizes to evaluate, sorted descending.

    Raises:
        ValueError: If strategy is not one of the valid options.

    Example:
        >>> compute_eval_sizes(100, 5, "geometric")
        [100, 50, 25, 12, 6, 5]
        >>> compute_eval_sizes(100, 5, "fine")
        [100, 75, 50, 37, 25, 18, 12, 9, 6, 5]
    """
    valid_strategies = {"geometric", "fine", "linear"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid step_strategy: '{strategy}'. " f"Must be one of {sorted(valid_strategies)}."
        )

    if max_size <= min_size:
        return [max_size]

    if strategy == "linear":
        return list(range(max_size, min_size - 1, -1))

    if strategy == "fine":
        sizes = []
        current = max_size
        while current > min_size:
            sizes.append(current)
            quarter_step = max(min_size, int(current * 0.75))
            if quarter_step not in sizes and quarter_step > min_size:
                sizes.append(quarter_step)
            half_step = max(min_size, current // 2)
            if half_step not in sizes and half_step > min_size:
                sizes.append(half_step)
            current = half_step
            if current in sizes and current <= min_size:
                break

        if min_size not in sizes:
            sizes.append(min_size)

        return sorted(set(sizes), reverse=True)

    # Geometric: powers of 2 plus min_size
    sizes = []
    current = max_size
    while current > min_size:
        sizes.append(current)
        current = max(min_size, current // 2)
        if current in sizes:
            break

    if min_size not in sizes:
        sizes.append(min_size)

    return sorted(set(sizes), reverse=True)


def find_recommended_panels(
    curve: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> dict[str, int]:
    """Find smallest panel sizes maintaining various AUROC thresholds.

    Args:
        curve: List of dicts with keys "size" and "auroc_val".
        thresholds: Fractions of max AUROC to target (default: [0.95, 0.90, 0.85]).

    Returns:
        Dict with keys like "min_size_95pct" -> panel size, plus "knee_point".

    Example:
        >>> curve = [
        ...     {"size": 100, "auroc_val": 0.90},
        ...     {"size": 50, "auroc_val": 0.88},
        ...     {"size": 25, "auroc_val": 0.85},
        ... ]
        >>> find_recommended_panels(curve)
        {'min_size_95pct': 50, 'min_size_90pct': 25, 'knee_point': 50, ...}
    """
    if thresholds is None:
        thresholds = [0.95, 0.90, 0.85]

    if not curve:
        return {}

    max_auroc = max(p["auroc_val"] for p in curve)
    recommended: dict[str, int] = {}

    for thresh in thresholds:
        target = max_auroc * thresh
        valid = [p for p in curve if p["auroc_val"] >= target]
        if valid:
            smallest = min(valid, key=lambda x: x["size"])
            key = f"min_size_{int(thresh * 100)}pct"
            recommended[key] = smallest["size"]

    recommended["knee_point"] = detect_knee_point(curve)

    return recommended


def detect_knee_point(curve: list[dict[str, Any]]) -> int:
    """Detect knee point (elbow) in the AUROC vs size curve.

    Uses the perpendicular distance method: finds the point with maximum
    distance from the line connecting the first and last points.

    Args:
        curve: List of dicts with "size" and "auroc_val".

    Returns:
        Panel size at the knee point.
    """
    if len(curve) < 3:
        return curve[0]["size"] if curve else 0

    sorted_curve = sorted(curve, key=lambda x: -x["size"])

    sizes = np.array([p["size"] for p in sorted_curve], dtype=float)
    aurocs = np.array([p["auroc_val"] for p in sorted_curve], dtype=float)

    size_range = sizes.max() - sizes.min()
    auroc_range = aurocs.max() - aurocs.min()

    if size_range == 0 or auroc_range == 0:
        return sorted_curve[0]["size"]

    x = (sizes - sizes.min()) / size_range
    y = (aurocs - aurocs.min()) / auroc_range

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    denom = np.sqrt(a**2 + b**2)

    if denom == 0:
        return sorted_curve[0]["size"]

    distances = np.abs(a * x + b * y + c) / denom

    knee_idx = int(np.argmax(distances))
    return int(sorted_curve[knee_idx]["size"])


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap percentile confidence interval for AUROC.

    Returns:
        Tuple of (std, ci_low, ci_high) where ci_low/ci_high are the
        percentile-based bounds at the requested confidence level.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        if len(np.unique(y_t)) < 2:
            continue

        try:
            aurocs.append(auroc(y_t, y_p))
        except Exception as e:
            logger.debug("Bootstrap AUROC calculation skipped: %s", e)
            continue

    if not aurocs:
        return 0.0, 0.0, 0.0

    alpha = 1.0 - ci_level
    ci_low = float(np.percentile(aurocs, 100 * alpha / 2))
    ci_high = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return float(np.std(aurocs)), ci_low, ci_high


def evaluate_panel_size(
    pipeline: Pipeline,
    X_train_subset: pd.DataFrame,
    y_train: np.ndarray,
    X_val_subset: pd.DataFrame,
    y_val: np.ndarray,
    cv_folds: int,
    random_state: int,
    panel_size: int,
    cv_n_jobs: int = 1,
) -> dict[str, float]:
    """Evaluate metrics for a given panel size.

    Args:
        pipeline: Fitted pipeline to evaluate.
        X_train_subset: Training features subset.
        y_train: Training labels.
        X_val_subset: Validation features subset.
        y_val: Validation labels.
        cv_folds: CV folds for OOF estimation (0 to skip).
        random_state: Random seed.
        panel_size: Current panel size (for logging).
        cv_n_jobs: Parallel jobs for cross_val_predict (default 1).

    Returns:
        Dict with all evaluation metrics.
    """
    metrics: dict[str, float] = {}

    if cv_folds > 1:
        logger.info(f"  Running {cv_folds}-fold CV for size {panel_size}...")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        oof_probs = cross_val_predict(
            clone(pipeline),
            X_train_subset,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=cv_n_jobs,
        )[:, 1]
        metrics["auroc_cv"] = auroc(y_train, oof_probs)
        metrics["prauc_cv"] = prauc(y_train, oof_probs)
        metrics["brier_cv"] = compute_brier_score(y_train, oof_probs)
        metrics["sens_at_95spec_cv"] = alpha_sensitivity_at_specificity(
            y_train, oof_probs, target_specificity=0.95
        )
        logger.info(
            f"  CV AUROC: {metrics['auroc_cv']:.4f}, "
            f"PR-AUC: {metrics['prauc_cv']:.4f}, "
            f"Brier: {metrics['brier_cv']:.4f}"
        )
        auroc_cv_std, _, _ = bootstrap_auroc_ci(y_train, oof_probs, n_bootstrap=50)
        metrics["auroc_cv_std"] = auroc_cv_std
    else:
        train_probs = pipeline.predict_proba(X_train_subset)[:, 1]
        metrics["auroc_cv"] = auroc(y_train, train_probs)
        metrics["prauc_cv"] = prauc(y_train, train_probs)
        metrics["brier_cv"] = compute_brier_score(y_train, train_probs)
        metrics["sens_at_95spec_cv"] = alpha_sensitivity_at_specificity(
            y_train, train_probs, target_specificity=0.95
        )
        metrics["auroc_cv_std"] = 0.0
        logger.info(
            f"  Train AUROC: {metrics['auroc_cv']:.4f}, "
            f"PR-AUC: {metrics['prauc_cv']:.4f}, "
            f"Brier: {metrics['brier_cv']:.4f}"
        )

    logger.info("  Computing validation metrics...")
    val_probs = pipeline.predict_proba(X_val_subset)[:, 1]
    metrics["auroc_val"] = auroc(y_val, val_probs)
    metrics["prauc_val"] = prauc(y_val, val_probs)
    metrics["brier_val"] = compute_brier_score(y_val, val_probs)
    metrics["sens_at_95spec_val"] = alpha_sensitivity_at_specificity(
        y_val, val_probs, target_specificity=0.95
    )
    auroc_val_std, auroc_val_ci_low, auroc_val_ci_high = bootstrap_auroc_ci(
        y_val, val_probs, n_bootstrap=200
    )
    metrics["auroc_val_std"] = auroc_val_std
    metrics["auroc_val_ci_low"] = auroc_val_ci_low
    metrics["auroc_val_ci_high"] = auroc_val_ci_high

    logger.info(
        f"[RFE k={panel_size}] Val AUROC={metrics['auroc_val']:.3f} "
        f"(95% CI: {auroc_val_ci_low:.3f}-{auroc_val_ci_high:.3f}), "
        f"PR-AUC={metrics['prauc_val']:.3f}, Brier={metrics['brier_val']:.4f}"
    )

    return metrics
