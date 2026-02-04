"""
Stratified bootstrap confidence intervals for binary classification metrics.

This module provides functions for computing bootstrap confidence intervals
using stratified resampling to maintain case/control ratios. The percentile
method is used for CI construction.
"""

import logging
from collections.abc import Callable

import numpy as np

from ced_ml.utils.constants import CI_LOWER_PCT, CI_UPPER_PCT, MIN_BOOTSTRAP_SAMPLES

logger = logging.getLogger(__name__)


def _safe_metric(metric_fn: Callable, y: np.ndarray, p: np.ndarray) -> float:
    """
    Safely compute metric, returning NaN on failure.

    Args:
        metric_fn: Metric function that takes (y_true, y_pred)
        y: True labels
        p: Predicted probabilities

    Returns:
        Metric value, or NaN if computation failed
    """
    try:
        return metric_fn(y, p)
    except Exception as e:
        logger.debug("Metric computation failed: %s", e)
        return np.nan


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 0,
    min_valid_frac: float = 0.1,
) -> tuple[float, float]:
    """
    Compute stratified bootstrap confidence interval for a metric.

    Performs stratified resampling (maintaining case/control ratio) and computes
    95% CI using percentile method. If fewer than `max(20, n_boot * min_valid_frac)`
    valid bootstrap samples are obtained, returns (NaN, NaN).

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities [0, 1]
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 0)
        min_valid_frac: Minimum fraction of valid samples required (default: 0.1)

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI, or (NaN, NaN) if insufficient
        valid samples

    Raises:
        ValueError: If fewer than 2 cases or 2 controls in y_true

    Examples:
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4])
        >>> ci_lower, ci_upper = stratified_bootstrap_ci(
        ...     y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        ... )
        >>> 0 <= ci_lower <= ci_upper <= 1
        True
    """
    rng = np.random.RandomState(seed)
    logger.info("Bootstrap CI: seed=%d, n_boot=%d, n_samples=%d", seed, n_boot, len(y_true))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)} elements"
        )

    # Identify cases and controls
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"Insufficient samples for stratified bootstrap: "
            f"{len(pos)} cases, {len(neg)} controls (need ≥2 each)"
        )

    # Perform stratified bootstrap
    vals = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])
        v = _safe_metric(metric_fn, y_true[idx], y_pred[idx])
        if np.isfinite(v):
            vals.append(v)

    # Check minimum valid samples threshold
    min_valid = max(MIN_BOOTSTRAP_SAMPLES, int(n_boot * min_valid_frac))
    if len(vals) < min_valid:
        return (np.nan, np.nan)

    # Compute 95% CI using percentile method
    return (
        float(np.percentile(vals, CI_LOWER_PCT)),
        float(np.percentile(vals, CI_UPPER_PCT)),
    )


def stratified_bootstrap_diff_ci(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 500,
    seed: int = 0,
    min_valid_frac: float = 0.1,
) -> tuple[float, float, float]:
    """
    Compute stratified bootstrap CI for difference between two models.

    Computes the metric difference (model1 - model2) on the full sample and
    on stratified bootstrap resamples, then returns the full-sample difference
    and its 95% CI.

    Args:
        y_true: True binary labels (0/1)
        p1: Predictions from model 1
        p2: Predictions from model 2
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations (default: 500)
        seed: Random seed for reproducibility (default: 0)
        min_valid_frac: Minimum fraction of valid samples required (default: 0.1)

    Returns:
        Tuple of (diff_full, lower_bound, upper_bound) where:
        - diff_full: Full-sample difference (model1 - model2)
        - lower_bound: 2.5th percentile of bootstrap distribution
        - upper_bound: 97.5th percentile of bootstrap distribution

        If insufficient valid samples, returns (diff_full, NaN, NaN)

    Raises:
        ValueError: If fewer than 2 cases or 2 controls in y_true, or if
                    array lengths don't match

    Examples:
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> p1 = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4])
        >>> p2 = np.array([0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.5, 0.5])
        >>> diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
        ...     y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        ... )
        >>> isinstance(diff, float)
        True
    """
    rng = np.random.RandomState(seed)
    logger.debug("Bootstrap diff CI using seed=%d, n_boot=%d", seed, n_boot)
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1).astype(float)
    p2 = np.asarray(p2).astype(float)

    # Validate inputs
    if not (len(y_true) == len(p1) == len(p2)):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, p1={len(p1)}, p2={len(p2)}")

    # Compute full-sample difference
    diff_full = float(metric_fn(y_true, p1) - metric_fn(y_true, p2))

    # Identify cases and controls
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"Insufficient samples for stratified bootstrap: "
            f"{len(pos)} cases, {len(neg)} controls (need ≥2 each)"
        )

    # Perform stratified bootstrap on differences
    diffs = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])

        m1 = _safe_metric(metric_fn, y_true[idx], p1[idx])
        m2 = _safe_metric(metric_fn, y_true[idx], p2[idx])
        if np.isfinite(m1) and np.isfinite(m2):
            diffs.append(m1 - m2)

    # Check minimum valid samples threshold
    min_valid = max(MIN_BOOTSTRAP_SAMPLES, int(n_boot * min_valid_frac))
    if len(diffs) < min_valid:
        return (diff_full, np.nan, np.nan)

    # Compute 95% CI using percentile method
    return (
        diff_full,
        float(np.percentile(diffs, CI_LOWER_PCT)),
        float(np.percentile(diffs, CI_UPPER_PCT)),
    )
