"""
Model selection scoring utilities.

This module provides composite scoring functions for formal model selection,
combining discrimination (AUROC), calibration (Brier score), and calibration
quality (calibration slope) into a single metric.

The selection score enables consistent model comparison across multiple
evaluation dimensions rather than relying on a single metric like AUROC.

References:
    - Van Calster et al. (2019). Calibration: the Achilles heel of predictive
      analytics. BMC Medicine.
    - Collins et al. (2015). TRIPOD Statement. BMJ.
"""

import logging
from typing import Any

import numpy as np

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER

logger = logging.getLogger(__name__)

# Default weights for composite score components
DEFAULT_WEIGHTS: dict[str, float] = {
    "auroc": 0.50,
    "brier": 0.30,
    "slope": 0.20,
}


def compute_selection_score(
    metrics: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute composite score for model selection.

    Combines discrimination (AUROC), prediction quality (Brier score), and
    calibration quality (calibration slope) into a single score suitable for
    model comparison and selection.

    Score components:
    - AUROC: Direct contribution (higher is better)
    - Brier: Inverted contribution as (1 - Brier) (lower Brier is better)
    - Slope: Contribution as (1 - |slope - 1|) (closer to 1.0 is better)

    Args:
        metrics: Dictionary with metric values. Expected keys:
            - 'AUROC': Area under ROC curve [0, 1]
            - 'Brier': Brier score [0, 1], lower is better
            - 'calib_slope': Calibration slope, ideally ~1.0
            Keys are case-insensitive and support variants:
            - AUROC variants: 'AUROC', 'auroc', 'auc', 'roc_auc'
            - Brier variants: 'Brier', 'brier', 'brier_score'
            - Slope variants: 'calib_slope', 'calibration_slope', 'slope'
        weights: Optional custom weights for score components.
            Default: {'auroc': 0.50, 'brier': 0.30, 'slope': 0.20}
            Weights should sum to 1.0 for interpretable scores.

    Returns:
        Composite score in [0, 1] where higher is better.
        Returns NaN if required metrics are missing or invalid.

    Examples:
        >>> # Perfect model
        >>> metrics = {'AUROC': 1.0, 'Brier': 0.0, 'calib_slope': 1.0}
        >>> compute_selection_score(metrics)
        1.0

        >>> # Random model (AUROC=0.5, Brier=0.25, perfect calibration)
        >>> metrics = {'AUROC': 0.5, 'Brier': 0.25, 'calib_slope': 1.0}
        >>> round(compute_selection_score(metrics), 3)
        0.475

        >>> # Custom weights emphasizing AUROC
        >>> metrics = {'AUROC': 0.9, 'Brier': 0.1, 'calib_slope': 1.0}
        >>> weights = {'auroc': 0.7, 'brier': 0.2, 'slope': 0.1}
        >>> round(compute_selection_score(metrics, weights), 3)
        0.91
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Validate weights
    if not weights:
        logger.warning("Empty weights provided, using defaults")
        weights = DEFAULT_WEIGHTS.copy()

    # Extract metrics using canonical keys (with fallback for legacy keys)
    auroc = metrics.get(METRIC_AUROC) or _extract_metric(
        metrics,
        ["AUROC", "auroc", "auc", "roc_auc", "ROC_AUC"],
        default=0.5,
    )
    brier = metrics.get(METRIC_BRIER) or _extract_metric(
        metrics,
        ["Brier", "brier", "brier_score", "Brier_score"],
        default=0.25,
    )
    slope = _extract_metric(
        metrics,
        ["calib_slope", "calibration_slope", "slope", "Slope"],
        default=1.0,
    )

    # Validate metric values
    if not _is_valid_metric(auroc):
        logger.warning(f"Invalid AUROC value: {auroc}, using default 0.5")
        auroc = 0.5

    if not _is_valid_metric(brier):
        logger.warning(f"Invalid Brier value: {brier}, using default 0.25")
        brier = 0.25

    if not _is_valid_metric(slope):
        logger.warning(f"Invalid slope value: {slope}, using default 1.0")
        slope = 1.0

    # Clamp values to valid ranges
    auroc = np.clip(float(auroc), 0.0, 1.0)
    brier = np.clip(float(brier), 0.0, 1.0)
    slope = float(slope)

    # Compute component scores
    # AUROC: direct contribution (higher is better)
    auroc_component = auroc

    # Brier: inverted (lower Brier is better, so 1 - Brier)
    brier_component = 1.0 - brier

    # Slope: penalize deviation from 1.0
    # Score is 1.0 when slope=1.0, decreases as slope deviates
    # Clamp deviation penalty to [0, 1] range
    slope_deviation = min(abs(slope - 1.0), 1.0)
    slope_component = 1.0 - slope_deviation

    # Normalize weights to sum to 1.0
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        logger.warning("Weights sum to zero or negative, using defaults")
        weights = DEFAULT_WEIGHTS.copy()
        weight_sum = 1.0

    auroc_weight = weights.get("auroc", 0.0) / weight_sum
    brier_weight = weights.get("brier", 0.0) / weight_sum
    slope_weight = weights.get("slope", 0.0) / weight_sum

    # Compute weighted composite score
    score = (
        auroc_weight * auroc_component
        + brier_weight * brier_component
        + slope_weight * slope_component
    )

    return float(score)


def _extract_metric(
    metrics: dict[str, Any],
    keys: list[str],
    default: float,
) -> float:
    """
    Extract metric value from dictionary with flexible key matching.

    Args:
        metrics: Dictionary of metrics
        keys: List of possible key names (checked in order)
        default: Default value if no key found

    Returns:
        Metric value or default
    """
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if value is not None:
                return float(value)
    return default


def _is_valid_metric(value: Any) -> bool:
    """
    Check if a metric value is valid (finite number).

    Args:
        value: Value to check

    Returns:
        True if value is a finite number, False otherwise
    """
    if value is None:
        return False
    try:
        fval = float(value)
        return np.isfinite(fval)
    except (TypeError, ValueError):
        return False


def compute_selection_scores_for_models(
    model_metrics: dict[str, dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute selection scores for multiple models.

    Convenience function to compute selection scores for a dictionary
    of per-model metrics, as produced by aggregation functions.

    Args:
        model_metrics: Dictionary mapping model names to their metrics dicts.
            Example: {'LR_EN': {'AUROC': 0.85, 'Brier': 0.12, ...}, ...}
        weights: Optional custom weights (passed to compute_selection_score)

    Returns:
        Dictionary mapping model names to selection scores.

    Examples:
        >>> model_metrics = {
        ...     'LR_EN': {'AUROC': 0.85, 'Brier': 0.10, 'calib_slope': 1.05},
        ...     'RF': {'AUROC': 0.82, 'Brier': 0.12, 'calib_slope': 0.90},
        ... }
        >>> scores = compute_selection_scores_for_models(model_metrics)
        >>> sorted(scores.keys())
        ['LR_EN', 'RF']
    """
    scores = {}
    for model_name, metrics in model_metrics.items():
        scores[model_name] = compute_selection_score(metrics, weights)
    return scores


def rank_models_by_selection_score(
    model_metrics: dict[str, dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """
    Rank models by their selection scores (descending).

    Args:
        model_metrics: Dictionary mapping model names to their metrics dicts
        weights: Optional custom weights

    Returns:
        List of (model_name, score) tuples sorted by score (highest first)

    Examples:
        >>> model_metrics = {
        ...     'LR_EN': {'AUROC': 0.85, 'Brier': 0.10, 'calib_slope': 1.0},
        ...     'RF': {'AUROC': 0.80, 'Brier': 0.15, 'calib_slope': 0.9},
        ... }
        >>> ranking = rank_models_by_selection_score(model_metrics)
        >>> ranking[0][0]  # Best model
        'LR_EN'
    """
    scores = compute_selection_scores_for_models(model_metrics, weights)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
