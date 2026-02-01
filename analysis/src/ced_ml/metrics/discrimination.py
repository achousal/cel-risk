"""
Discrimination metrics for binary classification models.

This module computes ranking-based performance metrics:
- AUROC (Area Under ROC Curve)
- PR-AUC (Precision-Recall Area Under Curve)
- Youden's J statistic
- Alpha (sensitivity at target specificity)

These metrics evaluate a model's ability to rank positive cases higher than
negative cases, independent of the chosen decision threshold.

References:
    - Hanley & McNeil (1982). The meaning and use of the area under a ROC curve.
    - Davis & Goadrich (2006). The relationship between PR and ROC curves.
    - Youden (1950). Index for rating diagnostic tests.
    - Fluss et al. (2005). Estimation of the Youden Index and its cutoff point.
"""

import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

from ced_ml.data.schema import METRIC_AUROC, METRIC_PRAUC
from ced_ml.utils.math_utils import EPSILON_LOGLOSS


def _validate_binary_labels(
    y_true: np.ndarray,
    metric_name: str,
    strict: bool = False,
) -> bool:
    """
    Validate that y_true contains both positive and negative classes.

    Args:
        y_true: True binary labels (0/1)
        metric_name: Name of metric for error/warning messages
        strict: If True, raise ValueError instead of warning and returning False

    Returns:
        True if both classes present, False otherwise (only when strict=False)

    Raises:
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present
    """
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        msg = (
            f"{metric_name} requires both classes (0 and 1) in y_true, "
            f"but only found {unique_classes.tolist()}."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(
            f"{msg} Returning NaN.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return True


def auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strict: bool = False,
) -> float:
    """
    Compute Area Under the ROC Curve (AUROC).

    AUROC measures the probability that a randomly chosen positive case has
    a higher predicted score than a randomly chosen negative case.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        strict: If True, raise ValueError when metric cannot be computed
            (e.g., single class). If False (default), return NaN with warning.

    Returns:
        AUROC score in [0.0, 1.0], or NaN if only one class present (strict=False)
            - 1.0: Perfect discrimination
            - 0.5: No discrimination (random classifier)
            - <0.5: Worse than random (usually indicates label swap)
            - NaN: Only one class present in y_true (strict=False only)

    Raises:
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> auroc(y_true, y_pred)
        1.0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    if not _validate_binary_labels(y_true, "AUROC", strict=strict):
        return np.nan

    return float(roc_auc_score(y_true, y_pred))


def prauc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strict: bool = False,
) -> float:
    """
    Compute Precision-Recall Area Under Curve (PR-AUC).

    PR-AUC is more informative than AUROC for imbalanced datasets where
    the positive class is rare. Unlike AUROC, PR-AUC focuses on positive
    class predictions and penalizes false positives more heavily.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        strict: If True, raise ValueError when metric cannot be computed
            (e.g., single class). If False (default), return NaN with warning.

    Returns:
        PR-AUC score in [0.0, 1.0], or NaN if only one class present (strict=False)
            - 1.0: Perfect precision and recall
            - Baseline: prevalence (random classifier)
            - NaN: Only one class present in y_true (strict=False only)

    Raises:
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> prauc(y_true, y_pred)  # doctest: +SKIP
        1.0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    if not _validate_binary_labels(y_true, "PR-AUC", strict=strict):
        return np.nan

    return float(average_precision_score(y_true, y_pred))


def youden_j(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strict: bool = False,
) -> float:
    """
    Compute Youden's J statistic (maximum TPR - FPR).

    Youden's J statistic represents the maximum vertical distance between the
    ROC curve and the diagonal (random classifier line). It identifies the
    threshold that optimally balances sensitivity and specificity.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        strict: If True, raise ValueError when metric cannot be computed
            (e.g., single class). If False (default), return NaN with warning.

    Returns:
        Youden's J statistic in [0.0, 1.0], or NaN if only one class present (strict=False)
            - 1.0: Perfect separation (TPR=1, FPR=0)
            - 0.0: No discrimination beyond chance
            - NaN: Only one class present in y_true (strict=False only)

    Raises:
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present

    Notes:
        J = max(TPR - FPR) = max(Sensitivity + Specificity - 1)

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> youden_j(y_true, y_pred)
        1.0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    if not _validate_binary_labels(y_true, "Youden's J", strict=strict):
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    j_scores = tpr - fpr
    return float(np.nanmax(j_scores))


def alpha_sensitivity_at_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_specificity: float = 0.95,
    strict: bool = False,
) -> float:
    """
    Compute sensitivity (TPR) at a target specificity level (Alpha metric).

    This metric evaluates model performance at high-specificity operating points,
    useful for clinical screening where false positives must be minimized.
    Common targets are 95% or 99% specificity.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        target_specificity: Target specificity level, must be in (0, 1)
            Default: 0.95 (95% specificity)
        strict: If True, raise ValueError when metric cannot be computed
            (e.g., single class). If False (default), return NaN with warning.

    Returns:
        Sensitivity achieved at or above the target specificity, or NaN if only one class present (strict=False)
            - If target is achievable: max sensitivity among thresholds meeting target
            - If target is unachievable: sensitivity at closest achievable specificity
            - NaN: Only one class present in y_true (strict=False only)

    Raises:
        ValueError: If target_specificity is not in (0, 1)
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
        1.0

        >>> # For imbalanced data with rare positives
        >>> y_true = np.array([0]*95 + [1]*5)
        >>> y_pred = np.random.random(100)
        >>> alpha = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.99)
        >>> 0.0 <= alpha <= 1.0
        True
    """
    if not 0 < target_specificity < 1:
        raise ValueError(f"target_specificity must be in (0, 1), got {target_specificity}")

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    if not _validate_binary_labels(y_true, "Alpha (sensitivity at specificity)", strict=strict):
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    specificity = 1.0 - fpr

    # Find thresholds meeting or exceeding target specificity
    meets_target = specificity >= target_specificity

    if np.any(meets_target):
        # Among thresholds meeting target, return maximum sensitivity
        return float(np.max(tpr[meets_target]))
    else:
        # If target is unachievable, return sensitivity at closest specificity
        closest_idx = int(np.argmin(np.abs(specificity - target_specificity)))
        return float(tpr[closest_idx])


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    include_youden: bool = True,
    include_alpha: bool = True,
    alpha_target_specificity: float = 0.95,
    strict: bool = False,
) -> dict[str, float]:
    """
    Compute all discrimination metrics in one pass.

    This is more efficient than calling individual metric functions when
    multiple metrics are needed, as it reuses computed ROC curves.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        include_youden: Whether to compute Youden's J statistic
        include_alpha: Whether to compute Alpha (sensitivity at target specificity)
        alpha_target_specificity: Target specificity for Alpha metric (default 0.95)
        strict: If True, raise ValueError when metrics cannot be computed
            (e.g., single class). If False (default), return NaN values with warning.

    Returns:
        Dictionary with metric names as keys and computed values:
            - "AUROC": Area under ROC curve (NaN if only one class, strict=False)
            - "PR_AUC": Precision-Recall area under curve (NaN if only one class, strict=False)
            - "Youden": Youden's J statistic (if include_youden=True, NaN if only one class, strict=False)
            - "Alpha": Sensitivity at target specificity (if include_alpha=True, NaN if only one class, strict=False)

    Raises:
        ValueError: If strict=True and only one class is present

    Warns:
        UserWarning: If strict=False and only one class is present

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        >>> metrics = compute_discrimination_metrics(y_true, y_pred)
        >>> sorted(metrics.keys())
        ['AUROC', 'Alpha', 'PR_AUC', 'Youden']
        >>> metrics['AUROC']
        1.0
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    # Check for single-class case
    if not _validate_binary_labels(y_true, "compute_discrimination_metrics", strict=strict):
        # Return NaN for all metrics
        metrics = {
            METRIC_AUROC: np.nan,
            METRIC_PRAUC: np.nan,
        }
        if include_youden:
            metrics["Youden"] = np.nan
        if include_alpha:
            metrics["Alpha"] = np.nan
        return metrics

    # Core discrimination metrics
    metrics = {
        METRIC_AUROC: float(roc_auc_score(y_true, y_pred)),
        METRIC_PRAUC: float(average_precision_score(y_true, y_pred)),
    }

    # Optional metrics requiring ROC curve computation
    if include_youden or include_alpha:
        fpr, tpr, _ = roc_curve(y_true, y_pred)

        if include_youden:
            j_scores = tpr - fpr
            metrics["Youden"] = float(np.nanmax(j_scores))

        if include_alpha:
            specificity = 1.0 - fpr
            meets_target = specificity >= alpha_target_specificity

            if np.any(meets_target):
                metrics["Alpha"] = float(np.max(tpr[meets_target]))
            else:
                closest_idx = int(np.argmin(np.abs(specificity - alpha_target_specificity)))
                metrics["Alpha"] = float(tpr[closest_idx])

    return metrics


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of predicted probabilities).

    While Brier score is technically a calibration metric, it's often reported
    alongside discrimination metrics. Lower is better.

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)

    Returns:
        Brier score in [0.0, 1.0]
            - 0.0: Perfect calibration and discrimination
            - 0.25: Baseline for balanced dataset (constant 0.5 prediction)

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.1, 0.9, 0.9])
        >>> compute_brier_score(y_true, y_pred)
        0.02
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    return float(brier_score_loss(y_true, y_pred))


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPSILON_LOGLOSS) -> float:
    """
    Compute log loss (cross-entropy) with numerical stability clipping.

    Log loss heavily penalizes confident wrong predictions and is sensitive
    to calibration. Probabilities are clipped to [eps, 1-eps] to avoid log(0).

    Args:
        y_true: True binary labels (0/1), shape (n_samples,)
        y_pred: Predicted probabilities for positive class, shape (n_samples,)
        eps: Clipping threshold to avoid log(0), default 1e-15 (EPSILON_LOGLOSS)

    Returns:
        Log loss >= 0.0 (lower is better)
            - 0.0: Perfect predictions (p=1 for y=1, p=0 for y=0)
            - log(2) ≈ 0.693: Random predictor (p=0.5 always)

    Examples:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.1, 0.9, 0.9])
        >>> compute_log_loss(y_true, y_pred)  # doctest: +ELLIPSIS
        0.105...
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    return float(log_loss(y_true, y_pred_clipped))
