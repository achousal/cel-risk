"""Threshold selection strategies using the Strategy pattern.

This module provides a unified interface for threshold selection strategies,
replacing ad-hoc config attribute checks with type-safe strategy objects.

The Protocol pattern allows:
- Type-safe threshold selection
- Easy addition of new strategies
- Testable, composable threshold logic
- Clear separation of configuration from computation

Usage:
    strategy = get_threshold_strategy(config)
    threshold = strategy.find_threshold(y_true, y_prob)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ced_ml.metrics.thresholds import (
    threshold_for_precision,
    threshold_for_specificity,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
)

if TYPE_CHECKING:
    from ced_ml.config.calibration_schema import ThresholdConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class ThresholdStrategy(Protocol):
    """Protocol for threshold selection strategies.

    All threshold strategies must implement:
    - find_threshold: Compute optimal threshold from labels and probabilities
    - name: Return a descriptive name for the strategy (for logging/serialization)

    Example:
        >>> strategy = FixedSpecificityThreshold(target_specificity=0.95)
        >>> threshold = strategy.find_threshold(y_true, y_prob)
        >>> print(f"Using {strategy.name} threshold: {threshold}")
    """

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find the optimal threshold for the given data.

        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities [0, 1]

        Returns:
            Optimal threshold value in [0, 1]
        """
        ...

    @property
    def name(self) -> str:
        """Return a descriptive name for this strategy."""
        ...


@dataclass(frozen=True)
class MaxF1Threshold:
    """Find threshold maximizing F1 score.

    The F1 score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    This strategy balances precision and recall equally.
    """

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold maximizing F1 score."""
        return threshold_max_f1(y_true, y_prob)

    @property
    def name(self) -> str:
        return "max_f1"


@dataclass(frozen=True)
class MaxFBetaThreshold:
    """Find threshold maximizing F-beta score.

    The F-beta score weights recall more than precision by a factor of beta:
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    Args:
        beta: Weight for recall vs precision.
            beta > 1 favors recall, beta < 1 favors precision.

    Examples:
        - beta=2: Recall is twice as important as precision
        - beta=0.5: Precision is twice as important as recall
    """

    beta: float = 1.0

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold maximizing F-beta score."""
        return threshold_max_fbeta(y_true, y_prob, beta=self.beta)

    @property
    def name(self) -> str:
        return f"max_fbeta_{self.beta:.2f}"


@dataclass(frozen=True)
class YoudensJThreshold:
    """Find threshold maximizing Youden's J statistic.

    Youden's J = sensitivity + specificity - 1 = TPR - FPR

    This maximizes the vertical distance from the ROC curve to the diagonal,
    balancing sensitivity and specificity equally.
    """

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold maximizing Youden's J statistic."""
        return threshold_youden(y_true, y_prob)

    @property
    def name(self) -> str:
        return "youden"


@dataclass(frozen=True)
class FixedSpecificityThreshold:
    """Find threshold achieving target specificity.

    Selects the lowest threshold (highest sensitivity) among those
    meeting the specificity target. Falls back to closest achievable
    specificity if target is unattainable.

    Args:
        target_specificity: Target specificity value in [0, 1].
            Default 0.95 for clinical screening (minimize false positives).
    """

    target_specificity: float = 0.95

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold achieving target specificity."""
        return threshold_for_specificity(y_true, y_prob, target_spec=self.target_specificity)

    @property
    def name(self) -> str:
        return f"fixed_spec_{self.target_specificity:.2f}"


@dataclass(frozen=True)
class FixedSensitivityThreshold:
    """Find threshold achieving target sensitivity.

    Selects the highest threshold (highest specificity) among those
    meeting the sensitivity target. Falls back to closest achievable
    sensitivity if target is unattainable.

    Args:
        target_sensitivity: Target sensitivity value in [0, 1].
            Default 0.95 for high recall scenarios.

    Note:
        This is equivalent to targeting (1 - false_negative_rate).
        High sensitivity = few missed cases but potentially more false alarms.
    """

    target_sensitivity: float = 0.95

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold achieving target sensitivity.

        Implementation: Since sensitivity = 1 - FNR, and FNR decreases as threshold
        decreases, we find the threshold where TPR >= target_sensitivity.
        """
        from sklearn.metrics import roc_curve

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        # Filter out NaN/inf values
        valid_mask = np.isfinite(y_prob) & np.isfinite(y_true)
        if not np.any(valid_mask):
            return 0.5
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

        # Single-class guard
        if len(np.unique(y_true)) < 2:
            return float(np.min(y_prob) - 1e-12) if len(y_prob) > 0 else 0.5

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        # Find thresholds where sensitivity (TPR) >= target
        ok = tpr >= self.target_sensitivity
        if np.any(ok):
            # Want highest threshold (most specific) meeting sensitivity target
            # In ROC curve, thresholds typically decrease with index
            idx = int(np.argmax(ok))
            threshold = thresholds[idx]
        else:
            # Fall back to closest achievable sensitivity
            idx = int(np.argmin(np.abs(tpr - self.target_sensitivity)))
            threshold = thresholds[idx]
            logger.warning(
                f"Target sensitivity {self.target_sensitivity:.3f} unattainable. "
                f"Using closest achievable sensitivity {tpr[idx]:.3f}. "
                f"Threshold set to {threshold:.6f}."
            )

        if not np.isfinite(threshold):
            threshold = float(np.min(y_prob) - 1e-12)
        return float(threshold)

    @property
    def name(self) -> str:
        return f"fixed_sens_{self.target_sensitivity:.2f}"


@dataclass(frozen=True)
class FixedPPVThreshold:
    """Find threshold achieving target positive predictive value (precision).

    Selects the lowest threshold (most inclusive) among those meeting the
    PPV target. Falls back to max-F1 threshold if target is unattainable.

    Args:
        target_ppv: Target positive predictive value in [0, 1].
            Default 0.5 for moderate precision requirement.

    Note:
        PPV = precision = TP / (TP + FP)
        Higher threshold typically increases PPV but reduces recall.
    """

    target_ppv: float = 0.5

    def find_threshold(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> float:
        """Find threshold achieving target PPV."""
        return threshold_for_precision(y_true, y_prob, target_ppv=self.target_ppv)

    @property
    def name(self) -> str:
        return f"fixed_ppv_{self.target_ppv:.2f}"


def get_threshold_strategy(config: ThresholdConfig) -> ThresholdStrategy:
    """Factory function to create threshold strategy from configuration.

    Maps the config objective string to the appropriate strategy class,
    passing any required parameters from the config.

    Args:
        config: ThresholdConfig with objective and parameter settings

    Returns:
        Appropriate ThresholdStrategy implementation

    Raises:
        ValueError: If config.objective is not a recognized strategy

    Example:
        >>> from ced_ml.config.schema import ThresholdConfig
        >>> config = ThresholdConfig(objective="fixed_spec", fixed_spec=0.95)
        >>> strategy = get_threshold_strategy(config)
        >>> isinstance(strategy, FixedSpecificityThreshold)
        True
    """
    objective = config.objective.lower().strip()

    if objective == "max_f1":
        return MaxF1Threshold()
    elif objective == "max_fbeta":
        return MaxFBetaThreshold(beta=config.fbeta)
    elif objective == "youden":
        return YoudensJThreshold()
    elif objective == "fixed_spec":
        return FixedSpecificityThreshold(target_specificity=config.fixed_spec)
    elif objective == "fixed_ppv":
        return FixedPPVThreshold(target_ppv=config.fixed_ppv)
    else:
        logger.warning(
            f"Unknown threshold objective '{objective}'. Falling back to max_f1. "
            f"Valid objectives: max_f1, max_fbeta, youden, fixed_spec, fixed_ppv"
        )
        return MaxF1Threshold()


def get_threshold_strategy_from_params(
    objective: str,
    fbeta: float = 1.0,
    fixed_spec: float = 0.90,
    fixed_ppv: float = 0.5,
    fixed_sens: float = 0.95,
) -> ThresholdStrategy:
    """Create threshold strategy from explicit parameters.

    Alternative factory function when you have raw parameters instead of
    a ThresholdConfig object. Useful for testing or CLI interfaces.

    Args:
        objective: One of 'max_f1', 'max_fbeta', 'youden', 'fixed_spec',
            'fixed_ppv', 'fixed_sens'
        fbeta: Beta parameter for max_fbeta objective
        fixed_spec: Target specificity for fixed_spec objective
        fixed_ppv: Target PPV for fixed_ppv objective
        fixed_sens: Target sensitivity for fixed_sens objective

    Returns:
        Appropriate ThresholdStrategy implementation
    """
    objective = objective.lower().strip()

    if objective == "max_f1":
        return MaxF1Threshold()
    elif objective == "max_fbeta":
        return MaxFBetaThreshold(beta=fbeta)
    elif objective == "youden":
        return YoudensJThreshold()
    elif objective == "fixed_spec":
        return FixedSpecificityThreshold(target_specificity=fixed_spec)
    elif objective == "fixed_ppv":
        return FixedPPVThreshold(target_ppv=fixed_ppv)
    elif objective == "fixed_sens":
        return FixedSensitivityThreshold(target_sensitivity=fixed_sens)
    else:
        logger.warning(
            f"Unknown threshold objective '{objective}'. Falling back to max_f1. "
            f"Valid objectives: max_f1, max_fbeta, youden, fixed_spec, fixed_ppv, fixed_sens"
        )
        return MaxF1Threshold()
