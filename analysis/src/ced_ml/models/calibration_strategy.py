"""
Calibration strategy pattern for model calibration.

This module provides:
- CalibrationStrategy protocol defining the interface for calibration strategies
- Concrete implementations: PerFoldCalibration, OOFPosthocCalibration,
  IsotonicCalibration, SigmoidCalibration (per-fold convenience), NoCalibration
- Factory function to get calibration strategy from config

The strategy pattern replaces direct config.calibration.* checks throughout
the codebase, providing a cleaner and more extensible interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV

if TYPE_CHECKING:
    from ced_ml.config import TrainingConfig
    from ced_ml.models.calibration import OOFCalibrator

logger = logging.getLogger(__name__)


@runtime_checkable
class CalibrationStrategy(Protocol):
    """Protocol for calibration strategies.

    Defines the interface that all calibration strategies must implement.
    This allows for clean separation of calibration logic from the training
    orchestration code.
    """

    def name(self) -> str:
        """Return the name of the calibration strategy.

        Returns:
            Strategy name (e.g., "per_fold", "oof_posthoc", "none")
        """
        ...

    def method(self) -> str | None:
        """Return the calibration method used.

        Returns:
            Calibration method or None if not applicable
        """
        ...

    def requires_oof_calibration(self) -> bool:
        """Check if this strategy requires OOF (out-of-fold) calibration.

        Returns:
            True if OOF calibration should be applied post-CV
        """
        ...

    def requires_per_fold_calibration(self) -> bool:
        """Check if this strategy requires per-fold calibration.

        Returns:
            True if calibration should be applied within each CV fold
        """
        ...

    def should_skip_for_model(self, model_name: str) -> bool:
        """Check if calibration should be skipped for a specific model.

        Args:
            model_name: Name of the model (e.g., "LR_EN", "LinSVM_cal")

        Returns:
            True if calibration should be skipped
        """
        ...

    def get_cv_folds(self) -> int:
        """Get the number of CV folds for per-fold calibration.

        Returns:
            Number of CV folds (only relevant for per_fold strategy)
        """
        ...


class NoCalibration:
    """No-op calibration strategy (returns probabilities unchanged).

    Used when calibration is disabled or for models that are already calibrated.
    """

    def name(self) -> str:
        """Return 'none' as the strategy name."""
        return "none"

    def method(self) -> str | None:
        """Return None since no method is used."""
        return None

    def requires_oof_calibration(self) -> bool:
        """No OOF calibration required."""
        return False

    def requires_per_fold_calibration(self) -> bool:
        """No per-fold calibration required."""
        return False

    def should_skip_for_model(self, model_name: str) -> bool:
        """Always skip calibration."""
        return True

    def get_cv_folds(self) -> int:
        """Return 0 since no CV is used."""
        return 0

    def __repr__(self) -> str:
        return "NoCalibration()"


class PerFoldCalibration:
    """Per-fold calibration using CalibratedClassifierCV.

    Applies calibration within each CV fold using sklearn's CalibratedClassifierCV.
    This is the default calibration strategy.
    """

    def __init__(self, method: str = "isotonic", cv: int = 5):
        """Initialize per-fold calibration strategy.

        Args:
            method: Calibration method ("isotonic" or "sigmoid")
            cv: Number of CV folds for calibration
        """
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got '{method}'")
        if cv < 2:
            raise ValueError(f"cv must be >= 2, got {cv}")
        self._method = method
        self._cv = cv

    def name(self) -> str:
        """Return 'per_fold' as the strategy name."""
        return "per_fold"

    def method(self) -> str:
        """Return the calibration method."""
        return self._method

    def requires_oof_calibration(self) -> bool:
        """No OOF calibration required."""
        return False

    def requires_per_fold_calibration(self) -> bool:
        """Per-fold calibration is required."""
        return True

    def should_skip_for_model(self, model_name: str) -> bool:
        """Skip for already-calibrated models (LinSVM_cal)."""
        from ced_ml.data.schema import ModelName

        return model_name == ModelName.LinSVM_cal

    def get_cv_folds(self) -> int:
        """Return the number of CV folds."""
        return self._cv

    def apply(
        self,
        estimator,
        model_name: str,
        X_train: NDArray,
        y_train: NDArray,
    ):
        """Apply per-fold calibration to an estimator.

        Args:
            estimator: Fitted sklearn estimator or pipeline
            model_name: Model name
            X_train: Training features
            y_train: Training labels

        Returns:
            Calibrated estimator or original if calibration should be skipped
        """
        from ced_ml.data.schema import ModelName

        # Skip for already-calibrated models
        if model_name == ModelName.LinSVM_cal:
            return estimator

        # Don't double-calibrate
        if isinstance(estimator, CalibratedClassifierCV):
            return estimator

        # Determine appropriate number of CV folds based on class sizes
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(np.min(class_counts))

        # Need at least 2 samples per fold for the minority class
        max_safe_cv = max(2, min_class_count)
        effective_cv = min(self._cv, max_safe_cv)

        # Wrap with calibration
        calibrated = CalibratedClassifierCV(
            estimator=estimator, method=self._method, cv=effective_cv
        )

        # Fit on training data
        calibrated.fit(X_train, y_train)
        return calibrated

    def __repr__(self) -> str:
        return f"PerFoldCalibration(method='{self._method}', cv={self._cv})"


class OOFPosthocCalibration:
    """Out-of-fold posthoc calibration strategy.

    Collects raw OOF predictions during CV, then fits a single calibrator
    post-hoc on the pooled OOF predictions. This avoids the optimistic bias
    introduced by per-fold CalibratedClassifierCV.
    """

    _VALID_METHODS = frozenset({"isotonic", "logistic_full", "logistic_intercept", "beta"})

    def __init__(self, method: str = "logistic_intercept"):
        """Initialize OOF posthoc calibration strategy.

        Args:
            method: Calibration method. One of "isotonic", "logistic_full",
                    "logistic_intercept", "beta".
        """
        if method not in self._VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(self._VALID_METHODS)}, got '{method}'")
        self._method = method

    def name(self) -> str:
        """Return 'oof_posthoc' as the strategy name."""
        return "oof_posthoc"

    def method(self) -> str:
        """Return the calibration method."""
        return self._method

    def requires_oof_calibration(self) -> bool:
        """OOF calibration is required."""
        return True

    def requires_per_fold_calibration(self) -> bool:
        """No per-fold calibration required."""
        return False

    def should_skip_for_model(self, model_name: str) -> bool:
        """Skip for already-calibrated models (LinSVM_cal)."""
        from ced_ml.data.schema import ModelName

        return model_name == ModelName.LinSVM_cal

    def get_cv_folds(self) -> int:
        """Return 0 since per-fold CV is not used."""
        return 0

    def create_calibrator(self) -> OOFCalibrator:
        """Create an OOFCalibrator with the configured method.

        Returns:
            Unfitted OOFCalibrator instance
        """
        from ced_ml.models.calibration import OOFCalibrator

        return OOFCalibrator(method=self._method)

    def __repr__(self) -> str:
        return f"OOFPosthocCalibration(method='{self._method}')"


class IsotonicCalibration(PerFoldCalibration):
    """Isotonic regression calibration (convenience class).

    A shorthand for PerFoldCalibration with method='isotonic'.
    """

    def __init__(self, cv: int = 5):
        """Initialize isotonic calibration.

        Args:
            cv: Number of CV folds for calibration
        """
        super().__init__(method="isotonic", cv=cv)

    def __repr__(self) -> str:
        return f"IsotonicCalibration(cv={self._cv})"


class SigmoidCalibration(PerFoldCalibration):
    """Platt scaling / sigmoid calibration (convenience class).

    A shorthand for PerFoldCalibration with method='sigmoid'.
    """

    def __init__(self, cv: int = 5):
        """Initialize sigmoid calibration.

        Args:
            cv: Number of CV folds for calibration
        """
        super().__init__(method="sigmoid", cv=cv)

    def __repr__(self) -> str:
        return f"SigmoidCalibration(cv={self._cv})"


def get_calibration_strategy(
    config: TrainingConfig,
    model_name: str | None = None,
) -> CalibrationStrategy:
    """Factory function to get calibration strategy from config.

    This function creates the appropriate CalibrationStrategy instance
    based on the configuration settings.

    Args:
        config: TrainingConfig with calibration settings
        model_name: Optional model name for per-model strategy overrides

    Returns:
        CalibrationStrategy instance

    Examples:
        >>> strategy = get_calibration_strategy(config)
        >>> if strategy.requires_oof_calibration():
        ...     # Handle OOF calibration
        >>> if strategy.requires_per_fold_calibration():
        ...     # Handle per-fold calibration
    """
    # Get effective strategy for this model
    if model_name:
        effective_strategy = config.calibration.get_strategy_for_model(model_name)
    else:
        # Use global strategy if no model specified
        if not config.calibration.enabled:
            effective_strategy = "none"
        else:
            effective_strategy = config.calibration.strategy

    # Create appropriate strategy instance
    if effective_strategy == "none":
        return NoCalibration()

    elif effective_strategy == "per_fold":
        # Map OOF-granularity method names to sklearn CalibratedClassifierCV terms.
        # sklearn only accepts "isotonic" or "sigmoid".
        _SKLEARN_METHOD_MAP = {
            "isotonic": "isotonic",
            "logistic_full": "sigmoid",
            "logistic_intercept": "sigmoid",
            "beta": "sigmoid",
        }
        sklearn_method = _SKLEARN_METHOD_MAP.get(config.calibration.method, "sigmoid")
        return PerFoldCalibration(
            method=sklearn_method,
            cv=config.calibration.cv,
        )

    elif effective_strategy == "oof_posthoc":
        return OOFPosthocCalibration(
            method=config.calibration.method,
        )

    else:
        logger.warning(
            f"Unknown calibration strategy '{effective_strategy}', " "falling back to NoCalibration"
        )
        return NoCalibration()


def get_strategy_display_name(strategy: CalibrationStrategy) -> str:
    """Get a human-readable display name for a calibration strategy.

    Args:
        strategy: CalibrationStrategy instance

    Returns:
        Display name like "isotonic (per_fold)" or "none"
    """
    strategy_name = strategy.name()
    method = strategy.method()

    if strategy_name == "none":
        return "none"

    if method:
        return f"{method} ({strategy_name})"

    return strategy_name


__all__ = [
    "CalibrationStrategy",
    "NoCalibration",
    "PerFoldCalibration",
    "OOFPosthocCalibration",
    "IsotonicCalibration",
    "SigmoidCalibration",
    "get_calibration_strategy",
    "get_strategy_display_name",
]
