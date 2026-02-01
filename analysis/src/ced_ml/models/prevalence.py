"""
Prevalence adjustment for risk models.

This module provides utilities for adjusting predicted probabilities to match
target population prevalence, critical for deployment where training prevalence
(with downsampling/enrichment) differs from real-world prevalence.

Key classes:
    PrevalenceAdjustedModel: sklearn-compatible wrapper that applies prevalence
        adjustment to predict_proba() outputs
"""

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ced_ml.utils.math_utils import EPSILON_PREVALENCE, inv_logit, logit

logger = logging.getLogger(__name__)


def _logit(p: np.ndarray) -> np.ndarray:
    """
    Compute logit (log-odds) from probabilities.

    Wrapper for shared logit function using prevalence epsilon.

    Args:
        p: Probabilities in [0, 1]

    Returns:
        Log-odds values
    """
    return logit(p, eps=EPSILON_PREVALENCE)


def _inv_logit(z: np.ndarray) -> np.ndarray:
    """
    Compute inverse logit (sigmoid) from log-odds.

    Wrapper for shared inv_logit function.

    Args:
        z: Log-odds values

    Returns:
        Probabilities in [0, 1]
    """
    return inv_logit(z)


def adjust_probabilities_for_prevalence(
    probs: np.ndarray,
    sample_prev: float,
    target_prev: float,
) -> np.ndarray:
    """
    Apply intercept shift so predicted probabilities reflect target prevalence.

    Uses the method:
        P(Y=1|X,prev_new) = sigmoid(logit(p) + logit(prev_new) - logit(prev_old))

    This is the theoretically correct adjustment when:
    - The model was trained on data with sample_prev prevalence
    - We want predictions calibrated to target_prev prevalence
    - The feature distributions P(X|Y) remain the same

    Reference:
        Saerens et al. (2002). Adjusting the outputs of a classifier to new a priori
        probabilities: a simple procedure. Neural Computation.

    Args:
        probs: Raw probabilities from classifier [0, 1]
        sample_prev: Observed prevalence in training sample (0, 1)
        target_prev: Target prevalence for deployment (0, 1)

    Returns:
        Adjusted probabilities reflecting target prevalence

    Examples:
        >>> # Model trained on 1:5 case:control (prevalence ~0.167)
        >>> # Target population has 0.0034 prevalence (0.34%)
        >>> probs = np.array([0.1, 0.3, 0.7, 0.9])
        >>> adjusted = adjust_probabilities_for_prevalence(probs, 0.167, 0.0034)
        >>> adjusted  # doctest: +SKIP
        array([0.002, 0.007, 0.056, 0.188])  # Much lower after adjustment
    """
    # Validate inputs
    if not np.isfinite(sample_prev) or not np.isfinite(target_prev):
        logger.warning(
            f"Prevalence adjustment skipped: non-finite prevalence values "
            f"(sample_prev={sample_prev}, target_prev={target_prev}). "
            "Returning raw probabilities."
        )
        return probs
    if not (0.0 < sample_prev < 1.0) or not (0.0 < target_prev < 1.0):
        logger.warning(
            f"Prevalence adjustment skipped: prevalence values at boundary "
            f"(sample_prev={sample_prev}, target_prev={target_prev}). "
            "Valid range is (0.0, 1.0) exclusive. Returning raw probabilities."
        )
        return probs

    # Compute prevalence shift on logit scale
    delta = np.log(target_prev / (1.0 - target_prev)) - np.log(sample_prev / (1.0 - sample_prev))

    # Apply shift: logit(p_adjusted) = logit(p_raw) + delta
    logits = _logit(probs)
    adjusted = _inv_logit(logits + delta)

    # Clip to avoid numerical instability
    return np.clip(adjusted, 1e-9, 1.0 - 1e-9)


class PrevalenceAdjustedModel(BaseEstimator):
    """
    Wraps a fitted classifier and applies prevalence adjustment to predict_proba().

    This ensures the serialized model artifact produces adjusted probabilities
    that reflect target deployment prevalence rather than training prevalence.

    Attributes:
        base_model: Fitted sklearn classifier with predict_proba() method
        sample_prevalence: Prevalence in training data (after downsampling/enrichment)
        target_prevalence: Target prevalence for deployment (real-world population)
        classes_: Class labels (inherited from base_model)

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.7, 0.3])
        >>> rf = RandomForestClassifier(n_estimators=10, random_state=42)
        >>> rf.fit(X, y)  # doctest: +SKIP
        >>>
        >>> # Wrap model with prevalence adjustment
        >>> # Training prevalence ~0.3, target population prevalence 0.01
        >>> wrapper = PrevalenceAdjustedModel(
        ...     base_model=rf,
        ...     sample_prevalence=0.3,
        ...     target_prevalence=0.01
        ... )
        >>> adjusted_probs = wrapper.predict_proba(X[:10])  # doctest: +SKIP
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        sample_prevalence: float,
        target_prevalence: float,
    ):
        """
        Initialize prevalence-adjusted model wrapper.

        Args:
            base_model: Fitted classifier with predict_proba() method
            sample_prevalence: Prevalence in training sample (0, 1)
            target_prevalence: Target prevalence for deployment (0, 1)
        """
        self.base_model = base_model
        self.sample_prevalence = float(sample_prevalence)
        self.target_prevalence = float(target_prevalence)
        self.classes_ = getattr(base_model, "classes_", None)

    def fit(self, X, y=None):  # noqa: ARG002
        """
        No-op fit method (base model is already fitted).

        This method is provided for sklearn pipeline compatibility but does nothing
        since the base model is already fitted when the wrapper is instantiated.

        Args:
            X: Feature matrix (ignored)
            y: Target labels (ignored)

        Returns:
            self
        """
        return self

    def _can_adjust(self) -> bool:
        """Check if prevalence adjustment is valid."""
        return (
            np.isfinite(self.sample_prevalence)
            and np.isfinite(self.target_prevalence)
            and 0.0 < self.sample_prevalence < 1.0
            and 0.0 < self.target_prevalence < 1.0
        )

    def _adjust_binary_probs(self, probs: np.ndarray) -> np.ndarray:
        """
        Adjust binary probability matrix (N, 2).

        Args:
            probs: Probability matrix with shape (n_samples, 2)

        Returns:
            Adjusted probability matrix with shape (n_samples, 2)
        """
        if probs.ndim != 2 or probs.shape[1] != 2:
            return probs

        # Adjust positive class probabilities
        pos = probs[:, 1]
        adj_pos = adjust_probabilities_for_prevalence(
            pos, self.sample_prevalence, self.target_prevalence
        )
        adj_pos = np.clip(adj_pos, 1e-9, 1.0 - 1e-9)

        # Reconstruct probability matrix
        adj = np.column_stack([1.0 - adj_pos, adj_pos])
        return adj

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with prevalence adjustment.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Adjusted probability matrix (n_samples, n_classes)
        """
        # Get base model predictions
        base_probs = self.base_model.predict_proba(X)
        base_probs = np.asarray(base_probs, dtype=float)

        # Skip adjustment if parameters invalid
        if not self._can_adjust():
            return base_probs

        # Handle 1D output (positive class only)
        if base_probs.ndim == 1:
            adj = adjust_probabilities_for_prevalence(
                base_probs, self.sample_prevalence, self.target_prevalence
            )
            return np.column_stack([1.0 - adj, adj])

        # Handle binary classification (most common case)
        if base_probs.shape[1] == 2:
            return self._adjust_binary_probs(base_probs)

        # Multi-class: return unadjusted (adjustment not well-defined)
        return base_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using adjusted probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)

        # Ensure 2D probability matrix
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])

        # Get class with highest probability
        idx = np.argmax(probs, axis=1)

        # Map to class labels if available
        if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
            classes = np.asarray(self.classes_)
            return classes[idx]

        return idx

    def get_base_model(self) -> BaseEstimator:
        """
        Access the underlying base model.

        Returns:
            Base classifier
        """
        return self.base_model

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to base model.

        This allows the wrapper to transparently expose base model attributes
        like feature_importances_, n_features_in_, etc.

        Args:
            name: Attribute name

        Returns:
            Attribute value from base model
        """
        return getattr(self.base_model, name)
