"""K-best feature selection using univariate statistical tests.

This module provides functions for selecting top-K features based on
univariate F-statistic (ANOVA F-value) between each feature and the target.

Key functions:
- select_kbest_features: Main interface for K-best selection
- compute_f_classif_scores: Compute F-statistics for each feature
- rank_features_by_score: Rank features by score (descending)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


def compute_f_classif_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute ANOVA F-value for each feature vs binary target.

    Uses sklearn.feature_selection.f_classif which computes the ANOVA
    F-statistic for each feature. Higher F-values indicate stronger
    univariate association with the target.

    Args:
        X: Feature matrix (n_samples, n_features), must be numeric
        y: Binary target vector (n_samples,)

    Returns:
        Array of F-statistics (n_features,)

    Raises:
        ValueError: If y has fewer than 2 classes

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 0, 1])
        >>> scores = compute_f_classif_scores(X, y)
        >>> scores.shape
        (2,)
    """
    y = np.asarray(y).astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError("Target y must have at least 2 classes")

    F_scores, _ = f_classif(X, y)
    return np.asarray(F_scores, dtype=float)


def rank_features_by_score(
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Rank features by score and return indices of top-K.

    Args:
        scores: Array of feature scores (higher is better)
        k: Number of top features to select

    Returns:
        Indices of top-K features (sorted by score descending)

    Example:
        >>> scores = np.array([0.5, 2.0, 1.0])
        >>> rank_features_by_score(scores, k=2)
        array([1, 2])
    """
    scores = np.asarray(scores, dtype=float)
    k_effective = max(1, min(len(scores), k))

    # Sort descending (highest scores first)
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices[:k_effective]


def build_kbest_pipeline_step(k: int) -> Any:
    """Build SelectKBest pipeline step for feature selection.

    Creates a sklearn SelectKBest transformer configured with F-test scoring.
    This isolates sklearn implementation details from CLI code.

    Args:
        k: Number of features to select

    Returns:
        Unfitted SelectKBest transformer

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> kbest = build_kbest_pipeline_step(k=100)
        >>> pipe = Pipeline([("sel", kbest), ("clf", classifier)])
    """
    return SelectKBest(score_func=f_classif, k=k)


class ScreeningTransformer:
    """Univariate screening transformer for feature pre-filtering.

    Wraps screen_proteins to work as an sklearn transformer.
    Screens features using Mann-Whitney or F-statistic, keeping top-N.
    """

    def __init__(
        self,
        method: str = "mannwhitney",
        top_n: int = 1000,
        protein_cols: list[str] | None = None,
        precomputed_features: list[str] | None = None,
    ):
        """Initialize screening transformer.

        Args:
            method: "mannwhitney" or "f_classif"
            top_n: Number of top features to keep
            protein_cols: List of protein column names (set during fit)
            precomputed_features: If provided, skip screening and use these
                features directly. Useful for learning curves where screening
                should be computed once on the full training set.
        """
        self.method = method
        self.top_n = top_n
        self.protein_cols = protein_cols or []
        self.precomputed_features = precomputed_features
        self.selected_features_ = None
        self.screening_stats_ = None

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """Fit screening transformer.

        Args:
            X: Feature matrix (DataFrame or array from ColumnTransformer)
            y: Binary target vector

        Returns:
            self
        """
        if y is None:
            raise ValueError("y must be provided for screening")

        import logging

        from ced_ml.features.screening import screen_proteins

        logger = logging.getLogger(__name__)

        # Store all column names for reconstruction in transform()
        if isinstance(X, pd.DataFrame):
            self.all_feature_names_ = X.columns.tolist()
        else:
            raise TypeError(
                "ScreeningTransformer expects DataFrame input. "
                "It should be placed before ColumnTransformer in the pipeline, "
                "not after. Current pipeline has ColumnTransformer -> ScreeningTransformer "
                "which is incompatible."
            )

        # Short-circuit: use precomputed features (e.g. for learning curves)
        if self.precomputed_features is not None:
            self.selected_features_ = list(self.precomputed_features)
            self.selected_proteins_ = list(self.precomputed_features)
            self.screening_stats_ = None
            return self

        # Determine protein columns if not set
        if not self.protein_cols:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.protein_cols = numeric_cols

        selected_prots, stats, was_cached = screen_proteins(
            X_train=X,
            y_train=y,
            protein_cols=self.protein_cols,
            method=self.method,
            top_n=self.top_n,
        )

        # Use DEBUG logging for cached results to reduce noise in repeated CV folds
        log_fn = logger.debug if was_cached else logger.info
        log_fn(
            f"Screening: {self.method} on {len(self.protein_cols)} proteins "
            f"(N={len(X)}, top_n={self.top_n}) -> {len(selected_prots)} selected"
            + (" [CACHED]" if was_cached else "")
        )

        self.selected_features_ = selected_prots
        self.selected_proteins_ = selected_prots
        self.screening_stats_ = stats

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform: keep only selected protein features + all non-protein columns.

        Args:
            X: Feature matrix (must be DataFrame)

        Returns:
            Subset of X with selected proteins + all metadata columns
        """
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ScreeningTransformer.transform() expects DataFrame input. "
                "Ensure ScreeningTransformer is placed before ColumnTransformer."
            )

        # Keep selected proteins + all non-protein columns (metadata)
        selected_proteins = [c for c in self.selected_features_ if c in X.columns]
        non_protein_cols = [c for c in X.columns if c not in self.protein_cols]
        cols_to_keep = selected_proteins + non_protein_cols

        return X[cols_to_keep]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        """Get output feature names (for sklearn compatibility)."""
        if self.selected_features_ is None:
            return np.array([])
        return np.array(self.selected_features_)
