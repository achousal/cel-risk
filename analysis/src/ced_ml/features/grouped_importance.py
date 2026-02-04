"""Correlation-aware grouped permutation feature importance.

This module implements permutation feature importance (PFI) that is robust to
correlated features by permuting feature clusters together. This addresses the
known issue where standard PFI can underestimate importance of correlated features.

Key features:
- Individual permutation importance (standard PFI)
- Grouped permutation importance (correlation-robust)
- Cluster-based feature grouping using existing correlation infrastructure
- Compatible with any sklearn-compatible estimator

Algorithm:
---------
For each feature f (or cluster C):
  1. For r in 1..n_repeats (default 30):
     - Permute f (or all features in C) in held-out X
     - Compute AUROC on permuted held-out
  2. importance[f] = mean(original_auroc - permuted_auroc)

The grouped version ensures that highly correlated features are permuted together,
preserving their correlation structure and providing more stable importance estimates.

References:
----------
- Breiman (2001). Random Forests. Machine Learning 45(1).
- Strobl et al. (2008). Conditional variable importance for random forests.
- Altmann et al. (2010). Permutation importance: a corrected feature importance measure.

Examples:
--------
>>> # Standard per-feature PFI
>>> importance_df = compute_permutation_importance(
...     estimator=fitted_model,
...     X=X_test,
...     y=y_test,
...     feature_names=X_test.columns.tolist(),
...     n_repeats=30,
...     random_state=42
... )

>>> # Grouped PFI (correlation-robust)
>>> individual_df, grouped_df = compute_oof_permutation_importance(
...     estimator=fitted_model,
...     X=X_test,
...     y=y_test,
...     feature_names=X_test.columns.tolist(),
...     corr_threshold=0.85,
...     n_repeats=30,
...     random_state=42
... )
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)
from ced_ml.metrics.discrimination import auroc

logger = logging.getLogger(__name__)


def build_feature_clusters(
    X: pd.DataFrame,
    feature_names: list[str],
    corr_threshold: float = 0.85,
    corr_method: str = "spearman",
) -> list[list[str]]:
    """Build feature clusters based on correlation threshold.

    Uses the existing correlation infrastructure from corr_prune to identify
    groups of highly correlated features that should be permuted together.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (typically held-out data)
    feature_names : List[str]
        List of feature names to cluster
    corr_threshold : float, default=0.85
        Absolute correlation threshold for grouping features together.
        Features with correlation >= threshold are placed in the same cluster.
    corr_method : {"pearson", "spearman"}, default="spearman"
        Correlation method to use

    Returns
    -------
    List[List[str]]
        List of feature clusters. Each cluster is a list of feature names.
        Singleton clusters (uncorrelated features) are also included.

    Notes
    -----
    - Uses DFS-based connected components to find correlation clusters
    - Handles missing values via median imputation (consistent with corr_prune)
    - Returns empty list if no valid features are found
    - Clusters are sorted internally for reproducibility

    Examples
    --------
    >>> X = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [1.1, 2.1, 3.1, 4.1],  # Highly correlated with A
    ...     'C': [10, 20, 15, 25]       # Uncorrelated
    ... })
    >>> clusters = build_feature_clusters(X, ['A', 'B', 'C'], corr_threshold=0.85)
    >>> len(clusters)  # Will have 2 clusters: {A, B} and {C}
    2
    """
    valid_features = [f for f in feature_names if f in X.columns]
    if len(valid_features) == 0:
        logger.warning("No valid features found in X; returning empty cluster list")
        return []

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(
        df=X,
        proteins=valid_features,
        method=corr_method,
    )

    if corr_matrix.empty:
        logger.warning("Correlation matrix is empty; returning singleton clusters")
        return [[f] for f in valid_features]

    # Build correlation graph and find connected components
    adjacency = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    clusters = find_connected_components(adjacency)

    logger.info(
        f"Built {len(clusters)} feature clusters from {len(valid_features)} features "
        f"(corr_threshold={corr_threshold:.2f}, corr_method={corr_method})"
    )

    # Log cluster size distribution
    cluster_sizes = [len(c) for c in clusters]
    n_singletons = sum(1 for size in cluster_sizes if size == 1)
    max_size = max(cluster_sizes) if cluster_sizes else 0
    logger.info(
        f"Cluster size distribution: {n_singletons} singletons, "
        f"max_size={max_size}, mean_size={np.mean(cluster_sizes):.1f}"
    )

    return clusters


def compute_permutation_importance(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute standard per-feature permutation importance.

    For each feature, permutes that feature's values and measures the drop in
    AUROC. Higher importance indicates the feature is more critical for model
    performance.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Fitted model with predict_proba method
    X : pd.DataFrame
        Feature matrix (held-out data)
    y : np.ndarray
        True binary labels (0/1)
    feature_names : List[str]
        List of feature names to compute importance for
    n_repeats : int, default=30
        Number of permutation repeats for variance estimation
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Importance statistics with columns:
        - feature: feature name
        - mean_importance: mean(original_auroc - permuted_auroc)
        - std_importance: standard deviation across repeats
        - n_repeats: number of permutation repeats
        - baseline_auroc: original model AUROC before permutation

    Notes
    -----
    - Uses AUROC as the performance metric
    - Higher importance = larger drop in AUROC when feature is permuted
    - Negative importance can occur if feature is noise or anti-correlated
    - Standard error = std_importance / sqrt(n_repeats)

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    >>> model = LogisticRegression().fit(X_df, y)
    >>> importance = compute_permutation_importance(
    ...     estimator=model,
    ...     X=X_df,
    ...     y=y,
    ...     feature_names=X_df.columns.tolist(),
    ...     n_repeats=10,
    ...     random_state=42
    ... )
    >>> importance.shape[0] == 5  # One row per feature
    True
    >>> 'mean_importance' in importance.columns
    True
    """
    y_clean = np.asarray(y).astype(int)
    valid_features = [f for f in feature_names if f in X.columns]

    if len(valid_features) == 0:
        logger.warning("No valid features found; returning empty importance DataFrame")
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_repeats",
                "baseline_auroc",
            ]
        )

    # Compute baseline performance
    y_pred_proba = estimator.predict_proba(X[valid_features])[:, 1]
    baseline_auroc = auroc(y_clean, y_pred_proba)
    logger.info(f"Baseline AUROC: {baseline_auroc:.4f}")

    # Initialize RNG
    rng = np.random.RandomState(random_state)

    # Compute importance for each feature
    rows = []
    for i, feature in enumerate(valid_features):
        importance_scores = []

        for _repeat in range(n_repeats):
            # Create permuted copy
            X_permuted = X[valid_features].copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].values)

            # Compute permuted performance
            y_pred_permuted = estimator.predict_proba(X_permuted)[:, 1]
            permuted_auroc = auroc(y_clean, y_pred_permuted)

            # Importance = drop in performance
            importance = baseline_auroc - permuted_auroc
            importance_scores.append(importance)

        # Aggregate across repeats
        mean_importance = float(np.mean(importance_scores))
        std_importance = float(np.std(importance_scores, ddof=1))

        rows.append(
            {
                "feature": feature,
                "mean_importance": mean_importance,
                "std_importance": std_importance,
                "n_repeats": n_repeats,
                "baseline_auroc": baseline_auroc,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info(f"Computed importance for {i + 1}/{len(valid_features)} features")

    result = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    logger.info(
        f"Computed permutation importance for {len(result)} features "
        f"(n_repeats={n_repeats}, baseline_auroc={baseline_auroc:.4f})"
    )

    return result


def compute_grouped_permutation_importance(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    clusters: list[list[str]],
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute grouped permutation importance for feature clusters.

    Permutes all features in each cluster together, preserving their correlation
    structure. This provides more stable importance estimates for groups of
    correlated features.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Fitted model with predict_proba method
    X : pd.DataFrame
        Feature matrix (held-out data)
    y : np.ndarray
        True binary labels (0/1)
    clusters : List[List[str]]
        List of feature clusters (from build_feature_clusters)
    n_repeats : int, default=30
        Number of permutation repeats
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Cluster importance statistics with columns:
        - cluster_id: cluster index (0-based)
        - cluster_features: JSON list of feature names in cluster
        - cluster_size: number of features in cluster
        - mean_importance: mean(original_auroc - permuted_auroc)
        - std_importance: standard deviation across repeats
        - n_repeats: number of permutation repeats
        - baseline_auroc: original model AUROC before permutation

    Notes
    -----
    - All features in a cluster are permuted together (same permutation order)
    - This preserves within-cluster correlation structure
    - Cluster importance reflects the joint contribution of all cluster members
    - For singleton clusters, this is equivalent to individual feature importance

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    >>> X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
    >>> model = LogisticRegression().fit(X_df, y)
    >>> clusters = [['feat_0', 'feat_1'], ['feat_2'], ['feat_3', 'feat_4']]
    >>> importance = compute_grouped_permutation_importance(
    ...     estimator=model,
    ...     X=X_df,
    ...     y=y,
    ...     clusters=clusters,
    ...     n_repeats=10,
    ...     random_state=42
    ... )
    >>> importance.shape[0] == 3  # One row per cluster
    True
    """
    y_clean = np.asarray(y).astype(int)

    # Validate clusters and get all unique features
    all_features_set = set()
    for cluster in clusters:
        all_features_set.update(cluster)

    # Preserve feature order from X.columns to avoid sklearn feature name errors
    valid_features = [f for f in X.columns if f in all_features_set]

    if len(valid_features) == 0:
        logger.warning("No valid features in clusters; returning empty importance DataFrame")
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_features",
                "cluster_size",
                "mean_importance",
                "std_importance",
                "n_repeats",
                "baseline_auroc",
            ]
        )

    # Compute baseline performance
    y_pred_proba = estimator.predict_proba(X[valid_features])[:, 1]
    baseline_auroc = auroc(y_clean, y_pred_proba)
    logger.info(f"Baseline AUROC for grouped importance: {baseline_auroc:.4f}")

    # Initialize RNG
    rng = np.random.RandomState(random_state)

    # Compute importance for each cluster
    rows = []
    for cluster_id, cluster_features in enumerate(clusters):
        # Filter to valid features in this cluster
        valid_cluster = [f for f in cluster_features if f in X.columns]
        if len(valid_cluster) == 0:
            logger.warning(f"Cluster {cluster_id} has no valid features; skipping")
            continue

        importance_scores = []

        for _repeat in range(n_repeats):
            # Create permuted copy
            X_permuted = X[valid_features].copy()

            # Permute all features in the cluster together
            # Use the same permutation indices for all cluster members
            perm_indices = rng.permutation(len(X_permuted))
            for feature in valid_cluster:
                X_permuted[feature] = X_permuted[feature].values[perm_indices]

            # Compute permuted performance
            y_pred_permuted = estimator.predict_proba(X_permuted)[:, 1]
            permuted_auroc = auroc(y_clean, y_pred_permuted)

            # Importance = drop in performance
            importance = baseline_auroc - permuted_auroc
            importance_scores.append(importance)

        # Aggregate across repeats
        mean_importance = float(np.mean(importance_scores))
        std_importance = float(np.std(importance_scores, ddof=1))

        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_features": json.dumps(sorted(valid_cluster)),
                "cluster_size": len(valid_cluster),
                "mean_importance": mean_importance,
                "std_importance": std_importance,
                "n_repeats": n_repeats,
                "baseline_auroc": baseline_auroc,
            }
        )

        if (cluster_id + 1) % 50 == 0:
            logger.info(
                f"Computed grouped importance for {cluster_id + 1}/{len(clusters)} clusters"
            )

    result = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    logger.info(
        f"Computed grouped permutation importance for {len(result)} clusters "
        f"(n_repeats={n_repeats}, baseline_auroc={baseline_auroc:.4f})"
    )

    return result
