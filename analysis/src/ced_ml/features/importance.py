"""Feature importance computation for OOF predictions.

This module provides correlation-robust feature importance computation on held-out data:
- Linear models (LR_EN, LR_L1, LinSVM_cal): Standardized absolute coefficients
- Tree models (RF, XGBoost): Built-in feature importances (Gini/gain)
- Grouped importance: Cluster-aware importance aggregation (optional)

All importance values are extracted from fitted models and aggregated across outer CV folds
to produce robust, unbiased importance estimates. This approach avoids data leakage by
using only OOF data for importance computation.

Key functions:
    extract_linear_importance: Extract standardized |coef| from linear models
    extract_tree_importance: Extract feature_importances_ from tree models
    extract_importance_from_model: Unified dispatcher based on model_name (supports grouped mode)
    aggregate_fold_importances: Aggregate importance across CV folds
    cluster_features_by_correlation: Group features by correlation threshold

Design notes:
    - Handles sklearn Pipeline wrappers (preprocessing, feature selection)
    - Handles CalibratedClassifierCV wrappers (averages across calibration folds)
    - Supports grouped/cluster-aware importance (permutation-based for trees,
      aggregation for linear)
    - Returns empty DataFrame on errors rather than raising exceptions
    - Uses logging for diagnostics
"""

import json
import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from ..data.schema import ModelName
from ..metrics.discrimination import auroc
from .corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)

logger = logging.getLogger(__name__)

__all__ = [
    "extract_linear_importance",
    "extract_tree_importance",
    "extract_importance_from_model",
    "aggregate_fold_importances",
    "cluster_features_by_correlation",
]


def _get_final_feature_names(pipeline: Pipeline) -> np.ndarray | None:
    """Extract feature names from pipeline after all transformations.

    Handles sklearn pipelines with preprocessing and feature selection steps.
    Supports both ProteinOnlySelector wrappers and bare selectors.

    Args:
        pipeline: Fitted sklearn Pipeline

    Returns:
        Array of feature names after all transformations, or None if unavailable
    """
    if not isinstance(pipeline, Pipeline):
        logger.warning("Expected Pipeline object, got %s", type(pipeline).__name__)
        return None

    # Start with preprocessor feature names
    if "pre" in pipeline.named_steps:
        pre = pipeline.named_steps["pre"]
        if hasattr(pre, "get_feature_names_out"):
            feature_names = pre.get_feature_names_out()
        else:
            logger.warning("Preprocessor does not have get_feature_names_out method")
            return None
    else:
        logger.warning("Pipeline does not have 'pre' step")
        return None

    # Apply K-best selector mask if present
    if "sel" in pipeline.named_steps:
        sel = pipeline.named_steps["sel"]
        # ProteinOnlySelector exposes get_feature_names_out directly
        if hasattr(sel, "selected_proteins_"):
            feature_names = sel.get_feature_names_out()
        elif hasattr(sel, "get_support"):
            support = sel.get_support()
            feature_names = feature_names[support]
        else:
            logger.warning("Selector 'sel' does not have get_support method")

    # Apply model-specific selector mask if present
    if "model_sel" in pipeline.named_steps:
        model_sel = pipeline.named_steps["model_sel"]
        if hasattr(model_sel, "selected_proteins_"):
            feature_names = model_sel.get_feature_names_out()
        elif hasattr(model_sel, "get_support"):
            support = model_sel.get_support()
            feature_names = feature_names[support]
        else:
            logger.warning("Selector 'model_sel' does not have get_support method")

    return feature_names


def extract_linear_importance(
    estimator,
    feature_names: np.ndarray | list[str],
) -> pd.DataFrame:
    """Extract standardized absolute coefficients from linear models.

    This function handles:
    - Standard linear models (LogisticRegression, LinearSVC)
    - CalibratedClassifierCV wrappers (averages across calibration folds)
    - Pipeline wrappers (extracts final classifier step)

    Args:
        estimator: Fitted sklearn estimator (may be wrapped in Pipeline or CalibratedClassifierCV)
        feature_names: Feature names corresponding to coefficients

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - importance: float, standardized |coef|
            - importance_type: str, always "abs_coef"

        Empty DataFrame if coefficients cannot be extracted.

    Notes:
        Standardization is performed by dividing |coef| by the sum of all |coef|,
        producing importance values that sum to 1.0. This makes importance
        values comparable across different models and folds.
    """
    # Handle Pipeline wrapper
    if isinstance(estimator, Pipeline):
        if "clf" not in estimator.named_steps:
            logger.warning("Pipeline does not have 'clf' step; cannot extract coefficients")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    # Handle CalibratedClassifierCV wrapper (e.g., LinSVM_cal)
    if isinstance(clf, CalibratedClassifierCV):
        if not hasattr(clf, "calibrated_classifiers_"):
            logger.warning("CalibratedClassifierCV is not fitted; cannot extract coefficients")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

        # Average coefficients across calibration folds
        coefs_list = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is None:
                est = getattr(cc, "base_estimator", None)  # Older sklearn versions
            if est and hasattr(est, "coef_"):
                coefs_list.append(est.coef_.ravel())

        if not coefs_list:
            logger.warning(
                "No coefficients found in CalibratedClassifierCV calibrated_classifiers_"
            )
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

        coefs = np.mean(np.vstack(coefs_list), axis=0)

    elif hasattr(clf, "coef_"):
        # Standard linear model
        coefs = clf.coef_.ravel()
    else:
        logger.warning(
            "Estimator %s does not have coef_ attribute; cannot extract coefficients",
            type(clf).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Validate dimensions
    feature_names = np.asarray(feature_names)
    if len(feature_names) != len(coefs):
        logger.warning(
            "Feature names length (%d) != coef length (%d); cannot extract importance",
            len(feature_names),
            len(coefs),
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Compute standardized |coef|
    abs_coefs = np.abs(coefs)
    coef_sum = np.sum(abs_coefs)
    if coef_sum == 0:
        logger.warning("All coefficients are zero; cannot standardize")
        standardized = abs_coefs
    else:
        standardized = abs_coefs / coef_sum

    # Build DataFrame
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": standardized,
            "importance_type": "abs_coef",
        }
    )


def extract_tree_importance(
    estimator,
    feature_names: np.ndarray | list[str],
) -> pd.DataFrame:
    """Extract feature importances from tree-based models.

    This function handles:
    - RandomForestClassifier (Gini importance)
    - XGBClassifier (gain importance)
    - Pipeline wrappers (extracts final classifier step)

    Args:
        estimator: Fitted sklearn estimator (may be wrapped in Pipeline)
        feature_names: Feature names corresponding to importances

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - importance: float, normalized importance (sums to 1.0)
            - importance_type: str, "gini" for RF, "gain" for XGBoost

        Empty DataFrame if importances cannot be extracted.

    Notes:
        Tree importances are already normalized by sklearn/XGBoost to sum to 1.0.
        For RF, this is mean decrease in impurity (Gini importance).
        For XGBoost, this is total gain from splits on each feature.
    """
    # Handle Pipeline wrapper
    if isinstance(estimator, Pipeline):
        if "clf" not in estimator.named_steps:
            logger.warning("Pipeline does not have 'clf' step; cannot extract importances")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    # Extract feature_importances_
    if not hasattr(clf, "feature_importances_"):
        logger.warning(
            "Estimator %s does not have feature_importances_ attribute; cannot extract importance",
            type(clf).__name__,
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    importances = clf.feature_importances_

    # Validate dimensions
    feature_names = np.asarray(feature_names)
    if len(feature_names) != len(importances):
        logger.warning(
            "Feature names length (%d) != importance length (%d); cannot extract importance",
            len(feature_names),
            len(importances),
        )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Determine importance type based on estimator class
    clf_type = type(clf).__name__
    if "RandomForest" in clf_type:
        importance_type = "gini"
    elif "XGB" in clf_type:
        importance_type = "gain"
    else:
        importance_type = "tree"  # Generic fallback

    # Build DataFrame
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "importance_type": importance_type,
        }
    )


def cluster_features_by_correlation(
    X: pd.DataFrame,
    feature_names: list[str],
    corr_threshold: float = 0.85,
    corr_method: str = "spearman",
) -> list[list[str]]:
    """Cluster features by correlation threshold.

    Features with |correlation| >= corr_threshold are grouped together using
    connected components in a correlation graph. Uses hierarchical clustering
    infrastructure from corr_prune module.

    Args:
        X: Feature matrix (used to compute correlations).
        feature_names: Feature names to cluster.
        corr_threshold: Correlation threshold for clustering (default 0.85).
                       Features with |corr| >= threshold are grouped.
        corr_method: Correlation method ("pearson" or "spearman", default "spearman").

    Returns:
        List of feature clusters (each cluster is a list of feature names).
        Singleton clusters (uncorrelated features) are also included.

    Notes:
        - Uses DFS-based connected components to find correlation clusters
        - Handles missing values via median imputation (consistent with corr_prune)
        - Returns empty list if no valid features are found
        - Clusters are sorted internally for reproducibility

    Examples:
        >>> X = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [1.1, 2.1, 3.1, 4.1],  # Highly correlated with A
        ...     'C': [10, 20, 15, 25]       # Uncorrelated
        ... })
        >>> clusters = cluster_features_by_correlation(X, ['A', 'B', 'C'], corr_threshold=0.85)
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


def _compute_grouped_permutation_importance(
    estimator: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_clusters: list[list[str]],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute grouped permutation importance for tree models.

    For each cluster, permute ALL features in the cluster together,
    then measure AUROC drop on validation data.

    Args:
        estimator: Fitted sklearn Pipeline with predict_proba method.
        X_val: Validation feature matrix.
        y_val: Validation binary labels (0/1).
        feature_clusters: List of feature clusters (from cluster_features_by_correlation).
        n_repeats: Number of permutation repeats for variance estimation (default 5).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            - cluster_id: int, cluster identifier (0-based)
            - cluster_features: str, JSON list of feature names
            - cluster_size: int, number of features in cluster
            - mean_importance: float, mean AUROC drop across repeats
            - std_importance: float, std dev of AUROC drop across repeats
            - n_repeats: int, number of permutation repeats
            - baseline_auroc: float, original model AUROC before permutation

    Notes:
        - All features in a cluster are permuted together (same permutation order)
        - This preserves within-cluster correlation structure
        - Cluster importance reflects joint contribution of all cluster members
        - For singleton clusters, this is equivalent to individual feature importance
    """
    y_clean = np.asarray(y_val).astype(int)

    # Validate clusters and get all unique features
    all_features_set = set()
    for cluster in feature_clusters:
        all_features_set.update(cluster)

    # Preserve feature order from X_val.columns to avoid sklearn feature name errors
    valid_features = [f for f in X_val.columns if f in all_features_set]

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
    # Pass full X_val to Pipeline so pre/sel steps can transform correctly
    y_pred_proba = estimator.predict_proba(X_val)[:, 1]
    baseline_auroc = auroc(y_clean, y_pred_proba)
    logger.info(f"Baseline AUROC for grouped permutation importance: {baseline_auroc:.4f}")

    # Initialize RNG
    rng = np.random.RandomState(random_state)

    # Compute importance for each cluster
    rows = []
    for cluster_id, cluster_features in enumerate(feature_clusters):
        # Filter to valid features in this cluster
        valid_cluster = [f for f in cluster_features if f in X_val.columns]
        if len(valid_cluster) == 0:
            logger.warning(f"Cluster {cluster_id} has no valid features; skipping")
            continue

        importance_scores = []

        for _repeat in range(n_repeats):
            # Create permuted copy (full DataFrame so Pipeline pre/sel steps work)
            X_permuted = X_val.copy()

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
                f"Computed grouped importance for {cluster_id + 1}/{len(feature_clusters)} clusters"
            )

    result = pd.DataFrame(rows).sort_values("mean_importance", ascending=False, ignore_index=True)
    logger.info(
        f"Computed grouped permutation importance for {len(result)} clusters "
        f"(n_repeats={n_repeats}, baseline_auroc={baseline_auroc:.4f})"
    )

    return result


def extract_importance_from_model(
    estimator,
    model_name: Literal["LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost"],
    feature_names: np.ndarray | list[str] | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    grouped: bool = False,
    corr_threshold: float = 0.85,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract feature importance from fitted model (dispatcher function).

    This function dispatches to the appropriate extraction method based on model_name:
    - Linear models (LR_EN, LR_L1, LinSVM_cal) -> extract_linear_importance
    - Tree models (RF, XGBoost) -> extract_tree_importance

    If estimator is a Pipeline, feature names are extracted automatically from the
    pipeline's preprocessing and feature selection steps.

    If grouped=True and X_val/y_val provided:
        - For trees (RF, XGBoost): Compute grouped permutation importance
          by permuting all features in each cluster together
        - For linear models: Aggregate coefficient importance by cluster

    Args:
        estimator: Fitted sklearn estimator (may be Pipeline)
        model_name: Model identifier (one of: "LR_EN", "LR_L1", "LinSVM_cal", "RF", "XGBoost")
        feature_names: Optional feature names. If None and estimator is Pipeline,
                      feature names are extracted from pipeline steps.
        X_val: Validation feature matrix (required for grouped mode with trees).
        y_val: Validation binary labels (required for grouped mode with trees).
        grouped: Enable cluster-aware importance mode (default False).
        corr_threshold: Correlation threshold for feature clustering (default 0.85).
        n_repeats: Number of permutation repeats for grouped tree importance (default 5).
        random_state: Random seed for permutation (default 42).

    Returns:
        DataFrame with columns (standard mode):
            - feature: str, feature name
            - importance: float, importance value
            - importance_type: str, "abs_coef", "gini", or "gain"

        DataFrame with additional columns (grouped mode):
            - cluster_id: int, cluster identifier (0-based)
            - cluster_features: str, JSON list of feature names in cluster
            - cluster_size: int, number of features in cluster
            - mean_importance: float, aggregated or permutation-based importance
            - std_importance: float, std dev (only for permutation-based)
            - n_repeats: int, number of repeats (only for permutation-based)
            - baseline_auroc: float, baseline AUROC (only for permutation-based)

        Empty DataFrame if importance cannot be extracted.

    Raises:
        ValueError: If model_name is unknown or if grouped=True but X_val/y_val missing for trees

    Examples:
        >>> # Standard extraction from Pipeline
        >>> df = extract_importance_from_model(fitted_pipeline, "LR_EN")
        >>>
        >>> # Grouped importance for tree model
        >>> df = extract_importance_from_model(
        ...     fitted_pipeline, "RF",
        ...     X_val=X_val, y_val=y_val,
        ...     grouped=True, corr_threshold=0.85
        ... )
    """
    # Extract feature names from Pipeline if needed
    if feature_names is None:
        if isinstance(estimator, Pipeline):
            feature_names = _get_final_feature_names(estimator)
            if feature_names is None:
                logger.warning("Could not extract feature names from Pipeline")
                return pd.DataFrame(columns=["feature", "importance", "importance_type"])
        else:
            logger.warning("feature_names is required for non-Pipeline estimators")
            return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    # Validate grouped mode requirements
    model_name_clean = str(model_name).strip()
    is_tree_model = model_name_clean in (ModelName.RF, ModelName.XGBoost)
    is_linear_model = model_name_clean in (ModelName.LR_EN, ModelName.LR_L1, ModelName.LinSVM_cal)

    if grouped and is_tree_model and (X_val is None or y_val is None):
        raise ValueError(
            "grouped=True for tree models requires X_val and y_val for permutation importance"
        )

    # Handle grouped mode
    if grouped:
        logger.info(f"Computing grouped importance (corr_threshold={corr_threshold})")

        # Build feature clusters
        if X_val is not None:
            logger.warning(
                "Grouped importance is clustering features using validation data. "
                "This may introduce information from the validation set into feature grouping. "
                "For stricter data hygiene, consider clustering on training data instead."
            )
            clusters = cluster_features_by_correlation(
                X=X_val,
                feature_names=list(feature_names),
                corr_threshold=corr_threshold,
                corr_method="spearman",
            )
        else:
            # For linear models without X_val, use singleton clusters
            logger.warning("No X_val provided for clustering; using singleton clusters")
            clusters = [[f] for f in feature_names]

        # Compute grouped importance based on model type
        if is_tree_model:
            # Use grouped permutation importance for trees
            return _compute_grouped_permutation_importance(
                estimator=estimator,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        elif is_linear_model:
            # For linear models, aggregate coefficient importance by cluster
            individual_importance = extract_linear_importance(estimator, feature_names)

            if individual_importance.empty:
                return pd.DataFrame(
                    columns=[
                        "cluster_id",
                        "cluster_features",
                        "cluster_size",
                        "mean_importance",
                        "importance_type",
                    ]
                )

            # Aggregate by cluster
            rows = []
            for cluster_id, cluster_features in enumerate(clusters):
                cluster_df = individual_importance[
                    individual_importance["feature"].isin(cluster_features)
                ]
                if cluster_df.empty:
                    continue

                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster_features": json.dumps(sorted(cluster_features)),
                        "cluster_size": len(cluster_features),
                        "mean_importance": float(cluster_df["importance"].sum()),
                        "importance_type": cluster_df["importance_type"].iloc[0],
                    }
                )

            result = pd.DataFrame(rows).sort_values(
                "mean_importance", ascending=False, ignore_index=True
            )
            logger.info(
                f"Aggregated linear importance into {len(result)} clusters "
                f"from {len(individual_importance)} features"
            )
            return result

    # Standard mode (not grouped)
    if is_linear_model:
        return extract_linear_importance(estimator, feature_names)
    elif is_tree_model:
        return extract_tree_importance(estimator, feature_names)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Expected one of: LR_EN, LR_L1, LinSVM_cal, RF, XGBoost"
        )


def aggregate_fold_importances(fold_importances: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate feature importances across CV folds.

    Computes mean, std, and non-zero count for each feature across folds.
    This produces robust importance estimates that account for variability
    across different train/test splits.

    Args:
        fold_importances: List of importance DataFrames (one per fold)
                         Each DataFrame must have a feature column plus either
                         importance (legacy) or mean_importance (aggregated format).

    Returns:
        DataFrame with columns:
            - feature: str, feature name
            - mean_importance: float, mean importance across folds
            - std_importance: float, standard deviation across folds
            - n_folds_nonzero: int, number of folds where importance > 0
            - importance_type: str, importance type (taken from first fold)

        Sorted by mean_importance descending.
        Empty DataFrame if input list is empty.

    Notes:
        - Features present in some folds but not others are assigned importance=0 for missing folds
        - Features with higher n_folds_nonzero are more consistently selected
        - Standard deviation indicates importance stability across folds

    Examples:
        >>> fold_dfs = [
        ...     extract_importance_from_model(pipeline1, "LR_EN"),
        ...     extract_importance_from_model(pipeline2, "LR_EN"),
        ... ]
        >>> agg_df = aggregate_fold_importances(fold_dfs)
        >>> print(agg_df.head())
    """
    if not fold_importances:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_folds_nonzero",
                "importance_type",
            ]
        )

    # Filter out empty DataFrames
    valid_dfs = [df for df in fold_importances if not df.empty]
    if not valid_dfs:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_folds_nonzero",
                "importance_type",
            ]
        )

    # Expand grouped (cluster-level) DataFrames to feature-level if needed.
    # Grouped DataFrames have 'cluster_features' (JSON list) and 'mean_importance'
    # but no 'feature' column. Expand each cluster row into one row per feature.
    # F3 fix: Distribute cluster importance evenly across member features to avoid
    # artificial inflation. Each feature gets importance / cluster_size.
    expanded_dfs = []
    for df in valid_dfs:
        if "cluster_features" in df.columns and "feature" not in df.columns:
            rows = []
            imp_col = "mean_importance" if "mean_importance" in df.columns else "importance"
            imp_type = (
                df["importance_type"].iloc[0]
                if "importance_type" in df.columns
                else "grouped_permutation"
            )
            for _, row in df.iterrows():
                cluster_feats = json.loads(row["cluster_features"])
                cluster_size = len(cluster_feats)
                # Distribute cluster importance across features to preserve total importance
                distributed_importance = (
                    float(row[imp_col]) / cluster_size if cluster_size > 0 else 0.0
                )
                for feat in cluster_feats:
                    rows.append(
                        {
                            "feature": feat,
                            "importance": distributed_importance,
                            "importance_type": imp_type,
                        }
                    )
            expanded_dfs.append(pd.DataFrame(rows) if rows else df)
        else:
            expanded_dfs.append(df)
    normalized_dfs = []
    for df in expanded_dfs:
        if df.empty or "feature" not in df.columns:
            continue
        if "importance" not in df.columns:
            if "mean_importance" in df.columns:
                df = df.copy()
                df["importance"] = df["mean_importance"]
            else:
                # Skip unrecognized schema instead of crashing aggregation.
                continue
        normalized_dfs.append(df)

    valid_dfs = normalized_dfs

    if not valid_dfs:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_importance",
                "std_importance",
                "n_folds_nonzero",
                "importance_type",
            ]
        )

    # Collect all unique features
    all_features = set()
    for df in valid_dfs:
        all_features.update(df["feature"].values)

    # Determine importance type (should be consistent across folds)
    importance_type = (
        valid_dfs[0]["importance_type"].iloc[0]
        if "importance_type" in valid_dfs[0].columns
        else "unknown"
    )

    # Build importance matrix: rows=features, cols=folds
    importance_matrix = {}
    for feature in all_features:
        importance_matrix[feature] = []
        for df in valid_dfs:
            if feature in df["feature"].values:
                val = float(df[df["feature"] == feature]["importance"].iloc[0])
            else:
                val = 0.0
            importance_matrix[feature].append(val)

    # Compute statistics
    results = []
    for feature, values in importance_matrix.items():
        values_array = np.array(values)
        results.append(
            {
                "feature": feature,
                "mean_importance": float(np.mean(values_array)),
                "std_importance": (
                    float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
                ),
                "n_folds_nonzero": int(np.sum(values_array > 0)),
                "importance_type": importance_type,
            }
        )

    # Build DataFrame and sort
    result_df = pd.DataFrame(results).sort_values(
        "mean_importance", ascending=False, ignore_index=True
    )

    return result_df
