"""Drop-column importance validation for clinical feature panels.

After feature selection (RFE, consensus, etc.) identifies a panel, validate which
features/clusters are essential by systematic drop-column refit.

Design:
- For each feature or cluster in final panel, remove it and refit the model
- Compare AUROC on held-out fold to original AUROC
- Delta AUROC represents feature/cluster importance
- Supports aggregation across CV folds for robust estimates

Typical workflow:
    1. Train model and select features -> final panel
    2. Run drop_column for each fold -> list of DropColumnResult
    3. Aggregate across folds -> DataFrame with mean/std delta_auroc

This helps clinicians understand which features drive predictions and supports
cost-benefit analysis for panels (e.g., "Removing feature X only costs 0.02 AUROC").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

__all__ = [
    "DropColumnResult",
    "compute_drop_column_importance",
    "aggregate_drop_column_results",
    "validate_panel_essentiality",
]


@dataclass
class DropColumnResult:
    """Result from dropping a single feature/cluster and refitting.

    Attributes:
        cluster_id: Unique identifier for this feature/cluster (0-indexed).
        cluster_features: List of feature names in this cluster/group.
        original_auroc: AUROC of model with all features (baseline).
        reduced_auroc: AUROC of model with cluster removed.
        delta_auroc: Importance measure = original_auroc - reduced_auroc.
            Positive values: feature/cluster is important.
            ~0: feature/cluster has little impact.
            Negative: removing features improved AUROC (noise or redundancy).
        n_folds: Number of CV folds this was evaluated over (for aggregation).
        model_name: Optional name of model used (for logging).
        fold_id: Optional fold index (for tracking which fold this result came from).
        error_msg: Optional error message if drop-column evaluation failed.
    """

    cluster_id: int
    cluster_features: list[str]
    original_auroc: float
    reduced_auroc: float
    delta_auroc: float
    n_folds: int = 1
    model_name: str = ""
    fold_id: int | None = None
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (JSON, CSV).

        Returns:
            Dictionary representation suitable for JSON/CSV export.
        """
        return {
            "cluster_id": self.cluster_id,
            "cluster_features": ",".join(self.cluster_features),
            "n_features_in_cluster": len(self.cluster_features),
            "original_auroc": round(self.original_auroc, 6),
            "reduced_auroc": round(self.reduced_auroc, 6),
            "delta_auroc": round(self.delta_auroc, 6),
            "n_folds": self.n_folds,
            "model_name": self.model_name,
            "fold_id": self.fold_id,
            "error_msg": self.error_msg,
        }


def compute_drop_column_importance(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_clusters: list[list[str]],
    random_state: int = 42,
) -> list[DropColumnResult]:
    """Compute drop-column importance by refitting model with each cluster removed.

    For each cluster in feature_clusters:
    1. Remove all features in the cluster from X_train and X_val
    2. Clone and refit the estimator on reduced training set
    3. Evaluate on validation set (AUROC)
    4. Compute delta_auroc = original_auroc - reduced_auroc

    Args:
        estimator: Fitted scikit-learn Pipeline or estimator with predict_proba.
        X_train: Training features (pd.DataFrame or np.ndarray).
        y_train: Training labels (0/1 binary).
        X_val: Validation features (same columns as X_train).
        y_val: Validation labels (0/1 binary).
        feature_clusters: List of feature clusters to drop.
            Each cluster is a list of feature names.
            Example: [['A', 'B'], ['C'], ['D', 'E', 'F']]
        random_state: Random state for consistent cloning/fitting.

    Returns:
        List of DropColumnResult objects, one per cluster.

    Raises:
        ValueError: If X_train/X_val have different columns or feature_clusters
                    contains unknown features.
        RuntimeError: If refitting or evaluation fails for a cluster.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.pipeline import Pipeline
        >>> estimator = Pipeline([('model', LogisticRegression())])
        >>> estimator.fit(X_train, y_train)
        >>> clusters = [['prot_A', 'prot_B'], ['prot_C']]
        >>> results = compute_drop_column_importance(
        ...     estimator, X_train, y_train, X_val, y_val,
        ...     clusters, random_state=42
        ... )
        >>> for res in results:
        ...     print(f"Cluster {res.cluster_id}: delta_auroc={res.delta_auroc:.4f}")
    """
    # Validate inputs
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_val = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train (n={len(X_train)}) and y_train (n={len(y_train)}) have different lengths"
        )
    if len(X_val) != len(y_val):
        raise ValueError(
            f"X_val (n={len(X_val)}) and y_val (n={len(y_val)}) have different lengths"
        )

    if X_train.shape[1] == 0:
        raise ValueError("X_train has no features")
    if X_val.shape[1] == 0:
        raise ValueError("X_val has no features")

    if set(X_train.columns) != set(X_val.columns):
        raise ValueError("X_train and X_val have different columns")

    # Flatten and validate clusters
    all_cluster_features = set()
    for cluster in feature_clusters:
        if not isinstance(cluster, list | tuple):
            raise ValueError(f"Each cluster must be a list or tuple, got {type(cluster)}")
        all_cluster_features.update(cluster)

    unknown_features = all_cluster_features - set(X_train.columns)
    if unknown_features:
        raise ValueError(f"Unknown features in clusters: {unknown_features}")

    # Compute baseline AUROC (all features)
    try:
        y_pred_proba_baseline = estimator.predict_proba(X_val)[:, 1]
        original_auroc = roc_auc_score(y_val, y_pred_proba_baseline)
        logger.debug(f"Baseline AUROC: {original_auroc:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute baseline AUROC: {e}")
        raise RuntimeError(f"Failed to compute baseline AUROC: {e}") from e

    results = []

    # For each cluster, drop and refit
    for cluster_id, cluster_features in enumerate(feature_clusters):
        result = _drop_and_evaluate_cluster(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cluster_id=cluster_id,
            cluster_features=cluster_features,
            original_auroc=original_auroc,
            random_state=random_state,
        )
        results.append(result)

    return results


def _drop_and_evaluate_cluster(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cluster_id: int,
    cluster_features: list[str],
    original_auroc: float,
    random_state: int,
) -> DropColumnResult:
    """Helper: drop a single cluster and evaluate AUROC.

    Args:
        estimator: Fitted estimator.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        cluster_id: Cluster identifier.
        cluster_features: List of feature names to drop.
        original_auroc: Baseline AUROC for comparison.
        random_state: Random state for consistent cloning.

    Returns:
        DropColumnResult with delta_auroc computed.
    """
    try:
        # Drop cluster from X_train and X_val
        features_to_keep = [f for f in X_train.columns if f not in cluster_features]

        if len(features_to_keep) == 0:
            logger.warning(f"Cluster {cluster_id}: dropping all features, skipping")
            return DropColumnResult(
                cluster_id=cluster_id,
                cluster_features=cluster_features,
                original_auroc=original_auroc,
                reduced_auroc=np.nan,
                delta_auroc=np.nan,
                error_msg="All features dropped from panel",
            )

        X_train_reduced = X_train[features_to_keep]
        X_val_reduced = X_val[features_to_keep]

        # Clone estimator and set random_state on base estimator (not Pipeline)
        estimator_clone = clone(estimator)
        # Try to set random_state on the final estimator in the pipeline
        if hasattr(estimator_clone, "steps") and len(estimator_clone.steps) > 0:
            final_estimator = estimator_clone.steps[-1][1]
            if hasattr(final_estimator, "random_state"):
                final_estimator.set_params(random_state=random_state)
        estimator_clone.fit(X_train_reduced, y_train)

        # Evaluate on validation set
        y_pred_proba_reduced = estimator_clone.predict_proba(X_val_reduced)[:, 1]
        reduced_auroc = roc_auc_score(y_val, y_pred_proba_reduced)

        delta_auroc = original_auroc - reduced_auroc

        logger.debug(
            f"Cluster {cluster_id} ({len(cluster_features)} features): "
            f"reduced_auroc={reduced_auroc:.6f}, delta_auroc={delta_auroc:.6f}"
        )

        return DropColumnResult(
            cluster_id=cluster_id,
            cluster_features=cluster_features,
            original_auroc=original_auroc,
            reduced_auroc=reduced_auroc,
            delta_auroc=delta_auroc,
        )

    except Exception as e:
        logger.error(f"Cluster {cluster_id}: Failed to refit and evaluate: {e}")
        return DropColumnResult(
            cluster_id=cluster_id,
            cluster_features=cluster_features,
            original_auroc=original_auroc,
            reduced_auroc=np.nan,
            delta_auroc=np.nan,
            error_msg=str(e),
        )


def aggregate_drop_column_results(
    results_per_fold: list[list[DropColumnResult]],
    agg_method: str = "mean",
) -> pd.DataFrame:
    """Aggregate drop-column results across multiple CV folds.

    For each cluster (identified by cluster_id), computes mean and std of
    delta_auroc across all folds.

    Args:
        results_per_fold: List of result lists, one per fold.
            Example: [[result_c1_fold0, result_c2_fold0, ...],
                      [result_c1_fold1, result_c2_fold1, ...], ...]
        agg_method: Aggregation method for delta_auroc ("mean", "median", "max").
            Default: "mean" (standard practice for cross-validation).

    Returns:
        DataFrame with columns:
            - cluster_id: Cluster identifier
            - cluster_features: Feature names (comma-separated)
            - n_features_in_cluster: Number of features
            - mean_delta_auroc: Average delta_auroc across folds
            - std_delta_auroc: Standard deviation (NaN if n_folds=1)
            - min_delta_auroc: Minimum delta_auroc
            - max_delta_auroc: Maximum delta_auroc
            - n_folds: Number of folds included
            - n_errors: Number of folds where evaluation failed

    Raises:
        ValueError: If results_per_fold is empty or has inconsistent cluster IDs.
    """
    if not results_per_fold or all(len(fold_results) == 0 for fold_results in results_per_fold):
        raise ValueError("results_per_fold is empty or contains no results")

    # Flatten and group by cluster_id
    all_results_flat = []
    for _fold_idx, fold_results in enumerate(results_per_fold):
        for result in fold_results:
            all_results_flat.append(result)

    if not all_results_flat:
        raise ValueError("No results found in results_per_fold")

    # Group by cluster_id
    grouped = {}
    for result in all_results_flat:
        cid = result.cluster_id
        if cid not in grouped:
            grouped[cid] = []
        grouped[cid].append(result)

    # Aggregate per cluster
    agg_rows = []
    for cluster_id in sorted(grouped.keys()):
        cluster_results = grouped[cluster_id]
        cluster_features = cluster_results[0].cluster_features
        n_features = len(cluster_features)

        # Collect valid delta_auroc values (skip NaN/errors)
        valid_deltas = [
            r.delta_auroc
            for r in cluster_results
            if r.delta_auroc is not None and not np.isnan(r.delta_auroc)
        ]
        n_errors = len(cluster_results) - len(valid_deltas)

        if len(valid_deltas) == 0:
            logger.warning(
                f"Cluster {cluster_id}: no valid delta_auroc values "
                f"across {len(cluster_results)} folds"
            )
            mean_delta = np.nan
            std_delta = np.nan
            min_delta = np.nan
            max_delta = np.nan
        elif agg_method == "mean":
            mean_delta = float(np.mean(valid_deltas))
            std_delta = float(np.std(valid_deltas, ddof=1)) if len(valid_deltas) > 1 else np.nan
            min_delta = float(np.min(valid_deltas))
            max_delta = float(np.max(valid_deltas))
        elif agg_method == "median":
            mean_delta = float(np.median(valid_deltas))
            std_delta = np.nan  # Not applicable for median
            min_delta = float(np.min(valid_deltas))
            max_delta = float(np.max(valid_deltas))
        elif agg_method == "max":
            mean_delta = float(np.max(valid_deltas))
            std_delta = np.nan
            min_delta = float(np.min(valid_deltas))
            max_delta = float(np.max(valid_deltas))
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")

        agg_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_features": ",".join(cluster_features),
                "n_features_in_cluster": n_features,
                "mean_delta_auroc": mean_delta,
                "std_delta_auroc": std_delta,
                "min_delta_auroc": min_delta,
                "max_delta_auroc": max_delta,
                "n_folds": len(cluster_results),
                "n_errors": n_errors,
            }
        )

    df = pd.DataFrame(agg_rows)

    # Sort by mean_delta_auroc descending (most important first)
    df = df.sort_values("mean_delta_auroc", ascending=False, na_position="last").reset_index(
        drop=True
    )

    logger.info(
        f"Aggregated drop-column results: {len(df)} clusters across {len(results_per_fold)} folds"
    )

    return df


def validate_panel_essentiality(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    panel_features: list[str],
    corr_threshold: float = 0.85,
    include_brier: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Validate panel robustness via grouped LOCO/refit.

    For each correlation cluster in the panel:
    1. Compute correlation matrix and cluster correlated features
    2. Drop each cluster and refit model
    3. Evaluate OOF AUROC (and optionally Brier)
    4. Compute delta_auroc, delta_brier, essentiality_rank

    Args:
        model: Fitted sklearn Pipeline with predict_proba.
        X_train: Training features (must contain panel_features).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        panel_features: List of feature names in the panel.
        corr_threshold: Correlation threshold for clustering (default: 0.85).
        include_brier: Whether to compute Brier score (default: False).
        random_state: Random state for model cloning.

    Returns:
        DataFrame with columns:
            - cluster_id: int
            - features: str (comma-separated feature names)
            - n_features: int
            - delta_auroc: float (importance measure)
            - delta_brier: float (if include_brier=True, else NaN)
            - essentiality_rank: int (1 = most essential)

        Sorted by delta_auroc descending (most essential first).

    Raises:
        ValueError: If panel_features are not in X_train or X_val.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.pipeline import Pipeline
        >>> model = Pipeline([('model', LogisticRegression())])
        >>> model.fit(X_train, y_train)
        >>> panel_features = ['prot_A_resid', 'prot_B_resid', 'prot_C_resid']
        >>> results = validate_panel_essentiality(
        ...     model, X_train, y_train, X_val, y_val,
        ...     panel_features, corr_threshold=0.85, random_state=42
        ... )
        >>> print(results[['cluster_id', 'n_features', 'delta_auroc', 'essentiality_rank']])
    """
    # Validate inputs
    X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    X_val = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    missing_train = set(panel_features) - set(X_train.columns)
    missing_val = set(panel_features) - set(X_val.columns)
    if missing_train:
        raise ValueError(f"Panel features not found in X_train: {missing_train}")
    if missing_val:
        raise ValueError(f"Panel features not found in X_val: {missing_val}")

    if len(panel_features) == 0:
        raise ValueError("panel_features is empty")

    logger.info(
        f"Validating panel essentiality: {len(panel_features)} features, "
        f"corr_threshold={corr_threshold}, include_brier={include_brier}"
    )

    # Step 1: Build correlation clusters
    feature_clusters = _cluster_panel_features(
        X_train=X_train,
        panel_features=panel_features,
        corr_threshold=corr_threshold,
    )

    logger.info(f"Identified {len(feature_clusters)} correlation clusters")
    for i, cluster in enumerate(feature_clusters):
        logger.debug(f"Cluster {i}: {len(cluster)} features - {cluster[:3]}...")

    # Step 2: Compute drop-column importance
    drop_results = compute_drop_column_importance(
        estimator=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_clusters=feature_clusters,
        random_state=random_state,
    )

    # Step 3: Build results DataFrame
    rows = []
    for result in drop_results:
        row = {
            "cluster_id": result.cluster_id,
            "features": ",".join(result.cluster_features),
            "n_features": len(result.cluster_features),
            "delta_auroc": result.delta_auroc,
            "delta_brier": np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Step 4: Compute Brier scores if requested
    if include_brier:
        logger.debug("Computing Brier scores for original and reduced models")
        df["delta_brier"] = _compute_brier_deltas(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=feature_clusters,
            random_state=random_state,
        )

    # Step 5: Add essentiality rank (1 = most essential)
    # Sort by delta_auroc descending, then assign rank
    df = df.sort_values("delta_auroc", ascending=False, na_position="last").reset_index(drop=True)
    df["essentiality_rank"] = (
        df["delta_auroc"].rank(method="dense", ascending=False, na_option="bottom").astype(int)
    )

    logger.info(
        f"Panel essentiality validation complete: {len(df)} clusters, "
        f"top cluster delta_auroc={df.iloc[0]['delta_auroc']:.4f}"
    )

    return df


def _cluster_panel_features(
    X_train: pd.DataFrame,
    panel_features: list[str],
    corr_threshold: float = 0.85,
) -> list[list[str]]:
    """Cluster panel features by Spearman correlation using hierarchical clustering.

    Args:
        X_train: Training features (must contain panel_features).
        panel_features: List of feature names to cluster.
        corr_threshold: Correlation threshold for clustering.

    Returns:
        List of feature clusters (list of lists of feature names).
    """
    if len(panel_features) == 1:
        logger.debug("Single feature panel, returning one cluster")
        return [panel_features]

    # Compute Spearman correlation matrix
    X_panel = X_train[panel_features]
    corr_matrix = X_panel.corr(method="spearman").abs()

    # Convert to distance: distance = 1 - |correlation|
    distance_matrix = 1 - corr_matrix

    # Handle NaN in distance matrix (can occur if a feature has zero variance)
    if distance_matrix.isna().any().any():
        logger.warning("NaN values in distance matrix, filling with 1.0 (no correlation)")
        distance_matrix = distance_matrix.fillna(1.0)

    # Convert to condensed distance matrix (required by linkage)
    # squareform expects a symmetric matrix and returns the upper triangle as a vector
    condensed_dist = squareform(distance_matrix.values, checks=False)

    # Perform hierarchical clustering (average linkage is standard for correlation-based clustering)
    linkage_matrix = linkage(condensed_dist, method="average")

    # Cut dendrogram at distance = 1 - corr_threshold
    distance_threshold = 1.0 - corr_threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")

    # Build feature clusters from labels
    feature_clusters = {}
    for feature_name, cluster_label in zip(panel_features, cluster_labels, strict=True):
        if cluster_label not in feature_clusters:
            feature_clusters[cluster_label] = []
        feature_clusters[cluster_label].append(feature_name)

    # Convert to list of lists (sorted by cluster ID for reproducibility)
    clusters_list = [feature_clusters[cid] for cid in sorted(feature_clusters.keys())]

    logger.debug(
        f"Clustered {len(panel_features)} features into {len(clusters_list)} clusters "
        f"at corr_threshold={corr_threshold}"
    )

    return clusters_list


def _compute_brier_deltas(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_clusters: list[list[str]],
    random_state: int,
) -> list[float]:
    """Compute delta_brier for each feature cluster.

    Note: Lower Brier score is better, so delta_brier = original_brier - reduced_brier.
    Positive delta_brier means removing the cluster degraded calibration.

    Args:
        model: Fitted sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        feature_clusters: List of feature clusters to drop.
        random_state: Random state for cloning.

    Returns:
        List of delta_brier values (one per cluster).
    """
    # Compute original Brier score
    try:
        y_pred_proba_original = model.predict_proba(X_val)[:, 1]
        original_brier = brier_score_loss(y_val, y_pred_proba_original)
        logger.debug(f"Original Brier score: {original_brier:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute original Brier score: {e}")
        return [np.nan] * len(feature_clusters)

    delta_brier_list = []

    for cluster_id, cluster_features in enumerate(feature_clusters):
        try:
            # Drop cluster and refit
            features_to_keep = [f for f in X_train.columns if f not in cluster_features]

            if len(features_to_keep) == 0:
                logger.warning(f"Cluster {cluster_id}: all features dropped, skipping Brier")
                delta_brier_list.append(np.nan)
                continue

            X_train_reduced = X_train[features_to_keep]
            X_val_reduced = X_val[features_to_keep]

            # Clone and refit
            model_clone = clone(model)
            if hasattr(model_clone, "steps") and len(model_clone.steps) > 0:
                final_estimator = model_clone.steps[-1][1]
                if hasattr(final_estimator, "random_state"):
                    final_estimator.set_params(random_state=random_state)

            model_clone.fit(X_train_reduced, y_train)

            # Evaluate Brier score
            y_pred_proba_reduced = model_clone.predict_proba(X_val_reduced)[:, 1]
            reduced_brier = brier_score_loss(y_val, y_pred_proba_reduced)

            # delta_brier = original_brier - reduced_brier
            # Positive: removing cluster degraded calibration (worse Brier)
            delta_brier = original_brier - reduced_brier
            delta_brier_list.append(delta_brier)

            logger.debug(
                f"Cluster {cluster_id}: original_brier={original_brier:.6f}, "
                f"reduced_brier={reduced_brier:.6f}, delta_brier={delta_brier:.6f}"
            )

        except Exception as e:
            logger.error(f"Cluster {cluster_id}: Failed to compute Brier score: {e}")
            delta_brier_list.append(np.nan)

    return delta_brier_list
