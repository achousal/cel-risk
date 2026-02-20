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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

# Valid refit modes for drop-column essentiality
RefitMode = Literal["fixed", "retune", "fixed_retune"]

logger = logging.getLogger(__name__)

__all__ = [
    "DropColumnResult",
    "RefitMode",
    "compute_drop_column_importance",
    "aggregate_drop_column_results",
    "validate_panel_essentiality",
]


def _propagate_random_state(estimator: Any, random_state: int) -> None:
    """Propagate random_state to nested estimators in a pipeline.

    Recursively sets random_state on all estimators that support it,
    including those nested within CalibratedClassifierCV or Pipeline steps.

    Args:
        estimator: Sklearn estimator (may be Pipeline or CalibratedClassifierCV).
        random_state: Random seed to set.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    # Set on the top-level estimator if it has random_state
    if hasattr(estimator, "random_state"):
        try:
            estimator.set_params(random_state=random_state)
        except Exception:
            pass

    # Use get_params(deep=True) to discover and set ALL nested random_state params.
    # This catches nested estimators that the explicit Pipeline/CalibratedClassifierCV
    # handling below might miss (e.g., deeply nested preprocessors or sub-estimators).
    if hasattr(estimator, "get_params"):
        try:
            params = estimator.get_params(deep=True)
            rs_params = {key: random_state for key in params if key.endswith("random_state")}
            if rs_params:
                estimator.set_params(**rs_params)
                logger.debug(
                    f"Set random_state={random_state} on {len(rs_params)} nested params: "
                    f"{list(rs_params.keys())}"
                )
        except Exception as e:
            logger.debug(f"Could not set nested random_state params: {e}")

    # Handle Pipeline: propagate to all steps
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _propagate_random_state(step, random_state)

    # Handle CalibratedClassifierCV: propagate to base estimator
    elif isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "estimator") and estimator.estimator is not None:
            _propagate_random_state(estimator.estimator, random_state)
        elif hasattr(estimator, "base_estimator") and estimator.base_estimator is not None:
            _propagate_random_state(estimator.base_estimator, random_state)


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
    # Retune fields (populated when refit_mode is "retune" or "fixed_retune")
    retune_auroc: float | None = None
    delta_auroc_retune: float | None = None
    retune_best_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (JSON, CSV).

        Returns:
            Dictionary representation suitable for JSON/CSV export.
        """
        d = {
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
        if self.retune_auroc is not None:
            d["retune_auroc"] = round(self.retune_auroc, 6)
        if self.delta_auroc_retune is not None:
            d["delta_auroc_retune"] = round(self.delta_auroc_retune, 6)
        if self.retune_best_params:
            d["retune_best_params"] = str(self.retune_best_params)
        return d


def compute_drop_column_importance(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_clusters: list[list[str]],
    random_state: int = 42,
    *,
    refit_mode: RefitMode = "fixed",
    model_name: str = "",
    cat_cols: list[str] | None = None,
    retune_n_trials: int = 20,
    retune_inner_folds: int = 3,
    retune_spaces: dict[str, dict[str, dict]] | None = None,
    n_jobs: int = 1,
) -> list[DropColumnResult]:
    """Compute drop-column importance by refitting model with each cluster removed.

    For each cluster in feature_clusters:
    1. Remove all features in the cluster from X_train and X_val
    2. Refit the estimator on reduced training set (strategy depends on refit_mode)
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
        refit_mode: Strategy for refitting after dropping features.
            - "fixed": clone with frozen hyperparams (fast, default).
            - "retune": full Optuna re-optimization per cluster drop.
            - "fixed_retune": run both, report side-by-side for compensation analysis.
        model_name: Model identifier (e.g., "LR_EN", "RF"). Required for retune modes.
        cat_cols: Categorical metadata columns. Required for retune modes.
        retune_n_trials: Number of Optuna trials per cluster (retune modes only).
        retune_inner_folds: Inner CV folds for retune's OptunaSearchCV.
        retune_spaces: Optional override search spaces (model_name -> {param: spec}).
            Passed to get_rfe_tune_space as config_overrides.
        n_jobs: Number of parallel jobs for cluster evaluation. Each cluster
            drop-and-refit is independent. Default 1 (sequential). Use -1 for
            all available CPUs. When using retune mode, keep inner Optuna
            n_jobs=1 to avoid over-subscription.

    Returns:
        List of DropColumnResult objects, one per cluster.

    Raises:
        ValueError: If X_train/X_val have different columns or feature_clusters
                    contains unknown features.
        RuntimeError: If refitting or evaluation fails for a cluster.
    """
    if refit_mode in ("retune", "fixed_retune") and not model_name:
        raise ValueError(f"model_name is required for refit_mode='{refit_mode}'")
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

    # Compute retune baseline AUROC if needed (re-optimized on full feature set)
    retune_baseline_auroc = None
    if refit_mode in ("retune", "fixed_retune"):
        retune_baseline_auroc = _compute_retune_baseline(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_cols=cat_cols or [],
            n_trials=retune_n_trials,
            cv_folds=retune_inner_folds,
            random_state=random_state,
            retune_spaces=retune_spaces,
        )

    # Build common kwargs for _drop_and_evaluate_cluster
    common_kwargs = {
        "estimator": estimator,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "original_auroc": original_auroc,
        "random_state": random_state,
        "refit_mode": refit_mode,
        "model_name": model_name,
        "cat_cols": cat_cols or [],
        "retune_n_trials": retune_n_trials,
        "retune_inner_folds": retune_inner_folds,
        "retune_spaces": retune_spaces,
        "retune_baseline_auroc": retune_baseline_auroc,
    }

    # Parallel or sequential cluster evaluation
    use_parallel = n_jobs != 1 and len(feature_clusters) > 1
    if use_parallel:
        from joblib import Parallel, delayed

        logger.info(f"Evaluating {len(feature_clusters)} clusters in parallel (n_jobs={n_jobs})")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_drop_and_evaluate_cluster)(
                cluster_id=cluster_id,
                cluster_features=cluster_features,
                **common_kwargs,
            )
            for cluster_id, cluster_features in enumerate(feature_clusters)
        )
    else:
        results = []
        for cluster_id, cluster_features in enumerate(feature_clusters):
            result = _drop_and_evaluate_cluster(
                cluster_id=cluster_id,
                cluster_features=cluster_features,
                **common_kwargs,
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
    refit_mode: RefitMode = "fixed",
    model_name: str = "",
    cat_cols: list[str] | None = None,
    retune_n_trials: int = 20,
    retune_inner_folds: int = 3,
    retune_spaces: dict[str, dict[str, dict]] | None = None,
    retune_baseline_auroc: float | None = None,
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
        original_auroc: Baseline AUROC for comparison (fixed mode).
        random_state: Random state for consistent cloning.
        refit_mode: "fixed", "retune", or "fixed_retune".
        model_name: Model identifier for retune modes.
        cat_cols: Categorical columns for retune pipeline rebuild.
        retune_n_trials: Optuna trials for retune modes.
        retune_inner_folds: Inner CV folds for retune.
        retune_spaces: Optional override search spaces.
        retune_baseline_auroc: Baseline AUROC from re-optimized model on full features.

    Returns:
        DropColumnResult with delta_auroc computed. If refit_mode includes retune,
        also populates retune_auroc, delta_auroc_retune, and retune_best_params.
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

        # --- Fixed mode (clone with frozen hyperparams) ---
        reduced_auroc = np.nan
        delta_auroc = np.nan
        if refit_mode in ("fixed", "fixed_retune"):
            estimator_clone = clone(estimator)
            _propagate_random_state(estimator_clone, random_state)
            estimator_clone.fit(X_train_reduced, y_train)

            y_pred_proba_reduced = estimator_clone.predict_proba(X_val_reduced)[:, 1]
            reduced_auroc = roc_auc_score(y_val, y_pred_proba_reduced)
            delta_auroc = original_auroc - reduced_auroc

            logger.debug(
                f"Cluster {cluster_id} [fixed] ({len(cluster_features)} features): "
                f"reduced_auroc={reduced_auroc:.6f}, delta_auroc={delta_auroc:.6f}"
            )

        # --- Retune mode (Optuna re-optimization) ---
        retune_auroc_val = None
        delta_auroc_retune_val = None
        retune_best_params = {}
        if refit_mode in ("retune", "fixed_retune"):
            retune_result = _retune_and_evaluate_cluster(
                model_name=model_name,
                X_train_reduced=X_train_reduced,
                y_train=y_train,
                X_val_reduced=X_val_reduced,
                y_val=y_val,
                cat_cols=cat_cols or [],
                n_trials=retune_n_trials,
                cv_folds=retune_inner_folds,
                random_state=random_state,
                retune_spaces=retune_spaces,
            )
            retune_auroc_val = retune_result["auroc"]
            retune_best_params = retune_result["best_params"]

            baseline = (
                retune_baseline_auroc if retune_baseline_auroc is not None else original_auroc
            )
            delta_auroc_retune_val = baseline - retune_auroc_val

            logger.debug(
                f"Cluster {cluster_id} [retune] ({len(cluster_features)} features): "
                f"retune_auroc={retune_auroc_val:.6f}, "
                f"delta_auroc_retune={delta_auroc_retune_val:.6f}"
            )

        # For pure retune mode, use retune values as primary
        if refit_mode == "retune":
            reduced_auroc = retune_auroc_val if retune_auroc_val is not None else np.nan
            delta_auroc = delta_auroc_retune_val if delta_auroc_retune_val is not None else np.nan

        return DropColumnResult(
            cluster_id=cluster_id,
            cluster_features=cluster_features,
            original_auroc=original_auroc,
            reduced_auroc=reduced_auroc,
            delta_auroc=delta_auroc,
            retune_auroc=retune_auroc_val,
            delta_auroc_retune=delta_auroc_retune_val,
            retune_best_params=retune_best_params,
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


def _retune_and_evaluate_cluster(
    model_name: str,
    X_train_reduced: pd.DataFrame,
    y_train: np.ndarray,
    X_val_reduced: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    n_trials: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
    retune_spaces: dict[str, dict[str, dict]] | None = None,
) -> dict[str, Any]:
    """Re-optimize hyperparameters on reduced features and evaluate.

    Uses quick_tune_at_k from rfe_tuning for Optuna-based re-optimization,
    then evaluates the re-tuned pipeline on the validation set.

    Args:
        model_name: Model identifier (e.g., "LR_EN", "RF").
        X_train_reduced: Training features with cluster removed.
        y_train: Training labels.
        X_val_reduced: Validation features with cluster removed.
        y_val: Validation labels.
        cat_cols: Categorical metadata columns.
        n_trials: Number of Optuna trials.
        cv_folds: Inner CV folds for Optuna.
        random_state: Random seed.
        retune_spaces: Optional override search spaces.

    Returns:
        Dict with keys: auroc, best_params.
    """
    from ced_ml.features.rfe_tuning import quick_tune_at_k

    feature_cols = [c for c in X_train_reduced.columns if c not in cat_cols]

    fitted_pipeline, best_params = quick_tune_at_k(
        model_name=model_name,
        X_train=X_train_reduced,
        y_train=y_train,
        feature_cols=feature_cols,
        cat_cols=[c for c in cat_cols if c in X_train_reduced.columns],
        cv_folds=cv_folds,
        n_trials=n_trials,
        n_jobs=1,
        random_state=random_state,
        rfe_tune_spaces=retune_spaces,
    )

    y_pred_proba = fitted_pipeline.predict_proba(X_val_reduced)[:, 1]
    auroc = roc_auc_score(y_val, y_pred_proba)

    return {"auroc": auroc, "best_params": best_params}


def _compute_retune_baseline(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    n_trials: int = 20,
    cv_folds: int = 3,
    random_state: int = 42,
    retune_spaces: dict[str, dict[str, dict]] | None = None,
) -> float:
    """Compute baseline AUROC from a re-optimized model on the full feature set.

    This ensures the retune delta is measured against a consistently re-optimized
    baseline, not the original training-time optimization which used a different
    pipeline structure (screen, sel, model_sel steps).

    Args:
        model_name: Model identifier.
        X_train: Full training features.
        y_train: Training labels.
        X_val: Full validation features.
        y_val: Validation labels.
        cat_cols: Categorical metadata columns.
        n_trials: Number of Optuna trials.
        cv_folds: Inner CV folds.
        random_state: Random seed.
        retune_spaces: Optional override search spaces.

    Returns:
        Baseline AUROC from re-optimized model.
    """
    logger.info(
        f"Computing retune baseline: {model_name} with {n_trials} trials "
        f"on {X_train.shape[1]} features"
    )
    result = _retune_and_evaluate_cluster(
        model_name=model_name,
        X_train_reduced=X_train,
        y_train=y_train,
        X_val_reduced=X_val,
        y_val=y_val,
        cat_cols=cat_cols,
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=random_state,
        retune_spaces=retune_spaces,
    )
    logger.info(f"Retune baseline AUROC: {result['auroc']:.6f}")
    return result["auroc"]


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

        row = {
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

        # Aggregate retune deltas if present
        valid_retune_deltas = [
            r.delta_auroc_retune
            for r in cluster_results
            if r.delta_auroc_retune is not None and not np.isnan(r.delta_auroc_retune)
        ]
        if valid_retune_deltas:
            row["mean_delta_auroc_retune"] = float(np.mean(valid_retune_deltas))
            row["std_delta_auroc_retune"] = (
                float(np.std(valid_retune_deltas, ddof=1))
                if len(valid_retune_deltas) > 1
                else np.nan
            )

        agg_rows.append(row)

    df = pd.DataFrame(agg_rows)

    # Sort by mean_delta_auroc descending (most important first)
    df = df.sort_values("mean_delta_auroc", ascending=False, na_position="last").reset_index(
        drop=True
    )

    # Add compensation flag for fixed_retune mode
    if "mean_delta_auroc_retune" in df.columns:
        compensation_threshold = 0.005
        df["compensation_flag"] = (df["mean_delta_auroc"].abs() < compensation_threshold) & (
            df["mean_delta_auroc_retune"] > compensation_threshold
        )
        df["compensation_delta"] = df["mean_delta_auroc_retune"] - df["mean_delta_auroc"]

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
    *,
    refit_mode: RefitMode = "fixed",
    model_name: str = "",
    cat_cols: list[str] | None = None,
    retune_n_trials: int = 20,
    retune_inner_folds: int = 3,
    retune_spaces: dict[str, dict[str, dict]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Validate panel robustness via grouped LOCO/refit.

    For each correlation cluster in the panel:
    1. Compute correlation matrix and cluster correlated features
    2. Drop each cluster and refit model (strategy depends on refit_mode)
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
        refit_mode: "fixed", "retune", or "fixed_retune".
        model_name: Model identifier (required for retune modes).
        cat_cols: Categorical metadata columns (required for retune modes).
        retune_n_trials: Optuna trials for retune modes.
        retune_inner_folds: Inner CV folds for retune.
        retune_spaces: Optional override search spaces.

    Returns:
        DataFrame with columns:
            - cluster_id: int
            - features: str (comma-separated feature names)
            - n_features: int
            - delta_auroc: float (importance measure)
            - delta_brier: float (if include_brier=True, else NaN)
            - essentiality_rank: int (1 = most essential)
            - delta_auroc_retune: float (if retune mode, else absent)
            - compensation_flag: bool (if fixed_retune mode, else absent)

        Sorted by delta_auroc descending (most essential first).

    Raises:
        ValueError: If panel_features are not in X_train or X_val.
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
        f"corr_threshold={corr_threshold}, include_brier={include_brier}, "
        f"refit_mode={refit_mode}"
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
        refit_mode=refit_mode,
        model_name=model_name,
        cat_cols=cat_cols,
        retune_n_trials=retune_n_trials,
        retune_inner_folds=retune_inner_folds,
        retune_spaces=retune_spaces,
        n_jobs=n_jobs,
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
    """Cluster panel features by Spearman correlation using connected components.

    Uses the same graph-based connected-component approach as corr_prune.py
    to ensure consistent clustering behavior across the pipeline.

    Args:
        X_train: Training features (must contain panel_features).
        panel_features: List of feature names to cluster.
        corr_threshold: Absolute Spearman correlation threshold for grouping.

    Returns:
        List of feature clusters (list of lists of feature names).
    """
    if len(panel_features) <= 1:
        return [panel_features] if panel_features else []

    from ced_ml.features.corr_prune import (
        build_correlation_graph,
        compute_correlation_matrix,
        find_connected_components,
    )

    # Compute absolute Spearman correlation matrix
    corr_matrix = compute_correlation_matrix(X_train, panel_features, method="spearman")

    # Build adjacency graph and find connected components
    adjacency = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    components = find_connected_components(adjacency)

    logger.debug(
        f"Clustered {len(panel_features)} features into {len(components)} clusters "
        f"at corr_threshold={corr_threshold}"
    )

    return components


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

            # Clone and refit with random_state propagation
            model_clone = clone(model)
            _propagate_random_state(model_clone, random_state)
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


def _compute_pr_auc_deltas(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_clusters: list[list[str]],
    random_state: int,
) -> list[float]:
    """Compute delta_pr_auc for each feature cluster.

    PR-AUC (average precision) is especially informative for imbalanced datasets.
    delta_pr_auc = original_pr_auc - reduced_pr_auc.
    Positive delta means removing the cluster degraded precision-recall performance.

    Args:
        model: Fitted sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        feature_clusters: List of feature clusters to drop.
        random_state: Random state for cloning.

    Returns:
        List of delta_pr_auc values (one per cluster).
    """
    # Compute original PR-AUC
    try:
        y_pred_proba_original = model.predict_proba(X_val)[:, 1]
        original_pr_auc = average_precision_score(y_val, y_pred_proba_original)
        logger.debug(f"Original PR-AUC: {original_pr_auc:.6f}")
    except Exception as e:
        logger.error(f"Failed to compute original PR-AUC: {e}")
        return [np.nan] * len(feature_clusters)

    delta_pr_auc_list = []

    for cluster_id, cluster_features in enumerate(feature_clusters):
        try:
            # Drop cluster and refit
            features_to_keep = [f for f in X_train.columns if f not in cluster_features]

            if len(features_to_keep) == 0:
                logger.warning(f"Cluster {cluster_id}: all features dropped, skipping PR-AUC")
                delta_pr_auc_list.append(np.nan)
                continue

            X_train_reduced = X_train[features_to_keep]
            X_val_reduced = X_val[features_to_keep]

            # Clone and refit with random_state propagation
            model_clone = clone(model)
            _propagate_random_state(model_clone, random_state)
            model_clone.fit(X_train_reduced, y_train)

            # Evaluate PR-AUC
            y_pred_proba_reduced = model_clone.predict_proba(X_val_reduced)[:, 1]
            reduced_pr_auc = average_precision_score(y_val, y_pred_proba_reduced)

            # delta = original - reduced (positive means cluster was important)
            delta_pr_auc = original_pr_auc - reduced_pr_auc
            delta_pr_auc_list.append(delta_pr_auc)

            logger.debug(
                f"Cluster {cluster_id}: original_pr_auc={original_pr_auc:.6f}, "
                f"reduced_pr_auc={reduced_pr_auc:.6f}, delta_pr_auc={delta_pr_auc:.6f}"
            )

        except Exception as e:
            logger.error(f"Cluster {cluster_id}: Failed to compute PR-AUC: {e}")
            delta_pr_auc_list.append(np.nan)

    return delta_pr_auc_list
