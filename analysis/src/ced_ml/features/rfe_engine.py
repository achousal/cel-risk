"""Core RFE elimination loop and clustering utilities.

Extracted from original rfe.py to reduce module complexity. This module now
focuses on the core elimination algorithm and correlation-aware clustering,
with evaluation, importance, and tuning functions delegated to specialized modules.

Imports from submodules are re-exported for backwards compatibility.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ced_ml.features.rfe_evaluation import (
    bootstrap_auroc_ci,
    compute_eval_sizes,
    detect_knee_point,
    evaluate_panel_size,
    find_recommended_panels,
)
from ced_ml.features.rfe_importance import compute_feature_importance
from ced_ml.features.rfe_tuning import (
    build_lightweight_pipeline,
    make_fresh_estimator,
    quick_tune_at_k,
)

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = [
    "bootstrap_auroc_ci",
    "build_lightweight_pipeline",
    "cluster_correlated_proteins_for_rfe",
    "compute_eval_sizes",
    "compute_feature_importance",
    "detect_knee_point",
    "elimination_loop",
    "evaluate_panel_size",
    "find_recommended_panels",
    "make_fresh_estimator",
    "quick_tune_at_k",
    "run_elimination_with_evaluation",
]


def cluster_correlated_proteins_for_rfe(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    protein_cols: list[str],
    selection_freq: dict[str, float] | None = None,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
) -> tuple[list[str], dict[str, dict]]:
    """Cluster correlated proteins and select representatives for RFE.

    Uses graph-based connected components (from corr_prune) to group
    proteins with |correlation| >= threshold, then selects one
    representative per cluster using composite criterion: stability
    frequency (primary) + Mann-Whitney p-value (tiebreak).

    All correlation analysis uses TRAIN data only.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Training labels.
    protein_cols : list[str]
        Input proteins to cluster.
    selection_freq : dict[str, float] or None
        Selection frequency from CV folds {protein: frequency}.
        If None, all proteins get equal weight.
    corr_threshold : float
        Absolute correlation threshold for clustering (0.0-1.0).
    corr_method : str
        Correlation method ("spearman" or "pearson").

    Returns
    -------
    representative_proteins : list[str]
        Cluster representatives to pass to RFE.
    cluster_map : dict[str, dict]
        Mapping representative -> {cluster_id, cluster_size, members}.
        Empty if clustering was skipped.
    """
    from ced_ml.features.corr_prune import prune_correlated_proteins

    if corr_threshold >= 1.0 or len(protein_cols) < 2:
        logger.info("Skipping correlation clustering (threshold >= 1.0 or < 2 proteins)")
        return list(protein_cols), {}

    df_map, kept_proteins = prune_correlated_proteins(
        df=X_train,
        y=y_train,
        proteins=protein_cols,
        selection_freq=selection_freq,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        tiebreak_method="freq_then_univariate",
    )

    if df_map.empty:
        return list(protein_cols), {}

    cluster_map: dict[str, dict] = {}
    for rep in kept_proteins:
        comp_rows = df_map[df_map["rep_protein"] == rep]
        if comp_rows.empty:
            logger.warning(f"No component rows found for representative '{rep}'; skipping")
            continue
        comp_id = int(comp_rows["component_id"].iloc[0])
        members = sorted(comp_rows["protein"].tolist())
        cluster_map[rep] = {
            "cluster_id": comp_id,
            "cluster_size": len(members),
            "members": members,
        }

    n_multi = sum(1 for v in cluster_map.values() if v["cluster_size"] > 1)
    logger.info(
        f"Clustered {len(protein_cols)} proteins into {len(cluster_map)} clusters "
        f"({len(cluster_map) - n_multi} singletons, {n_multi} multi-protein)"
    )
    if n_multi > 0:
        top_clusters = sorted(
            [(rep, info) for rep, info in cluster_map.items() if info["cluster_size"] > 1],
            key=lambda x: -x[1]["cluster_size"],
        )[:5]
        for rep, info in top_clusters:
            preview = ", ".join(info["members"][:3])
            ellipsis = "..." if len(info["members"]) > 3 else ""
            logger.info(f"  {rep}: {info['cluster_size']} members ({preview}{ellipsis})")

    return kept_proteins, cluster_map


def elimination_loop(
    current_proteins: list[str],
    target_size: int,
    min_size: int,
    base_pipeline: Pipeline,
    cat_cols: list[str],
    meta_num_cols: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    random_state: int,
    n_perm_repeats: int,
    feature_ranking: dict[str, int],
    elimination_order: int,
) -> tuple[list[str], dict[str, int], int]:
    """Execute elimination loop to reduce panel to target size.

    Args:
        current_proteins: List of proteins currently in panel.
        target_size: Target panel size.
        min_size: Minimum allowed panel size.
        base_pipeline: Base pipeline to clone classifier from.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        X_train: Training features.
        y_train: Training labels.
        model_name: Model identifier.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats for importance.
        feature_ranking: Dict to update with eliminated proteins.
        elimination_order: Current elimination order counter.

    Returns:
        Tuple of (updated_proteins, updated_ranking, updated_order).
    """
    proteins_to_eliminate = len(current_proteins) - target_size
    logger.info(f"Eliminating {proteins_to_eliminate} proteins to reach target size {target_size}")

    elimination_count = 0
    while len(current_proteins) > target_size and len(current_proteins) > min_size:
        elimination_count += 1

        if elimination_count % 10 == 0:
            logger.info(
                f"  Elimination progress: {elimination_count}/{proteins_to_eliminate} "
                f"({elimination_count/proteins_to_eliminate*100:.1f}%), "
                f"current panel size: {len(current_proteins)}"
            )

        try:
            pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}")
            break

        feature_cols = current_proteins + cat_cols + meta_num_cols
        X_train_subset = X_train[feature_cols]

        try:
            pipeline.fit(X_train_subset, y_train)
        except Exception as e:
            logger.error(f"Pipeline fit failed at size {len(current_proteins)}: {e}")
            break

        importances = compute_feature_importance(
            pipeline,
            model_name,
            current_proteins,
            X_train_subset,
            y_train,
            random_state,
            n_perm_repeats,
        )

        protein_importances = {p: importances.get(p, 0.0) for p in current_proteins}

        unique_importances = set(protein_importances.values())
        if len(unique_importances) == 1 and len(current_proteins) > 1:
            logger.warning(
                f"All {len(current_proteins)} remaining features have identical importance "
                f"({list(unique_importances)[0]:.4f}). Breaking ties randomly with "
                f"seed={random_state + elimination_order} for reproducibility."
            )
            tie_rng = np.random.default_rng(random_state + elimination_order)
            worst_protein = str(tie_rng.choice(list(protein_importances.keys())))
        else:
            worst_protein = min(protein_importances, key=protein_importances.get)

        feature_ranking[worst_protein] = elimination_order
        current_proteins.remove(worst_protein)
        elimination_order += 1

        logger.debug(
            f"Eliminated {worst_protein} (importance={protein_importances[worst_protein]:.4f}), "
            f"panel size now {len(current_proteins)}"
        )

    return current_proteins, feature_ranking, elimination_order


def run_elimination_with_evaluation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_pipeline: Pipeline,
    model_name: str,
    current_proteins: list[str],
    cat_cols: list[str],
    meta_num_cols: list[str],
    eval_sizes: list[int],
    min_size: int,
    cv_folds: int,
    random_state: int,
    n_perm_repeats: int,
    can_retune: bool,
    retune_n_trials: int,
    retune_cv_folds: int,
    retune_n_jobs: int,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None,
    min_auroc_frac: float,
) -> tuple[list[dict[str, Any]], dict[str, int], float, list[dict]]:
    """Run RFE elimination loop with per-size evaluation.

    Core engine that executes elimination, evaluation, and optional tuning
    at each target panel size.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        base_pipeline: Base pipeline to clone from.
        model_name: Model identifier.
        current_proteins: Starting protein list.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        eval_sizes: List of panel sizes to evaluate.
        min_size: Minimum panel size.
        cv_folds: CV folds for OOF estimation.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats.
        can_retune: Whether per-k tuning is enabled.
        retune_n_trials: Optuna trials for tuning.
        retune_cv_folds: CV folds for tuning.
        retune_n_jobs: Parallel jobs for tuning.
        rfe_tune_spaces: Optional config-driven tune spaces.
        min_auroc_frac: Early stopping threshold.

    Returns:
        Tuple of (curve, feature_ranking, max_auroc, all_best_params).
    """
    curve: list[dict[str, Any]] = []
    feature_ranking: dict[str, int] = {}
    elimination_order = 0
    max_auroc_seen = 0.0
    all_best_params: list[dict] = []

    start_time = time.time()
    total_eval_points = len(eval_sizes)
    eval_point_idx = 0

    for target_size in eval_sizes:
        eval_point_idx += 1
        elapsed = time.time() - start_time
        logger.info(
            f"\n{'='*60}\n"
            f"RFE Progress: {eval_point_idx}/{total_eval_points} evaluation points "
            f"({eval_point_idx/total_eval_points*100:.1f}%)\n"
            f"Elapsed time: {elapsed/60:.1f} min\n"
            f"Target panel size: {target_size}\n"
            f"{'='*60}"
        )

        current_proteins, feature_ranking, elimination_order = elimination_loop(
            current_proteins=current_proteins,
            target_size=target_size,
            min_size=min_size,
            base_pipeline=base_pipeline,
            cat_cols=cat_cols,
            meta_num_cols=meta_num_cols,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            random_state=random_state,
            n_perm_repeats=n_perm_repeats,
            feature_ranking=feature_ranking,
            elimination_order=elimination_order,
        )

        if len(current_proteins) < min_size:
            logger.warning(
                f"Panel size {len(current_proteins)} below min_size {min_size}, stopping"
            )
            break

        logger.info(f"\nEvaluating panel size {len(current_proteins)}...")

        feature_cols = current_proteins + cat_cols + meta_num_cols
        X_train_subset = X_train[feature_cols]
        X_val_subset = X_val[feature_cols]

        eval_start = time.time()
        best_params = {}
        pipeline = None

        if can_retune:
            try:
                pipeline, best_params = quick_tune_at_k(
                    model_name=model_name,
                    X_train=X_train_subset,
                    y_train=y_train,
                    feature_cols=feature_cols,
                    cat_cols=cat_cols,
                    cv_folds=retune_cv_folds,
                    n_trials=retune_n_trials,
                    n_jobs=retune_n_jobs,
                    random_state=random_state,
                    rfe_tune_spaces=rfe_tune_spaces,
                )
                all_best_params.append(best_params)
            except Exception as exc:
                logger.warning(
                    "[RFE k=%d] Hyperparameter re-tuning failed (%s). "
                    "Falling back to baseline estimator.",
                    len(current_proteins),
                    exc,
                )

        if pipeline is None:
            try:
                pipeline = build_lightweight_pipeline(base_pipeline, current_proteins, cat_cols)
                pipeline.fit(X_train_subset, y_train)
            except Exception as e:
                logger.error(f"Evaluation failed at size {len(current_proteins)}: {e}")
                continue

        try:
            metrics = evaluate_panel_size(
                pipeline=pipeline,
                X_train_subset=X_train_subset,
                y_train=y_train,
                X_val_subset=X_val_subset,
                y_val=y_val,
                cv_folds=cv_folds,
                random_state=random_state,
                panel_size=len(current_proteins),
            )
        except Exception as e:
            logger.error(f"Evaluation failed at size {len(current_proteins)}: {e}")
            continue

        eval_elapsed = time.time() - eval_start
        logger.info(f"[RFE k={len(current_proteins)}] Completed in {eval_elapsed:.1f}s")

        point = {
            "size": len(current_proteins),
            "auroc_cv": metrics["auroc_cv"],
            "auroc_cv_std": metrics["auroc_cv_std"],
            "auroc_val": metrics["auroc_val"],
            "auroc_val_std": metrics["auroc_val_std"],
            "auroc_val_ci_low": metrics["auroc_val_ci_low"],
            "auroc_val_ci_high": metrics["auroc_val_ci_high"],
            "prauc_cv": metrics["prauc_cv"],
            "prauc_val": metrics["prauc_val"],
            "brier_cv": metrics["brier_cv"],
            "brier_val": metrics["brier_val"],
            "sens_at_95spec_cv": metrics["sens_at_95spec_cv"],
            "sens_at_95spec_val": metrics["sens_at_95spec_val"],
            "proteins": list(current_proteins),
            "best_params": best_params,
        }
        curve.append(point)
        logger.info(
            f"\n  *** Panel size {len(current_proteins)} results: ***\n"
            f"    AUROC_cv:        {metrics['auroc_cv']:.4f} +/- {metrics['auroc_cv_std']:.4f}\n"
            f"    AUROC_val:       {metrics['auroc_val']:.4f} "
            f"(95% CI: {metrics['auroc_val_ci_low']:.4f}-{metrics['auroc_val_ci_high']:.4f})\n"
            f"    PR-AUC_val:      {metrics['prauc_val']:.4f}\n"
            f"    Brier_val:       {metrics['brier_val']:.4f}\n"
            f"    Sens@95%Spec:    {metrics['sens_at_95spec_val']:.4f}\n"
        )

        if metrics["auroc_val"] > max_auroc_seen:
            max_auroc_seen = metrics["auroc_val"]
            logger.info(f"  New maximum AUROC: {max_auroc_seen:.4f}")

        min_evaluations = 3
        if (
            eval_point_idx >= min_evaluations
            and metrics["auroc_val"] < max_auroc_seen * min_auroc_frac
        ):
            logger.info(
                f"\n*** Early stopping triggered ***\n"
                f"AUROC {metrics['auroc_val']:.4f} < "
                f"{min_auroc_frac:.0%} of max {max_auroc_seen:.4f}\n"
                f"(after {eval_point_idx} evaluation rounds)\n"
            )
            break

    return curve, feature_ranking, max_auroc_seen, all_best_params
