"""Post-hoc RFE for clinical deployment panel optimization.

Identifies the smallest protein panel that maintains acceptable AUROC through
iterative feature removal. Runs AFTER training on a single trained model to
generate Pareto curves showing size-performance trade-offs for clinical
decision-making.

Design:
- Uses validation set AUROC for elimination decisions (test reserved for final eval)
- Supports geometric step strategy for efficiency (~45× faster than nested RFECV)
- Model-specific importance: coefficients for linear, permutation for trees
- Outputs: curve CSV, recommendations JSON, feature ranking, Pareto plots

Complementary to features/nested_rfe.py (robust feature discovery):
- rfe.py (this module): Clinical deployment after training
  → "What's the minimum panel size maintaining AUROC ≥ 0.90?"
  → Use for: Stakeholder decisions, cost-benefit analysis, rapid iteration
  → Output: Pareto curve (panel size vs. AUROC)
  → Speed: Fast (single model evaluation per size)

- nested_rfe.py (during training): Scientific discovery within CV
  → "What features are robustly selected across CV folds?"
  → Use for: Publishing, understanding stability, early discovery
  → Output: Consensus panel (features in ≥80% of folds)
  → Speed: Slower (~45× more model fits due to nested CV)

Typical workflow: Use both sequentially
  1. Enable rfe_enabled: true during training (robust discovery)
  2. Run ced optimize-panel after training (deployment trade-offs)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Re-export from rfe_engine for backward compatibility
from ced_ml.features.rfe_engine import (
    bootstrap_auroc_ci as _bootstrap_auroc_ci,
)
from ced_ml.features.rfe_engine import (
    build_lightweight_pipeline,
    cluster_correlated_proteins_for_rfe,
    compute_eval_sizes,
    compute_feature_importance,
    detect_knee_point,
    find_recommended_panels,
    make_fresh_estimator,
)
from ced_ml.features.rfe_engine import (
    quick_tune_at_k as _quick_tune_at_k,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RFEResult",
    "recursive_feature_elimination",
    "save_rfe_results",
    "aggregate_rfe_results",
    "_bootstrap_auroc_ci",
    "build_lightweight_pipeline",
    "cluster_correlated_proteins_for_rfe",
    "compute_eval_sizes",
    "compute_feature_importance",
    "detect_knee_point",
    "find_recommended_panels",
    "make_fresh_estimator",
    "_quick_tune_at_k",
]


@dataclass
class RFEResult:
    """Results from recursive feature elimination.

    Attributes:
        curve: List of evaluation points with size, AUROC, and selected proteins.
        feature_ranking: Dict mapping protein -> elimination order (0 = removed first).
        recommended_panels: Dict with minimum sizes at various AUROC thresholds.
        max_auroc: Maximum AUROC achieved.
        model_name: Name of model used.
        cluster_map: Dict mapping representative protein -> cluster metadata
            (cluster_id, cluster_size, members). Empty if correlation-aware
            pre-filtering was not used.
    """

    curve: list[dict[str, Any]] = field(default_factory=list)
    feature_ranking: dict[str, int] = field(default_factory=dict)
    recommended_panels: dict[str, int] = field(default_factory=dict)
    max_auroc: float = 0.0
    model_name: str = ""
    cluster_map: dict[str, dict] = field(default_factory=dict)


def recursive_feature_elimination(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_pipeline: Pipeline,
    model_name: str,
    initial_proteins: list[str],
    cat_cols: list[str],
    meta_num_cols: list[str],
    min_size: int = 5,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    min_auroc_frac: float = 0.90,
    random_state: int = 42,
    n_perm_repeats: int = 5,
    retune_n_trials: int = 40,
    retune_cv_folds: int = 3,
    retune_n_jobs: int = 1,
    corr_aware: bool = True,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
    selection_freq: dict[str, float] | None = None,
    rfe_tune_spaces: dict[str, dict[str, dict]] | None = None,
) -> RFEResult:
    """Perform recursive feature elimination to find minimum viable panel.

    Iteratively removes least important proteins, evaluating AUROC at each
    step. At each evaluation point, hyperparameters are re-tuned on TRAIN
    via a quick Optuna search so each Pareto curve point reflects the best
    achievable AUROC at that panel size.

    WARNING: When cv_folds > 0, the OOF AUROC estimates are optimistically
    biased because feature ranking is computed on the full training set before
    CV. The validation AUROC (auroc_val) is the honest metric. Default
    cv_folds=0 skips CV entirely to avoid confusion.

    Args:
        X_train: Training features (DataFrame with protein + metadata columns).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        base_pipeline: Trained pipeline to clone classifier from.
        model_name: Model identifier (e.g., "LR_EN", "RF").
        initial_proteins: Starting protein panel.
        cat_cols: Categorical metadata columns.
        meta_num_cols: Numeric metadata columns.
        min_size: Smallest panel to evaluate.
        cv_folds: CV folds for OOF AUROC estimation (default 0 = skip CV).
        step_strategy: Elimination strategy ("geometric", "fine", "linear").
        min_auroc_frac: Early stop if AUROC drops below this fraction of max.
        random_state: Random seed.
        n_perm_repeats: Permutation repeats for tree importance.
        retune_n_trials: Optuna trials for per-k hyperparameter re-tuning.
        retune_cv_folds: CV folds for per-k Optuna search.
        retune_n_jobs: Parallel jobs for per-k Optuna CV evaluation.
        corr_aware: If True, cluster correlated proteins before RFE and
            run elimination on representatives only.
        corr_threshold: Correlation threshold for clustering (0.0-1.0).
        corr_method: Correlation method ("spearman" or "pearson").
        selection_freq: Stability selection frequencies for representative
            selection. If None, uses uniform weights.
        rfe_tune_spaces: Optional config-driven per-model search spaces from
            optimize_panel.yaml. Overrides hardcoded RFE_TUNE_SPACES defaults.

    Returns:
        RFEResult with curve, feature_ranking, and recommended_panels.
    """
    import time

    from ced_ml.features.rfe_engine import run_elimination_with_evaluation
    from ced_ml.models.hyperparams import RFE_TUNE_SPACES

    start_time = time.time()
    logger.info("Starting recursive feature elimination")
    logger.info(f"  initial_proteins: {len(initial_proteins)} proteins")
    logger.info(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"  model_name: {model_name}")

    if rfe_tune_spaces:
        can_retune = model_name in rfe_tune_spaces
    else:
        can_retune = model_name in RFE_TUNE_SPACES
    if can_retune:
        logger.info(
            f"  Per-k hyperparameter re-tuning: enabled "
            f"({retune_n_trials} trials, cv={retune_cv_folds}, n_jobs={retune_n_jobs})"
        )
    else:
        logger.info(
            f"  Per-k hyperparameter re-tuning: disabled "
            f"(no tune space defined for {model_name})"
        )

    cluster_map: dict[str, dict] = {}
    if corr_aware:
        logger.info(
            f"Correlation-aware pre-filtering: threshold={corr_threshold}, " f"method={corr_method}"
        )
        current_proteins, cluster_map = cluster_correlated_proteins_for_rfe(
            X_train=X_train,
            y_train=y_train,
            protein_cols=initial_proteins,
            selection_freq=selection_freq,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
        )
    else:
        current_proteins = list(initial_proteins)

    logger.debug(f"Computing eval sizes: current={len(current_proteins)}, min={min_size}")
    eval_sizes = compute_eval_sizes(len(current_proteins), min_size, step_strategy)
    logger.debug(f"Eval sizes computed: {eval_sizes}")
    logger.info(f"RFE: Starting with {len(current_proteins)} proteins, target sizes: {eval_sizes}")
    logger.info(f"RFE: Will evaluate {len(eval_sizes)} panel sizes")

    curve, feature_ranking, max_auroc_seen, all_best_params = run_elimination_with_evaluation(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        base_pipeline=base_pipeline,
        model_name=model_name,
        current_proteins=current_proteins,
        cat_cols=cat_cols,
        meta_num_cols=meta_num_cols,
        eval_sizes=eval_sizes,
        min_size=min_size,
        cv_folds=cv_folds,
        random_state=random_state,
        n_perm_repeats=n_perm_repeats,
        can_retune=can_retune,
        retune_n_trials=retune_n_trials,
        retune_cv_folds=retune_cv_folds,
        retune_n_jobs=retune_n_jobs,
        rfe_tune_spaces=rfe_tune_spaces,
        min_auroc_frac=min_auroc_frac,
    )

    if can_retune and all_best_params:
        param_ranges: dict[str, list] = {}
        for bp in all_best_params:
            for k, v in bp.items():
                param_ranges.setdefault(k, []).append(v)

        range_summary = ", ".join(
            (
                f"{k.replace('clf__', '')} [{min(vs):.4g}, {max(vs):.4g}]"
                if isinstance(vs[0], float)
                else f"{k.replace('clf__', '')} [{min(vs)}, {max(vs)}]"
            )
            for k, vs in param_ranges.items()
        )
        logger.info(
            f"\nRFE Summary (seed {random_state}):\n"
            f"  Eval points: {len(curve)} | "
            f"Total elapsed: {(time.time() - start_time)/60:.1f} min\n"
            f"  Hyperparams varied: {range_summary}"
        )

    logger.info("\nComputing recommended panel sizes...")
    recommended = find_recommended_panels(curve)

    total_elapsed = time.time() - start_time
    logger.info(
        f"\n{'='*60}\n"
        f"RFE Completed\n"
        f"Total time: {total_elapsed/60:.1f} min\n"
        f"Evaluation points: {len(curve)}\n"
        f"Max AUROC: {max_auroc_seen:.4f}\n"
        f"{'='*60}\n"
    )

    return RFEResult(
        curve=curve,
        feature_ranking=feature_ranking,
        recommended_panels=recommended,
        max_auroc=max_auroc_seen,
        model_name=model_name,
        cluster_map=cluster_map,
    )


def save_rfe_results(
    result: RFEResult,
    output_dir: str,
    model_name: str,
    split_seed: int,
) -> dict[str, str]:
    """Save RFE results to output directory.

    Args:
        result: RFEResult from recursive_feature_elimination.
        output_dir: Directory to save outputs.
        model_name: Model name for metadata.
        split_seed: Split seed for metadata.

    Returns:
        Dict mapping artifact name -> file path.
    """
    import os
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # Determine suffix for aggregated results (split_seed=-1 indicates aggregated)
    suffix = "_aggregated" if split_seed == -1 else ""

    # 1. Panel curve CSV
    curve_path = os.path.join(output_dir, f"panel_curve{suffix}.csv")
    curve_records = []
    for p in result.curve:
        record = {
            "size": p["size"],
            "auroc_cv": p["auroc_cv"],
            "auroc_cv_std": p["auroc_cv_std"],
            "auroc_val": p["auroc_val"],
            "auroc_val_std": p.get("auroc_val_std", 0.0),
            "auroc_val_ci_low": p.get("auroc_val_ci_low", 0.0),
            "auroc_val_ci_high": p.get("auroc_val_ci_high", 0.0),
            "prauc_cv": p.get("prauc_cv", float("nan")),
            "prauc_val": p.get("prauc_val", float("nan")),
            "brier_cv": p.get("brier_cv", float("nan")),
            "brier_val": p.get("brier_val", float("nan")),
            "sens_at_95spec_cv": p.get("sens_at_95spec_cv", float("nan")),
            "sens_at_95spec_val": p.get("sens_at_95spec_val", float("nan")),
            "proteins": json.dumps(p["proteins"]),
        }
        if result.cluster_map:
            total_members = sum(
                result.cluster_map.get(prot, {}).get("cluster_size", 1) for prot in p["proteins"]
            )
            record["total_cluster_members"] = total_members
            members_map = {
                prot: result.cluster_map[prot]["members"]
                for prot in p["proteins"]
                if prot in result.cluster_map
            }
            record["cluster_members_json"] = json.dumps(members_map)
        curve_records.append(record)
    curve_df = pd.DataFrame(curve_records)
    curve_df.to_csv(curve_path, index=False)
    paths["panel_curve"] = curve_path

    # 2. Feature ranking CSV
    ranking_path = os.path.join(output_dir, f"feature_ranking{suffix}.csv")
    ranking_records = []
    for p, order in sorted(result.feature_ranking.items(), key=lambda x: x[1]):
        record = {"protein": p, "elimination_order": order}
        if result.cluster_map and p in result.cluster_map:
            info = result.cluster_map[p]
            record["cluster_id"] = info["cluster_id"]
            record["cluster_size"] = info["cluster_size"]
            record["cluster_members"] = json.dumps(info["members"])
        elif result.cluster_map:
            record["cluster_id"] = None
            record["cluster_size"] = 1
            record["cluster_members"] = json.dumps([p])
        ranking_records.append(record)
    ranking_df = pd.DataFrame(ranking_records)
    ranking_df.to_csv(ranking_path, index=False)
    paths["feature_ranking"] = ranking_path

    # 2b. Cluster mapping CSV (if clusters were used)
    if result.cluster_map:
        cluster_map_path = os.path.join(output_dir, f"cluster_mapping{suffix}.csv")
        cluster_records = []
        for rep, info in result.cluster_map.items():
            for member in info["members"]:
                cluster_records.append(
                    {
                        "representative": rep,
                        "member_protein": member,
                        "cluster_id": info["cluster_id"],
                        "cluster_size": info["cluster_size"],
                        "is_representative": member == rep,
                    }
                )
        cluster_df = pd.DataFrame(cluster_records).sort_values(
            ["cluster_id", "is_representative"],
            ascending=[True, False],
        )
        cluster_df.to_csv(cluster_map_path, index=False)
        paths["cluster_mapping"] = cluster_map_path
        logger.info(f"Saved cluster mapping to {cluster_map_path}")

    # 3. Recommended panels JSON
    rec_path = os.path.join(output_dir, f"recommended_panels{suffix}.json")
    rec_data = {
        "model": model_name,
        "split_seed": split_seed,
        "max_auroc": result.max_auroc,
        "recommended_panels": result.recommended_panels,
        "timestamp": datetime.now().isoformat(),
    }
    with open(rec_path, "w") as f:
        json.dump(rec_data, f, indent=2)
    paths["recommended_panels"] = rec_path

    # 4. Metrics summary CSV (panel size vs all metrics)
    metrics_summary_path = os.path.join(output_dir, f"metrics_summary{suffix}.csv")
    metrics_df = pd.DataFrame(
        [
            {
                "size": p["size"],
                "auroc_cv": p["auroc_cv"],
                "auroc_cv_std": p["auroc_cv_std"],
                "auroc_val": p["auroc_val"],
                "auroc_val_std": p.get("auroc_val_std", 0.0),
                "auroc_val_ci_low": p.get("auroc_val_ci_low", 0.0),
                "auroc_val_ci_high": p.get("auroc_val_ci_high", 0.0),
                "prauc_cv": p.get("prauc_cv", float("nan")),
                "prauc_val": p.get("prauc_val", float("nan")),
                "brier_cv": p.get("brier_cv", float("nan")),
                "brier_val": p.get("brier_val", float("nan")),
                "sens_at_95spec_cv": p.get("sens_at_95spec_cv", float("nan")),
                "sens_at_95spec_val": p.get("sens_at_95spec_val", float("nan")),
            }
            for p in result.curve
        ]
    )
    metrics_df.to_csv(metrics_summary_path, index=False)
    paths["metrics_summary"] = metrics_summary_path

    logger.info(f"Saved RFE results to {output_dir}")
    return paths


def aggregate_rfe_results(results: list[RFEResult]) -> RFEResult:
    """Aggregate RFE results across multiple split seeds.

    Combines per-seed Pareto curves into a single curve with cross-seed
    mean and percentile-based 95% confidence intervals. Feature rankings
    are aggregated via mean elimination order.

    Args:
        results: List of RFEResult objects, one per split seed.

    Returns:
        Aggregated RFEResult with mean validation metrics, cross-seed CIs,
        mean feature rankings, and recommendations from the aggregated curve.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    if len(results) == 1:
        logger.info("Single seed: skipping aggregation, returning as-is")
        return results[0]

    n_seeds = len(results)
    logger.info(f"Aggregating RFE curves across {n_seeds} seeds")

    # -- Aggregate curves by panel size --
    # Collect all curve points keyed by size
    size_to_metrics: dict[int, list[dict[str, Any]]] = {}
    size_to_proteins: dict[int, list[list[str]]] = {}
    for r in results:
        for point in r.curve:
            size = point["size"]
            if size not in size_to_metrics:
                size_to_metrics[size] = []
                size_to_proteins[size] = []
            size_to_metrics[size].append(point)
            size_to_proteins[size].append(point["proteins"])

    # Only keep sizes present in ALL seeds for a clean curve
    all_seed_sizes = [size for size, points in size_to_metrics.items() if len(points) == n_seeds]
    all_seed_sizes.sort(reverse=True)

    if not all_seed_sizes:
        # Fallback: use sizes present in at least half the seeds
        all_seed_sizes = [
            size for size, points in size_to_metrics.items() if len(points) >= max(1, n_seeds // 2)
        ]
        all_seed_sizes.sort(reverse=True)
        logger.warning(
            f"No panel sizes common to all {n_seeds} seeds; "
            f"using {len(all_seed_sizes)} sizes present in >= {max(1, n_seeds // 2)} seeds"
        )

    val_metrics = ["auroc_val", "prauc_val", "brier_val", "sens_at_95spec_val"]
    cv_metrics = ["auroc_cv", "prauc_cv", "brier_cv", "sens_at_95spec_cv"]

    aggregated_curve: list[dict[str, Any]] = []
    for size in all_seed_sizes:
        points = size_to_metrics[size]
        agg: dict[str, Any] = {"size": size}

        for metric in val_metrics + cv_metrics:
            values = np.array([p.get(metric, np.nan) for p in points])
            values = values[~np.isnan(values)]
            if len(values) > 0:
                agg[metric] = float(np.mean(values))
                agg[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            else:
                agg[metric] = np.nan
                agg[f"{metric}_std"] = 0.0

        # Cross-seed percentile CI for auroc_val
        auroc_vals = np.array([p["auroc_val"] for p in points])
        if len(auroc_vals) > 1:
            agg["auroc_val_ci_low"] = float(np.percentile(auroc_vals, 2.5))
            agg["auroc_val_ci_high"] = float(np.percentile(auroc_vals, 97.5))
        else:
            agg["auroc_val_ci_low"] = agg["auroc_val"]
            agg["auroc_val_ci_high"] = agg["auroc_val"]

        agg["auroc_cv_std"] = agg.get("auroc_cv_std", 0.0)
        agg["n_seeds"] = len(points)

        # Use proteins from the first seed at this size (ordering is seed-dependent)
        agg["proteins"] = size_to_proteins[size][0]

        aggregated_curve.append(agg)

    # -- Aggregate feature rankings via mean elimination order --
    all_proteins: set[str] = set()
    for r in results:
        all_proteins.update(r.feature_ranking.keys())

    aggregated_ranking: dict[str, float] = {}
    for protein in all_proteins:
        orders = [r.feature_ranking[protein] for r in results if protein in r.feature_ranking]
        aggregated_ranking[protein] = float(np.mean(orders))

    # Convert to int-keyed dict sorted by mean order (for compatibility)
    sorted_ranking = {
        p: rank
        for rank, (p, _) in enumerate(sorted(aggregated_ranking.items(), key=lambda x: x[1]))
    }

    # -- Recommendations from aggregated curve --
    recommended = find_recommended_panels(aggregated_curve)

    # -- Max AUROC from aggregated curve --
    max_auroc = max(
        (p["auroc_val"] for p in aggregated_curve),
        default=0.0,
    )

    model_name = results[0].model_name

    logger.info(
        f"Aggregated {len(aggregated_curve)} panel sizes across {n_seeds} seeds, "
        f"max mean AUROC={max_auroc:.4f}"
    )

    # Use first seed's cluster map as canonical (same proteins -> same clusters)
    cluster_map = results[0].cluster_map if results[0].cluster_map else {}

    return RFEResult(
        curve=aggregated_curve,
        feature_ranking=sorted_ranking,
        recommended_panels=recommended,
        max_auroc=max_auroc,
        model_name=model_name,
        cluster_map=cluster_map,
    )
