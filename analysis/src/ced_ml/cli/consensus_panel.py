"""CLI implementation for cross-model consensus panel generation.

This module provides the `ced consensus-panel` command for creating consensus
protein panels from multiple models via Robust Rank Aggregation.

USAGE:
    ced consensus-panel --run-id 20260127_115115

WORKFLOW:
    1. Train base models (ced train --model LR_EN/RF/XGBoost/LinSVM_cal)
    2. Aggregate results (ced aggregate-splits)
    3. (Optional) Run panel optimization (ced optimize-panel)
    4. Generate consensus: ced consensus-panel --run-id <RUN_ID>
    5. Validate: ced train --fixed-panel results/consensus_panel/.../final_panel.txt --split-seed 10

OUTPUT:
    results/consensus_panel/run_<RUN_ID>/
        final_panel.txt          # One protein per line (for --fixed-panel)
        final_panel.csv          # Panel with uncertainty metrics
        consensus_ranking.csv    # All proteins with RRA scores and uncertainty
        uncertainty_summary.csv  # Focused uncertainty report for final panel
        per_model_rankings.csv   # Per-model composite rankings
        correlation_clusters.csv # Cluster assignments
        consensus_metadata.json  # Run parameters, statistics, and uncertainty summary

UNCERTAINTY METRICS:
    - n_models_present: Number of models with this protein (cross-model agreement)
    - agreement_strength: Fraction of models agreeing (0-1)
    - rank_std: Standard deviation of ranks across models
    - rank_cv: Coefficient of variation (std/mean) - lower = more stable ranking
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from ced_ml.cli.discovery import (
    auto_detect_data_paths,
    discover_models_with_aggregated_results,
)
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.features.consensus import (
    ConsensusResult,
    build_consensus_panel,
    save_consensus_results,
)
from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)
from ced_ml.features.drop_column import (
    aggregate_drop_column_results,
    compute_drop_column_importance,
)
from ced_ml.models.calibration import OOFCalibratedModel

logger = logging.getLogger(__name__)


def _configure_screen_step_for_panel_refit(pipeline, panel_features: list[str]) -> None:
    """Configure pipeline steps for fixed-panel refits used in essentiality validation.

    During within-panel essentiality, we refit on a reduced feature matrix
    (panel proteins, optionally plus metadata). This helper adjusts:

    1) ``screen`` step: rewire protein_cols and pin precomputed_features so
       screening passes through only the panel proteins.
    2) ``sel`` / ``model_sel`` steps: replace with ``"passthrough"`` because
       the panel is already fixed and SelectKBest(k=tuned_k) would raise
       ValueError when tuned_k exceeds the (smaller) panel size.

    That keeps drop-column refits stable across all model types.
    """
    if not hasattr(pipeline, "named_steps"):
        return

    # -- Screen step: pin to panel proteins --
    screen_step = pipeline.named_steps.get("screen")
    if screen_step is not None:
        panel_copy = list(panel_features)
        if hasattr(screen_step, "protein_cols"):
            screen_step.protein_cols = panel_copy
        if hasattr(screen_step, "precomputed_features"):
            screen_step.precomputed_features = panel_copy

    # -- Feature selection steps: bypass (panel is already fixed) --
    for step_name in ("sel", "model_sel"):
        if step_name in pipeline.named_steps:
            pipeline.set_params(**{step_name: "passthrough"})


def _extract_bundle_metadata(model_path: Path) -> dict:
    """Extract only metadata keys from a model bundle without full deserialization.

    When the training and loading environments differ (e.g. pandas version
    mismatch causing StringDtype pickle errors), the sklearn pipeline inside
    the bundle cannot be unpickled.  This function uses pickle's Unpickler
    with a permissive class loader so that unresolvable objects are replaced
    by stubs, allowing the top-level dict and its lightweight values
    (``resolved_columns``, ``scenario``) to be recovered.
    """
    import io
    import pickle

    class _StubUnpickler(pickle.Unpickler):
        """Unpickler that replaces unresolvable classes with a placeholder."""

        def find_class(self, module: str, name: str):
            try:
                return super().find_class(module, name)
            except Exception:
                # Return a no-op callable that absorbs any constructor args
                return lambda *a, **_: None

    with open(model_path, "rb") as fh:
        raw = fh.read()

    bundle = _StubUnpickler(io.BytesIO(raw)).load()
    if not isinstance(bundle, dict):
        raise ValueError(f"Metadata-only extraction failed: expected dict, got {type(bundle)}")
    return bundle


def load_aggregated_significance(run_dir: Path) -> pd.DataFrame | None:
    """Load aggregated significance results for all models in a run.

    Args:
        run_dir: Path to run directory (results/run_{ID}/)

    Returns:
        DataFrame with columns [model, empirical_p_value, significant, ...] or None if not found.
    """
    sig_files = list(run_dir.glob("*/significance/aggregated_significance.csv"))
    if not sig_files:
        return None

    dfs = []
    for f in sig_files:
        try:
            df = pd.read_csv(f)
            if "model" not in df.columns:
                model_name = f.parent.parent.name
                df["model"] = model_name
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def load_model_stability(
    aggregated_dir: Path,
    stability_threshold: float = 0.0,
) -> pd.DataFrame:
    """Load feature stability summary from aggregated results.

    Args:
        aggregated_dir: Path to model's aggregated directory.
        stability_threshold: Minimum selection fraction (0 = load all).

    Returns:
        DataFrame with columns [protein, selection_fraction, ...].

    Raises:
        FileNotFoundError: If stability file not found.
    """
    stability_file = aggregated_dir / "panels" / "feature_stability_summary.csv"

    if not stability_file.exists():
        raise FileNotFoundError(
            f"Feature stability file not found: {stability_file}\nRun 'ced aggregate-splits' first."
        )

    df = pd.read_csv(stability_file)

    # Clean protein names (may have extra quotes)
    if "protein" in df.columns:
        df["protein"] = df["protein"].str.strip('"')

    # Filter by threshold if requested
    if stability_threshold > 0 and "selection_fraction" in df.columns:
        df = df[df["selection_fraction"] >= stability_threshold].copy()

    return df


def load_model_oof_importance(
    aggregated_dir: Path,
    model_name: str = "",
) -> pd.DataFrame | None:
    """Load OOF grouped importance from aggregated results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.
        model_name: Model name (e.g., "LR_EN") used to locate the importance
            file written by aggregate_importance() as
            ``importance/oof_importance__{model_name}.csv``.

    Returns:
        DataFrame with OOF importance (columns: feature, mean_importance, etc.), or None.
    """
    oof_file = None

    # Primary: match what aggregate_importance() actually writes
    if model_name:
        candidate = aggregated_dir / "importance" / f"oof_importance__{model_name}.csv"
        if candidate.exists():
            oof_file = candidate

    # Legacy fallbacks
    if oof_file is None:
        candidate = aggregated_dir / "importance" / "aggregated_oof_importance.csv"
        if candidate.exists():
            oof_file = candidate

    if oof_file is None:
        candidate = aggregated_dir / "importance" / "importance_aggregated.csv"
        if candidate.exists():
            oof_file = candidate

    if oof_file is None:
        return None

    try:
        df = pd.read_csv(oof_file)
        # Standardize column names
        if "feature" not in df.columns and "protein" in df.columns:
            df = df.rename(columns={"protein": "feature"})
        if "importance" in df.columns and "mean_importance" not in df.columns:
            df = df.rename(columns={"importance": "mean_importance"})
        return df
    except Exception as e:
        logger.warning(f"Failed to load OOF importance from {oof_file}: {e}")
        return None


def load_model_oof_shap_importance(
    aggregated_dir: Path,
    model_name: str = "",
) -> pd.DataFrame | None:
    """Load aggregated OOF SHAP importance from aggregated results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.
        model_name: Model name used to locate
            ``importance/oof_shap_importance__{model_name}.csv``.

    Returns:
        DataFrame with standardized columns ``feature`` and ``mean_importance``,
        or None if SHAP importance is unavailable.
    """
    shap_file = None

    if model_name:
        candidate = aggregated_dir / "importance" / f"oof_shap_importance__{model_name}.csv"
        if candidate.exists():
            shap_file = candidate

    if shap_file is None:
        candidate = aggregated_dir / "importance" / "aggregated_oof_shap_importance.csv"
        if candidate.exists():
            shap_file = candidate

    if shap_file is None:
        return None

    try:
        df = pd.read_csv(shap_file)

        # Standardize feature name
        if "feature" not in df.columns and "protein" in df.columns:
            df = df.rename(columns={"protein": "feature"})

        # Standardize score column expected by compute_per_model_ranking
        if "mean_importance" not in df.columns:
            if "mean_abs_shap" in df.columns:
                df = df.rename(columns={"mean_abs_shap": "mean_importance"})
            elif "importance" in df.columns:
                df = df.rename(columns={"importance": "mean_importance"})

        if "feature" not in df.columns or "mean_importance" not in df.columns:
            logger.warning("SHAP importance file missing required columns: %s", shap_file)
            return None

        return df
    except Exception as e:
        logger.warning(f"Failed to load OOF SHAP importance from {shap_file}: {e}")
        return None


def load_model_essentiality(
    aggregated_dir: Path,
    threshold: str = "95pct",
) -> pd.DataFrame | None:
    """Load drop-column essentiality from optimize-panel results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.
        threshold: Panel threshold to load (e.g., "95pct", "99pct").

    Returns:
        DataFrame with essentiality data (columns: cluster_id, mean_delta_auroc, etc.), or None.
    """
    # Check for essentiality file
    ess_file = (
        aggregated_dir / "optimize_panel" / "essentiality" / f"panel_{threshold}_essentiality.csv"
    )

    if not ess_file.exists():
        # Try alternative: single drop_column_validation.csv
        ess_file = aggregated_dir / "optimize_panel" / "drop_column_validation.csv"

    if not ess_file.exists():
        logger.debug(
            f"No essentiality file found in {aggregated_dir / 'optimize_panel'}. "
            f"Essentiality signal will be absent from composite ranking. "
            f"This is expected unless a drop-column validation step has been run."
        )
        return None

    try:
        df = pd.read_csv(ess_file)
        return df
    except Exception as e:
        logger.warning(f"Failed to load essentiality from {ess_file}: {e}")
        return None


def _run_multimodel_essentiality_validation(
    model_dirs: dict[str, Path],
    split_dirs: list[Path],
    split_indices_dir: Path,
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    y_all: pd.Series | None,
    panel_features: list[str],
    resolved_cols: dict,
    scenario: str,
    essentiality_dir: Path,
    essentiality_corr_threshold: float,
    include_brier: bool = True,
    include_pr_auc: bool = True,
    refit_mode: str = "fixed",
    retune_n_trials: int = 20,
    retune_inner_folds: int = 3,
    n_jobs: int = 1,
) -> dict:
    """Run drop-column essentiality validation for all models and aggregate.

    Model-specific cluster importance reveals:
    - Universally essential clusters (important in all/most models) - highest confidence
    - Model-specific clusters (important in some models only) - model architecture differences
    - Uncertainty estimates (std dev across models)

    Args:
        model_dirs: Dict of model_name -> aggregated_dir_path
        split_dirs: List of split directories (from first model, for reference)
        split_indices_dir: Directory containing shared train_idx_*.csv and val_idx_*.csv files
        df: Full data
        df_train: Training subset
        y_all: Binary target vector
        panel_features: Panel protein features
        resolved_cols: Resolved column metadata
        scenario: Scenario name (e.g., 'IncidentPlusPrevalent')
        essentiality_dir: Output directory for essentiality results
        essentiality_corr_threshold: Correlation threshold for clustering
        include_brier: Whether to compute delta-Brier
        include_pr_auc: Whether to compute delta-PR-AUC
        refit_mode: "fixed", "retune", or "fixed_retune"
        retune_n_trials: Optuna trials per cluster (retune modes only)
        retune_inner_folds: Inner CV folds for retune

    Returns:
        Dict with essentiality summary including per-model and cross-model statistics
    """
    import json

    import joblib
    import numpy as np

    from ced_ml.features.drop_column import (
        _compute_brier_deltas,
        _compute_pr_auc_deltas,
    )

    logger.info(
        f"Running within-panel essentiality validation for {len(model_dirs)} models "
        f"(refit_mode={refit_mode})"
    )

    # Keep resolved metadata columns in refits
    metadata_features = resolved_cols.get("numeric_metadata", []) + resolved_cols.get(
        "categorical_metadata", []
    )
    metadata_features = [
        c for c in metadata_features if c in df.columns and c not in panel_features
    ]
    refit_features = panel_features + metadata_features
    logger.info(
        f"Refit features: {len(panel_features)} panel proteins + "
        f"{len(metadata_features)} metadata covariates"
    )

    # Build correlation clusters for the consensus panel
    X_corr = df_train[panel_features]
    corr_matrix = compute_correlation_matrix(X_corr, panel_features, method="spearman")
    adj_graph = build_correlation_graph(corr_matrix, threshold=essentiality_corr_threshold)
    clusters = find_connected_components(adj_graph)
    logger.info(f"Found {len(clusters)} correlation clusters for essentiality validation")

    # Store results: model_name -> {seed: drop_column_df}
    per_model_results = {}

    logger.debug(f"Split indices directory: {split_indices_dir}")

    # Loop through each model
    for model_name, _aggregated_dir in model_dirs.items():
        logger.info(f"\nProcessing model: {model_name}")

        # Discover splits for this specific model
        model_root = _aggregated_dir.parent  # e.g., results/run_ID/ModelName/
        model_splits_dir = model_root / "splits"

        if not model_splits_dir.exists():
            logger.warning(f"  {model_name}: No splits directory found at {model_splits_dir}")
            continue

        model_split_dirs = sorted(model_splits_dir.glob("split_seed*"))
        if not model_split_dirs:
            logger.warning(f"  {model_name}: No split_seed* subdirectories found")
            continue

        # Auto-partition cores: outer seed-level vs inner cluster-level
        import os

        total_jobs = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
        n_seeds = len(model_split_dirs)
        seed_jobs = min(n_seeds, max(1, total_jobs))
        cluster_jobs = max(1, total_jobs // seed_jobs)

        def _essentiality_for_seed(
            split_dir_path: Path,
            *,
            _model_name: str = model_name,
            _cluster_jobs: int = cluster_jobs,
        ) -> dict | None:
            """Run essentiality for a single (model, seed) pair."""
            seed = int(split_dir_path.name.replace("split_seed", ""))
            model_path = split_dir_path / "core" / f"{_model_name}__final_model.joblib"

            if not model_path.exists():
                logger.debug(f"  Seed {seed}: {_model_name} not found, skipping")
                return None

            train_file = split_indices_dir / f"train_idx_{scenario}_seed{seed}.csv"
            val_file = split_indices_dir / f"val_idx_{scenario}_seed{seed}.csv"

            if not train_file.exists() or not val_file.exists():
                logger.debug(
                    f"  Seed {seed}: split indices not found ({train_file.name}, {val_file.name}), skipping"
                )
                return None

            train_idx = pd.read_csv(train_file).squeeze().values
            val_idx = pd.read_csv(val_file).squeeze().values

            try:
                seed_bundle = joblib.load(model_path)
                original_pipeline = seed_bundle.get("model")

                if original_pipeline is None:
                    logger.warning(f"  Seed {seed}: model bundle missing 'model' key, skipping")
                    return None

                if isinstance(original_pipeline, OOFCalibratedModel):
                    original_pipeline = original_pipeline.base_model

                panel_pipeline = clone(original_pipeline)
                _configure_screen_step_for_panel_refit(panel_pipeline, panel_features)

                X_train_seed = df.iloc[train_idx][refit_features]
                y_train_seed = y_all[train_idx]
                X_val_seed = df.iloc[val_idx][refit_features]
                y_val_seed = y_all[val_idx]

                logger.debug(
                    f"  Seed {seed}: refitting {_model_name} on "
                    f"{len(panel_features)} panel proteins"
                )
                panel_pipeline.fit(X_train_seed, y_train_seed)

                cat_cols = resolved_cols.get("categorical_metadata", [])
                cat_cols = [c for c in cat_cols if c in X_train_seed.columns]

                fold_results = compute_drop_column_importance(
                    estimator=panel_pipeline,
                    X_train=X_train_seed,
                    y_train=y_train_seed,
                    X_val=X_val_seed,
                    y_val=y_val_seed,
                    feature_clusters=clusters,
                    random_state=seed,
                    refit_mode=refit_mode,
                    model_name=_model_name,
                    cat_cols=cat_cols,
                    retune_n_trials=retune_n_trials,
                    retune_inner_folds=retune_inner_folds,
                    n_jobs=_cluster_jobs,
                )

                result = {"drop_column": fold_results}

                if include_brier:
                    result["brier"] = _compute_brier_deltas(
                        model=panel_pipeline,
                        X_train=X_train_seed,
                        y_train=y_train_seed,
                        X_val=X_val_seed,
                        y_val=y_val_seed,
                        feature_clusters=clusters,
                        random_state=seed,
                    )

                if include_pr_auc:
                    result["pr_auc"] = _compute_pr_auc_deltas(
                        model=panel_pipeline,
                        X_train=X_train_seed,
                        y_train=y_train_seed,
                        X_val=X_val_seed,
                        y_val=y_val_seed,
                        feature_clusters=clusters,
                        random_state=seed,
                    )

                logger.debug(f"  Seed {seed}: completed essentiality for {_model_name}")
                return result

            except Exception as e:
                logger.warning(f"  Seed {seed}: error processing {_model_name}: {e}")
                return None

        # Dispatch seed-level parallelism
        if seed_jobs > 1 and n_seeds > 1:
            from joblib import Parallel, delayed

            logger.info(
                f"  Parallel essentiality: {seed_jobs} seed jobs x "
                f"{cluster_jobs} cluster jobs/seed ({total_jobs} total cores)"
            )
            raw_results = Parallel(n_jobs=seed_jobs)(
                delayed(_essentiality_for_seed)(sd) for sd in model_split_dirs
            )
        else:
            raw_results = [_essentiality_for_seed(sd) for sd in model_split_dirs]

        # Unpack results
        drop_column_results_per_fold = []
        brier_deltas_per_fold = []
        pr_auc_deltas_per_fold = []
        for r in raw_results:
            if r is None:
                continue
            drop_column_results_per_fold.append(r["drop_column"])
            if "brier" in r:
                brier_deltas_per_fold.append(r["brier"])
            if "pr_auc" in r:
                pr_auc_deltas_per_fold.append(r["pr_auc"])

        # Aggregate results for this model across folds
        if drop_column_results_per_fold:
            model_df = aggregate_drop_column_results(drop_column_results_per_fold)

            # Add Brier deltas
            if include_brier and brier_deltas_per_fold:
                brier_arr = np.array(brier_deltas_per_fold)
                model_df["mean_delta_brier"] = brier_arr.mean(axis=0)
                model_df["std_delta_brier"] = brier_arr.std(axis=0)

            # Add PR-AUC deltas
            if include_pr_auc and pr_auc_deltas_per_fold:
                pr_auc_arr = np.array(pr_auc_deltas_per_fold)
                model_df["mean_delta_pr_auc"] = pr_auc_arr.mean(axis=0)
                model_df["std_delta_pr_auc"] = pr_auc_arr.std(axis=0)

            per_model_results[model_name] = model_df

            # Save per-model results
            model_ess_dir = essentiality_dir / "per_model"
            model_ess_dir.mkdir(parents=True, exist_ok=True)
            per_model_path = model_ess_dir / f"essentiality_{model_name}.csv"
            model_df.to_csv(per_model_path, index=False)
            logger.info(f"  Saved {model_name} essentiality to {per_model_path}")

        else:
            logger.warning(f"  {model_name}: No folds completed")

    if not per_model_results:
        logger.warning("No models completed essentiality validation")
        return {}

    # Cross-model aggregation
    logger.info(f"\nAggregating essentiality across {len(per_model_results)} models...")

    # Merge all model results on cluster_id.
    # Per-model outputs from aggregate_drop_column_results currently expose
    # `n_features_in_cluster` (not `n_features`), so handle both schemas.
    merged_df = None
    for model_name, model_df in per_model_results.items():
        n_features_col = (
            "n_features"
            if "n_features" in model_df.columns
            else "n_features_in_cluster" if "n_features_in_cluster" in model_df.columns else None
        )
        if n_features_col is None:
            raise ValueError(
                f"Essentiality dataframe for {model_name} missing feature-count column. "
                "Expected one of: n_features, n_features_in_cluster"
            )

        if merged_df is None:
            merged_df = model_df[["cluster_id", n_features_col, "mean_delta_auroc"]].copy()
            merged_df = merged_df.rename(columns={n_features_col: "n_features"})
            merged_df = merged_df.rename(columns={"mean_delta_auroc": f"delta_auroc_{model_name}"})
        else:
            # Right join to keep all clusters
            merged_df = merged_df.merge(
                model_df[["cluster_id", "mean_delta_auroc"]].rename(
                    columns={"mean_delta_auroc": f"delta_auroc_{model_name}"}
                ),
                on="cluster_id",
                how="left",
            )

    # Compute cross-model statistics
    delta_auroc_cols = [c for c in merged_df.columns if c.startswith("delta_auroc_")]

    merged_df["n_models_with_importance"] = merged_df[delta_auroc_cols].notna().sum(axis=1)
    merged_df["mean_delta_auroc_cross_model"] = merged_df[delta_auroc_cols].mean(axis=1)
    merged_df["std_delta_auroc_cross_model"] = merged_df[delta_auroc_cols].std(axis=1)
    merged_df["max_delta_auroc_cross_model"] = merged_df[delta_auroc_cols].max(axis=1)
    merged_df["min_delta_auroc_cross_model"] = merged_df[delta_auroc_cols].min(axis=1)

    # Identify universally essential clusters (high importance in most models)
    n_models = len(per_model_results)
    majority_threshold = np.ceil(n_models * 0.5)  # At least 50% of models
    merged_df["is_universal"] = merged_df["n_models_with_importance"] >= majority_threshold

    # Sort by cross-model importance
    merged_df = merged_df.sort_values(
        "mean_delta_auroc_cross_model", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # Save cross-model essentiality
    cross_model_path = essentiality_dir / "cross_model_essentiality.csv"
    merged_df.to_csv(cross_model_path, index=False)
    logger.info(f"Saved cross-model essentiality to {cross_model_path}")

    # Create summary
    universal_clusters = merged_df[merged_df["is_universal"]]
    model_specific = merged_df[~merged_df["is_universal"]]

    summary = {
        "validation_type": "multimodel_within_panel",
        "refit_mode": refit_mode,
        "n_models": n_models,
        "models_used": list(per_model_results.keys()),
        "panel_size": len(panel_features),
        "n_clusters": len(merged_df),
        "n_universal_clusters": len(universal_clusters),
        "n_model_specific_clusters": len(model_specific),
        "mean_delta_auroc_cross_model": float(merged_df["mean_delta_auroc_cross_model"].mean()),
        "max_delta_auroc_cross_model": float(merged_df["mean_delta_auroc_cross_model"].max()),
        "cross_model_std": float(merged_df["std_delta_auroc_cross_model"].mean()),
        "top_cluster_id": str(merged_df.iloc[0]["cluster_id"]),
        "top_cluster_delta_auroc": float(merged_df.iloc[0]["mean_delta_auroc_cross_model"]),
        "top_cluster_n_models": int(merged_df.iloc[0]["n_models_with_importance"]),
        "top_cluster_is_universal": bool(merged_df.iloc[0]["is_universal"]),
    }

    # Per-model summary statistics
    model_summary = {}
    for model_name in per_model_results.keys():
        col = f"delta_auroc_{model_name}"
        if col in merged_df.columns:
            model_data = merged_df[col].dropna()
            model_summary[model_name] = {
                "mean_delta_auroc": float(model_data.mean()),
                "max_delta_auroc": float(model_data.max()),
                "n_clusters_evaluated": int(len(model_data)),
            }

    summary["per_model_summary"] = model_summary

    # Save summary
    summary_path = essentiality_dir / "multimodel_essentiality_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved cross-model essentiality summary to {summary_path}")

    # Print key findings
    logger.info(
        f"  Universal clusters (>={majority_threshold:.0f}/{n_models} models): {len(universal_clusters)}"
    )
    logger.info(f"  Model-specific clusters: {len(model_specific)}")
    logger.info(f"  Mean cross-model delta AUROC: {summary['mean_delta_auroc_cross_model']:+.4f}")
    logger.info(f"  Cross-model std dev: {summary['cross_model_std']:.4f}")

    return summary


def run_consensus_panel(
    run_id: str,
    infile: str | None = None,
    split_dir: str | None = None,
    stability_threshold: float = 0.90,
    corr_threshold: float = 0.85,
    target_size: int = 25,
    rra_method: str = "geometric_mean",
    ranking_signal: str = "oof_importance",
    shap_explicit_normalization: bool = False,
    outdir: str | None = None,
    log_level: int | None = None,
    require_significance: bool = False,
    significance_alpha: float = 0.05,
    min_significant_models: int = 2,
    rra_significance: bool = False,
    rra_significance_perms: int = 10_000,
    rra_significance_alpha: float = 0.05,
    rra_universe_size: int | None = None,
    run_essentiality: bool = True,
    essentiality_corr_threshold: float = 0.75,
    include_brier: bool = True,
    include_pr_auc: bool = True,
    essentiality_refit_mode: str = "fixed",
    essentiality_retune_n_trials: int = 20,
    essentiality_retune_inner_folds: int = 3,
    n_jobs: int = 1,
) -> ConsensusResult:
    """Run consensus panel generation from multiple models.

    Three-step workflow:
        1. Per-model ranking: stability as hard filter, OOF importance as rank
        2. Cross-model RRA: geometric mean rank aggregation
        3. Post-hoc drop-column on final panel (interpretation, not ranking input)

    Args:
        run_id: Run ID to process.
        infile: Input data file (auto-detected if None).
        split_dir: Split directory (auto-detected if None).
        stability_threshold: Minimum selection fraction for stable proteins.
        corr_threshold: Correlation threshold for clustering.
        target_size: Target panel size.
        rra_method: RRA aggregation method.
        ranking_signal: Signal used for per-model ranking ("oof_importance" or "oof_shap").
        shap_explicit_normalization: Apply explicit SHAP normalization for cross-model
            aggregation when ranking_signal is "oof_shap".
        outdir: Output directory (default: results/consensus_panel/run_<RUN_ID>).
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        require_significance: Whether to filter models by permutation test significance.
        significance_alpha: P-value threshold for significance filtering.
        min_significant_models: Minimum number of significant models required.
        rra_significance: Run permutation null on RRA consensus scores. When True,
            overrides target_size with the count of significant proteins.
        rra_significance_perms: Number of permutations for RRA significance test.
        rra_significance_alpha: BH-corrected alpha for RRA significance.
        rra_universe_size: Total features in the original search space (e.g. 2920).
            When set, calibrates the permutation null and BH correction against
            the full feature universe instead of the pre-filtered subset.
        run_essentiality: Whether to run within-panel essentiality validation (default True).
            Refits a model on only the consensus panel features and runs drop-column
            to measure each cluster's contribution. This is a post-hoc
            interpretation artifact, not an input to panel selection.
        essentiality_corr_threshold: Correlation threshold for clustering in drop-column (default 0.75).
        include_brier: Whether to compute delta-Brier in post-hoc essentiality (default True).
        include_pr_auc: Whether to compute delta-PR-AUC in post-hoc essentiality (default True).
        essentiality_refit_mode: Refit strategy for essentiality validation.
            "fixed" (clone, fast), "retune" (Optuna re-optimization),
            or "fixed_retune" (both side-by-side).
        essentiality_retune_n_trials: Optuna trials per cluster (retune modes).
        essentiality_retune_inner_folds: Inner CV folds for retune.

    Returns:
        ConsensusResult with final panel and intermediate data.

    Raises:
        FileNotFoundError: If required files not found.
        ValueError: If insufficient models or proteins.
    """
    # Setup logging
    from ced_ml.utils.logging import setup_command_logger

    if ranking_signal not in {"oof_importance", "oof_shap"}:
        raise ValueError(
            f"ranking_signal must be 'oof_importance' or 'oof_shap', got '{ranking_signal}'"
        )

    if ranking_signal != "oof_shap":
        shap_explicit_normalization = False

    if log_level is None:
        log_level = logging.INFO

    # Auto-file-logging
    logger = setup_command_logger(
        command="consensus-panel",
        log_level=log_level,
        outdir=outdir or "results",
        run_id=run_id,
        logger_name=f"ced_ml.consensus_panel.{run_id}",
    )
    logger.info(f"Consensus panel generation started for run_id={run_id}")

    # Determine results root (CED_RESULTS_DIR env var for testability)
    import os

    from ced_ml.utils.paths import get_project_root

    results_dir_env = os.environ.get("CED_RESULTS_DIR")
    if results_dir_env:
        results_dir = Path(results_dir_env)
    else:
        results_dir = get_project_root() / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Discover models with aggregated results
    logger.info("Discovering models with aggregated stability results...")
    model_dirs = discover_models_with_aggregated_results(
        run_id=run_id,
        results_dir=results_dir,
        skip_ensemble=True,
    )

    logger.info(f"Found {len(model_dirs)} models: {list(model_dirs.keys())}")

    if require_significance and min_significant_models > len(model_dirs):
        raise ValueError(
            f"Consensus requires at least {min_significant_models} significant model(s), "
            f"but only {len(model_dirs)} model(s) were discovered: {list(model_dirs.keys())}. "
            "Lower significance.min_significant_models in consensus_panel.yaml "
            "(or pass --min-significant-models), or train additional base models."
        )

    # Significance filtering
    if require_significance:
        run_path = results_dir / f"run_{run_id}"
        sig_df = load_aggregated_significance(run_path)

        if sig_df is not None:
            # Filter to significant models only
            sig_models = sig_df[sig_df["empirical_p_value"] < significance_alpha]["model"].tolist()

            original_count = len(model_dirs)
            model_dirs = {m: p for m, p in model_dirs.items() if m in sig_models}

            if len(model_dirs) < original_count:
                skipped = original_count - len(model_dirs)
                logger.info(
                    f"Significance filtering: {skipped} model(s) excluded "
                    f"(p >= {significance_alpha})"
                )

            if len(model_dirs) < min_significant_models:
                raise ValueError(
                    f"Only {len(model_dirs)} significant model(s) found "
                    f"(need {min_significant_models}). "
                    f"Significant models: {list(model_dirs.keys())}\n"
                    f"Run permutation tests first: ced permutation-test --run-id {run_id}"
                )

            logger.info(
                f"Proceeding with {len(model_dirs)} significant model(s): {list(model_dirs.keys())}"
            )
        else:
            logger.warning(
                f"No significance data found for run {run_id}. "
                f"Run permutation tests first: ced permutation-test --run-id {run_id}\n"
                f"Proceeding without significance filtering."
            )

    # Auto-detect data paths if not provided
    if not infile or not split_dir:
        auto_infile, auto_split_dir = auto_detect_data_paths(run_id, results_dir)
        if not infile:
            infile = auto_infile
        if not split_dir:
            split_dir = auto_split_dir

    if not infile:
        raise ValueError(
            f"Could not auto-detect input file for run_id={run_id}.\n"
            f"Searched in: {results_dir}\n"
            f"Found models: {list(model_dirs.keys())}\n"
            f"Please provide --infile explicitly."
        )
    if not split_dir:
        raise ValueError(
            f"Could not auto-detect split directory for run_id={run_id}.\n"
            f"Searched in: {results_dir}\n"
            f"Found models: {list(model_dirs.keys())}\n"
            f"Please provide --split-dir explicitly."
        )

    logger.info(f"Input file: {infile}")
    logger.info(f"Split directory: {split_dir}")

    # Load stability data for each model
    logger.info("Loading ranking data from each model...")
    model_stability = {}
    model_oof_importance = {}

    for model_name, aggregated_dir in model_dirs.items():
        # Load ranking signal first when SHAP is required.
        if ranking_signal == "oof_shap":
            oof_df = load_model_oof_shap_importance(aggregated_dir, model_name)
        else:
            oof_df = load_model_oof_importance(aggregated_dir, model_name)

        if ranking_signal == "oof_shap" and oof_df is None:
            logger.warning(
                f"  {model_name}: SHAP ranking requested but no OOF SHAP importance file found. "
                "Model excluded from consensus."
            )
            continue

        # Load stability (always needed for models retained in consensus)
        stability_df = load_model_stability(aggregated_dir, stability_threshold=0.0)
        model_stability[model_name] = stability_df

        n_stable = (stability_df["selection_fraction"] >= stability_threshold).sum()

        has_signal = oof_df is not None
        if has_signal:
            model_oof_importance[model_name] = oof_df

        logger.info(
            f"  {model_name}: {len(stability_df)} total proteins, "
            f"{n_stable} stable (>={stability_threshold}), "
            f"{ranking_signal}={'yes' if has_signal else 'no'}"
        )

    if ranking_signal == "oof_shap" and len(model_stability) < 2:
        raise ValueError(
            "SHAP cross-model aggregation requires at least 2 models with "
            "aggregated OOF SHAP importance files. Run 'ced aggregate-splits' "
            "after SHAP-enabled training."
        )

    # Keep only models that contributed ranking inputs.
    model_dirs = {m: p for m, p in model_dirs.items() if m in model_stability}

    # Load training data for correlation computation
    logger.info(f"Loading training data from {infile}...")
    df_raw = read_proteomics_file(infile, validate=True)

    # Get metadata columns from first available model
    first_model = next(iter(model_dirs.keys()))
    first_aggregated = model_dirs[first_model]
    run_dir = first_aggregated.parent

    # Find representative split for metadata
    split_dirs = sorted(run_dir.glob("splits/split_seed*"))
    if not split_dirs:
        split_dirs = sorted(run_dir.glob("split_seed*"))
    if not split_dirs:
        raise FileNotFoundError(f"No split directories found in {run_dir}")

    representative_split = split_dirs[0]
    representative_seed = int(representative_split.name.replace("split_seed", ""))

    # Load model bundle for metadata (only need resolved_columns + scenario)
    model_path = representative_split / "core" / f"{first_model}__final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    import joblib

    try:
        bundle = joblib.load(model_path)
    except TypeError as exc:
        # Pickle deserialization can fail when pandas versions differ between
        # the environment that trained the model and the one loading it
        # (e.g. StringDtype signature changes).  We only need two lightweight
        # metadata keys, so attempt a selective unpickle.
        logger.warning(f"Full bundle load failed ({exc}); attempting metadata-only extraction")
        bundle = _extract_bundle_metadata(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentPlusPrevalent")
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    # Apply row filters
    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} -> {filter_stats['n_out']:,} rows")

    # Load train indices
    split_path = Path(split_dir)
    train_file = split_path / f"train_idx_{scenario}_seed{representative_seed}.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Train indices not found: {train_file}\n"
            f"Expected scenario-specific format: train_idx_{{scenario}}_seed{{N}}.csv\n"
            f"Scenario: {scenario}, Seed: {representative_seed}\n"
            f"Run 'ced save-splits' to generate splits with the modern format."
        )

    train_idx = pd.read_csv(train_file).squeeze().values
    df_train = df.iloc[train_idx].copy()

    logger.info(f"Training data: {len(df_train)} samples")

    # --- RRA significance test (optional, overrides target_size) ---
    rra_sig_result = None
    if rra_significance:
        from ced_ml.features.consensus.ranking import compute_per_model_ranking
        from ced_ml.features.consensus.significance import rra_permutation_test

        logger.info("Computing per-model rankings for RRA significance test...")
        sig_rankings = {}
        for model_name, stability_df in model_stability.items():
            stable_df = stability_df[
                stability_df["selection_fraction"] >= stability_threshold
            ].copy()
            if len(stable_df) == 0:
                continue
            oof_df = model_oof_importance.get(model_name) if model_oof_importance else None
            sig_rankings[model_name] = compute_per_model_ranking(
                stability_df=stable_df,
                stability_col="selection_fraction",
                oof_importance_df=oof_df,
            )

        if len(sig_rankings) >= 2:
            rra_sig_result = rra_permutation_test(
                per_model_rankings=sig_rankings,
                n_perms=rra_significance_perms,
                alpha=rra_significance_alpha,
                universe_size=rra_universe_size,
            )
            n_significant = int(rra_sig_result["significant"].sum())
            logger.info(
                f"RRA significance: {n_significant} proteins significant "
                f"(BH alpha={rra_significance_alpha})"
            )
            if n_significant > 0:
                target_size = n_significant
                logger.info(f"Overriding target_size to {target_size}")
            else:
                logger.warning(
                    "No proteins reached significance — keeping original "
                    f"target_size={target_size}"
                )
        else:
            logger.warning("RRA significance requires >= 2 models with rankings; skipping")

    # Build consensus panel
    logger.info("Building consensus panel...")
    result = build_consensus_panel(
        model_stability=model_stability,
        df_train=df_train,
        stability_threshold=stability_threshold,
        corr_threshold=corr_threshold,
        target_size=target_size,
        rra_method=rra_method,
        ranking_signal=ranking_signal,
        shap_explicit_normalization=shap_explicit_normalization,
        model_oof_importance=model_oof_importance,
    )

    # Save results
    if outdir is None:
        outdir = results_dir / f"run_{run_id}" / "consensus"
    else:
        outdir = Path(outdir)

    _paths = save_consensus_results(result, outdir)  # noqa: F841

    # Save RRA significance artifact
    if rra_sig_result is not None:
        rra_sig_path = outdir / "rra_significance.csv"
        rra_sig_result.to_csv(rra_sig_path, index=False)
        logger.info(f"Saved RRA significance results to {rra_sig_path}")

    # --- Within-panel essentiality validation (post-hoc interpretation) ---
    # After the consensus panel is built, refit all models on the panel proteins
    # (plus resolved metadata covariates) and run drop-column to measure each
    # cluster's contribution. Aggregate across models to identify:
    # - Universally essential clusters (important in all/most models)
    # - Model-specific clusters (important in some models only)
    # This is a validation/interpretation artifact, NOT an input to selection.
    essentiality_summary = None
    if run_essentiality and len(result.final_panel) > 0:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("Running within-panel essentiality validation for all models...")
            logger.info("=" * 60)

            # Create binary target vector
            if TARGET_COL not in df.columns:
                raise ValueError(f"Target column '{TARGET_COL}' not found in data")

            positive_label = get_positive_label(scenario)
            y_all = (df[TARGET_COL] == positive_label).astype(int).values
            logger.info(f"Target vector: {y_all.sum()} positive cases out of {len(y_all)} samples")

            # Create essentiality output directory
            essentiality_dir = outdir / "essentiality"
            essentiality_dir.mkdir(parents=True, exist_ok=True)

            # Get consensus panel features
            panel_features = [p for p in result.final_panel if p in df.columns]
            logger.info(f"Validating {len(panel_features)} panel features")

            # Run multi-model essentiality validation
            essentiality_summary = _run_multimodel_essentiality_validation(
                model_dirs=model_dirs,
                split_dirs=split_dirs,
                split_indices_dir=Path(split_dir),
                df=df,
                df_train=df_train,
                y_all=y_all,
                panel_features=panel_features,
                resolved_cols=resolved_cols,
                scenario=scenario,
                essentiality_dir=essentiality_dir,
                essentiality_corr_threshold=essentiality_corr_threshold,
                include_brier=include_brier,
                include_pr_auc=include_pr_auc,
                refit_mode=essentiality_refit_mode,
                retune_n_trials=essentiality_retune_n_trials,
                retune_inner_folds=essentiality_retune_inner_folds,
                n_jobs=n_jobs,
            )

        except Exception as e:
            logger.warning(f"Within-panel essentiality validation failed: {e}", exc_info=True)
            logger.info("Continuing without essentiality validation results")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Consensus Panel Generation Complete")
    print(f"{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(model_dirs.keys())}")
    print("\nParameters:")
    print(f"  Ranking signal: {ranking_signal}")
    print(
        "  SHAP explicit normalization: "
        f"{'on' if ranking_signal == 'oof_shap' and shap_explicit_normalization else 'off'}"
    )
    print(f"  Stability threshold: {stability_threshold}")
    print(f"  Correlation threshold: {corr_threshold}")
    print(f"  Target size: {target_size}")
    print(f"  RRA method: {rra_method}")

    print("\nResults:")
    print(f"  Total proteins across models: {len(result.consensus_ranking)}")
    print(f"  Clusters after correlation pruning: {result.metadata['results']['n_clusters']}")
    print(f"  Final panel size: {len(result.final_panel)}")

    if require_significance:
        print("\nSignificance filtering:")
        print(f"  Required alpha: {significance_alpha}")
        print(f"  Minimum models: {min_significant_models}")
        print(f"  Significant models used: {list(model_dirs.keys())}")

    if rra_sig_result is not None:
        n_sig = int(rra_sig_result["significant"].sum())
        n_total = len(rra_sig_result)
        print(f"\nRRA Permutation Significance ({rra_significance_perms} permutations):")
        print(f"  Significant proteins: {n_sig}/{n_total} (BH alpha={rra_significance_alpha})")
        print(f"  Target size override: {target_size}")
        # Show top significant proteins
        sig_proteins = rra_sig_result[rra_sig_result["significant"]].head(10)
        if not sig_proteins.empty:
            print("  Top significant proteins:")
            for _, row in sig_proteins.iterrows():
                print(
                    f"    {row['protein']}: RRA={row['observed_rra']:.4f}, "
                    f"p={row['bh_adjusted_p']:.4f}"
                )

    print("\nTop 10 proteins in consensus panel:")
    for i, protein in enumerate(result.final_panel[:10], 1):
        protein_row = result.consensus_ranking[result.consensus_ranking["protein"] == protein]
        score = protein_row["consensus_score"].iloc[0]
        n_models = protein_row["n_models_present"].iloc[0]
        agreement = protein_row["agreement_strength"].iloc[0]
        print(
            f"  {i:2d}. {protein} (score: {score:.4f}, "
            f"{n_models}/{len(model_dirs)} models, agreement: {agreement:.2f})"
        )

    # Print uncertainty summary
    if "uncertainty" in result.metadata:
        unc = result.metadata["uncertainty"]
        print("\nUncertainty Summary:")
        print(f"  Mean agreement strength: {unc['mean_agreement_strength']:.2f}")
        print(f"  Min agreement strength: {unc['min_agreement_strength']:.2f}")
        print(f"  Mean rank CV: {unc['mean_rank_cv']:.3f}")
        print(f"  Max rank CV: {unc['max_rank_cv']:.3f}")
        print(
            f"  Proteins in all models: {unc['proteins_in_all_models']}/{len(result.final_panel)}"
        )
        majority = unc["proteins_in_majority_models"]
        print(f"  Proteins in majority: {majority}/{len(result.final_panel)}")

    # Print essentiality summary if available
    if essentiality_summary:
        if essentiality_summary.get("validation_type") == "multimodel_within_panel":
            print("\nWithin-Panel Essentiality Validation (Multi-Model):")
            print(f"  Models evaluated: {', '.join(essentiality_summary['models_used'])}")
            print(f"  Clusters validated: {essentiality_summary['n_clusters']}")
            print(
                f"  Universal clusters (>50% models): {essentiality_summary['n_universal_clusters']}"
            )
            print(f"  Model-specific clusters: {essentiality_summary['n_model_specific_clusters']}")
            print(
                f"  Mean delta AUROC (cross-model): {essentiality_summary['mean_delta_auroc_cross_model']:+.4f}"
            )
            print(f"  Cross-model std dev: {essentiality_summary['cross_model_std']:.4f}")
            print(
                f"  Max delta AUROC (cross-model): {essentiality_summary['max_delta_auroc_cross_model']:+.4f}"
            )
            print(
                f"  Top cluster: {essentiality_summary['top_cluster_id']} (delta AUROC={essentiality_summary['top_cluster_delta_auroc']:+.4f}, found in {essentiality_summary['top_cluster_n_models']}/{essentiality_summary['n_models']} models)"
            )
            if essentiality_summary["top_cluster_is_universal"]:
                print("    -> Universal cluster (found in >50% of models)")
            print("\n  Per-model summary:")
            for model, stats in essentiality_summary.get("per_model_summary", {}).items():
                print(
                    f"    {model}: mean delta AUROC={stats['mean_delta_auroc']:+.4f}, max={stats['max_delta_auroc']:+.4f}"
                )
        else:
            print(
                f"\nWithin-Panel Essentiality Validation (model: {essentiality_summary['model_used']}):"
            )
            print(f"  Clusters validated: {essentiality_summary['n_clusters']}")
            print(f"  Folds validated: {essentiality_summary['n_folds_validated']}")
            print(f"  Mean delta AUROC: {essentiality_summary['mean_delta_auroc']:+.4f}")
            print(f"  Max delta AUROC: {essentiality_summary['max_delta_auroc']:+.4f}")
            if "mean_delta_pr_auc" in essentiality_summary:
                print(f"  Mean delta PR-AUC: {essentiality_summary['mean_delta_pr_auc']:+.4f}")
            if "mean_delta_brier" in essentiality_summary:
                print(f"  Mean delta Brier: {essentiality_summary['mean_delta_brier']:+.4f}")
            print(
                f"  Top cluster: {essentiality_summary['top_cluster_id']} "
                f"(delta AUROC={essentiality_summary['top_cluster_delta_auroc']:+.4f})"
            )

    print(f"\nOutput saved to: {outdir}")
    print("  - final_panel.txt (for --fixed-panel)")
    print("  - final_panel.csv (with uncertainty metrics)")
    print("  - consensus_ranking.csv (all proteins with uncertainty)")
    print("  - uncertainty_summary.csv (focused uncertainty report)")
    print("  - per_model_rankings.csv")
    print("  - correlation_clusters.csv")
    print("  - consensus_metadata.json")
    if rra_sig_result is not None:
        print("  - rra_significance.csv (permutation test results)")
    if essentiality_summary:
        if essentiality_summary.get("validation_type") == "multimodel_within_panel":
            print("  - essentiality/per_model/essentiality_*.csv (per-model results)")
            print("  - essentiality/cross_model_essentiality.csv (aggregated)")
            print("  - essentiality/multimodel_essentiality_summary.json")
        else:
            print("  - essentiality/within_panel_essentiality.csv")
            print("  - essentiality/essentiality_summary.json")

    print("\nNext step: Validate with new split seed:")
    print(f"  ced train --model LR_EN --fixed-panel {outdir}/final_panel.txt --split-seed 10")
    print(f"{'=' * 60}\n")

    logger.info("Consensus panel generation completed successfully")

    return result
