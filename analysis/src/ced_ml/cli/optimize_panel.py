"""CLI implementation for panel size optimization via RFE.

This module provides the `ced optimize-panel` command for finding minimum
viable protein panels through Recursive Feature Elimination.

USAGE MODES:
    1. Single model: --model-path /path/to/model.joblib
    2. Run-based (all base models): --run-id 20260127_115115
    3. Run-based (specific model): --run-id 20260127_115115 --model LR_EN

ENSEMBLE HANDLING:
    ENSEMBLE models are automatically SKIPPED when using --run-id (they operate
    on 4 meta-features, not 2,920 proteins). RFE runs on all base models instead.

WORKFLOW:
    1. Train base models across splits (ced train)
    2. Train ensemble meta-learner (ced train-ensemble) [optional]
    3. Optimize panel size:
       - For single model: ced optimize-panel --model-path ...
       - For all base models: ced optimize-panel --run-id <RUN_ID>
    4. Retrain on optimized panel (see FEATURE_SELECTION.md)

NOTE: For consensus panels from multiple base models, use the aggregation
pipeline (ced aggregate-splits) which automatically computes feature stability
and consensus panels via frequency-based voting.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    save_rfe_results,
)
from ced_ml.plotting.panel_curve import plot_feature_ranking, plot_pareto_curve

logger = logging.getLogger(__name__)


def load_aggregated_significance(run_dir: Path) -> pd.DataFrame | None:
    """Load pre-computed aggregated significance results for all models in a run.

    Expects aggregated_significance.csv files produced by the permutation
    aggregation step (``ced permutation-test --run-id <ID> --model <MODEL>``).

    Args:
        run_dir: Path to run directory (results/run_{ID}/)

    Returns:
        DataFrame with columns [model, empirical_p_value, significant, ...] or None if not found.
    """
    sig_files = list(run_dir.glob("*/significance/aggregated_significance.csv"))
    if not sig_files:
        logger.warning(
            "No aggregated_significance.csv found in %s. "
            "Run 'ced permutation-test --run-id <ID> --model <MODEL>' to produce it.",
            run_dir,
        )
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
            logger.warning("Failed to load %s: %s", f, e)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def _extract_model_name_from_aggregated_path(results_path: Path) -> str:
    """Extract model name from aggregated results path.

    Expected: .../run_{id}/{MODEL}/aggregated -> returns MODEL
    """
    if results_path.name == "aggregated":
        return results_path.parent.name
    return results_path.parent.parent.name


def discover_models_by_run_id(
    run_id: str,
    results_dir: Path | str,
    model_filter: str | None = None,
) -> dict[str, Path]:
    """Discover models with aggregated results for a given run_id.

    NOTE: This is a compatibility wrapper. New code should use
    discover_models_with_aggregated_results from ced_ml.cli.discovery.

    Args:
        run_id: Run ID to search for (e.g., "20260127_115115")
        results_dir: Root results directory (e.g., ../results)
        model_filter: Optional model name to filter by (e.g., "LR_EN")

    Returns:
        Dictionary mapping model names to their aggregated results directories.
        Example: {"LR_EN": Path("../results/LR_EN/run_20260127_115115/aggregated")}

    Raises:
        FileNotFoundError: If results_dir does not exist
    """
    from ced_ml.cli.discovery import discover_models_with_aggregated_results

    # Validate results_dir exists (original behavior)
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    try:
        return discover_models_with_aggregated_results(
            run_id=run_id,
            results_dir=results_dir,
            model_filter=model_filter,
            skip_ensemble=True,  # This function always skips ENSEMBLE
            require_stability=True,
        )
    except FileNotFoundError:
        # Return empty dict for "no models found" (original behavior for missing run_id)
        return {}


def run_optimize_panel_single_seed(
    results_dir: str | Path,
    infile: str,
    split_dir: str,
    seed: int,
    model_name: str | None = None,
    stability_threshold: float = 0.90,
    start_size: int | None = None,
    min_size: int = 5,
    min_auroc_frac: float = 0.90,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    log_level: int | None = None,
    retune_n_trials: int = 60,
    retune_n_jobs: int = 1,
    corr_aware: bool = True,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
    rfe_tune_spaces: dict[str, dict[str, dict]] | None = None,
) -> RFEResult:
    """Run panel optimization RFE for a single split seed and save the result.

    This is the HPC-parallelizable unit: one (model, seed) pair. The result
    is saved as a joblib file under optimize_panel/seeds/seed_{N}/rfe_result.joblib
    for later aggregation by run_optimize_panel_aggregated().

    Args:
        results_dir: Path to model's aggregated results directory.
        infile: Path to input data file.
        split_dir: Directory containing split indices.
        seed: The split seed to run RFE for.
        model_name: Model name (auto-detected if None).
        stability_threshold: Minimum selection frequency (default: 0.90).
        start_size: Cap starting panel to top N proteins by selection frequency.
            None or 0 means no cap (use all stable proteins).
        min_size: Minimum panel size.
        min_auroc_frac: Early stop threshold.
        cv_folds: CV folds for RFE.
        step_strategy: Elimination strategy.
        log_level: Logging level.
        retune_n_trials: Optuna trials per evaluation point.
        retune_n_jobs: Parallel jobs for Optuna CV.

    Returns:
        RFEResult for this seed.
    """
    from ced_ml.cli.panel_optimization_helpers import (
        extract_retune_cv_folds,
        load_and_filter_data,
        load_model_bundle,
        load_stability_panel,
        run_rfe_for_seed,
    )
    from ced_ml.utils.logging import log_hpc_context, setup_command_logger

    results_path = Path(results_dir)

    if not model_name:
        model_name = _extract_model_name_from_aggregated_path(results_path)

    if log_level is None:
        log_level = logging.INFO

    _run_id = None
    _run_level_dir = results_path
    for parent in [results_path] + list(results_path.parents):
        if parent.name.startswith("run_"):
            _run_id = parent.name[4:]
            _run_level_dir = parent
            break

    logger = setup_command_logger(
        command="optimize-panel",
        log_level=log_level,
        outdir=_run_level_dir,
        run_id=_run_id,
        model=model_name,
        split_seed=seed,
        logger_name=f"ced_ml.optimize_panel.{model_name}_seed{seed}",
    )
    logger.info(f"Single-seed panel optimization: model={model_name}, seed={seed}")
    logger.info(f"Run ID: {_run_id or 'unknown'}")
    log_hpc_context(logger)

    stability_file = results_path / "panels" / "feature_stability_summary.csv"
    importance_file = results_path / "importance" / f"oof_importance__{model_name}.csv"
    stable_proteins, selection_freq = load_stability_panel(
        stability_file, stability_threshold, start_size, importance_file=importance_file
    )

    run_dir = results_path.parent
    seed_dir = run_dir / "splits" / f"split_seed{seed}"
    model_path = seed_dir / "core" / f"{model_name}__final_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for seed {seed}: {model_path}")

    bundle = load_model_bundle(model_path)

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentOnly")
    protein_cols = resolved_cols.get("protein_cols", [])
    cat_cols = resolved_cols.get("categorical_metadata", [])
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    retune_cv_folds = extract_retune_cv_folds(bundle.get("config", {}))

    feature_cols = protein_cols + cat_cols + meta_num_cols
    df, feature_cols, protein_cols = load_and_filter_data(
        infile, feature_cols, protein_cols, meta_num_cols, scenario=scenario
    )

    initial_proteins = [p for p in stable_proteins if p in df.columns]
    if len(initial_proteins) < min_size:
        raise ValueError(
            f"Only {len(initial_proteins)} stable proteins available, less than min_size={min_size}"
        )

    filtered_cat_cols = [c for c in cat_cols if c in df.columns]
    filtered_meta_num_cols = [c for c in meta_num_cols if c in df.columns]

    result = run_rfe_for_seed(
        seed=seed,
        model_path=model_path,
        df=df,
        feature_cols=feature_cols,
        split_dir=Path(split_dir),
        scenario=scenario,
        model_name=model_name,
        initial_proteins=initial_proteins,
        cat_cols=filtered_cat_cols,
        meta_num_cols=filtered_meta_num_cols,
        min_size=min_size,
        cv_folds=cv_folds,
        step_strategy=step_strategy,
        min_auroc_frac=min_auroc_frac,
        retune_n_trials=retune_n_trials,
        retune_cv_folds=retune_cv_folds,
        retune_n_jobs=retune_n_jobs,
        corr_aware=corr_aware,
        corr_threshold=corr_threshold,
        corr_method=corr_method,
        selection_freq=selection_freq,
        rfe_tune_spaces=rfe_tune_spaces,
    )

    outdir = results_path / "optimize_panel" / "seeds" / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "rfe_result.joblib"
    joblib.dump(result, result_path)
    logger.info(f"Saved seed {seed} RFE result to {result_path}")

    print(f"Single-seed panel optimization complete: {model_name} seed {seed}")
    print(f"  Max AUROC: {result.max_auroc:.4f}")
    print(f"  Saved to: {result_path}")

    return result


def run_drop_column_validation_for_panels(
    result: RFEResult,
    results_path: Path,
    df: pd.DataFrame,
    split_dir: Path,
    scenario: str,
    model_name: str,
    split_dirs: list[Path],
    cat_cols: list[str],
    meta_num_cols: list[str],
    corr_threshold: float,
    corr_method: str,
    essentiality_corr_threshold: float | None = None,
) -> None:
    """Run drop-column essentiality validation for each recommended panel.

    Args:
        result: RFE result with recommended panels
        results_path: Path to model's aggregated results directory
        df: Scenario-aligned filtered data (same index space as split files)
        split_dir: Root directory containing split indices
        scenario: Data scenario (e.g., "IncidentOnly")
        model_name: Model name
        split_dirs: List of split directories
        cat_cols: Categorical metadata columns
        meta_num_cols: Numeric metadata columns
        corr_threshold: Correlation threshold for RFE pruning (used as fallback)
        corr_method: Correlation method
        essentiality_corr_threshold: Separate correlation threshold for
            essentiality clustering (default: 0.75). Lower than RFE threshold
            to produce meaningful multi-feature clusters for interpretation.
    """
    from sklearn.base import clone

    from ced_ml.cli.consensus_panel import _configure_screen_step_for_panel_refit
    from ced_ml.cli.panel_optimization_helpers import load_split_indices
    from ced_ml.data.schema import TARGET_COL, get_positive_label
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

    # Use dedicated essentiality threshold; fall back to RFE threshold if not set
    ess_corr = essentiality_corr_threshold if essentiality_corr_threshold is not None else 0.75
    if ess_corr != corr_threshold:
        logger.info(
            f"Essentiality clustering uses corr_threshold={ess_corr:.2f} "
            f"(RFE corr_threshold={corr_threshold:.2f})"
        )

    if not result.recommended_panels:
        logger.warning("No recommended panels found, skipping drop-column validation")
        return

    # Reuse scenario-aligned filtered dataframe from optimize-panel workflow
    y_all = (df[TARGET_COL] == get_positive_label(scenario)).astype(int).values

    essentiality_dir = results_path / "optimize_panel" / "essentiality"
    essentiality_dir.mkdir(parents=True, exist_ok=True)

    # Process each recommended panel threshold
    for threshold_name, panel_size in result.recommended_panels.items():
        logger.info(f"\nProcessing {threshold_name} panel (size={panel_size})...")

        # Find the curve point matching this panel size
        matching_points = [p for p in result.curve if p["size"] == panel_size]
        if not matching_points:
            logger.warning(f"  No curve point found for size={panel_size}, skipping")
            continue

        panel_proteins = matching_points[0]["proteins"]
        logger.info(f"  Panel proteins: {len(panel_proteins)}")

        # Cluster proteins by correlation
        feature_cols = panel_proteins + cat_cols + meta_num_cols
        X_all = df[feature_cols]

        if len(panel_proteins) > 1:
            corr_matrix = compute_correlation_matrix(df, panel_proteins, method=corr_method)
            G = build_correlation_graph(corr_matrix, ess_corr)
            clusters = find_connected_components(G)
            logger.info(
                f"  Correlation clustering: {len(panel_proteins)} -> {len(clusters)} clusters"
            )
        else:
            clusters = [[panel_proteins[0]]]

        # Run drop-column across all seeds
        per_seed_results = []
        for seed_dir in split_dirs:
            seed = int(seed_dir.name.replace("split_seed", ""))
            model_path = seed_dir / "core" / f"{model_name}__final_model.joblib"

            if not model_path.exists():
                logger.warning(f"  Seed {seed}: model not found at {model_path}, skipping")
                continue

            # Load split indices
            train_idx, val_idx = load_split_indices(split_dir, scenario, seed)

            # Split data
            X_train = X_all.iloc[train_idx]
            y_train = y_all[train_idx]
            X_val = X_all.iloc[val_idx]
            y_val = y_all[val_idx]

            # Load model, clone, reconfigure, and refit on panel features
            try:
                bundle = joblib.load(model_path)
                original_pipeline = bundle.get("model")

                if original_pipeline is None:
                    logger.warning(f"  Seed {seed}: model bundle missing 'model' key, skipping")
                    continue

                # Unwrap OOFCalibratedModel to get the fittable base model
                if isinstance(original_pipeline, OOFCalibratedModel):
                    original_pipeline = original_pipeline.base_model

                # Clone preserves tuned hyperparameters but produces unfitted estimator
                panel_pipeline = clone(original_pipeline)
                _configure_screen_step_for_panel_refit(panel_pipeline, panel_proteins)

                # Refit on panel features only
                logger.debug(f"  Seed {seed}: refitting on {len(panel_proteins)} panel proteins")
                panel_pipeline.fit(X_train, y_train)

                seed_results = compute_drop_column_importance(
                    estimator=panel_pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    feature_clusters=clusters,
                    random_state=seed,
                )
                per_seed_results.append(seed_results)
                logger.info(f"  Seed {seed}: {len(seed_results)} clusters evaluated")
            except Exception as e:
                logger.warning(f"  Seed {seed}: drop-column failed: {e}")

        if not per_seed_results:
            logger.warning(f"  No successful drop-column results for {threshold_name}, skipping")
            continue

        # Aggregate across seeds
        agg_df = aggregate_drop_column_results(per_seed_results)

        # Add representative column (first protein in each cluster)
        agg_df["representative"] = agg_df["cluster_features"].str.split(",").str[0]

        # Reorder columns for clarity
        cols = [
            "cluster_id",
            "representative",
            "cluster_features",
            "n_features_in_cluster",
            "mean_delta_auroc",
            "std_delta_auroc",
            "min_delta_auroc",
            "max_delta_auroc",
            "n_folds",
            "n_errors",
        ]
        agg_df = agg_df[[c for c in cols if c in agg_df.columns]]

        # Save
        out_file = essentiality_dir / f"panel_{threshold_name}_essentiality.csv"
        agg_df.to_csv(out_file, index=False)
        logger.info(f"  Saved essentiality results to {out_file}")
        logger.info(
            "  Top 3 essential features: " + ", ".join(agg_df.head(3)["representative"].tolist())
        )


def run_optimize_panel_aggregated(
    results_dir: str | Path,
    infile: str,
    split_dir: str,
    model_name: str | None = None,
    stability_threshold: float = 0.90,
    start_size: int | None = None,
    min_size: int = 5,
    min_auroc_frac: float = 0.90,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    outdir: str | None = None,
    log_level: int | None = None,
    n_jobs: int = 1,
    retune_n_trials: int = 60,
    corr_aware: bool = True,
    corr_threshold: float = 0.80,
    corr_method: str = "spearman",
    rfe_tune_spaces: dict[str, dict[str, dict]] | None = None,
    require_significance: bool = False,
    significance_alpha: float = 0.05,
    essentiality_corr_threshold: float | None = None,
) -> RFEResult | None:
    """Run panel optimization using aggregated stability panel.

    Runs RFE independently on each available split seed, then aggregates
    the validation curves (mean + cross-seed 95% CI). This produces a
    more robust Pareto curve than single-seed optimization.

    At each evaluation point, hyperparameters are re-tuned via a quick
    Optuna search so each Pareto curve point answers "best possible AUROC
    at panel size k".

    Args:
        results_dir: Path to model's aggregated results directory
        infile: Path to input data file
        split_dir: Directory containing split indices
        model_name: Model name (auto-detected if None)
        stability_threshold: Minimum selection frequency (default: 0.90)
        start_size: Cap starting panel to top N proteins by selection frequency.
            None or 0 means no cap (use all stable proteins).
        min_size: Minimum panel size
        min_auroc_frac: Early stop threshold
        cv_folds: CV folds for RFE
        step_strategy: Elimination strategy ("geometric", "fine", "linear")
        outdir: Output directory (default: results_dir/optimize_panel)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        n_jobs: Total parallel cores (auto-split between seeds and Optuna)
        retune_n_trials: Optuna trials per evaluation point for re-tuning
        require_significance: If True, skip models that fail significance test
        significance_alpha: P-value threshold for significance testing (default: 0.05)

    Returns:
        RFEResult with optimization curve and recommendations, or None if skipped
        due to significance

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If stability panel is too small
    """
    import logging

    from ced_ml.utils.logging import log_hpc_context, setup_command_logger

    results_path = Path(results_dir)

    # Auto-detect model name from path if not provided
    if not model_name:
        # results_dir is typically: ../results/run_20260127/LR_EN/aggregated
        model_name = _extract_model_name_from_aggregated_path(results_path)

    # Setup logging
    if log_level is None:
        log_level = logging.INFO

    # Derive run_id from path (pattern: .../run_{ID}/...)
    _run_id = None
    _run_level_dir = results_path  # fallback
    for parent in [results_path] + list(results_path.parents):
        if parent.name.startswith("run_"):
            _run_id = parent.name[4:]
            _run_level_dir = parent
            break

    # Auto-file-logging
    logger = setup_command_logger(
        command="optimize-panel",
        log_level=log_level,
        outdir=_run_level_dir,
        run_id=_run_id,
        model=model_name,
        logger_name=f"ced_ml.optimize_panel.{model_name}",
    )
    logger.info(f"Aggregated panel optimization started for {model_name}")
    logger.info(f"Run ID: {_run_id or 'unknown'}")
    logger.info(f"Results dir: {results_dir}")
    log_hpc_context(logger)

    # Significance gating
    if require_significance:
        if results_path.name == "aggregated":
            run_dir = results_path.parent.parent
        else:
            run_dir = results_path.parent
        sig_df = load_aggregated_significance(run_dir)

        if sig_df is not None:
            model_sig = sig_df[sig_df["model"] == model_name]
            if len(model_sig) > 0:
                p_value = model_sig["empirical_p_value"].iloc[0]
                if p_value >= significance_alpha:
                    logger.warning(
                        f"Skipping {model_name}: p-value={p_value:.4f} >= "
                        f"alpha={significance_alpha}. Model not significant."
                    )
                    print(
                        f"\nSkipped {model_name}: not statistically significant "
                        f"(p={p_value:.4f} >= {significance_alpha})"
                    )
                    return None
                else:
                    logger.info(
                        f"{model_name} is significant (p={p_value:.4f} < "
                        f"{significance_alpha}), proceeding with RFE"
                    )
            else:
                logger.warning(
                    f"No significance data found for {model_name}, skipping significance check"
                )
        else:
            logger.warning(
                f"No aggregated significance data found in {run_dir}, skipping significance check"
            )

    # Load aggregated stability panel (ranked by OOF importance when available)
    from ced_ml.cli.panel_optimization_helpers import load_stability_panel

    stability_file = results_path / "panels" / "feature_stability_summary.csv"
    importance_file = results_path / "importance" / f"oof_importance__{model_name}.csv"
    stable_proteins, selection_freq = load_stability_panel(
        stability_file, stability_threshold, start_size, importance_file=importance_file
    )

    # Discover available split seeds
    run_dir = results_path.parent
    split_dirs = sorted(run_dir.glob("splits/split_seed*"))
    if not split_dirs:
        raise FileNotFoundError(
            f"No split directories found in {run_dir}/splits/. "
            f"Expected at least one split_seed* directory."
        )

    # Load metadata from first split (columns, scenario are identical across seeds)
    representative_split = split_dirs[0]
    model_path = representative_split / "core" / f"{model_name}__final_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Representative model not found: {model_path}\n"
            f"Tried first available split: {representative_split.name}"
        )

    from ced_ml.cli.panel_optimization_helpers import extract_retune_cv_folds, load_model_bundle

    logger.info(f"Loading metadata from {model_path}")
    bundle = load_model_bundle(model_path)

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentOnly")

    protein_cols = resolved_cols.get("protein_cols", [])
    cat_cols = resolved_cols.get("categorical_metadata", [])
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    if not protein_cols:
        raise ValueError("Model bundle missing protein_cols")

    logger.info(f"Model metadata: {len(protein_cols)} total proteins, scenario={scenario}")

    retune_cv_folds = extract_retune_cv_folds(bundle.get("config", {}))
    logger.info(
        f"Per-k re-tuning: {retune_n_trials} trials, cv={retune_cv_folds} "
        f"(from bundle config inner_folds)"
    )

    # Auto-split parallelism: seed_jobs vs optuna_jobs
    n_seeds = len(split_dirs)
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1

    seed_jobs = min(n_seeds, max(1, n_jobs))
    optuna_jobs = max(1, n_jobs // seed_jobs)
    logger.info(
        f"Parallelism: {n_jobs} total cores -> "
        f"{seed_jobs} seed jobs x {optuna_jobs} Optuna jobs/seed"
    )

    # Load data once (shared across all seeds)
    from ced_ml.cli.panel_optimization_helpers import load_and_filter_data

    feature_cols = protein_cols + cat_cols + meta_num_cols
    df, feature_cols, protein_cols = load_and_filter_data(
        infile, feature_cols, protein_cols, meta_num_cols, scenario=scenario
    )

    # Filter stable proteins
    initial_proteins = [p for p in stable_proteins if p in df.columns]
    if len(initial_proteins) < min_size:
        raise ValueError(
            f"Only {len(initial_proteins)} stable proteins available, less than min_size={min_size}"
        )

    filtered_cat_cols = [c for c in cat_cols if c in df.columns]
    filtered_meta_num_cols = [c for c in meta_num_cols if c in df.columns]
    logger.info(f"Starting multi-seed RFE with {len(initial_proteins)} stable proteins")
    logger.info(
        f"Seeds: {len(split_dirs)}, "
        f"Categorical cols: {len(filtered_cat_cols)}, "
        f"Numeric metadata: {len(filtered_meta_num_cols)}"
    )

    # -- Check for pre-computed per-seed results (HPC mode) --
    seeds_cache_dir = results_path / "optimize_panel" / "seeds"
    precomputed_results: list[RFEResult] = []
    precomputed_seeds: set[int] = set()
    if seeds_cache_dir.exists():
        for seed_dir_candidate in sorted(seeds_cache_dir.glob("seed_*")):
            joblib_path = seed_dir_candidate / "rfe_result.joblib"
            if joblib_path.exists():
                seed_num = int(seed_dir_candidate.name.replace("seed_", ""))
                precomputed_seeds.add(seed_num)

    available_seeds = {int(sd.name.replace("split_seed", "")) for sd in split_dirs}

    if precomputed_seeds and precomputed_seeds >= available_seeds:
        # All seeds have pre-computed results; load and skip to aggregation
        logger.info(
            f"Found pre-computed RFE results for all {len(available_seeds)} seeds, "
            f"skipping RFE and proceeding to aggregation"
        )
        for seed_num in sorted(available_seeds):
            joblib_path = seeds_cache_dir / f"seed_{seed_num}" / "rfe_result.joblib"
            loaded_result = joblib.load(joblib_path)
            precomputed_results.append(loaded_result)
            logger.info(f"  Loaded seed {seed_num}: max AUROC={loaded_result.max_auroc:.4f}")

    # -- Run RFE for each split seed (if not pre-computed) --
    import time

    overall_start = time.time()

    if precomputed_results:
        per_seed_results = precomputed_results
        multi_seed_elapsed = time.time() - overall_start
    else:

        def _run_rfe_for_seed(seed_dir: Path) -> RFEResult:
            """Run RFE for a single split seed."""
            from ced_ml.cli.panel_optimization_helpers import run_rfe_for_seed

            seed = int(seed_dir.name.replace("split_seed", ""))
            seed_model_path = seed_dir / "core" / f"{model_name}__final_model.joblib"

            if not seed_model_path.exists():
                raise FileNotFoundError(f"Model not found for seed {seed}: {seed_model_path}")

            return run_rfe_for_seed(
                seed=seed,
                model_path=seed_model_path,
                df=df,
                feature_cols=feature_cols,
                split_dir=Path(split_dir),
                scenario=scenario,
                model_name=model_name,
                initial_proteins=initial_proteins,
                cat_cols=filtered_cat_cols,
                meta_num_cols=filtered_meta_num_cols,
                min_size=min_size,
                cv_folds=cv_folds,
                step_strategy=step_strategy,
                min_auroc_frac=min_auroc_frac,
                retune_n_trials=retune_n_trials,
                retune_cv_folds=retune_cv_folds,
                retune_n_jobs=optuna_jobs,
                corr_aware=corr_aware,
                corr_threshold=corr_threshold,
                corr_method=corr_method,
                selection_freq=selection_freq,
                rfe_tune_spaces=rfe_tune_spaces,
            )

        if len(split_dirs) > 1 and seed_jobs != 1:
            from sklearn.utils.parallel import Parallel, delayed

            logger.info(
                f"\n{'=' * 60}\n"
                f"Multi-seed RFE execution\n"
                f"{'=' * 60}\n"
                f"Total seeds: {len(split_dirs)}\n"
                f"Parallel seed jobs: {seed_jobs}\n"
                f"Optuna jobs per seed: {optuna_jobs}\n"
                f"Starting panel: {len(initial_proteins)} proteins\n"
                f"{'=' * 60}\n"
            )
            try:
                per_seed_results = Parallel(n_jobs=seed_jobs)(
                    delayed(_run_rfe_for_seed)(sd) for sd in split_dirs
                )
            except (PermissionError, NotImplementedError, OSError) as exc:
                logger.warning(
                    "Parallel seed execution unavailable in current runtime (%s). "
                    "Falling back to sequential RFE.",
                    exc,
                )
                per_seed_results = []
                for idx, sd in enumerate(split_dirs, 1):
                    logger.info(f"\nProcessing seed {idx}/{len(split_dirs)}")
                    result = _run_rfe_for_seed(sd)
                    per_seed_results.append(result)
        else:
            logger.info(
                f"\n{'=' * 60}\n"
                f"Sequential RFE execution\n"
                f"{'=' * 60}\n"
                f"Total seeds: {len(split_dirs)}\n"
                f"Starting panel: {len(initial_proteins)} proteins\n"
                f"{'=' * 60}\n"
            )
            per_seed_results = []
            for idx, sd in enumerate(split_dirs, 1):
                logger.info(f"\nProcessing seed {idx}/{len(split_dirs)}")
                result = _run_rfe_for_seed(sd)
                per_seed_results.append(result)

    multi_seed_elapsed = time.time() - overall_start

    # Aggregated summary
    mean_auroc = np.mean([r.max_auroc for r in per_seed_results])
    std_auroc = np.std([r.max_auroc for r in per_seed_results])
    logger.info(
        f"\n{'=' * 60}\n"
        f"All seeds completed\n"
        f"Total time: {multi_seed_elapsed / 60:.1f} min\n"
        f"Average per seed: {multi_seed_elapsed / len(split_dirs) / 60:.1f} min\n"
        f"{'=' * 60}\n"
    )

    # Aggregate across seeds
    logger.info("Aggregating results across seeds...")
    result = aggregate_rfe_results(per_seed_results)

    # Aggregated summary log
    rec_summary = ", ".join(f"{k}={v}" for k, v in result.recommended_panels.items())
    logger.info(
        f"\nAggregated RFE ({n_seeds} seeds):\n"
        f"  Mean best AUROC: {mean_auroc:.3f} +/- {std_auroc:.3f}\n"
        f"  Recommended: {rec_summary}\n"
        f"  Wall time: {multi_seed_elapsed / 60:.1f} min ({n_jobs} cores)"
    )
    logger.info(
        f"Aggregation complete:\n"
        f"  Aggregated max AUROC: {result.max_auroc:.4f}\n"
        f"  Evaluation points: {len(result.curve)}\n"
        f"  Recommended panels: {len(result.recommended_panels)}\n"
    )

    # Save results
    if outdir is None:
        outdir = results_path / "optimize_panel"
    else:
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # Use split_seed=-1 as identifier for aggregated results (across all splits)
    paths = save_rfe_results(result, str(outdir), model_name, split_seed=-1)
    logger.info(f"Saved RFE results to {outdir}")

    # Load full-model AUROC as reference (from aggregated pooled val metrics)
    _full_model_auroc = None
    _val_metrics_file = results_path / "metrics" / "pooled_val_metrics.csv"
    if _val_metrics_file.exists():
        try:
            _val_df = pd.read_csv(_val_metrics_file)
            if "auroc" in _val_df.columns and len(_val_df) > 0:
                _full_model_auroc = float(_val_df["auroc"].iloc[0])
                logger.info(f"Full-model reference AUROC (pooled val): {_full_model_auroc:.4f}")
        except Exception as e:
            logger.debug(f"Could not load full-model AUROC: {e}")

    # Generate plots
    try:
        plot_path = Path(outdir) / "panel_curve_aggregated.png"
        plot_pareto_curve(
            curve=result.curve,
            recommended=result.recommended_panels,
            out_path=plot_path,
            title=f"Panel Size Optimization ({model_name}, {n_seeds} seeds)",
            model_name=model_name,
            n_splits=n_seeds,
            feature_selection_method="Aggregated RFE",
            run_id=_run_id,
            full_model_auroc=_full_model_auroc,
        )
        paths["panel_curve_plot"] = str(plot_path)
        logger.info(f"Saved panel curve plot to {plot_path}")

        ranking_plot_path = Path(outdir) / "feature_ranking_aggregated.png"
        plot_feature_ranking(
            feature_ranking=result.feature_ranking,
            out_path=ranking_plot_path,
            top_n=30,
            title=f"Feature Importance Ranking ({model_name}, {n_seeds} seeds)",
            n_splits=n_seeds,
            feature_selection_method="Aggregated RFE",
            run_id=_run_id,
        )
        paths["feature_ranking_plot"] = str(ranking_plot_path)
        logger.info(f"Saved feature ranking plot to {ranking_plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # Print summary
    from ced_ml.cli.panel_optimization_helpers import print_optimization_summary

    print_optimization_summary(
        model_name=model_name,
        result=result,
        n_seeds=n_seeds,
        initial_panel_size=len(initial_proteins),
        outdir=Path(outdir),
        aggregated=True,
    )

    logger.info("Aggregated panel optimization completed successfully")

    # Run drop-column essentiality validation for recommended panels
    try:
        logger.info("\n" + "=" * 60)
        logger.info("Running drop-column essentiality validation...")
        logger.info("=" * 60)
        run_drop_column_validation_for_panels(
            result=result,
            results_path=results_path,
            df=df,
            split_dir=Path(split_dir),
            scenario=scenario,
            model_name=model_name,
            split_dirs=split_dirs,
            cat_cols=filtered_cat_cols,
            meta_num_cols=filtered_meta_num_cols,
            corr_threshold=corr_threshold,
            corr_method=corr_method,
            essentiality_corr_threshold=essentiality_corr_threshold,
        )
        logger.info("Drop-column essentiality validation completed successfully")
    except Exception as e:
        logger.warning(f"Drop-column validation failed: {e}", exc_info=True)
        logger.warning("Continuing without essentiality validation")

    return result
