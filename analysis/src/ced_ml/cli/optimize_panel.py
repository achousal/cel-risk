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
from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    save_rfe_results,
)
from ced_ml.plotting.panel_curve import plot_feature_ranking, plot_pareto_curve

logger = logging.getLogger(__name__)


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
                # Extract model name from path
                model_name = f.parent.parent.name
                df["model"] = model_name
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


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
    stable_proteins, selection_freq = load_stability_panel(
        stability_file, stability_threshold, start_size
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
        infile, feature_cols, protein_cols, meta_num_cols
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
                    f"No significance data found for {model_name}, " "skipping significance check"
                )
        else:
            logger.warning(
                f"No aggregated significance data found in {run_dir}, "
                "skipping significance check"
            )

    # Load aggregated stability panel
    from ced_ml.cli.panel_optimization_helpers import load_stability_panel

    stability_file = results_path / "panels" / "feature_stability_summary.csv"
    stable_proteins, selection_freq = load_stability_panel(
        stability_file, stability_threshold, start_size
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
        infile, feature_cols, protein_cols, meta_num_cols
    )

    # Filter stable proteins
    initial_proteins = [p for p in stable_proteins if p in df.columns]
    if len(initial_proteins) < min_size:
        raise ValueError(
            f"Only {len(initial_proteins)} stable proteins available, "
            f"less than min_size={min_size}"
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
                f"\n{'='*60}\n"
                f"Multi-seed RFE execution\n"
                f"{'='*60}\n"
                f"Total seeds: {len(split_dirs)}\n"
                f"Parallel seed jobs: {seed_jobs}\n"
                f"Optuna jobs per seed: {optuna_jobs}\n"
                f"Starting panel: {len(initial_proteins)} proteins\n"
                f"{'='*60}\n"
            )
            per_seed_results: list[RFEResult] = Parallel(n_jobs=seed_jobs)(
                delayed(_run_rfe_for_seed)(sd) for sd in split_dirs
            )
        else:
            logger.info(
                f"\n{'='*60}\n"
                f"Sequential RFE execution\n"
                f"{'='*60}\n"
                f"Total seeds: {len(split_dirs)}\n"
                f"Starting panel: {len(initial_proteins)} proteins\n"
                f"{'='*60}\n"
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
        f"\n{'='*60}\n"
        f"All seeds completed\n"
        f"Total time: {multi_seed_elapsed/60:.1f} min\n"
        f"Average per seed: {multi_seed_elapsed/len(split_dirs)/60:.1f} min\n"
        f"{'='*60}\n"
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
        f"  Wall time: {multi_seed_elapsed/60:.1f} min ({n_jobs} cores)"
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

    # --- Drop-column essentiality validation per recommended panel threshold ---
    try:
        logger.info("\n" + "=" * 60)
        logger.info("Running drop-column essentiality validation on recommended panels...")
        logger.info("=" * 60)

        # Create binary target vector (needed for drop-column validation)
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in data")

        positive_label = get_positive_label(scenario)
        y_all = (df[TARGET_COL] == positive_label).astype(int).values
        logger.info(
            f"Target vector created: {y_all.sum()} positive cases out of {len(y_all)} samples"
        )

        # Create essentiality output directory
        essentiality_dir = outdir / "essentiality"
        essentiality_dir.mkdir(parents=True, exist_ok=True)

        # Extract feature ranking for panel extraction
        all_features = [row["feature"] for row in result.feature_ranking]

        # Summary storage for all thresholds
        essentiality_summaries = []

        # Run essentiality validation for each recommended panel threshold
        for threshold_name, panel_size in result.recommended_panels.items():
            logger.info(f"\n--- Essentiality validation: {threshold_name} (size={panel_size}) ---")

            # Extract panel features (top-k by elimination order)
            panel_features = all_features[:panel_size]
            logger.info(f"Panel features: {len(panel_features)}")

            # Build correlation clusters for this panel
            first_split = split_dirs[0]
            seed = int(first_split.name.replace("split_seed", ""))
            train_file = Path(split_dir) / f"train_idx_{scenario}_seed{seed}.csv"
            train_idx = pd.read_csv(train_file).squeeze().values

            X_corr = df.iloc[train_idx][panel_features]
            corr_matrix = compute_correlation_matrix(X_corr, panel_features, method=corr_method)
            adj_graph = build_correlation_graph(corr_matrix, threshold=corr_threshold)
            clusters = find_connected_components(adj_graph)
            logger.info(f"Found {len(clusters)} correlation clusters for {threshold_name}")

            # Run drop-column across all folds
            drop_column_results_per_fold = []
            for split_dir_path in split_dirs:
                seed = int(split_dir_path.name.replace("split_seed", ""))
                model_path = split_dir_path / "core" / f"{model_name}__final_model.joblib"

                train_file = Path(split_dir) / f"train_idx_{scenario}_seed{seed}.csv"
                val_file = Path(split_dir) / f"val_idx_{scenario}_seed{seed}.csv"
                train_idx = pd.read_csv(train_file).squeeze().values
                val_idx = pd.read_csv(val_file).squeeze().values

                bundle = joblib.load(model_path)
                pipeline = bundle.get("model")

                if pipeline is None:
                    logger.warning(f"Seed {seed}: model bundle missing 'model' key, skipping")
                    continue

                X_train_seed = df.iloc[train_idx][panel_features]
                y_train_seed = y_all[train_idx]
                X_val_seed = df.iloc[val_idx][panel_features]
                y_val_seed = y_all[val_idx]

                fold_results = compute_drop_column_importance(
                    estimator=pipeline,
                    X_train=X_train_seed,
                    y_train=y_train_seed,
                    X_val=X_val_seed,
                    y_val=y_val_seed,
                    feature_clusters=clusters,
                    random_state=seed,
                )
                drop_column_results_per_fold.append(fold_results)

            # Aggregate and save per-threshold results
            drop_column_df = aggregate_drop_column_results(drop_column_results_per_fold)

            # Clean threshold name for filename (e.g., "95%" -> "95pct")
            threshold_clean = threshold_name.replace("%", "pct").replace(" ", "_").lower()
            threshold_path = essentiality_dir / f"panel_{threshold_clean}_essentiality.csv"
            drop_column_df.to_csv(threshold_path, index=False)
            logger.info(f"Saved {threshold_name} essentiality to {threshold_path}")

            # Collect summary stats
            essentiality_summaries.append(
                {
                    "threshold": threshold_name,
                    "panel_size": panel_size,
                    "n_clusters": len(clusters),
                    "mean_delta_auroc": drop_column_df["mean_delta_auroc"].mean(),
                    "max_delta_auroc": drop_column_df["mean_delta_auroc"].max(),
                    "top_cluster_id": drop_column_df.iloc[0]["cluster_id"],
                    "top_cluster_delta": drop_column_df.iloc[0]["mean_delta_auroc"],
                }
            )

        # Save essentiality summary
        summary_df = pd.DataFrame(essentiality_summaries)
        summary_path = essentiality_dir / "essentiality_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved essentiality summary to {summary_path}")

        # Print summary
        print("\nDrop-column essentiality validation complete:")
        print(f"  Validated {len(result.recommended_panels)} recommended panels")
        print(f"  Across {len(split_dirs)} folds")
        print(f"  Results saved to: {essentiality_dir}")
        for row in essentiality_summaries:
            print(
                f"    - {row['threshold']}: {row['n_clusters']} clusters, "
                f"top delta={row['top_cluster_delta']:+.4f}"
            )

    except Exception as e:
        logger.warning(f"Drop-column essentiality validation failed: {e}", exc_info=True)
        logger.info("Continuing without essentiality validation results")

    logger.info("Aggregated panel optimization completed successfully")

    return result
