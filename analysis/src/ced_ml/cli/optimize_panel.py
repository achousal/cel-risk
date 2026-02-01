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
import os
from pathlib import Path

import joblib
import pandas as pd

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.features.rfe import (
    RFEResult,
    aggregate_rfe_results,
    recursive_feature_elimination,
    save_rfe_results,
)
from ced_ml.features.stability import compute_selection_frequencies, rank_proteins_by_frequency
from ced_ml.plotting.panel_curve import plot_feature_ranking, plot_pareto_curve


def find_model_paths_for_run(
    run_id: str | None = None,
    model: str | None = None,
    split_seed: int = 0,
    skip_ensemble: bool = True,
) -> list[str]:
    """Auto-detect model paths from run_id.

    Directory layout: results/run_{RUN_ID}/{MODEL}/splits/split_seed{N}/core/

    Args:
        run_id: Run ID (e.g., "20260127_104409"). If None, auto-detects latest.
        model: Model name (e.g., "LR_EN"). If None, returns all base models.
        split_seed: Split seed (default: 0)
        skip_ensemble: If True, exclude ENSEMBLE models from results (default: True)

    Returns:
        List of paths to model files

    Raises:
        FileNotFoundError: If no models found
    """
    import os
    from pathlib import Path

    from ced_ml.utils.paths import get_project_root

    results_dir_env = os.environ.get("CED_RESULTS_DIR")
    if results_dir_env:
        results_dir = Path(results_dir_env)
    else:
        results_dir = get_project_root() / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Auto-detect run_id: scan results/run_*/
    if not run_id:
        run_ids = [d.name.replace("run_", "") for d in results_dir.glob("run_*") if d.is_dir()]
        if not run_ids:
            raise FileNotFoundError("No runs found in results directory")
        run_ids.sort(reverse=True)
        run_id = run_ids[0]

    run_path = results_dir / f"run_{run_id}"

    # Find models under results/run_{id}/
    if model:
        models_to_check = [model]
    else:
        models_to_check = []
        if run_path.exists():
            for model_dir in sorted(run_path.glob("*/")):
                if model_dir.name.startswith(".") or model_dir.name in (
                    "investigations",
                    "consensus",
                ):
                    continue
                if skip_ensemble and model_dir.name == "ENSEMBLE":
                    continue
                models_to_check.append(model_dir.name)

        if not models_to_check:
            msg = f"No base models found for run {run_id}"
            if skip_ensemble:
                msg += " (ENSEMBLE excluded)"
            raise FileNotFoundError(msg)

    # Build paths: results/run_{id}/{model}/splits/split_seed{N}/core/{model}__final_model.joblib
    model_paths = []
    for model_name in models_to_check:
        model_file = (
            run_path
            / model_name
            / "splits"
            / f"split_seed{split_seed}"
            / "core"
            / f"{model_name}__final_model.joblib"
        )

        if model_file.exists():
            model_paths.append(str(model_file))
        else:
            if model:
                raise FileNotFoundError(
                    f"Model not found: {model_file}\n"
                    f"Run: {run_id}, Model: {model_name}, Split: {split_seed}"
                )

    if not model_paths:
        raise FileNotFoundError(f"No valid model files found for run {run_id}, split {split_seed}")

    return model_paths


def run_optimize_panel(
    model_path: str,
    infile: str,
    split_dir: str,
    split_seed: int = 0,
    start_size: int = 100,
    min_size: int = 5,
    min_auroc_frac: float = 0.90,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    outdir: str | None = None,
    use_stability_panel: bool = True,
    log_level: int | None = None,
) -> RFEResult:
    """Run panel optimization via Recursive Feature Elimination.

    Loads a trained model, extracts the stability panel (or uses all features),
    and performs RFE to find the minimum viable panel maintaining acceptable AUROC.

    Args:
        model_path: Path to trained model bundle (.joblib).
        infile: Path to input data file (Parquet/CSV).
        split_dir: Directory containing split indices.
        split_seed: Split seed to use.
        start_size: Starting panel size (top N from stability ranking).
        min_size: Minimum panel size to evaluate.
        min_auroc_frac: Early stop if AUROC drops below this fraction of max.
        cv_folds: CV folds for OOF AUROC estimation.
        step_strategy: Elimination strategy ("geometric", "fine", "linear").
        outdir: Output directory (default: alongside model in optimize_panel/).
        use_stability_panel: If True, start from stability ranking; else all proteins.
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)

    Returns:
        RFEResult with curve, feature_ranking, and recommendations.

    Raises:
        FileNotFoundError: If model or data files not found.
        ValueError: If required data is missing from model bundle.
    """
    # Setup logging to file and console
    if log_level is None:
        log_level = logging.INFO

    # Use unique logger per model to avoid handler accumulation in batch mode
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"{__name__}.{model_path.replace('/', '_')}_{timestamp}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent propagation to root logger

    # Create logs/features directory at root level (parallel to results/)
    from ced_ml.utils.paths import get_project_root

    log_dir = get_project_root() / "logs" / "features"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    log_file = log_dir / f"optimize_panel_{timestamp}.log"

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Always add handlers (unique logger per invocation)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Panel optimization started at {log_file}")
    logger.info(f"Loading model from {model_path}")

    # Load model bundle
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary (not bare model)")

    model_name = bundle.get("model_name", "unknown")
    pipeline = bundle.get("model")
    resolved_cols = bundle.get("resolved_columns", {})

    # Validate that this is not an ENSEMBLE model (should be filtered upstream)
    if model_name == "ENSEMBLE":
        logger.warning(
            "ENSEMBLE model detected - skipping (operates on meta-features, not proteins)"
        )
        raise ValueError(
            "ENSEMBLE models cannot be optimized directly (they operate on 4 meta-features, not proteins).\n"
            "This should have been filtered by find_model_paths_for_run()."
        )

    if pipeline is None:
        raise ValueError("Model bundle missing 'model' key")

    protein_cols = resolved_cols.get("protein_cols", [])
    cat_cols = resolved_cols.get("categorical_metadata", [])
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    if not protein_cols:
        raise ValueError("Model bundle missing protein_cols in resolved_columns")

    logger.info(f"Model: {model_name}, {len(protein_cols)} proteins available")

    # Load data
    logger.info(f"Loading data from {infile}")
    df_raw = read_proteomics_file(infile, validate=True)

    # Apply row filters (must match split generation)
    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} → {filter_stats['n_out']:,} rows")
    logger.info(f"  Removed {filter_stats['n_removed_uncertain_controls']} uncertain controls")
    logger.info(f"  Removed {filter_stats['n_removed_dropna_meta_num']} rows with missing metadata")

    # Load split indices (CSV format)
    split_path = Path(split_dir)
    scenario = bundle.get("scenario", "IncidentOnly")

    train_file = split_path / f"train_idx_{scenario}_seed{split_seed}.csv"
    val_file = split_path / f"val_idx_{scenario}_seed{split_seed}.csv"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            f"Split files not found: {train_file}, {val_file}\n"
            f"Run 'ced save-splits' to generate splits with scenario={scenario}"
        )

    logger.info(f"Loading splits from {split_path}")
    train_idx = pd.read_csv(train_file).squeeze().values
    val_idx = pd.read_csv(val_file).squeeze().values

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split file missing train_idx or val_idx")

    # Prepare data
    feature_cols = protein_cols + cat_cols + meta_num_cols

    # Check for missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns in data: {missing[:10]}...")
        feature_cols = [c for c in feature_cols if c in df.columns]
        protein_cols = [c for c in protein_cols if c in df.columns]

    # Create binary target using schema-defined target column and positive label
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in data. "
            f"Available columns: {list(df.columns[:10])}..."
        )

    positive_label = get_positive_label(scenario)
    y_all = (df[TARGET_COL] == positive_label).astype(int).values

    # Subset to train/val (use iloc for position-based indexing)
    X_train = df.iloc[train_idx][feature_cols].copy()
    y_train = y_all[train_idx]

    X_val = df.iloc[val_idx][feature_cols].copy()
    y_val = y_all[val_idx]

    logger.info(f"Train: {len(X_train)} samples, {y_train.sum()} cases")
    logger.info(f"Val: {len(X_val)} samples, {y_val.sum()} cases")

    # Determine initial proteins
    if use_stability_panel:
        # Try to load stability panel from model directory
        model_dir = Path(model_path).parent.parent  # Go up from core/
        stable_panel_path = model_dir / "reports" / "stable_panel" / "stable_panel__KBest.csv"
        selected_proteins_path = model_dir / "cv" / "selected_proteins_per_split.csv"

        initial_proteins = None

        # Priority 1: Load from stable_panel__KBest.csv (authoritative source)
        # NOTE: When using this file, start_size is IGNORED - all 'kept' proteins are used.
        # This provides the most robust starting point based on cross-validated stability.
        if stable_panel_path.exists():
            logger.info(f"Loading stable panel from {stable_panel_path}")
            stable_df = pd.read_csv(stable_panel_path)
            # Ensure kept column is boolean (handle string "True"/"False" if present)
            if stable_df["kept"].dtype == object:
                stable_df["kept"] = stable_df["kept"].astype(str).str.lower() == "true"
            initial_proteins = stable_df[stable_df["kept"]]["protein"].tolist()
            logger.info(
                f"Loaded {len(initial_proteins)} stable proteins (≥75% threshold). "
                f"start_size parameter ignored when using stable_panel__KBest.csv."
            )

            # Validate all proteins exist in data
            missing = [p for p in initial_proteins if p not in X_train.columns]
            if missing:
                logger.warning(
                    f"{len(missing)} stable proteins not found in data: {missing[:5]}..."
                )
                initial_proteins = [p for p in initial_proteins if p in X_train.columns]
                logger.info(f"After filtering: {len(initial_proteins)} proteins available")

        # Alternative: Load from cv/selected_proteins_per_split.csv if stable panel not found
        elif selected_proteins_path.exists():
            logger.info(f"Loading stability panel from {selected_proteins_path}")
            sel_df = pd.read_csv(selected_proteins_path)

            # Compute selection frequencies
            freq = compute_selection_frequencies(sel_df, selection_col="selected_proteins_split")
            if not freq:
                freq = compute_selection_frequencies(sel_df, selection_col="selected_proteins")
            if not freq:
                freq = compute_selection_frequencies(
                    sel_df, selection_col="selected_proteins_final"
                )

            if freq:
                ranked = rank_proteins_by_frequency(freq)
                initial_proteins = ranked[:start_size]
                logger.info(f"Using top {len(initial_proteins)} proteins from stability ranking")

        if initial_proteins is None:
            logger.warning("Could not load stability panel, using all proteins")
            initial_proteins = protein_cols[:start_size]
    else:
        initial_proteins = protein_cols[:start_size]

    # Ensure initial proteins are in the data (for non-stability path)
    if not use_stability_panel:
        initial_proteins = [p for p in initial_proteins if p in X_train.columns]

    if len(initial_proteins) < min_size:
        raise ValueError(
            f"Only {len(initial_proteins)} valid proteins, less than min_size={min_size}"
        )

    logger.info(f"Starting RFE with {len(initial_proteins)} proteins")
    logger.info(
        f"RFE parameters: step_strategy={step_strategy}, cv_folds={cv_folds}, "
        f"min_auroc_frac={min_auroc_frac}, min_size={min_size}"
    )

    # Run RFE
    logger.info("Filtering categorical columns...")
    filtered_cat_cols = [c for c in cat_cols if c in X_train.columns]
    logger.info(f"  {len(filtered_cat_cols)} categorical columns found")

    logger.info("Filtering numeric metadata columns...")
    filtered_meta_num_cols = [c for c in meta_num_cols if c in X_train.columns]
    logger.info(f"  {len(filtered_meta_num_cols)} numeric metadata columns found")

    logger.info("Calling recursive_feature_elimination...")
    result = recursive_feature_elimination(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        base_pipeline=pipeline,
        model_name=model_name,
        initial_proteins=initial_proteins,
        cat_cols=filtered_cat_cols,
        meta_num_cols=filtered_meta_num_cols,
        min_size=min_size,
        cv_folds=cv_folds,
        step_strategy=step_strategy,
        min_auroc_frac=min_auroc_frac,
        random_state=split_seed,
    )

    logger.info(f"RFE complete. Max AUROC: {result.max_auroc:.4f}")

    # Determine output directory
    if outdir is None:
        model_dir = Path(model_path).parent.parent
        outdir = str(model_dir / "optimize_panel")

    os.makedirs(outdir, exist_ok=True)

    # Save results
    paths = save_rfe_results(result, outdir, model_name, split_seed)
    logger.info(f"Saved RFE results to {outdir}")

    # Generate plots
    try:
        plot_path = Path(outdir) / "panel_curve.png"
        plot_pareto_curve(
            curve=result.curve,
            recommended=result.recommended_panels,
            out_path=plot_path,
            title="Panel Size vs AUROC (RFE)",
            model_name=model_name,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
            n_train_cases=int(y_train.sum()),
            n_val_cases=int(y_val.sum()),
            feature_selection_method="Single-split RFE",
        )
        paths["panel_curve_plot"] = str(plot_path)
        logger.info(f"Saved panel curve plot to {plot_path}")

        ranking_plot_path = Path(outdir) / "feature_ranking.png"
        plot_feature_ranking(
            feature_ranking=result.feature_ranking,
            out_path=ranking_plot_path,
            top_n=30,
            title=f"Feature Importance Ranking ({model_name})",
            feature_selection_method="Single-split RFE",
        )
        paths["feature_ranking_plot"] = str(ranking_plot_path)
        logger.info(f"Saved feature ranking plot to {ranking_plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Panel Optimization Complete: {model_name}")
    print(f"{'='*60}")
    print(f"Starting panel size: {len(initial_proteins)}")
    print(f"Max AUROC: {result.max_auroc:.4f}")

    # Show metrics for best panel
    if result.curve:
        best_point = max(result.curve, key=lambda x: x["auroc_val"])
        print(f"\nBest panel (size={best_point['size']}) validation metrics:")
        print(f"  AUROC:           {best_point['auroc_val']:.4f}")
        print(f"  PR-AUC:          {best_point.get('prauc_val', float('nan')):.4f}")
        print(f"  Brier Score:     {best_point.get('brier_val', float('nan')):.4f}")
        print(f"  Sens@95%Spec:    {best_point.get('sens_at_95spec_val', float('nan')):.4f}")

    print("\nRecommended panel sizes:")
    for key, size in result.recommended_panels.items():
        print(f"  {key}: {size} proteins")

    print(f"\nResults saved to: {outdir}")
    print("  - panel_curve.csv (full curve with all metrics)")
    print("  - metrics_summary.csv (metrics at each panel size)")
    print("  - recommended_panels.json (threshold-based recommendations)")
    print("  - feature_ranking.csv (protein elimination order)")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")

    logger.info("Panel optimization completed successfully")

    return result


def discover_models_by_run_id(
    run_id: str,
    results_root: Path | str,
    model_filter: str | None = None,
) -> dict[str, Path]:
    """Discover models with aggregated results for a given run_id.

    Args:
        run_id: Run ID to search for (e.g., "20260127_115115")
        results_root: Root results directory (e.g., ../results)
        model_filter: Optional model name to filter by (e.g., "LR_EN")

    Returns:
        Dictionary mapping model names to their aggregated results directories.
        Example: {"LR_EN": Path("../results/LR_EN/run_20260127_115115/aggregated")}

    Raises:
        FileNotFoundError: If results_root does not exist
    """
    # Convert to Path and validate
    results_root = Path(results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    model_dirs = {}

    # New layout: results/run_{RUN_ID}/{MODEL}/aggregated/
    run_dir = results_root / f"run_{run_id}"
    if not run_dir.exists():
        return model_dirs

    for model_dir in sorted(run_dir.glob("*/")):
        model_name = model_dir.name

        # Skip hidden/special directories and ensemble models
        if model_name.startswith(".") or model_name in ("investigations", "ENSEMBLE", "consensus"):
            continue

        # Apply model filter if specified
        if model_filter and model_name != model_filter:
            continue

        # Check for aggregated results
        aggregated_dir = model_dir / "aggregated"
        if not aggregated_dir.exists():
            continue

        # Check for required aggregated files
        feature_stability_file = aggregated_dir / "panels" / "feature_stability_summary.csv"
        if not feature_stability_file.exists():
            continue

        model_dirs[model_name] = aggregated_dir

    return model_dirs


def run_optimize_panel_aggregated(
    results_dir: str | Path,
    infile: str,
    split_dir: str,
    model_name: str | None = None,
    stability_threshold: float = 0.75,
    min_size: int = 5,
    min_auroc_frac: float = 0.90,
    cv_folds: int = 0,
    step_strategy: str = "geometric",
    outdir: str | None = None,
    log_level: int | None = None,
    n_jobs: int = 1,
) -> RFEResult:
    """Run panel optimization using aggregated stability panel.

    Runs RFE independently on each available split seed, then aggregates
    the validation curves (mean + cross-seed 95% CI). This produces a
    more robust Pareto curve than single-seed optimization.

    Args:
        results_dir: Path to model's aggregated results directory
        infile: Path to input data file
        split_dir: Directory containing split indices
        model_name: Model name (auto-detected if None)
        stability_threshold: Minimum selection frequency (default: 0.75)
        min_size: Minimum panel size
        min_auroc_frac: Early stop threshold
        cv_folds: CV folds for RFE
        step_strategy: Elimination strategy ("geometric", "fine", "linear")
        outdir: Output directory (default: results_dir/optimize_panel)
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        n_jobs: Parallel jobs for multi-seed RFE (1=sequential, -1=all CPUs)

    Returns:
        RFEResult with optimization curve and recommendations

    Raises:
        FileNotFoundError: If required files not found
        ValueError: If stability panel is too small
    """
    import logging

    from ced_ml.utils.logging import auto_log_path, setup_logger

    results_path = Path(results_dir)

    # Auto-detect model name from path if not provided
    if not model_name:
        # results_dir is typically: ../results/run_20260127/LR_EN/aggregated
        model_name = results_path.parent.parent.name

    # Setup logging
    if log_level is None:
        log_level = logging.INFO

    # Derive run_id from path (pattern: .../run_{ID}/...)
    _run_id = None
    for parent in [results_path] + list(results_path.parents):
        if parent.name.startswith("run_"):
            _run_id = parent.name[4:]
            break

    # Auto-file-logging
    log_file = auto_log_path(
        command="optimize-panel",
        outdir=results_path.parent.parent if _run_id else results_path,
        run_id=_run_id,
        model=model_name,
    )
    logger = setup_logger(
        f"ced_ml.optimize_panel.{model_name}",
        level=log_level,
        log_file=log_file,
    )
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Aggregated panel optimization started for {model_name}")
    logger.info(f"Results dir: {results_dir}")

    # Load aggregated stability panel
    stability_file = results_path / "panels" / "feature_stability_summary.csv"
    if not stability_file.exists():
        raise FileNotFoundError(
            f"Feature stability file not found: {stability_file}\n"
            "Run 'ced aggregate-splits' first to generate aggregated results."
        )

    logger.info(f"Loading aggregated stability panel from {stability_file}")
    stability_df = pd.read_csv(stability_file)

    # Filter by stability threshold (column is 'selection_fraction', not 'selection_frequency')
    stable_proteins = stability_df[stability_df["selection_fraction"] >= stability_threshold][
        "protein"
    ].tolist()

    if not stable_proteins:
        raise ValueError(
            f"No proteins meet stability threshold {stability_threshold:.2f}. "
            f"Try lowering --stability-threshold."
        )

    logger.info(
        f"Found {len(stable_proteins)} stable proteins (≥{stability_threshold:.2f} threshold)"
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

    logger.info(f"Loading metadata from {model_path}")
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentOnly")

    protein_cols = resolved_cols.get("protein_cols", [])
    cat_cols = resolved_cols.get("categorical_metadata", [])
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    if not protein_cols:
        raise ValueError("Model bundle missing protein_cols")

    logger.info(f"Model metadata: {len(protein_cols)} total proteins, scenario={scenario}")

    # Load data once (shared across all seeds)
    logger.info(f"Loading data from {infile}")
    df_raw = read_proteomics_file(infile, validate=True)

    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} → {filter_stats['n_out']:,} rows")

    feature_cols = protein_cols + cat_cols + meta_num_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing[:10]}...")
        feature_cols = [c for c in feature_cols if c in df.columns]
        protein_cols = [c for c in protein_cols if c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    positive_label = get_positive_label(scenario)
    y_all = (df[TARGET_COL] == positive_label).astype(int).values

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

    # -- Run RFE for each split seed --
    def _run_rfe_for_seed(seed_dir: Path) -> RFEResult:
        """Run RFE for a single split seed."""
        seed = int(seed_dir.name.replace("split_seed", ""))
        seed_model_path = seed_dir / "core" / f"{model_name}__final_model.joblib"

        if not seed_model_path.exists():
            raise FileNotFoundError(f"Model not found for seed {seed}: {seed_model_path}")

        seed_bundle = joblib.load(seed_model_path)
        pipeline = seed_bundle.get("model")
        if pipeline is None:
            raise ValueError(f"Model bundle missing 'model' key for seed {seed}")

        split_path = Path(split_dir)
        train_file = split_path / f"train_idx_{scenario}_seed{seed}.csv"
        val_file = split_path / f"val_idx_{scenario}_seed{seed}.csv"

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(
                f"Split files not found for seed {seed}: {train_file}, {val_file}"
            )

        train_idx = pd.read_csv(train_file).squeeze().values
        val_idx = pd.read_csv(val_file).squeeze().values

        X_train = df.iloc[train_idx][feature_cols].copy()
        y_train = y_all[train_idx]
        X_val = df.iloc[val_idx][feature_cols].copy()
        y_val = y_all[val_idx]

        logger.info(
            f"Seed {seed}: train={len(X_train)} ({y_train.sum()} cases), "
            f"val={len(X_val)} ({y_val.sum()} cases)"
        )

        return recursive_feature_elimination(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            base_pipeline=pipeline,
            model_name=model_name,
            initial_proteins=initial_proteins,
            cat_cols=filtered_cat_cols,
            meta_num_cols=filtered_meta_num_cols,
            min_size=min_size,
            cv_folds=cv_folds,
            step_strategy=step_strategy,
            min_auroc_frac=min_auroc_frac,
            random_state=seed,
        )

    if len(split_dirs) > 1 and n_jobs != 1:
        from joblib import Parallel, delayed

        effective_jobs = n_jobs if n_jobs != 0 else 1
        logger.info(f"Running RFE across {len(split_dirs)} seeds (n_jobs={effective_jobs})")
        per_seed_results: list[RFEResult] = Parallel(n_jobs=effective_jobs)(
            delayed(_run_rfe_for_seed)(sd) for sd in split_dirs
        )
    else:
        per_seed_results = [_run_rfe_for_seed(sd) for sd in split_dirs]

    # Aggregate across seeds
    result = aggregate_rfe_results(per_seed_results)

    # Save results
    if outdir is None:
        outdir = results_path / "optimize_panel"
    else:
        outdir = Path(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # Use split_seed=-1 as identifier for aggregated results (across all splits)
    paths = save_rfe_results(result, str(outdir), model_name, split_seed=-1)
    logger.info(f"Saved RFE results to {outdir}")

    # Generate plots
    n_seeds = len(split_dirs)
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
    print(f"\n{'='*60}")
    print(f"Aggregated Panel Optimization Complete: {model_name}")
    print(f"{'='*60}")
    print(f"Seeds aggregated: {n_seeds}")
    print(f"Starting panel size: {len(initial_proteins)} (aggregated stable proteins)")
    print(f"Max mean AUROC: {result.max_auroc:.4f}")

    if result.curve:
        best_point = max(result.curve, key=lambda x: x["auroc_val"])
        print(f"\nBest panel (size={best_point['size']}):")
        print(f"  AUROC:           {best_point['auroc_val']:.4f}")
        print(f"  PR-AUC:          {best_point.get('prauc_val', float('nan')):.4f}")
        print(f"  Brier Score:     {best_point.get('brier_val', float('nan')):.4f}")
        print(f"  Sens@95%Spec:    {best_point.get('sens_at_95spec_val', float('nan')):.4f}")

    print("\nRecommended panel sizes:")
    for key, size in result.recommended_panels.items():
        print(f"  {key}: {size} proteins")

    print(f"\nResults saved to: {outdir}")
    print("  - panel_curve_aggregated.csv (full curve with all metrics)")
    print("  - metrics_summary_aggregated.csv (metrics at each panel size)")
    print("  - recommended_panels_aggregated.json (threshold-based recommendations)")
    print("  - feature_ranking_aggregated.csv (protein elimination order)")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")

    logger.info("Aggregated panel optimization completed successfully")

    return result
