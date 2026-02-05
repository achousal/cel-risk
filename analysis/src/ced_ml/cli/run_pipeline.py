"""
Full pipeline orchestration: train, aggregate, ensemble, consensus, optimize.

This module provides a single command to run the complete ML workflow from
training through panel optimization and consensus generation.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ced_ml.cli.aggregate_splits import run_aggregate_splits
from ced_ml.cli.consensus_panel import run_consensus_panel
from ced_ml.cli.optimize_panel import (
    discover_models_by_run_id,
    run_optimize_panel_aggregated,
)
from ced_ml.cli.permutation_test import run_permutation_test_cli
from ced_ml.cli.save_splits import run_save_splits
from ced_ml.cli.train import run_train
from ced_ml.cli.train_ensemble import run_train_ensemble
from ced_ml.data.schema import ModelName
from ced_ml.utils.logging import setup_command_logger, setup_logger
from ced_ml.utils.paths import get_analysis_dir, get_project_root


def _ensure_splits_exist(
    split_dir: Path,
    infile: Path,
    config_file: Path | None,
    overrides: list[str] | None = None,
    log_level: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Generate splits if needed, otherwise use existing.

    Reads the ``overwrite`` flag from splits_config.yaml to decide whether
    to regenerate splits when they already exist.

    Args:
        split_dir: Directory for split files
        infile: Input data file path
        config_file: Optional splits config file
        overrides: Optional config overrides
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        logger: Optional logger instance

    Note:
        This is the single source for split generation logic used by both
        local and HPC pipeline modes. Scenario is determined by splits_config.yaml.
    """
    from ced_ml.config.loader import load_splits_config

    if logger is None:
        logger = logging.getLogger(__name__)

    # Load splits config to check overwrite flag
    splits_config = load_splits_config(config_file=config_file, overrides=overrides or [])
    overwrite = splits_config.overwrite

    # Check if splits need to be generated
    needs_generation = overwrite or not any(split_dir.glob("train_idx_*_seed*.csv"))

    if needs_generation:
        logger.info("Generating splits...")
        splits_cli_args = {
            "infile": str(infile),
            "outdir": str(split_dir),
        }
        run_save_splits(
            config_file=config_file,
            cli_args=splits_cli_args,
            overrides=overrides or [],
            log_level=log_level,
        )
        logger.info("Splits generated.")
    else:
        logger.info(f"Using existing splits from: {split_dir}")


# Hardcoded fallback defaults (used when no config and no CLI override)
_PIPELINE_DEFAULTS: dict[str, Any] = {
    "models": [ModelName.LR_EN, ModelName.RF, ModelName.XGBoost],
    "ensemble": True,
    "consensus": True,
    "optimize_panel": True,
    "permutation_test": False,  # Disabled by default (computationally expensive)
    "permutation_n_perms": 200,
    "permutation_n_jobs": -1,
    "n_boot": 500,
    "dry_run": False,
}


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    """Load a pipeline YAML config and return a flat dict of resolved values.

    Merges ``paths``, ``configs``, ``pipeline``, and ``hpc`` sections into a
    single namespace.  Paths are resolved relative to the config file's parent
    directory (assumed to be ``analysis/configs/``), making them root-aware.

    Args:
        config_path: Absolute or relative path to a ``pipeline_*.yaml`` file.

    Returns:
        Dict with keys like ``infile``, ``splits_dir``, ``results_dir``,
        ``training_config``, ``models``, ``split_seeds``, ``ensemble``, etc.
    """
    from ced_ml.config.loader import load_yaml
    from ced_ml.utils.paths import get_project_root

    config_path = Path(config_path).resolve()
    raw = load_yaml(config_path)

    # Determine base directory for path resolution
    # If config is in analysis/configs/, base is analysis/ (config.parent.parent)
    # Otherwise use project root
    try:
        project_root = get_project_root()
        if "analysis" in config_path.parts and "configs" in config_path.parts:
            # Config is in analysis/configs/, resolve relative to analysis/
            base_dir = config_path.parent.parent
        else:
            # Fallback: use project root
            base_dir = project_root
    except Exception as e:
        # Ultimate fallback: assume config is in configs/ and go up two levels
        logging.getLogger(__name__).debug(
            "Could not determine project root, using config parent: %s", e
        )
        base_dir = config_path.parent.parent

    result: dict[str, Any] = {}

    # -- paths ---------------------------------------------------------------
    paths = raw.get("paths", {})
    for key in ("infile", "splits_dir", "results_dir", "logs_dir"):
        val = paths.get(key)
        if val is not None:
            result[key] = (base_dir / val).resolve()

    # -- configs (resolve to absolute paths) ---------------------------------
    configs = raw.get("configs", {})
    if configs.get("training"):
        result["training_config"] = (base_dir / configs["training"]).resolve()
    if configs.get("splits"):
        result["splits_config"] = (base_dir / configs["splits"]).resolve()

    # -- pipeline section (flat copy) ----------------------------------------
    pipeline = raw.get("pipeline", {})
    for key in (
        "models",
        "ensemble",
        "consensus",
        "optimize_panel",
        "permutation_test",
        "permutation_n_perms",
        "permutation_n_jobs",
        "n_boot",
        "dry_run",
    ):
        if key in pipeline:
            result[key] = pipeline[key]

    # -- hpc section (pass through) ------------------------------------------
    if "hpc" in raw:
        result["hpc"] = raw["hpc"]

    result["environment"] = raw.get("environment", "local")
    return result


def resolve_pipeline_config_path(hpc: bool = False) -> Path | None:
    """Return the default pipeline config path based on mode.

    Looks for ``configs/pipeline_hpc.yaml`` (if *hpc*) or
    ``configs/pipeline_local.yaml`` relative to either cwd or analysis/.
    Returns ``None`` if the file does not exist.
    """
    from ced_ml.utils.paths import get_default_paths

    name = "pipeline_hpc.yaml" if hpc else "pipeline_local.yaml"

    # Try multiple locations
    try:
        defaults = get_default_paths()
        # 1. From analysis/configs/
        candidate = defaults["configs"] / name
        if candidate.exists():
            return candidate

        # 2. From cwd/configs/
        candidate = Path("configs") / name
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        # Fallback: try cwd/configs/
        candidate = Path("configs") / name
        if candidate.exists():
            return candidate.resolve()

    return None


def _discover_input_file(
    config_file: Path | None,
    outdir: Path | None,
    logger: logging.Logger,
) -> Path:
    """
    Auto-discover input data file from common locations.

    Search order:
    1. From pipeline config (configs/pipeline_local.yaml or pipeline_hpc.yaml)
    2. From training config (if provided)
    3. Common data locations (data/, ../data/)
    4. Raise error if not found

    Args:
        config_file: Optional training config path
        outdir: Results output directory (used to find project root)
        logger: Logger instance

    Returns:
        Path to discovered input file

    Raises:
        FileNotFoundError: If no input file can be discovered
    """
    from pathlib import Path

    import yaml

    from ced_ml.utils.paths import get_default_paths, get_project_root

    # Get project root and default paths
    try:
        root = get_project_root()
        defaults = get_default_paths()
        logger.debug(f"Project root: {root}")
    except Exception as e:
        logger.warning(f"Could not determine project root: {e}. Using cwd.")
        root = Path.cwd()
        defaults = {
            "project_root": root,
            "analysis": root / "analysis",
            "data": root / "data",
        }

    # 1. Check pipeline configs (most likely source)
    for pipeline_config in ["configs/pipeline_local.yaml", "configs/pipeline_hpc.yaml"]:
        # Try both from root and from analysis/
        for base in [
            defaults["project_root"],
            defaults.get("analysis", root / "analysis"),
        ]:
            pipeline_path = base / pipeline_config
            if pipeline_path.exists():
                try:
                    with open(pipeline_path) as f:
                        config = yaml.safe_load(f)
                        if "paths" in config and "infile" in config["paths"]:
                            rel_path = config["paths"]["infile"]
                            # Resolve relative to config file's parent.parent (analysis/ -> root/)
                            infile = (pipeline_path.parent.parent / rel_path).resolve()
                            if infile.exists():
                                logger.info(f"Discovered from {pipeline_config}: {infile}")
                                return infile
                except Exception as e:
                    logger.debug(f"Could not load {pipeline_config}: {e}")

    # 2. Check training config
    if config_file and config_file.exists():
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
                if "infile" in config:
                    infile = Path(config["infile"])
                    if not infile.is_absolute():
                        infile = (config_file.parent / infile).resolve()
                    if infile.exists():
                        logger.info(f"Discovered from training config: {infile}")
                        return infile
        except Exception as e:
            logger.debug(f"Could not load training config: {e}")

    # 3. Check common data locations
    data_dir = defaults.get("data", root / "data")
    common_locations = [
        data_dir / "Celiac_dataset_proteomics_w_demo.parquet",
        data_dir / "celiac.parquet",
        data_dir / "Celiac_dataset_proteomics_w_demo.csv",
    ]

    for location in common_locations:
        if location.exists():
            logger.info(f"Discovered from common location: {location}")
            return location

    # 4. Not found - provide helpful error
    raise FileNotFoundError(
        "Could not auto-discover input data file. Tried:\n"
        f"  - Pipeline configs: {defaults.get('analysis', root / 'analysis')}/configs/pipeline_*.yaml\n"
        f"  - Training config: {config_file}\n"
        f"  - Common locations: {data_dir}/\n"
        "\nPlease provide --infile explicitly or ensure data file exists in expected location."
    )


def _run_hpc_mode(
    *,
    config_file: Path | None,
    splits_config_file: Path | None,
    infile: Path | None,
    split_dir: Path | None,
    models: list[str],
    split_seeds: list[int],
    run_id: str | None,
    outdir: Path | None,
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
    enable_permutation_test: bool,
    permutation_n_perms: int,
    permutation_n_jobs: int,
    hpc_config_file: Path | None,
    dry_run: bool,
    log_level: int,
) -> None:
    """Submit pipeline to HPC via LSF job dependency chains.

    Generates splits locally, then submits per-seed training jobs and a
    post-processing job that depends on all training jobs completing.

    Note:
        Panel optimization and consensus generation are automatically skipped
        when using feature_selection_strategy='fixed_panel'.

        Permutation testing submits separate jobs per model/seed combination,
        each depending on the corresponding training job completing first.
    """
    from datetime import datetime

    from ced_ml.hpc.lsf import (
        load_hpc_config,
        submit_hpc_pipeline,
    )

    hpc_logger = setup_logger("ced_ml.hpc", level=log_level)

    # Detect if using fixed panel strategy and auto-disable optimization/consensus
    using_fixed_panel = _detect_fixed_panel_strategy(config_file, hpc_logger)
    if using_fixed_panel:
        if enable_optimize_panel or enable_consensus:
            hpc_logger.info(
                "Detected feature_selection_strategy='fixed_panel' - "
                "disabling panel optimization and consensus generation"
            )
        enable_optimize_panel = False
        enable_consensus = False

    # Load HPC config (returns HPCConfig schema)
    if hpc_config_file is None:
        hpc_config_file = resolve_pipeline_config_path(hpc=True)
        if hpc_config_file is None:
            hpc_config_file = Path("configs/pipeline_hpc.yaml")
    hpc_config = load_hpc_config(hpc_config_file)

    # Load pipeline config for paths and other settings (returns dict with paths/configs/pipeline sections)
    pipeline_config = load_pipeline_config(hpc_config_file)

    # Auto-discover infile from HPC config if not provided
    if infile is None:
        hpc_logger.info("Auto-discovering input data file...")
        infile = _discover_input_file(config_file, outdir, hpc_logger)
    infile = infile.resolve()

    # Resolve paths from pipeline config
    if outdir is None:
        outdir = pipeline_config.get("results_dir")
        if outdir is None:
            outdir = (get_project_root() / "results").resolve()
    else:
        outdir = outdir.resolve()

    if split_dir is None:
        split_dir = pipeline_config.get("splits_dir")
        if split_dir is None:
            split_dir = outdir.parent / "splits"
    split_dir = split_dir.resolve()

    if config_file is None:
        config_file = pipeline_config.get("training_config")

    if splits_config_file is None:
        splits_config_file = pipeline_config.get("splits_config")

    logs_dir = pipeline_config.get("logs_dir")
    if logs_dir is None:
        logs_dir = (get_project_root() / "logs").resolve()

    # Generate run_id
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    outdir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate splits locally (fast, no HPC job needed)
    _ensure_splits_exist(
        split_dir=split_dir,
        infile=infile,
        config_file=splits_config_file,
        overrides=[],
        log_level=log_level,
        logger=hpc_logger,
    )

    # Log summary
    hpc_logger.info("=" * 70)
    hpc_logger.info("CeD-ML HPC Pipeline Submission")
    hpc_logger.info("=" * 70)
    hpc_logger.info(f"Run ID: {run_id}")
    hpc_logger.info(f"Models: {', '.join(models)}")
    hpc_logger.info(f"Split seeds: {split_seeds}")
    hpc_logger.info(
        f"HPC: {hpc_config.project} / {hpc_config.queue} / "
        f"{hpc_config.walltime} / {hpc_config.cores}c / {hpc_config.mem_per_core}MB"
    )
    hpc_logger.info(f"Dry run: {dry_run}")
    if enable_permutation_test:
        hpc_logger.info(
            f"Permutation testing: enabled ({permutation_n_perms} perms, "
            f"{permutation_n_jobs} jobs)"
        )
    hpc_logger.info("=" * 70)

    # Submit jobs
    result = submit_hpc_pipeline(
        config_file=config_file,
        infile=infile,
        split_dir=split_dir,
        outdir=outdir,
        models=models,
        split_seeds=split_seeds,
        run_id=run_id,
        enable_ensemble=enable_ensemble,
        enable_consensus=enable_consensus,
        enable_optimize_panel=enable_optimize_panel,
        enable_permutation_test=enable_permutation_test,
        permutation_n_perms=permutation_n_perms,
        permutation_n_jobs=permutation_n_jobs,
        hpc_config=hpc_config,
        logs_dir=logs_dir,
        dry_run=dry_run,
        pipeline_logger=hpc_logger,
    )

    # Print monitoring instructions
    hpc_logger.info("")
    hpc_logger.info("=" * 70)
    hpc_logger.info("Pipeline Submission Complete")
    hpc_logger.info("=" * 70)
    hpc_logger.info(f"Run ID: {result['run_id']}")
    hpc_logger.info(f"Training jobs: {len(result['training_jobs'])}")
    hpc_logger.info(f"Post-processing job: {result['postprocessing_job']}")

    if result.get("panel_optimization_jobs"):
        hpc_logger.info(
            f"Panel optimization jobs: {len(result['panel_optimization_jobs'])} (parallel)"
        )

    if result.get("consensus_job"):
        hpc_logger.info(f"Consensus panel job: {result['consensus_job']}")

    if result.get("permutation_jobs"):
        hpc_logger.info(f"Permutation test jobs: {len(result['permutation_jobs'])}")

    hpc_logger.info("")
    hpc_logger.info("Monitor jobs:")
    hpc_logger.info("  bjobs -w | grep CeD_")
    hpc_logger.info("")
    hpc_logger.info("Training logs:")
    hpc_logger.info(f"  tail -f {outdir.parent}/logs/training/run_{result['run_id']}/*.log")
    hpc_logger.info("")
    hpc_logger.info("LSF error logs (job failures only):")
    hpc_logger.info(f"  cat {result['logs_dir']}/*.err")
    hpc_logger.info("")
    hpc_logger.info(f"Results: {outdir}/run_{result['run_id']}/")
    hpc_logger.info("=" * 70)


def _log_aggregated_metrics_summary(
    logger: logging.Logger,
    outdir: Path,
    run_id: str,
    models: list[str],
) -> None:
    """Log a compact metrics summary table after aggregation.

    Reads test_metrics_summary.csv from each model's aggregated/ dir
    and logs key metrics (AUROC, PR-AUC, n_features) in a table.
    Failures are logged as warnings and do not raise.
    """
    import csv

    rows: list[dict[str, str]] = []
    for model_name in models:
        summary_path = (
            outdir
            / f"run_{run_id}"
            / model_name
            / "aggregated"
            / "metrics"
            / "test_metrics_summary.csv"
        )
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("statistic") == "mean":
                        rows.append({"model": model_name, **row})
                        break
        except Exception as exc:
            logger.warning(f"Could not read metrics summary for {model_name}: {exc}")

    if not rows:
        return

    logger.info("")
    logger.info("Aggregated test metrics (mean across splits):")
    header = f"  {'Model':<15s} {'AUROC':>8s} {'PR-AUC':>8s}"
    logger.info(header)
    logger.info("  " + "-" * len(header.strip()))
    for row in rows:
        auroc = row.get("AUROC", row.get("auroc", "N/A"))
        prauc = row.get("PR_AUC", row.get("pr_auc", "N/A"))
        try:
            auroc = f"{float(auroc):.4f}"
        except (ValueError, TypeError):
            pass
        try:
            prauc = f"{float(prauc):.4f}"
        except (ValueError, TypeError):
            pass
        logger.info(f"  {row['model']:<15s} {auroc:>8s} {prauc:>8s}")
    logger.info("")


def _detect_fixed_panel_strategy(config_file: Path | None, logger: logging.Logger) -> bool:
    """
    Detect if training config uses fixed_panel feature selection strategy.

    Args:
        config_file: Path to training config YAML
        logger: Logger instance

    Returns:
        True if feature_selection_strategy is 'fixed_panel', False otherwise
    """
    if config_file is None or not config_file.exists():
        return False

    try:
        from ced_ml.config.loader import load_training_config

        # Load config without overrides (just check base config)
        config = load_training_config(
            config_file=config_file,
            overrides=[],
        )
        strategy = config.features.feature_selection_strategy
        logger.debug(f"Detected feature_selection_strategy: {strategy}")
        return strategy == "fixed_panel"
    except Exception as e:
        logger.warning(f"Could not detect feature selection strategy: {e}")
        return False


def run_pipeline(
    config_file: Path | None,
    splits_config_file: Path | None,
    infile: Path | None,
    split_dir: Path | None,
    models: list[str],
    split_seeds: list[int],
    run_id: str | None,
    outdir: Path | None,
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
    enable_permutation_test: bool,
    permutation_n_perms: int,
    permutation_n_jobs: int,
    log_file: Path | None,
    cli_args: dict,
    overrides: list[str],
    log_level: int,
    hpc: bool = False,
    hpc_config_file: Path | None = None,
    dry_run: bool = False,
):
    """
    Run the complete ML pipeline end-to-end.

    Workflow:
    1. Generate splits (if needed)
    2. Train base models (all specified models x all seeds)
    3. Aggregate results per model
    4. Train ensemble (if enabled)
    5. Aggregate ensemble results
    6. Optimize panel per model (if enabled and not using fixed_panel)
    7. Generate consensus panel (if enabled and not using fixed_panel)
    8. Permutation testing for statistical significance (if enabled)

    Note:
        Panel optimization and consensus generation are automatically skipped
        when using feature_selection_strategy='fixed_panel', since the panel
        is pre-specified and cannot be optimized.

    Args:
        config_file: Path to training config YAML
        splits_config_file: Path to splits config YAML (for split generation)
        infile: Input data file (Parquet/CSV, auto-discovered if None)
        split_dir: Directory for split indices
        models: List of model names to train
        split_seeds: List of split seeds to train
        run_id: Shared run identifier (auto-generated if None)
        outdir: Results output directory
        enable_ensemble: Train stacking ensemble
        enable_consensus: Generate cross-model consensus panel
        enable_optimize_panel: Run panel optimization
        enable_permutation_test: Run permutation testing for significance
        permutation_n_perms: Number of permutations (default: 200)
        permutation_n_jobs: Parallel jobs for permutation testing (-1 for all)
        log_file: Path to save pipeline logs (None for console only)
        cli_args: Additional CLI arguments for training
        overrides: Config override strings
        log_level: Logging level constant
    """
    # HPC mode: submit LSF jobs and exit
    if hpc:
        _run_hpc_mode(
            config_file=config_file,
            splits_config_file=splits_config_file,
            infile=infile,
            split_dir=split_dir,
            models=models,
            split_seeds=split_seeds,
            run_id=run_id,
            outdir=outdir,
            enable_ensemble=enable_ensemble,
            enable_consensus=enable_consensus,
            enable_optimize_panel=enable_optimize_panel,
            enable_permutation_test=enable_permutation_test,
            permutation_n_perms=permutation_n_perms,
            permutation_n_jobs=permutation_n_jobs,
            hpc_config_file=hpc_config_file,
            dry_run=dry_run,
            log_level=log_level,
        )
        return

    # Eager run_id: generate before logger so log file is named correctly
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Auto-file-logging: use explicit log_file if provided, otherwise auto-generate
    if log_file is None:
        logger = setup_command_logger(
            command="run-pipeline",
            log_level=log_level,
            outdir=outdir,
            run_id=run_id,
            logger_name="ced_ml.pipeline",
        )
    else:
        logger = setup_logger("ced_ml.pipeline", level=log_level, log_file=log_file)
        logger.info(f"Logging to file: {log_file}")

    pipeline_t0 = time.monotonic()
    step_timings: list[tuple[str, float]] = []

    # Auto-discover input file if not provided
    if infile is None:
        logger.info("Auto-discovering input data file...")
        infile = _discover_input_file(config_file, outdir, logger)
        logger.info(f"Using input file: {infile}")

    # Detect if using fixed panel strategy
    using_fixed_panel = _detect_fixed_panel_strategy(config_file, logger)

    # Auto-disable panel optimization and consensus for fixed panels
    if using_fixed_panel:
        if enable_optimize_panel or enable_consensus:
            logger.info(
                "Detected feature_selection_strategy='fixed_panel' - "
                "disabling panel optimization and consensus generation"
            )
        enable_optimize_panel = False
        enable_consensus = False

    logger.info("=" * 70)
    logger.info("CeD-ML Full Pipeline")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Split seeds: {split_seeds}")
    logger.info(f"Input file: {infile}")
    logger.info(f"Output dir: {outdir}")
    logger.info(f"Config file: {config_file or 'default'}")
    logger.info(f"Ensemble: {'enabled' if enable_ensemble else 'disabled'}")
    logger.info(f"Consensus panel: {'enabled' if enable_consensus else 'disabled'}")
    logger.info(f"Panel optimization: {'enabled' if enable_optimize_panel else 'disabled'}")
    logger.info(f"Permutation test: {'enabled' if enable_permutation_test else 'disabled'}")
    if enable_permutation_test:
        logger.info(f"  n_perms={permutation_n_perms}, n_jobs={permutation_n_jobs}")
    if using_fixed_panel:
        logger.info("Feature selection: fixed_panel (optimization/consensus N/A)")
    logger.info(
        f"Total training jobs: {len(models)} models x {len(split_seeds)} seeds "
        f"= {len(models) * len(split_seeds)}"
    )
    logger.info("=" * 70)

    # Step 1: Generate splits (if needed)
    if split_dir is None:
        split_dir = outdir.parent / "splits"

    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Generate Splits")
    logger.info("=" * 70)

    t0 = time.monotonic()
    _ensure_splits_exist(
        split_dir=split_dir,
        infile=infile,
        config_file=splits_config_file,
        overrides=overrides,
        log_level=log_level,
        logger=logger,
    )
    step_timings.append(("Splits", time.monotonic() - t0))

    # Step 2: Train base models
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Train Base Models")
    logger.info("=" * 70)

    shared_run_id = run_id

    t0 = time.monotonic()
    for model_name in models:
        for split_seed in split_seeds:
            logger.info(f"\nTraining {model_name} with split_seed={split_seed}")

            train_cli_args = {
                "infile": str(infile),
                "split_dir": str(split_dir),
                "model": model_name,
                "split_seed": split_seed,
                "outdir": str(outdir),
                "run_id": shared_run_id,
                **cli_args,
            }

            run_train(
                config_file=config_file,
                cli_args=train_cli_args,
                overrides=overrides,
                log_level=log_level,
            )
            logger.info(f"Completed {model_name} seed={split_seed}")

    step_timings.append(("Training", time.monotonic() - t0))

    # Step 3: Aggregate base models
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Aggregate Base Model Results")
    logger.info("=" * 70)

    t0 = time.monotonic()
    for model_name in models:
        logger.info(f"\nAggregating {model_name}")

        model_dir = outdir / f"run_{shared_run_id}" / model_name

        run_aggregate_splits(
            results_dir=str(model_dir),
            stability_threshold=0.75,
            target_specificity=0.95,
            plot_formats=["png"],
            n_boot=500,
            log_level=log_level,
        )
    step_timings.append(("Aggregation", time.monotonic() - t0))

    # Log aggregated metrics summary
    _log_aggregated_metrics_summary(
        logger=logger,
        outdir=outdir,
        run_id=shared_run_id,
        models=models,
    )

    # Step 4: Train ensemble (if enabled)
    if enable_ensemble:
        logger.info("\n" + "=" * 70)
        logger.info("Step 4: Train Ensemble Meta-Learner")
        logger.info("=" * 70)

        t0 = time.monotonic()
        for split_seed in split_seeds:
            logger.info(f"\nTraining ensemble with split_seed={split_seed}")

            run_train_ensemble(
                config_file=config_file,
                run_id=shared_run_id,
                split_seed=split_seed,
                base_models=None,  # Auto-detect from run_id
                results_dir=None,  # Auto-detect from run_id
                outdir=None,  # Use default
                meta_penalty=None,
                meta_c=None,
                log_level=log_level,
            )

        # Step 5: Aggregate ensemble
        logger.info("\n" + "=" * 70)
        logger.info("Step 5: Aggregate Ensemble Results")
        logger.info("=" * 70)

        ensemble_dir = outdir / f"run_{shared_run_id}" / "ENSEMBLE"

        run_aggregate_splits(
            results_dir=str(ensemble_dir),
            stability_threshold=0.75,
            target_specificity=0.95,
            plot_formats=["png"],
            n_boot=500,
            log_level=log_level,
        )
        step_timings.append(("Ensemble", time.monotonic() - t0))

        # Log ensemble metrics
        _log_aggregated_metrics_summary(
            logger=logger,
            outdir=outdir,
            run_id=shared_run_id,
            models=["ENSEMBLE"],
        )

    # Step 6: Optimize panel (if enabled)
    if enable_optimize_panel:
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: Optimize Panel Sizes")
        logger.info("=" * 70)

        t0 = time.monotonic()
        # Auto-discover models with aggregated results
        results_dir = outdir
        model_dirs = discover_models_by_run_id(
            run_id=shared_run_id,
            results_dir=results_dir,
            model_filter=None,
        )

        for model_name, results_dir in model_dirs.items():
            logger.info(f"\nOptimizing panel for {model_name}")

            # Load rfe_tune_spaces from optimize_panel.yaml if available
            _op_config_path = get_analysis_dir() / "configs" / "optimize_panel.yaml"
            _op_rfe_spaces = None
            _op_cfg = {}
            if _op_config_path.exists():
                import yaml

                with open(_op_config_path) as _f:
                    _op_cfg = yaml.safe_load(_f) or {}
                _op_rfe_spaces = _op_cfg.get("rfe_tune_spaces")
                if _op_rfe_spaces:
                    logger.info("Loaded rfe_tune_spaces from %s", _op_config_path)

            run_optimize_panel_aggregated(
                results_dir=results_dir,
                infile=str(infile),
                split_dir=str(split_dir),
                model_name=model_name,
                stability_threshold=_op_cfg.get("stability_threshold", 0.90),
                start_size=_op_cfg.get("start_size"),
                min_size=_op_cfg.get("min_size", 5),
                min_auroc_frac=_op_cfg.get("min_auroc_frac", 0.90),
                cv_folds=_op_cfg.get("cv_folds", 5),
                step_strategy=_op_cfg.get("step_strategy", "fine"),
                retune_n_trials=_op_cfg.get("retune_trials", 60),
                outdir=None,
                log_level=log_level,
                n_jobs=-1,
                rfe_tune_spaces=_op_rfe_spaces,
            )
        step_timings.append(("Panel optimization", time.monotonic() - t0))

    # Step 7: Generate consensus panel (if enabled)
    if enable_consensus:
        logger.info("\n" + "=" * 70)
        logger.info("Step 7: Generate Consensus Panel")
        logger.info("=" * 70)

        t0 = time.monotonic()
        # Load consensus config for parameters (corr_threshold, stability_threshold, etc.)
        from ced_ml.config.loader import load_yaml

        consensus_config_path = get_analysis_dir() / "configs" / "consensus_panel.yaml"
        consensus_cfg = {}
        if consensus_config_path.exists():
            consensus_cfg = load_yaml(consensus_config_path)

        run_consensus_panel(
            run_id=shared_run_id,
            infile=str(infile),
            split_dir=str(split_dir),
            stability_threshold=consensus_cfg.get("stability_threshold", 0.90),
            corr_threshold=consensus_cfg.get("corr_threshold", 0.85),
            target_size=consensus_cfg.get("target_size", 25),
            rfe_weight=consensus_cfg.get("rfe_weight", 0.5),
            rra_method=consensus_cfg.get("rra_method", "geometric_mean"),
            outdir=None,
            log_level=log_level,
        )
        step_timings.append(("Consensus panel", time.monotonic() - t0))

    # Step 8: Permutation testing (if enabled)
    if enable_permutation_test:
        logger.info("\n" + "=" * 70)
        logger.info("Step 8: Permutation Testing for Statistical Significance")
        logger.info("=" * 70)

        t0 = time.monotonic()
        # Test all base models (exclude ENSEMBLE - permutation test requires retraining)
        models_to_test = [m for m in models if m != "ENSEMBLE"]

        for model_name in models_to_test:
            for split_seed in split_seeds:
                logger.info(f"\nTesting {model_name} (split_seed={split_seed})")

                try:
                    run_permutation_test_cli(
                        run_id=shared_run_id,
                        model=model_name,
                        split_seed=split_seed,
                        n_perms=permutation_n_perms,
                        n_jobs=permutation_n_jobs,
                        log_level=log_level,
                    )
                    logger.info(f"Completed permutation test for {model_name} seed={split_seed}")
                except Exception as e:
                    logger.warning(
                        f"Permutation test failed for {model_name} seed={split_seed}: {e}"
                    )

        step_timings.append(("Permutation testing", time.monotonic() - t0))

    # Final summary
    total_elapsed = time.monotonic() - pipeline_t0
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"Run ID: {shared_run_id}")
    logger.info(f"Results: {outdir / f'run_{shared_run_id}'}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    logger.info("Step timings:")
    for step_name, elapsed in step_timings:
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        logger.info(f"  {step_name:<25s} {h:02d}:{m:02d}:{s:02d}")
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)
    logger.info(f"  {'TOTAL':<25s} {h:02d}:{m:02d}:{s:02d}")
    logger.info("=" * 70)
