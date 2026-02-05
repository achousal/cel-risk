"""
Main CLI entry point for CeD-ML pipeline.

Provides subcommands:
  - ced save-splits: Generate train/val/test splits
  - ced train: Train ML models
  - ced aggregate-splits: Aggregate results across split seeds
  - ced eval-holdout: Evaluate on holdout set
"""

import click

from ced_ml import __version__
from ced_ml.data.schema import ModelName


@click.group()
@click.version_option(version=__version__, prog_name="ced")
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging level (default: info). Use 'debug' for detailed algorithm insights.",
)
@click.pass_context
def cli(ctx, log_level):
    """
    CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

    A modular, reproducible ML pipeline for predicting incident Celiac Disease
    risk from proteomics biomarkers.
    """
    import logging

    from ced_ml.utils.random import apply_seed_global

    # Store global options in context
    ctx.ensure_object(dict)

    # Convert log_level string to logging constant
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    ctx.obj["log_level"] = log_level_map[log_level.lower()]

    # Apply SEED_GLOBAL if set (for single-threaded reproducibility debugging)
    seed_applied = apply_seed_global()
    if seed_applied is not None:
        ctx.obj["seed_global"] = seed_applied


@cli.command("save-splits")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default="splits",
    help="Output directory for splits (default: splits/)",
)
@click.option(
    "--mode",
    type=click.Choice(["development", "holdout"]),
    default=None,
    help="Split mode: development (TRAIN/VAL/TEST) or holdout (TRAIN/VAL/TEST + HOLDOUT)",
)
@click.option(
    "--scenarios",
    multiple=True,
    default=None,
    help="Scenarios to generate (can be repeated)",
)
@click.option(
    "--n-splits",
    type=int,
    default=None,
    help="Number of repeated splits with different seeds",
)
@click.option(
    "--val-size",
    type=float,
    default=None,
    help="Validation set proportion (0-1)",
)
@click.option(
    "--test-size",
    type=float,
    default=None,
    help="Test set proportion (0-1)",
)
@click.option(
    "--holdout-size",
    type=float,
    default=None,
    help="Holdout set proportion (only if mode=holdout)",
)
@click.option(
    "--seed-start",
    type=int,
    default=None,
    help="Starting random seed",
)
@click.option(
    "--prevalent-train-only",
    is_flag=True,
    default=None,
    help="Restrict prevalent cases to TRAIN set only (prevents reverse causality)",
)
@click.option(
    "--prevalent-train-frac",
    type=float,
    default=None,
    help="Fraction of prevalent cases to include in TRAIN (0-1)",
)
@click.option(
    "--train-control-per-case",
    type=float,
    default=None,
    help="Downsample TRAIN controls to N per case (e.g., 5 for 1:5 ratio)",
)
@click.option(
    "--eval-control-per-case",
    type=float,
    default=None,
    help="Downsample VAL/TEST controls to N per case",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing split files",
)
@click.option(
    "--temporal-split",
    is_flag=True,
    default=None,
    help="Enable temporal (chronological) validation splits",
)
@click.option(
    "--temporal-column",
    type=str,
    default=None,
    help="Column name for temporal ordering (e.g., 'CeD_date')",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.pass_context
def save_splits(ctx, config, **kwargs):
    """Generate train/val/test splits with stratification and optional downsampling."""
    from ced_ml.cli.save_splits import run_save_splits

    # Collect CLI args
    cli_args = {k: v for k, v in kwargs.items() if k != "override"}
    overrides = list(kwargs.get("override", []))

    # Run split generation
    run_save_splits(
        config_file=config,
        cli_args=cli_args,
        overrides=overrides,
        log_level=ctx.obj.get("log_level"),
    )


@cli.command("train")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    help="Directory containing split indices",
)
@click.option(
    "--scenario",
    default=None,
    help="Scenario name (must match split scenario)",
)
@click.option(
    "--model",
    default=ModelName.LR_EN,
    help="Model to train (LR, LR_EN, RF, XGBoost, LinSVM_cal, etc.)",
)
@click.option(
    "--split-seed",
    type=int,
    default=None,
    help="Split seed to use (if multiple splits generated). Mutually exclusive with --split-seeds.",
)
@click.option(
    "--split-seeds",
    type=str,
    default=None,
    help="Comma-separated list of split seeds (e.g., '0,1,2'). Mutually exclusive with --split-seed.",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default="results",
    help="Output directory for results",
)
@click.option(
    "--fixed-panel",
    type=click.Path(exists=True),
    default=None,
    help="Path to CSV with fixed feature panel (bypasses feature selection)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Shared run ID (default: auto-generated timestamp). Use to group multiple splits/models under one run.",
)
@click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Submit LSF jobs to HPC cluster instead of running locally",
)
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    default=None,
    help="HPC config file (default: configs/pipeline_hpc.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview HPC job submission without executing (--hpc mode only)",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.pass_context
def train(ctx, config, split_seed, split_seeds, **kwargs):
    """Train machine learning models with nested cross-validation.

    LOCAL MODE (default):
        Train a single model on a single split seed locally.

    HPC MODE (--hpc):
        Submit LSF job(s) to HPC cluster. Supports:
        - Single seed: ced train --model LR_EN --split-seed 0 --hpc
        - Multiple seeds: ced train --model LR_EN --split-seeds 0,1,2 --hpc

    Examples:

        # Local training (single model, single seed)
        ced train --infile data/celiac.parquet --split-dir splits/ --model LR_EN --split-seed 0

        # HPC: Submit single job
        ced train --infile data/celiac.parquet --split-dir splits/ \\
            --model LR_EN --split-seed 0 --hpc

        # HPC: Submit multiple jobs for all seeds
        ced train --infile data/celiac.parquet --split-dir splits/ \\
            --model LR_EN --split-seeds 0,1,2,3,4 --hpc

        # HPC: Preview without submitting
        ced train --infile data/celiac.parquet --split-dir splits/ \\
            --model LR_EN --split-seeds 0,1,2 --hpc --dry-run
    """
    from pathlib import Path

    hpc_flag = kwargs.pop("hpc", False)
    hpc_config_cli = kwargs.pop("hpc_config", None)
    dry_run_flag = kwargs.pop("dry_run", False)

    # Validate mutually exclusive options
    if split_seed is not None and split_seeds is not None:
        raise click.UsageError(
            "--split-seed and --split-seeds are mutually exclusive. Use one or the other."
        )

    # Determine which seeds to process
    if split_seeds is not None:
        seed_list = [int(s.strip()) for s in split_seeds.split(",")]
    elif split_seed is not None:
        seed_list = [split_seed]
    else:
        # Default to seed 0 for local mode, require explicit seeds for HPC mode
        if hpc_flag:
            raise click.UsageError(
                "HPC mode (--hpc) requires either --split-seed or --split-seeds. "
                "Example: --split-seed 0 or --split-seeds 0,1,2,3,4"
            )
        seed_list = [0]

    # HPC mode: submit jobs and exit
    if hpc_flag:
        from ced_ml.hpc.lsf import (
            _build_training_command,
            build_job_script,
            detect_environment,
            load_hpc_config,
            submit_job,
        )
        from ced_ml.utils.paths import get_project_root

        # Resolve paths
        infile = Path(kwargs["infile"]).resolve()
        split_dir = Path(kwargs["split_dir"]).resolve() if kwargs.get("split_dir") else None
        outdir = Path(kwargs.get("outdir", "results")).resolve()
        config_file = Path(config).resolve() if config else None
        model = kwargs.get("model", ModelName.LR_EN)

        if not split_dir:
            raise click.UsageError("HPC mode requires --split-dir to be specified.")

        # Generate run_id if not provided
        run_id = kwargs.get("run_id")
        if not run_id:
            from datetime import datetime

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load HPC config
        if not hpc_config_cli:
            root = get_project_root()
            candidates = [
                root / "configs" / "pipeline_hpc.yaml",
                root / "analysis" / "configs" / "pipeline_hpc.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    hpc_config_cli = str(candidate)
                    break
            else:
                raise click.UsageError(
                    "HPC mode requires --hpc-config or configs/pipeline_hpc.yaml to exist. "
                    "Searched: configs/ and analysis/configs/"
                )

        hpc_config = load_hpc_config(Path(hpc_config_cli))
        env_info = detect_environment(get_project_root())

        # Build bsub parameters
        default_resources = hpc_config.get_resources("default")
        root = get_project_root()
        log_dir = root / "logs" / "hpc" / "training" / f"run_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        bsub_params = {
            "project": hpc_config.project,
            "env_activation": env_info.activation_cmd,
            "log_dir": log_dir,
            **default_resources,
        }

        # Submit jobs
        n_jobs = len(seed_list)
        click.echo(f"\nSubmitting {n_jobs} training job(s) to HPC:")
        click.echo(f"  Model: {model}")
        click.echo(f"  Seeds: {seed_list}")
        click.echo(f"  Run ID: {run_id}")
        click.echo(f"  Output: {outdir}")

        submitted_jobs = []
        for seed in seed_list:
            job_name = f"CeD_{run_id}_{model}_s{seed}"

            # Build training command
            cmd_kwargs = {
                "infile": infile,
                "split_dir": split_dir,
                "outdir": outdir,
                "model": model,
                "split_seed": seed,
                "run_id": run_id,
            }
            if config_file:
                cmd_kwargs["config_file"] = config_file
            else:
                # Use a default config path if available
                default_config = root / "configs" / "training_config.yaml"
                if not default_config.exists():
                    default_config = root / "analysis" / "configs" / "training_config.yaml"
                if default_config.exists():
                    cmd_kwargs["config_file"] = default_config
                else:
                    raise click.UsageError(
                        "HPC mode requires --config or configs/training_config.yaml to exist."
                    )

            command = _build_training_command(**cmd_kwargs)

            script = build_job_script(
                job_name=job_name,
                command=command,
                **bsub_params,
            )

            job_id = submit_job(script, dry_run=dry_run_flag)

            if job_id:
                submitted_jobs.append((seed, job_id))
                click.echo(f"  Submitted seed {seed}: job_id={job_id}")
            elif dry_run_flag:
                click.echo(f"  [DRY RUN] seed {seed}: {job_name}")
            else:
                click.echo(f"  seed {seed}: Submission failed", err=True)

        if dry_run_flag:
            click.echo("\n[DRY RUN] No jobs were actually submitted.")
        elif submitted_jobs:
            click.echo(f"\nSuccessfully submitted {len(submitted_jobs)} job(s)")
            click.echo(f"Monitor with: bjobs -J 'CeD_{run_id}_{model}_*'")
            click.echo(f"Logs in: {log_dir}")

        return

    # Local mode: run training directly
    from ced_ml.cli.train import run_train

    # For local mode with multiple seeds, run sequentially
    for i, seed in enumerate(seed_list):
        if len(seed_list) > 1:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"Training seed {seed} ({i + 1}/{len(seed_list)})")
            click.echo(f"{'=' * 70}\n")

        # Collect CLI args
        cli_args = {k: v for k, v in kwargs.items() if k != "override"}
        cli_args["split_seed"] = seed
        overrides = list(kwargs.get("override", []))

        # Run training
        run_train(
            config_file=config,
            cli_args=cli_args,
            overrides=overrides,
            log_level=ctx.obj.get("log_level"),
        )

    if len(seed_list) > 1:
        click.echo(f"\n{'=' * 70}")
        click.echo(f"Training complete for all {len(seed_list)} seed(s)")
        click.echo(f"{'=' * 70}\n")


@cli.command("aggregate-splits")
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing split_seedX subdirectories (mutually exclusive with --run-id)",
)
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Run ID for auto-detection (e.g., 20260127_115115, mutually exclusive with --results-dir)",
)
@click.option(
    "--model",
    type=str,
    required=False,
    help="Model name for --run-id mode (e.g., LR_EN). If not specified with --run-id, aggregates all models for the run.",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=0.75,
    help="Fraction of splits a feature must appear in to be 'stable' (default: 0.75)",
)
@click.option(
    "--target-specificity",
    type=float,
    default=0.95,
    help="Target specificity for alpha threshold (default: 0.95)",
)
@click.option(
    "--plot-formats",
    multiple=True,
    default=["png"],
    help="Plot output formats (can be repeated, e.g., --plot-formats png --plot-formats pdf)",
)
@click.option(
    "--n-boot",
    type=int,
    default=500,
    help="Number of bootstrap iterations for CIs (reserved for future use)",
)
@click.pass_context
def aggregate_splits(ctx, **kwargs):
    """
    Aggregate results across multiple split seeds.

    Discovers split_seedX subdirectories, collects metrics, computes pooled
    metrics, generates aggregated plots with CI bands, and builds consensus
    feature panels. Results are saved to an aggregated/ subdirectory.

    Output structure:
        aggregated/
          core/                 # Pooled and summary metrics
          cv/                   # CV metrics summary
          preds/                # Pooled predictions
          reports/              # Feature stability and consensus panels
          plots/                # Aggregated ROC, PR, calibration, DCA plots
          diagnostics/          # Diagnostic CSV files (calibration, DCA, screening, learning curves)

    Usage:
        # Explicit path (original)
        ced aggregate-splits --results-dir results/run_20260127_115115/LR_EN/

        # Auto-detection - single model
        ced aggregate-splits --run-id 20260127_115115 --model LR_EN

        # Auto-detection - all models for run (new)
        ced aggregate-splits --run-id 20260127_115115

    Example:
        ced aggregate-splits --results-dir results_local/
        ced aggregate-splits --results-dir results_local/ --stability-threshold 0.80
        ced aggregate-splits --results-dir results_local/ --plot-formats png --plot-formats pdf
        ced aggregate-splits --run-id 20260127_115115 --model LR_EN
        ced aggregate-splits --run-id 20260127_115115  # All models
    """

    from ced_ml.cli.aggregate_splits import run_aggregate_splits
    from ced_ml.cli.discovery import (
        discover_models_for_run,
        get_run_path,
        resolve_run_id,
    )

    # Validate mutually exclusive options
    results_dir = kwargs.get("results_dir")
    run_id = kwargs.get("run_id")
    model = kwargs.get("model")

    if not results_dir and not run_id:
        raise click.UsageError(
            "Either --results-dir or --run-id must be provided.\n"
            "Examples:\n"
            "  ced aggregate-splits --results-dir results/run_20260127_115115/LR_EN/\n"
            "  ced aggregate-splits --run-id 20260127_115115 --model LR_EN\n"
            "  ced aggregate-splits --run-id 20260127_115115  # All models"
        )

    if results_dir and run_id:
        raise click.UsageError(
            "--results-dir and --run-id are mutually exclusive.\n"
            "Use --results-dir for explicit path OR --run-id for auto-detection."
        )

    # Convert tuple to list for plot_formats
    kwargs["plot_formats"] = list(kwargs["plot_formats"]) if kwargs["plot_formats"] else ["png"]

    # Auto-detect results_dir from run_id
    if run_id:
        try:
            # Resolve run_id and get run path
            run_id = resolve_run_id(run_id)
            run_path = get_run_path(run_id)

            # Discover models
            if model:
                # Validate specific model exists
                model_path = run_path / model
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Results directory not found for model {model}, run {run_id}.\n"
                        f"Tried: {model_path}"
                    )
                model_dirs = {model: str(model_path)}
            else:
                # Find all models
                models = discover_models_for_run(run_id, skip_ensemble=False)
                if not models:
                    raise FileNotFoundError(f"No models found for run {run_id}")
                model_dirs = {m: str(run_path / m) for m in models}

            # Display what we found
            if len(model_dirs) == 1:
                model_name, results_dir = next(iter(model_dirs.items()))
                click.echo(f"Auto-detected results directory: {results_dir}")
            else:
                click.echo(
                    f"Auto-discovered {len(model_dirs)} model(s) with results for run {run_id}:"
                )
                for model_name in sorted(model_dirs.keys()):
                    click.echo(f"  - {model_name}")
                click.echo("")

            # Process each model
            for model_name, results_dir in model_dirs.items():
                if len(model_dirs) > 1:
                    click.echo(f"\n{'=' * 70}")
                    click.echo(f"Aggregating splits for: {model_name}")
                    click.echo(f"{'=' * 70}\n")

                # Prepare kwargs for this model
                model_kwargs = kwargs.copy()
                model_kwargs["results_dir"] = results_dir
                model_kwargs.pop("run_id", None)
                model_kwargs.pop("model", None)

                run_aggregate_splits(**model_kwargs, log_level=ctx.obj.get("log_level"))

            if len(model_dirs) > 1:
                click.echo(f"\n{'=' * 70}")
                click.echo(f"Aggregation complete for all {len(model_dirs)} model(s)")
                click.echo(f"{'=' * 70}\n")

            return

        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            ctx.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            ctx.exit(1)

    # Single explicit results_dir provided
    kwargs.pop("run_id", None)
    kwargs.pop("model", None)

    run_aggregate_splits(**kwargs, log_level=ctx.obj.get("log_level"))


@cli.command("eval-holdout")
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--model-artifact",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model (.joblib file)",
)
@click.option(
    "--holdout-idx",
    type=click.Path(exists=True),
    required=True,
    help="Path to holdout indices CSV",
)
@click.option(
    "--outdir",
    type=click.Path(),
    required=True,
    help="Output directory for holdout evaluation results",
)
@click.option(
    "--compute-dca",
    is_flag=True,
    help="Compute decision curve analysis",
)
@click.pass_context
def eval_holdout(ctx, **kwargs):
    """Evaluate trained model on holdout set (run ONCE only)."""
    from ced_ml.cli.eval_holdout import run_eval_holdout

    run_eval_holdout(**kwargs)


@cli.command("train-ensemble")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file (uses ensemble section)",
)
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing base model results (auto-detected if --run-id provided)",
)
@click.option(
    "--base-models",
    type=str,
    default=None,
    help="Comma-separated list of base models (auto-detected if --run-id provided)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run ID for auto-detection (e.g., 20260127_115115). Auto-discovers results-dir and base-models.",
)
@click.option(
    "--split-seed",
    type=int,
    default=None,
    help="Single split seed (default: auto-detect all from --run-id)",
)
@click.option(
    "--split-seeds",
    type=str,
    default=None,
    help="Comma-separated list of split seeds (e.g., '72,73,74'). Overrides --split-seed.",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results_dir/ENSEMBLE/split_{seed})",
)
@click.option(
    "--meta-penalty",
    type=click.Choice(["l2", "l1", "elasticnet", "none"]),
    default=None,
    help="Meta-learner regularization penalty",
)
@click.option(
    "--meta-C",
    type=float,
    default=None,
    help="Meta-learner regularization strength (inverse)",
)
@click.pass_context
def train_ensemble(ctx, config, base_models, split_seed, split_seeds, **kwargs):
    """Train stacking ensemble from base model OOF predictions.

    This command collects out-of-fold (OOF) predictions from previously trained
    base models and trains a meta-learner (Logistic Regression) to combine them.

    AUTO-DETECTION MODE (recommended):
        Use --run-id to automatically discover results directory, base models,
        and split seeds:
            ced train-ensemble --run-id 20260127_115115

    EXPLICIT SPLITS:
        Specify which splits to train:
            ced train-ensemble --run-id 20260127_115115 --split-seeds 72,73,74
            ced train-ensemble --run-id 20260127_115115 --split-seed 0

    MANUAL MODE:
        Explicitly specify results directory and base models:
            ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost --split-seed 0

    Requirements:
        - Base models must be trained first using 'ced train'
        - OOF predictions must exist in results_dir/{model}/splits/split_seed{N}/preds/

    Examples:
        # Auto-detect everything from run-id (simplest, trains ALL splits)
        ced train-ensemble --run-id 20260127_115115

        # Train specific splits
        ced train-ensemble --run-id 20260127_115115 --split-seeds 72,73,74,75

        # Train single split
        ced train-ensemble --run-id 20260127_115115 --split-seed 0

        # With config file
        ced train-ensemble --config configs/training_config.yaml --run-id 20260127_115115
    """
    from ced_ml.cli.discovery import discover_split_seeds_for_run
    from ced_ml.cli.train_ensemble import run_train_ensemble

    run_id = kwargs.get("run_id")

    # Determine split seeds to process
    seeds_to_train: list[int] = []

    if split_seeds:
        # Explicit list provided
        seeds_to_train = [int(s.strip()) for s in split_seeds.split(",")]
    elif split_seed is not None:
        # Single seed provided
        seeds_to_train = [split_seed]
    elif run_id:
        # Auto-discover from run_id
        try:
            seeds_to_train = discover_split_seeds_for_run(
                run_id=run_id,
                results_dir=kwargs.get("results_dir"),
                skip_ensemble=True,
            )
            click.echo(f"Auto-discovered {len(seeds_to_train)} split seed(s): {seeds_to_train}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            ctx.exit(1)
    else:
        # No run_id and no seeds - require explicit seed
        click.echo(
            "Error: Either --run-id (for auto-detection) or --split-seed must be provided.",
            err=True,
        )
        ctx.exit(1)

    # Parse base models from comma-separated string
    base_model_list = None
    if base_models:
        base_model_list = [m.strip() for m in base_models.split(",")]

    # Train ensemble for each split seed
    n_seeds = len(seeds_to_train)
    for i, seed in enumerate(seeds_to_train, 1):
        if n_seeds > 1:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"Training ensemble for split_seed={seed} ({i}/{n_seeds})")
            click.echo(f"{'=' * 70}\n")

        try:
            run_train_ensemble(
                config_file=config,
                base_models=base_model_list,
                split_seed=seed,
                **kwargs,
                log_level=ctx.obj.get("log_level"),
            )
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            ctx.exit(1)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            ctx.exit(1)

    if n_seeds > 1:
        click.echo(f"\n{'=' * 70}")
        click.echo(f"Ensemble training complete for all {n_seeds} split(s)")
        click.echo(f"{'=' * 70}\n")


@cli.command("optimize-panel")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--results-dir",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Path to model results directory (e.g., results/run_20260127_115115/LR_EN/). Mutually exclusive with --run-id.",
)
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Run ID to auto-discover all models (e.g., 20260127_115115). Mutually exclusive with --results-dir.",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=False,
    help="Input data file (Parquet/CSV). Auto-detected from run metadata if using --run-id.",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing split indices. Auto-detected from run metadata if using --run-id.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (auto-detected from results directory if not provided, or filter when using --run-id)",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=None,
    help="Minimum selection fraction for stable proteins (default: 0.75)",
)
@click.option(
    "--start-size",
    type=int,
    default=None,
    help="Cap starting panel to top N proteins by selection frequency (default: no cap)",
)
@click.option(
    "--min-size",
    type=int,
    default=None,
    help="Minimum panel size to evaluate (default: 5)",
)
@click.option(
    "--min-auroc-frac",
    type=float,
    default=None,
    help="Early stop if AUROC drops below this fraction of max (default: 0.90)",
)
@click.option(
    "--cv-folds",
    type=int,
    default=None,
    help="CV folds for OOF AUROC estimation (default: 0 = skip, validation-only)",
)
@click.option(
    "--step-strategy",
    type=click.Choice(["geometric", "fine", "linear"]),
    default=None,
    help="Feature elimination strategy (default: geometric)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results_dir/aggregated/optimize_panel/)",
)
@click.option(
    "--n-jobs",
    type=int,
    default=None,
    help="Parallel jobs for multi-seed RFE (1=sequential, -1=all CPUs, default: 1 local / -1 HPC)",
)
@click.option(
    "--retune-trials",
    type=int,
    default=None,
    help="Optuna trials per evaluation point for hyperparameter re-tuning (default: 20)",
)
@click.option(
    "--split-seed",
    type=int,
    default=None,
    help="Run RFE for a single split seed only (for HPC parallelization). "
    "Saves result as joblib for later aggregation.",
)
@click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Submit LSF jobs to HPC cluster instead of running locally",
)
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    default=None,
    help="HPC config file (default: configs/pipeline_hpc.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview HPC job submission without executing (--hpc mode only)",
)
@click.option(
    "--corr-aware/--no-corr-aware",
    default=True,
    help="Enable correlation-aware pre-filtering (clusters correlated proteins before RFE)",
)
@click.option(
    "--corr-threshold",
    type=float,
    default=0.80,
    help="Correlation threshold for clustering (0.0-1.0, default: 0.80)",
)
@click.option(
    "--corr-method",
    type=click.Choice(["spearman", "pearson"], case_sensitive=False),
    default="spearman",
    help="Correlation method for clustering (default: spearman)",
)
@click.option(
    "--require-significance/--no-require-significance",
    default=None,
    help="Skip models that are not statistically significant (requires prior permutation testing)",
)
@click.option(
    "--significance-alpha",
    type=float,
    default=None,
    help="Significance threshold for gating (default: 0.05)",
)
@click.pass_context
def optimize_panel(ctx, config, **kwargs):
    """Find minimum viable panel from aggregated cross-split results.

    This command runs RFE on consensus stable proteins derived from ALL splits,
    providing a single authoritative panel size recommendation. Benefits:

    1. Uses consensus stable proteins from all splits (eliminates variability)
    2. Pools train/val data for maximum robustness
    3. Generates a single authoritative feature ranking
    4. Matches the aggregated analysis philosophy

    Requires prior aggregation:
        ced aggregate-splits --results-dir results/run_X/LR_EN

    Examples:

        # Use config file
        ced optimize-panel --config configs/optimize_panel.yaml

        # Optimize ALL models with aggregated results under a run-id (recommended)
        ced optimize-panel --run-id 20260127_115115

        # Optimize single model (explicit path)
        ced optimize-panel \\
          --results-dir results/run_20260127_115115/LR_EN \\
          --infile data/Celiac_dataset_proteomics_w_demo.parquet \\
          --split-dir splits/

        # Optimize specific model(s) by run-id
        ced optimize-panel --run-id 20260127_115115 --model LR_EN

        # Override config values with CLI args
        ced optimize-panel --config configs/optimize_panel.yaml --cv-folds 10

        # HPC mode - submit parallel jobs for all models
        ced optimize-panel --run-id 20260127_115115 --hpc

        # HPC mode - optimize specific model(s)
        ced optimize-panel --run-id 20260127_115115 --model LR_EN --hpc

        # HPC dry run - preview without submitting
        ced optimize-panel --run-id 20260127_115115 --hpc --dry-run

        # HPC with custom config
        ced optimize-panel --run-id 20260127_115115 --hpc --hpc-config configs/pipeline_custom.yaml

    Outputs (in results_dir/aggregated/optimize_panel/)
        - panel_curve_aggregated.csv: AUROC vs panel size
        - feature_ranking_aggregated.csv: Protein elimination order
        - recommended_panels_aggregated.json: Minimum sizes at thresholds
        - panel_curve_aggregated.png: Pareto curve visualization
    """
    import json
    from pathlib import Path

    from ced_ml.cli.optimize_panel import (
        discover_models_by_run_id,
        run_optimize_panel_aggregated,
    )

    # Load config file: use provided path, or auto-detect default if it exists
    config_params = {}
    default_config = Path(__file__).parent.parent.parent.parent / "configs" / "optimize_panel.yaml"
    config_path = config or (default_config if default_config.exists() else None)

    if config_path:
        import yaml

        with open(config_path) as f:
            config_params = yaml.safe_load(f) or {}
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Merge config with CLI args (CLI takes precedence)
    # Only use config values if CLI arg is None (not provided)
    for key in [
        "results_dir",
        "run_id",
        "infile",
        "split_dir",
        "model",
        "stability_threshold",
        "start_size",
        "min_size",
        "min_auroc_frac",
        "cv_folds",
        "step_strategy",
        "outdir",
        "n_jobs",
        "retune_trials",
        "retune_cv_folds",
        "corr_aware",
        "corr_threshold",
        "corr_method",
        "require_significance",
        "significance_alpha",
    ]:
        if kwargs.get(key) is None and key in config_params:
            kwargs[key] = config_params[key]
            click.echo(f"Using config value for {key}: {config_params[key]}")

    # Validate mutually exclusive options
    if kwargs.get("results_dir") and kwargs.get("run_id"):
        raise click.UsageError(
            "--results-dir and --run-id are mutually exclusive. Use one or the other."
        )

    if not kwargs.get("results_dir") and not kwargs.get("run_id"):
        raise click.UsageError("Either --results-dir or --run-id is required.")

    # Single-seed mode: run RFE for one seed and save joblib, then exit
    if kwargs.get("split_seed") is not None:
        import json
        import os
        from pathlib import Path

        from ced_ml.cli.optimize_panel import (
            discover_models_by_run_id,
            run_optimize_panel_single_seed,
        )
        from ced_ml.utils.paths import get_project_root

        if not kwargs.get("run_id"):
            raise click.UsageError("--split-seed requires --run-id.")

        run_id = kwargs["run_id"]
        seed = kwargs["split_seed"]

        results_dir_env = os.environ.get("CED_RESULTS_DIR")
        results_dir = Path(results_dir_env) if results_dir_env else get_project_root() / "results"

        model_dirs = discover_models_by_run_id(
            run_id=run_id,
            results_dir=results_dir,
            model_filter=kwargs.get("model"),
        )

        if not model_dirs:
            raise click.ClickException(
                f"No models found with run_id={run_id} and aggregated results"
            )

        # Auto-detect infile and split_dir from run metadata
        first_model_dir = next(iter(model_dirs.values()))
        run_level_dir = first_model_dir.parent.parent
        metadata_file = run_level_dir / "run_metadata.json"

        if not metadata_file.exists():
            model_dir = first_model_dir.parent
            split_dirs_list = list(model_dir.glob("splits/split_seed*"))
            if split_dirs_list:
                metadata_file = split_dirs_list[0] / "run_metadata.json"

        infile = kwargs.get("infile")
        split_dir = kwargs.get("split_dir")

        if metadata_file.exists() and (not infile or not split_dir):
            with open(metadata_file) as f:
                metadata = json.load(f)
            first_model_name = next(iter(model_dirs))
            model_meta = metadata.get("models", {}).get(first_model_name, {})
            if not infile:
                infile = model_meta.get("infile") or metadata.get("infile")
            if not split_dir:
                split_dir = (
                    model_meta.get("split_dir")
                    or metadata.get("split_dir")
                    or metadata.get("splits_dir")
                )

        if not infile or not split_dir:
            raise click.ClickException(
                "Could not auto-detect infile and split_dir. "
                "Please specify --infile and --split-dir manually."
            )

        for model_name, agg_dir in model_dirs.items():
            click.echo(f"Running single-seed RFE: {model_name} seed {seed}")
            run_optimize_panel_single_seed(
                results_dir=agg_dir,
                infile=infile,
                split_dir=split_dir,
                seed=seed,
                model_name=model_name,
                stability_threshold=kwargs.get("stability_threshold", 0.90),
                start_size=kwargs.get("start_size"),
                min_size=kwargs.get("min_size", 5),
                min_auroc_frac=kwargs.get("min_auroc_frac", 0.50),
                cv_folds=kwargs.get("cv_folds", 5),
                step_strategy=kwargs.get("step_strategy", "fine"),
                log_level=ctx.obj.get("log_level"),
                retune_n_trials=kwargs.get("retune_trials", 60),
                retune_n_jobs=kwargs.get("n_jobs") or 1,
                corr_aware=kwargs.get("corr_aware", True),
                corr_threshold=kwargs.get("corr_threshold", 0.80),
                corr_method=kwargs.get("corr_method", "spearman"),
                rfe_tune_spaces=config_params.get("rfe_tune_spaces"),
            )

        return

    # HPC mode: submit jobs and exit
    hpc_flag = kwargs.get("hpc", False)
    dry_run_flag = kwargs.get("dry_run", False)

    if hpc_flag:
        import os
        from pathlib import Path

        from ced_ml.hpc.lsf import (
            build_job_script,
            detect_environment,
            load_hpc_config,
            submit_job,
        )
        from ced_ml.utils.paths import get_project_root

        # Only --run-id mode is supported for HPC (not --results-dir)
        if not kwargs.get("run_id"):
            raise click.UsageError(
                "HPC mode (--hpc) requires --run-id. "
                "Use --run-id instead of --results-dir for parallel HPC execution."
            )

        run_id = kwargs["run_id"]

        # Load HPC config
        hpc_config_path = kwargs.get("hpc_config")
        if not hpc_config_path:
            root = get_project_root()
            candidates = [
                root / "configs" / "pipeline_hpc.yaml",
                root / "analysis" / "configs" / "pipeline_hpc.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    hpc_config_path = str(candidate)
                    break
            else:
                raise click.UsageError(
                    "HPC mode requires --hpc-config or configs/pipeline_hpc.yaml to exist. "
                    "Searched: configs/ and analysis/configs/"
                )

        hpc_config = load_hpc_config(Path(hpc_config_path))
        env_info = detect_environment(get_project_root())

        # Discover models to submit jobs for
        results_dir_env = os.environ.get("CED_RESULTS_DIR")
        results_dir = Path(results_dir_env) if results_dir_env else get_project_root() / "results"

        from ced_ml.cli.optimize_panel import discover_models_by_run_id

        model_dirs = discover_models_by_run_id(
            run_id=run_id,
            results_dir=results_dir,
            model_filter=kwargs.get("model"),
        )

        if not model_dirs:
            if kwargs.get("model"):
                raise click.ClickException(
                    f"No models matching '{kwargs['model']}' found with run_id={run_id} "
                    f"and aggregated results in {results_dir}"
                )
            else:
                raise click.ClickException(
                    f"No models found with run_id={run_id} and aggregated results in {results_dir}"
                )

        click.echo(f"\nSubmitting {len(model_dirs)} panel optimization job(s) to HPC:")
        for model_name in model_dirs.keys():
            click.echo(f"  - {model_name}")

        # Build bsub parameters from config + environment
        default_resources = hpc_config.get_resources("default")
        root = get_project_root()
        log_dir = root / "logs" / "hpc"
        log_dir.mkdir(parents=True, exist_ok=True)

        bsub_params = {
            "project": hpc_config.project,
            "env_activation": env_info.activation_cmd,
            "log_dir": log_dir,
            **default_resources,
        }

        # Build and submit jobs for each model
        from ced_ml.hpc.lsf import _build_panel_optimization_command

        submitted_jobs = []
        for model_name in model_dirs.keys():
            cmd = _build_panel_optimization_command(run_id=run_id, model=model_name)
            job_name = f"opt_panel_{model_name}_{run_id}"

            script = build_job_script(
                job_name=job_name,
                command=cmd,
                **bsub_params,
            )

            job_id = submit_job(script, dry_run=dry_run_flag)

            if job_id:
                submitted_jobs.append((model_name, job_id))
                click.echo(f"  Submitted {model_name}: job_id={job_id}")
            elif dry_run_flag:
                click.echo(f"  [DRY RUN] {model_name}: {job_name}")
            else:
                click.echo(f"  {model_name}: Submission failed", err=True)

        if dry_run_flag:
            click.echo("\n[DRY RUN] No jobs were actually submitted.")
        elif submitted_jobs:
            click.echo(f"\nSuccessfully submitted {len(submitted_jobs)} job(s)")
            click.echo("Monitor with: bjobs")

        return

    # Auto-discover models if using --run-id
    if kwargs.get("run_id"):
        run_id = kwargs["run_id"]

        # Support CED_RESULTS_DIR environment variable for testing
        import os

        from ced_ml.utils.paths import get_project_root

        results_dir_env = os.environ.get("CED_RESULTS_DIR")
        if results_dir_env:
            results_dir = Path(results_dir_env)
        else:
            results_dir = get_project_root() / "results"

        click.echo(f"Auto-discovering models for run_id={run_id} in {results_dir}")

        model_dirs = discover_models_by_run_id(
            run_id=run_id,
            results_dir=results_dir,
            model_filter=kwargs.get("model"),
        )

        if not model_dirs:
            if kwargs.get("model"):
                raise click.ClickException(
                    f"No models matching '{kwargs['model']}' found with run_id={run_id} "
                    f"and aggregated results in {results_dir}"
                )
            else:
                raise click.ClickException(
                    f"No models found with run_id={run_id} and aggregated results in {results_dir}"
                )

        click.echo(f"Found {len(model_dirs)} model(s) with aggregated results:")
        for model_name, results_dir in model_dirs.items():
            click.echo(f"  - {model_name}: {results_dir}")

        # Auto-detect infile and split_dir from first model's run metadata
        # (they should be the same across all models in the same run)
        first_model_dir = next(iter(model_dirs.values()))
        # model_dirs values are aggregated directories (run_{id}/MODEL/aggregated)
        # Go up two levels to reach run-level dir for shared run_metadata.json
        run_level_dir = first_model_dir.parent.parent
        metadata_file = run_level_dir / "run_metadata.json"

        # If not found at run level, search in split directories under model dir
        if not metadata_file.exists():
            model_dir = first_model_dir.parent
            split_dirs = list(model_dir.glob("splits/split_seed*"))
            if not split_dirs:
                split_dirs = list(model_dir.glob("split_seed*"))
            if split_dirs:
                metadata_file = split_dirs[0] / "run_metadata.json"

        if not metadata_file.exists():
            # Fallback: require manual specification
            if not kwargs.get("infile") or not kwargs.get("split_dir"):
                raise click.ClickException(
                    f"Could not find run_metadata.json in {first_model_dir} or split directories. "
                    f"Please specify --infile and --split-dir manually."
                )
            infile = kwargs["infile"]
            split_dir = kwargs["split_dir"]
        else:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Use metadata values, but allow CLI overrides
            # Shared metadata format: top-level or under models/{model_name}
            first_model_name = next(iter(model_dirs))
            model_meta = metadata.get("models", {}).get(first_model_name, {})
            infile = kwargs.get("infile") or model_meta.get("infile") or metadata.get("infile")
            split_dir = (
                kwargs.get("split_dir")
                or model_meta.get("split_dir")
                or metadata.get("split_dir")
                or metadata.get("splits_dir")
            )

            if not infile or not split_dir:
                raise click.ClickException(
                    "Could not auto-detect infile and split_dir from run metadata. "
                    "Please specify --infile and --split-dir manually."
                )

            click.echo("\nAuto-detected from run metadata:")
            click.echo(f"  Input file: {infile}")
            click.echo(f"  Split dir:  {split_dir}")

        click.echo("")

        # Run optimization for each discovered model
        for model_name, results_dir in model_dirs.items():
            click.echo(f"\n{'=' * 70}")
            click.echo(f"Optimizing panel for: {model_name}")
            click.echo(f"{'=' * 70}\n")

            # Default n_jobs: -1 on HPC (via config), -1 locally (all CPUs)
            n_jobs = kwargs.get("n_jobs") or config_params.get("n_jobs_hpc", -1)

            run_optimize_panel_aggregated(
                results_dir=results_dir,
                infile=infile,
                split_dir=split_dir,
                model_name=model_name,
                stability_threshold=kwargs.get("stability_threshold", 0.90),
                start_size=kwargs.get("start_size"),
                min_size=kwargs.get("min_size", 5),
                min_auroc_frac=kwargs.get("min_auroc_frac", 0.50),
                cv_folds=kwargs.get("cv_folds", 5),
                step_strategy=kwargs.get("step_strategy", "fine"),
                outdir=kwargs.get("outdir"),
                log_level=ctx.obj.get("log_level"),
                n_jobs=n_jobs,
                retune_n_trials=kwargs.get("retune_trials", 60),
                corr_aware=kwargs.get("corr_aware", True),
                corr_threshold=kwargs.get("corr_threshold", 0.80),
                corr_method=kwargs.get("corr_method", "spearman"),
                rfe_tune_spaces=config_params.get("rfe_tune_spaces"),
                require_significance=kwargs.get("require_significance")
                or config_params.get("significance", {}).get("require_significance", False),
                significance_alpha=kwargs.get("significance_alpha")
                or config_params.get("significance", {}).get("alpha", 0.05),
            )

        click.echo(f"\n{'=' * 70}")
        click.echo(f"Panel optimization complete for all {len(model_dirs)} model(s)")
        click.echo(f"{'=' * 70}\n")

    else:
        # Single model path provided explicitly
        # Require infile and split_dir for explicit path mode
        if not kwargs.get("infile") or not kwargs.get("split_dir"):
            raise click.UsageError(
                "When using --results-dir, both --infile and --split-dir are required."
            )

        run_optimize_panel_aggregated(
            results_dir=kwargs["results_dir"],
            infile=kwargs["infile"],
            split_dir=kwargs["split_dir"],
            model_name=kwargs.get("model"),
            stability_threshold=kwargs.get("stability_threshold") or 0.90,
            start_size=kwargs.get("start_size"),
            min_size=kwargs.get("min_size") or 5,
            min_auroc_frac=kwargs.get("min_auroc_frac") or 0.90,
            cv_folds=kwargs.get("cv_folds") or 0,
            step_strategy=kwargs.get("step_strategy") or "geometric",
            outdir=kwargs.get("outdir"),
            log_level=ctx.obj.get("log_level"),
            n_jobs=kwargs.get("n_jobs") or 1,
            retune_n_trials=kwargs.get("retune_trials") or 40,
            corr_aware=kwargs.get("corr_aware", True),
            corr_threshold=kwargs.get("corr_threshold") or 0.80,
            corr_method=kwargs.get("corr_method") or "spearman",
            rfe_tune_spaces=config_params.get("rfe_tune_spaces"),
            require_significance=kwargs.get("require_significance")
            or config_params.get("significance", {}).get("require_significance", False),
            significance_alpha=kwargs.get("significance_alpha")
            or config_params.get("significance", {}).get("alpha", 0.05),
        )


@cli.command("consensus-panel")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML configuration file (auto-detects consensus_panel.yaml if exists)",
)
@click.option(
    "--run-id",
    type=str,
    required=True,
    help="Run ID to process (e.g., 20260127_115115). Auto-discovers all models.",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    default=None,
    help="Input data file (auto-detected from run metadata if not provided)",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing split indices (auto-detected if not provided)",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=None,
    help="Minimum selection fraction for stable proteins (default: 0.75)",
)
@click.option(
    "--corr-threshold",
    type=float,
    default=None,
    help="Correlation threshold for clustering (default: 0.85)",
)
@click.option(
    "--target-size",
    type=int,
    default=None,
    help="Target panel size (default: 25)",
)
@click.option(
    "--rfe-weight",
    type=float,
    default=None,
    help="Weight for RFE rank vs stability (0-1, default: 0.5)",
)
@click.option(
    "--rra-method",
    type=click.Choice(["geometric_mean", "borda", "median"]),
    default=None,
    help="RRA aggregation method (default: geometric_mean)",
)
@click.option(
    "--require-significance/--no-require-significance",
    default=None,
    help="Only include statistically significant models in consensus (requires prior permutation testing)",
)
@click.option(
    "--significance-alpha",
    type=float,
    default=None,
    help="Significance threshold for filtering models (default: 0.05)",
)
@click.option(
    "--min-significant-models",
    type=int,
    default=None,
    help="Minimum number of significant models required (default: 2)",
)
@click.option(
    "--use-stringent-ranking/--no-use-stringent-ranking",
    default=None,
    help="Use stringent ranking (OOF importance + essentiality + stability) instead of legacy (stability + RFE)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results/run_<RUN_ID>/consensus)",
)
@click.pass_context
def consensus_panel(ctx, config, **kwargs):
    """Generate consensus protein panel from multiple models via RRA.

    Aggregates protein rankings from all base models (LR_EN, RF, XGBoost, etc.)
    to create a single consensus panel for clinical deployment. The workflow:

    1. Loads stability rankings from each model's aggregated results
    2. (Optional) Incorporates RFE rankings if available
    3. Aggregates via Robust Rank Aggregation (geometric mean)
    4. Clusters correlated proteins and selects representatives
    5. Outputs final panel for fixed-panel validation

    Requires prior aggregation:
        ced aggregate-splits --results-dir results/run_<RUN_ID>/<MODEL>

    Examples:

        # Basic usage (auto-discovers all models)
        ced consensus-panel --run-id 20260127_115115

        # Custom parameters
        ced consensus-panel --run-id 20260127_115115 \\
            --target-size 30 \\
            --corr-threshold 0.90 \\
            --rfe-weight 0.3

        # Validate the resulting panel
        ced train --model LR_EN \\
            --fixed-panel results/run_20260127_115115/consensus/final_panel.txt \\
            --split-seed 10

    Outputs (in results/run_<RUN_ID>/consensus/):
        - final_panel.txt: One protein per line (for --fixed-panel)
        - final_panel.csv: With consensus scores
        - consensus_ranking.csv: All proteins with RRA scores
        - per_model_rankings.csv: Per-model composite rankings
        - correlation_clusters.csv: Cluster assignments
        - consensus_metadata.json: Run parameters and statistics
    """

    from ced_ml.cli.consensus_panel import run_consensus_panel

    # Load config file if provided or auto-detect
    from ced_ml.utils.paths import get_analysis_dir

    config_params = {}
    try:
        default_config = get_analysis_dir() / "configs" / "consensus_panel.yaml"
    except RuntimeError:
        default_config = None
    config_path = config or (default_config if default_config and default_config.exists() else None)

    if config_path:
        import yaml

        with open(config_path) as f:
            config_params = yaml.safe_load(f) or {}
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Merge config with CLI args (CLI takes precedence)
    param_keys = [
        "run_id",
        "infile",
        "split_dir",
        "stability_threshold",
        "corr_threshold",
        "target_size",
        "rfe_weight",
        "rra_method",
        "outdir",
        "require_significance",
        "significance_alpha",
        "min_significant_models",
        "use_stringent_ranking",
    ]
    for key in param_keys:
        if kwargs.get(key) is None and key in config_params:
            kwargs[key] = config_params[key]

    # Run consensus panel generation
    run_consensus_panel(
        run_id=kwargs["run_id"],
        infile=kwargs.get("infile"),
        split_dir=kwargs.get("split_dir"),
        stability_threshold=kwargs.get("stability_threshold") or 0.90,
        corr_threshold=kwargs.get("corr_threshold") or 0.85,
        target_size=kwargs.get("target_size") or 25,
        rfe_weight=kwargs.get("rfe_weight") or 0.5,
        rra_method=kwargs.get("rra_method") or "geometric_mean",
        outdir=kwargs.get("outdir"),
        log_level=ctx.obj.get("log_level"),
        require_significance=kwargs.get("require_significance")
        or config_params.get("significance", {}).get("require_significance", False),
        significance_alpha=kwargs.get("significance_alpha")
        or config_params.get("significance", {}).get("alpha", 0.05),
        min_significant_models=kwargs.get("min_significant_models")
        or config_params.get("significance", {}).get("min_significant_models", 2),
        use_stringent_ranking=kwargs.get("use_stringent_ranking")
        or config_params.get("use_stringent_ranking", False),
    )


@cli.command("permutation-test")
@click.option(
    "--run-id",
    type=str,
    required=True,
    help="Run ID to test (e.g., 20260127_115115)",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Specific model to test (default: all base models)",
)
@click.option(
    "--split-seed",
    type=int,
    default=0,
    help="Split seed to use (default: 0)",
)
@click.option(
    "--n-perms",
    type=int,
    default=200,
    help="Number of permutations (default: 200)",
)
@click.option(
    "--perm-index",
    type=int,
    default=None,
    help="Single permutation index for HPC job arrays (optional)",
)
@click.option(
    "--metric",
    type=str,
    default="auroc",
    help="Metric to use (default: auroc, only auroc supported per ADR-007)",
)
@click.option(
    "--n-jobs",
    type=int,
    default=1,
    help="Parallel jobs for in-process parallelization (default: 1)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: {run_dir}/{model}/significance/)",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
@click.option(
    "--hpc",
    is_flag=True,
    help="Submit as HPC job array (LSF/Slurm)",
)
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    help="HPC config file (default: configs/pipeline_hpc.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview HPC job submission without executing (--hpc mode only)",
)
@click.pass_context
def permutation_test(ctx, **kwargs):
    """Test model significance via label permutation testing.

    Tests the null hypothesis that model performance is no better than chance
    by comparing observed AUROC against a null distribution from B label permutations.

    For each permutation, the full pipeline (screening, feature selection,
    hyperparameter optimization) is re-run on permuted labels to obtain a
    valid null distribution without data leakage.

    WORKFLOW:
        1. Train models: ced train --run-id <RUN_ID>
        2. Test significance: ced permutation-test --run-id <RUN_ID>

    MODES:
        Local parallel: Runs all permutations locally with n_jobs parallelism
        HPC mode (--hpc): Submits job array to LSF/Slurm cluster

    Examples:

        # Test all models locally with 4 parallel jobs
        ced permutation-test --run-id 20260127_115115 --n-jobs 4

        # Test specific model
        ced permutation-test --run-id 20260127_115115 --model LR_EN --n-perms 200

        # Submit as HPC job array (recommended for large n_perms)
        ced permutation-test --run-id 20260127_115115 --model LR_EN --hpc

        # Preview HPC submission without executing
        ced permutation-test --run-id 20260127_115115 --model LR_EN --hpc --dry-run

    Output (in results/run_{RUN_ID}/{MODEL}/significance/):
        - permutation_test_results.csv: Summary metrics
        - null_distribution.csv: Full null distributions
        - perm_{i}.joblib: Individual permutation results (HPC mode)

    Interpretation:
        p < 0.05: Strong evidence of generalization above chance
        p < 0.10: Marginal evidence
        p >= 0.10: No evidence above chance
    """
    from pathlib import Path

    # HPC mode: submit job array
    hpc_flag = kwargs.get("hpc", False)
    dry_run_flag = kwargs.get("dry_run", False)

    if hpc_flag:
        from ced_ml.hpc.lsf import (
            build_job_script,
            detect_environment,
            load_hpc_config,
            submit_job,
        )
        from ced_ml.utils.paths import get_project_root

        run_id = kwargs["run_id"]
        model = kwargs.get("model")
        n_perms = kwargs["n_perms"]

        if not model:
            raise click.UsageError(
                "HPC mode (--hpc) requires --model. Submit separate jobs for each model."
            )

        # Load HPC config
        hpc_config_path = kwargs.get("hpc_config")
        if not hpc_config_path:
            root = get_project_root()
            candidates = [
                root / "configs" / "pipeline_hpc.yaml",
                root / "analysis" / "configs" / "pipeline_hpc.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    hpc_config_path = str(candidate)
                    break
            else:
                raise click.UsageError(
                    "HPC mode requires --hpc-config or configs/pipeline_hpc.yaml to exist. "
                    "Searched: configs/ and analysis/configs/"
                )

        hpc_config = load_hpc_config(Path(hpc_config_path))
        env_info = detect_environment(get_project_root())

        # Build bsub parameters
        default_resources = hpc_config.get_resources("default")
        root = get_project_root()
        log_dir = root / "logs" / "hpc"
        log_dir.mkdir(parents=True, exist_ok=True)

        bsub_params = {
            "project": hpc_config.project,
            "env_activation": env_info.activation_cmd,
            "log_dir": log_dir,
            **default_resources,
        }

        # Build permutation test command for job array
        from ced_ml.hpc.lsf import _build_permutation_test_command

        cmd = _build_permutation_test_command(
            run_id=run_id,
            model=model,
            split_seed=kwargs["split_seed"],
            n_perms=n_perms,
            random_state=kwargs["random_state"],
        )

        # Job array name: perm_<model>_<run_id>[0-N]
        job_name = f"perm_{model}_{run_id}[0-{n_perms - 1}]"

        script = build_job_script(
            job_name=job_name,
            command=cmd,
            **bsub_params,
        )

        click.echo(f"\nSubmitting permutation test job array for {model}:")
        click.echo(f"  Run ID: {run_id}")
        click.echo(f"  Model: {model}")
        click.echo(f"  Permutations: {n_perms}")
        click.echo(f"  Job array: {job_name}")

        job_id = submit_job(script, dry_run=dry_run_flag)

        if job_id:
            click.echo(f"  Submitted: job_id={job_id}")
            click.echo(f"\nMonitor with: bjobs -J 'perm_{model}_{run_id}*'")
            click.echo(
                f"After completion, run aggregation:\n"
                f"  ced permutation-test --run-id {run_id} --model {model}"
            )
        elif dry_run_flag:
            click.echo("  [DRY RUN] Job not submitted")

        return

    # Local mode: run permutation test directly
    from ced_ml.cli.permutation_test import run_permutation_test_cli

    run_permutation_test_cli(
        run_id=kwargs["run_id"],
        model=kwargs.get("model"),
        split_seed=kwargs["split_seed"],
        n_perms=kwargs["n_perms"],
        perm_index=kwargs.get("perm_index"),
        metric=kwargs["metric"],
        n_jobs=kwargs["n_jobs"],
        outdir=kwargs.get("outdir"),
        random_state=kwargs["random_state"],
        log_level=ctx.obj.get("log_level"),
    )


@cli.group("config")
@click.pass_context
def config_group(ctx):
    """Configuration management tools (validate, diff)."""
    pass


@config_group.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--command",
    type=click.Choice(["save-splits", "train"]),
    default="train",
    help="Command type (default: train)",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as errors",
)
@click.pass_context
def config_validate(ctx, config_file, command, strict):
    """Validate configuration file and report issues."""
    from pathlib import Path

    from ced_ml.cli.config_tools import run_config_validate

    run_config_validate(
        config_file=Path(config_file),
        command=command,
        strict=strict,
        log_level=ctx.obj.get("log_level"),
    )


@config_group.command("diff")
@click.argument("config_file1", type=click.Path(exists=True))
@click.argument("config_file2", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for diff report",
)
@click.pass_context
def config_diff(ctx, config_file1, config_file2, output):
    """Compare two configuration files."""
    from pathlib import Path

    from ced_ml.cli.config_tools import run_config_diff

    run_config_diff(
        config_file1=Path(config_file1),
        config_file2=Path(config_file2),
        output_file=Path(output) if output else None,
        log_level=ctx.obj.get("log_level"),
    )


@cli.command("run-pipeline")
@click.option(
    "--pipeline-config",
    type=click.Path(exists=True),
    default=None,
    help="Pipeline config file (auto-detected: pipeline_local.yaml or pipeline_hpc.yaml)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Input data file (Parquet/CSV, auto-discovered if not provided)",
)
@click.option(
    "--split-dir",
    type=click.Path(),
    default=None,
    help="Directory for split indices (auto-generated if not provided)",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help="Comma-separated list of models to train (default from pipeline config)",
)
@click.option(
    "--split-seeds",
    type=str,
    default=None,
    help="Comma-separated list of split seeds (default from pipeline config)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Shared run ID for all models (default: auto-generated timestamp)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory for results (default from pipeline config or 'results/')",
)
@click.option(
    "--ensemble/--no-ensemble",
    default=None,
    help="Train stacking ensemble meta-learner (default from pipeline config)",
)
@click.option(
    "--consensus/--no-consensus",
    default=None,
    help="Generate cross-model consensus panel (default from pipeline config)",
)
@click.option(
    "--optimize-panel/--no-optimize-panel",
    default=None,
    help="Run panel size optimization (default from pipeline config)",
)
@click.option(
    "--permutation-test/--no-permutation-test",
    default=None,
    help="Run permutation testing for statistical significance (default from pipeline config)",
)
@click.option(
    "--permutation-n-perms",
    type=int,
    default=None,
    help="Number of permutations for significance testing (default: 200)",
)
@click.option(
    "--permutation-n-jobs",
    type=int,
    default=None,
    help="Parallel jobs for permutation testing (-1 = all cores, default: -1)",
)
@click.option(
    "--overwrite-splits",
    is_flag=True,
    help="Regenerate splits even if they exist",
)
@click.option(
    "--log-file",
    type=click.Path(),
    default=None,
    help="Save pipeline logs to file (in addition to console output)",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Submit LSF jobs to HPC cluster instead of running locally",
)
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    default=None,
    help="HPC pipeline config (default: configs/pipeline_hpc.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview HPC job submission without executing (--hpc mode only)",
)
@click.pass_context
def run_pipeline(ctx, config, models, split_seeds, **kwargs):
    """
    Run the complete ML pipeline end-to-end.

    This command orchestrates the full workflow from training to panel optimization:

    1. Generate splits (if needed)
    2. Train base models (all models x all seeds)
    3. Aggregate results per model
    4. Train ensemble meta-learner (if enabled)
    5. Aggregate ensemble results
    6. Optimize panel sizes per model (if enabled)
    7. Generate cross-model consensus panel (if enabled)
    8. Permutation testing for statistical significance (if enabled)

    This is equivalent to running:
        ced save-splits (if needed)
        ced train (for each model x seed)
        ced aggregate-splits (for each model)
        ced train-ensemble (if enabled, for each seed)
        ced aggregate-splits --model ENSEMBLE (if ensemble enabled)
        ced optimize-panel --run-id <ID> (if enabled)
        ced consensus-panel --run-id <ID> (if enabled)
        ced permutation-test --run-id <ID> (if enabled)

    Examples:

        # Quick start - auto-discover data file (SIMPLEST)
        ced run-pipeline

        # Or specify data file explicitly
        ced run-pipeline --infile data/celiac.parquet

        # Custom models and seeds
        ced run-pipeline \\
            --models LR_EN,RF \\
            --split-seeds 0,1,2,3,4

        # Skip ensemble and consensus (faster)
        ced run-pipeline \\
            --no-ensemble \\
            --no-consensus

        # Custom run ID and output directory
        ced run-pipeline \\
            --run-id production_v1 \\
            --outdir ../results_production

        # With config file
        ced run-pipeline \\
            --config configs/training_config.yaml

        # Save logs to file
        ced run-pipeline \\
            --log-file logs/run_$(date +%Y%m%d_%H%M%S).log

        # HPC mode - submit LSF jobs with dependency chains
        ced run-pipeline --hpc

        # HPC dry run - preview without submitting
        ced run-pipeline --hpc --dry-run

        # HPC with custom config
        ced run-pipeline --hpc --hpc-config configs/pipeline_custom.yaml

    Output structure:
        results/
          run_<RUN_ID>/
            LR_EN/
              splits/split_seed0/, split_seed1/, ...
              aggregated/
                optimize_panel/  (if enabled)
            RF/
              splits/split_seed0/, split_seed1/, ...
              aggregated/
                optimize_panel/  (if enabled)
            XGBoost/
              splits/...
            ENSEMBLE/  (if enabled)
              splits/...
              aggregated/
            consensus/  (if enabled)
              final_panel.txt
              consensus_ranking.csv
    """
    from pathlib import Path

    from ced_ml.cli.run_pipeline import (
        _PIPELINE_DEFAULTS,
        load_pipeline_config,
        resolve_pipeline_config_path,
    )
    from ced_ml.cli.run_pipeline import (
        run_pipeline as run_pipeline_impl,
    )

    def _derive_split_seeds_from_config(pcfg: dict) -> list[int]:
        """Derive split_seeds from splits_config.yaml (single source of truth).

        Reads n_splits and seed_start from the splits config referenced by
        the pipeline config. Falls back to [0, 1, 2] if no config found.
        """
        splits_config_path = pcfg.get("splits_config")
        if splits_config_path and Path(splits_config_path).exists():
            import yaml

            with open(splits_config_path) as f:
                splits_raw = yaml.safe_load(f) or {}
            seed_start = splits_raw.get("seed_start", 0)
            n_splits = splits_raw.get("n_splits", 3)
            return list(range(seed_start, seed_start + n_splits))
        # Fallback: no splits config found
        return [0, 1, 2]

    # --- Extract raw CLI values (None = not provided by user) ---------------
    hpc_flag = kwargs.pop("hpc", False) or False
    hpc_config_cli = kwargs.pop("hpc_config", None)
    dry_run_cli = kwargs.pop("dry_run", None)
    pipeline_config_cli = kwargs.pop("pipeline_config", None)

    # --- Load pipeline config -----------------------------------------------
    # Priority: --pipeline-config > auto-detect (local/hpc) > hardcoded defaults
    pcfg_path = (
        Path(pipeline_config_cli)
        if pipeline_config_cli
        else resolve_pipeline_config_path(hpc=hpc_flag)
    )
    pcfg = load_pipeline_config(pcfg_path) if pcfg_path else {}

    # Helper: first non-None wins (CLI > config > fallback)
    def _pick(cli_val, cfg_key, fallback=None):
        if cli_val is not None:
            return cli_val
        if cfg_key in pcfg:
            return pcfg[cfg_key]
        return fallback

    # --- Resolve every parameter --------------------------------------------
    # Models & seeds (CLI is comma-string, config is list)
    if models is not None:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = pcfg.get("models", _PIPELINE_DEFAULTS["models"])

    if split_seeds is not None:
        seed_list = [int(s.strip()) for s in split_seeds.split(",")]
    else:
        # Derive from splits_config (single source of truth for split seeds)
        seed_list = _derive_split_seeds_from_config(pcfg)

    # Paths
    infile = Path(kwargs["infile"]) if kwargs.get("infile") else pcfg.get("infile")
    split_dir = Path(kwargs["split_dir"]) if kwargs.get("split_dir") else pcfg.get("splits_dir")
    outdir_raw = kwargs.get("outdir")
    outdir = Path(outdir_raw) if outdir_raw else pcfg.get("results_dir", Path("results"))

    config_path = Path(config) if config else pcfg.get("training_config")
    splits_config_path = pcfg.get("splits_config")
    log_file = Path(kwargs["log_file"]) if kwargs.get("log_file") else None

    hpc_config_path = Path(hpc_config_cli) if hpc_config_cli else (pcfg_path if hpc_flag else None)
    dry_run_flag = _pick(dry_run_cli, "dry_run", _PIPELINE_DEFAULTS["dry_run"])

    # Boolean toggles
    enable_ensemble = _pick(kwargs.get("ensemble"), "ensemble", _PIPELINE_DEFAULTS["ensemble"])
    enable_consensus = _pick(kwargs.get("consensus"), "consensus", _PIPELINE_DEFAULTS["consensus"])
    enable_optimize_panel = _pick(
        kwargs.get("optimize_panel"),
        "optimize_panel",
        _PIPELINE_DEFAULTS["optimize_panel"],
    )
    enable_permutation_test = _pick(
        kwargs.get("permutation_test"),
        "permutation_test",
        _PIPELINE_DEFAULTS["permutation_test"],
    )
    permutation_n_perms = _pick(
        kwargs.get("permutation_n_perms"),
        "permutation_n_perms",
        _PIPELINE_DEFAULTS["permutation_n_perms"],
    )
    permutation_n_jobs = _pick(
        kwargs.get("permutation_n_jobs"),
        "permutation_n_jobs",
        _PIPELINE_DEFAULTS["permutation_n_jobs"],
    )
    # Collect remaining CLI args
    cli_args = {}
    overrides = list(kwargs.get("override", []))

    try:
        run_pipeline_impl(
            config_file=config_path,
            splits_config_file=splits_config_path,
            infile=infile,
            split_dir=split_dir,
            models=model_list,
            split_seeds=seed_list,
            run_id=kwargs.get("run_id"),
            outdir=outdir,
            enable_ensemble=enable_ensemble,
            enable_consensus=enable_consensus,
            enable_optimize_panel=enable_optimize_panel,
            enable_permutation_test=enable_permutation_test,
            permutation_n_perms=permutation_n_perms,
            permutation_n_jobs=permutation_n_jobs,
            log_file=log_file,
            cli_args=cli_args,
            overrides=overrides,
            log_level=ctx.obj.get("log_level"),
            hpc=hpc_flag,
            hpc_config_file=hpc_config_path,
            dry_run=dry_run_flag,
        )
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        ctx.exit(1)


@cli.command("convert-to-parquet")
@click.argument(
    "csv_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output Parquet file path (default: same as input with .parquet extension)",
)
@click.option(
    "--compression",
    type=click.Choice(["snappy", "gzip", "brotli", "zstd", "none"]),
    default="snappy",
    help="Compression algorithm (default: snappy)",
)
def convert_to_parquet(csv_path, output, compression):
    """
    Convert proteomics CSV file to Parquet format.

    This command reads a CSV file and converts it to Parquet format with
    compression. Only columns needed for modeling are included (same as
    the training pipeline).

    CSV_PATH: Path to input CSV file

    Example:
        ced convert-to-parquet data/celiac_proteomics.csv
        ced convert-to-parquet data/celiac_proteomics.csv -o data/celiac.parquet --compression gzip
    """

    from ced_ml.data.io import convert_csv_to_parquet

    try:
        parquet_path = convert_csv_to_parquet(
            csv_path=csv_path,
            parquet_path=output,
            compression=compression,
        )
        click.echo(f"Successfully converted to: {parquet_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


def main():
    """Entry point for console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
