"""
Training commands for the CeD-ML CLI.

Provides:
  - train: Train ML models with nested cross-validation
  - train-ensemble: Train stacking ensemble from base model OOF predictions
"""

import click

from ced_ml.cli.options import (
    config_option,
    dry_run_option,
    experiment_option,
    hpc_option,
    infile_option,
    run_id_option,
)
from ced_ml.cli.utils.seed_parsing import parse_seed_list
from ced_ml.cli.utils.validation import validate_mutually_exclusive
from ced_ml.data.schema import ModelName


@click.command("train")
@config_option
@infile_option
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
    "--split-index",
    type=int,
    default=None,
    help="Ordinal position of this seed among all seeds (0-based). Used for max_plot_splits gate.",
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
@run_id_option()
@experiment_option
@hpc_option
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    default=None,
    help="HPC config file (default: configs/pipeline_hpc.yaml)",
)
@dry_run_option
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
    try:
        validate_mutually_exclusive(
            "--split-seed",
            split_seed,
            "--split-seeds",
            split_seeds,
            "--split-seed and --split-seeds are mutually exclusive. Use one or the other.",
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    # Determine which seeds to process
    if split_seeds is not None:
        try:
            seed_list = parse_seed_list(split_seeds)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
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

        from ced_ml.cli.hpc import submit_train_jobs

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
            from ced_ml.utils.paths import make_run_id

            run_id = make_run_id(kwargs.get("experiment"))

        # Submit jobs
        submit_train_jobs(
            seed_list=seed_list,
            model=model,
            infile=infile,
            split_dir=split_dir,
            outdir=outdir,
            run_id=run_id,
            config_file=config_file,
            hpc_config_path=hpc_config_cli,
            dry_run=dry_run_flag,
        )

        return

    # Local mode: run training directly
    from ced_ml.cli.train import run_train

    # If --split-index was explicitly provided via CLI (e.g. HPC), use it; otherwise derive from loop
    cli_split_index = kwargs.pop("split_index", None)

    # For local mode with multiple seeds, run sequentially
    for i, seed in enumerate(seed_list):
        if len(seed_list) > 1:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"Training seed {seed} ({i + 1}/{len(seed_list)})")
            click.echo(f"{'=' * 70}\n")

        # Collect CLI args
        cli_args = {k: v for k, v in kwargs.items() if k != "override"}
        cli_args["split_seed"] = seed
        cli_args["split_index"] = cli_split_index if cli_split_index is not None else i
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


@click.command("train-ensemble")
@config_option
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=False,
    help="Results directory containing base model OOF predictions",
)
@click.option(
    "--base-models",
    type=str,
    default=None,
    help="Comma-separated list of base models (auto-detected if --run-id provided)",
)
@run_id_option(
    help_text="Run ID for auto-detection (e.g., 20260127_115115). Auto-discovers results-dir and base-models.",
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
@click.option(
    "--split-index",
    type=int,
    default=None,
    help="Ordinal position of this seed among all seeds (0-based). Used for max_plot_splits gate.",
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
        try:
            seeds_to_train = parse_seed_list(split_seeds)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
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
        try:
            # Use the same parsing logic as seeds (comma-separated list)
            base_model_list = [m.strip() for m in base_models.split(",")]
        except Exception as e:
            raise click.UsageError(f"Failed to parse base models: {e}") from e

    # Train ensemble for each split seed
    n_seeds = len(seeds_to_train)
    cli_split_index = kwargs.pop("split_index", None)
    for i, seed in enumerate(seeds_to_train, 1):
        if n_seeds > 1:
            click.echo(f"\n{'=' * 70}")
            click.echo(f"Training ensemble for split_seed={seed} ({i}/{n_seeds})")
            click.echo(f"{'=' * 70}\n")

        # Use CLI --split-index if provided (HPC single-seed calls), otherwise derive from loop
        effective_split_index = cli_split_index if cli_split_index is not None else (i - 1)

        try:
            run_train_ensemble(
                config_file=config,
                base_models=base_model_list,
                split_seed=seed,
                split_index=effective_split_index,
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
