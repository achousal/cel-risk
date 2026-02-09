"""
Pipeline orchestration commands.

Commands:
  - run-pipeline: End-to-end workflow execution
"""

import click

from ced_ml.cli.options import dry_run_option, hpc_option


@click.command("run-pipeline")
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
@hpc_option
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    default=None,
    help="HPC pipeline config (default: configs/pipeline_hpc.yaml)",
)
@dry_run_option
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
              splits/split_seed0/, split_seed1/, ...\n              aggregated/
                optimize_panel/  (if enabled)
            RF/
              splits/split_seed0/, split_seed1/, ...\n              aggregated/
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
    from ced_ml.cli.utils.seed_parsing import parse_seed_list, parse_seed_range

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
            return parse_seed_range(seed_start, n_splits)
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
        try:
            model_list = [m.strip() for m in models.split(",")]
        except Exception as e:
            raise click.UsageError(f"Failed to parse models: {e}") from e
    else:
        model_list = pcfg.get("models", _PIPELINE_DEFAULTS["models"])

    if split_seeds is not None:
        try:
            seed_list = parse_seed_list(split_seeds)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
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
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        click.echo(f"Pipeline failed: {e}", err=True)
        ctx.exit(1)
