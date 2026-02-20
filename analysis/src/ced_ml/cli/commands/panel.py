"""
Panel optimization and consensus commands.

Commands:
  - optimize-panel: Find minimum viable panel via RFE
  - consensus-panel: Generate cross-model consensus panel via RRA
"""

import click

from ced_ml.cli.options import config_option, run_id_option, run_id_required_option
from ced_ml.cli.utils.validation import validate_mutually_exclusive


def _resolve_cli_or_config(cli_value, config_value, default):
    """Resolve option precedence while preserving explicit False/0 CLI values."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


@click.command("optimize-panel")
@config_option
@click.option(
    "--results-dir",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Path to model results directory (e.g., results/run_20260127_115115/LR_EN/). Mutually exclusive with --run-id.",
)
@run_id_option(
    required=False,
    help_text="Run ID to auto-discover all models (e.g., 20260127_115115). Mutually exclusive with --results-dir.",
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
        - rfe_feature_report_aggregated.csv: Importance + retention summary
        - recommended_panels_aggregated.json: Minimum sizes at thresholds
        - panel_curve_aggregated.png: Pareto curve visualization
    """
    import json
    from pathlib import Path

    from ced_ml.cli.optimize_panel import (
        discover_models_by_run_id,
        run_optimize_panel_aggregated,
    )
    from ced_ml.cli.utils.config_merge import load_config_file, merge_config_with_cli

    # Load config file: use provided path, or auto-detect default if it exists
    default_config = Path(__file__).parent.parent.parent.parent / "configs" / "optimize_panel.yaml"
    config_path = config or (default_config if default_config.exists() else None)

    if config_path:
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Load full config for nested sections
    config_params = load_config_file(Path(config_path) if config_path else None)
    significance_cfg = config_params.get("significance", {})
    essentiality_cfg = config_params.get("essentiality", {})

    # Merge config with CLI args (CLI takes precedence)
    param_keys = [
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
        "corr_aware",
        "corr_threshold",
        "corr_method",
        "require_significance",
        "significance_alpha",
    ]
    kwargs.update(
        merge_config_with_cli(
            Path(config_path) if config_path else None, kwargs, param_keys, verbose=True
        )
    )

    # Validate mutually exclusive options
    try:
        validate_mutually_exclusive(
            "--results-dir",
            kwargs.get("results_dir"),
            "--run-id",
            kwargs.get("run_id"),
            "--results-dir and --run-id are mutually exclusive. Use one or the other.",
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

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
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise click.ClickException(
                    f"Failed to read run metadata from {metadata_file}: {e}. "
                    "The metadata file may be corrupt. Please specify --infile and --split-dir manually."
                ) from e
            first_model_name = next(iter(model_dirs))
            if not infile:
                infile = metadata.get("infile")
            if not split_dir:
                split_dir = metadata.get("split_dir") or metadata.get("splits_dir")
            if not infile or not split_dir:
                model_meta = metadata.get("models", {}).get(first_model_name, {})
                if not infile:
                    infile = model_meta.get("infile")
                if not split_dir:
                    split_dir = model_meta.get("split_dir")

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
                stability_threshold=kwargs.get("stability_threshold") or 0.90,
                start_size=kwargs.get("start_size"),
                min_size=kwargs.get("min_size") or 5,
                min_auroc_frac=kwargs.get("min_auroc_frac") or 0.50,
                cv_folds=kwargs.get("cv_folds") or 5,
                step_strategy=kwargs.get("step_strategy") or "fine",
                log_level=ctx.obj.get("log_level"),
                retune_n_trials=kwargs.get("retune_trials") or 60,
                retune_n_jobs=kwargs.get("n_jobs") or 1,
                corr_aware=kwargs.get("corr_aware", True),
                corr_threshold=kwargs.get("corr_threshold") or 0.80,
                corr_method=kwargs.get("corr_method") or "spearman",
                rfe_tune_spaces=config_params.get("rfe_tune_spaces"),
            )

        return

    # HPC mode: submit jobs and exit
    hpc_flag = kwargs.get("hpc", False)
    dry_run_flag = kwargs.get("dry_run", False)

    if hpc_flag:
        import os
        from pathlib import Path

        from ced_ml.hpc import (
            build_job_script,
            detect_environment,
            get_scheduler,
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
        scheduler = get_scheduler(hpc_config.scheduler)

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

        # Discover available split seeds per model
        from ced_ml.hpc.common import _build_panel_optimization_command

        model_seeds: dict[str, list[int]] = {}
        for model_name, agg_dir in model_dirs.items():
            model_root = agg_dir.parent  # e.g., results/run_ID/ModelName/
            splits_dir = model_root / "splits"
            if splits_dir.exists():
                seeds = sorted(
                    int(d.name.replace("split_seed", ""))
                    for d in splits_dir.glob("split_seed*")
                    if d.is_dir()
                )
                model_seeds[model_name] = seeds
            else:
                model_seeds[model_name] = []

        total_seed_jobs = sum(len(seeds) for seeds in model_seeds.values())
        total_agg_jobs = len(model_dirs)
        click.echo(
            f"\nSubmitting {total_seed_jobs} per-seed + {total_agg_jobs} aggregation "
            f"job(s) to HPC ({len(model_dirs)} models):"
        )
        for model_name, seeds in model_seeds.items():
            click.echo(f"  - {model_name}: {len(seeds)} seeds {seeds}")

        # Build job parameters from config + environment
        default_resources = hpc_config.get_resources("default")
        root = get_project_root()
        log_dir = root / "logs" / "hpc"
        log_dir.mkdir(parents=True, exist_ok=True)

        job_params = {
            "scheduler": scheduler,
            "project": hpc_config.project,
            "env_activation": env_info.activation_cmd,
            "log_dir": log_dir,
            **default_resources,
        }

        # Submit per-model per-seed jobs, then per-model aggregation jobs
        submitted_jobs = []
        for model_name, seeds in model_seeds.items():
            if not seeds:
                click.echo(f"  {model_name}: No seeds found, submitting single job")
                cmd = _build_panel_optimization_command(run_id=run_id, model=model_name)
                job_name = f"opt_panel_{model_name}_{run_id}"
                script = build_job_script(job_name=job_name, command=cmd, **job_params)
                job_id = submit_job(script, scheduler=scheduler, dry_run=dry_run_flag)
                if job_id:
                    submitted_jobs.append((job_name, job_id))
                    click.echo(f"  Submitted {job_name}: job_id={job_id}")
                elif dry_run_flag:
                    click.echo(f"  [DRY RUN] {job_name}")
                continue

            # Per-seed RFE jobs
            for seed in seeds:
                cmd = _build_panel_optimization_command(
                    run_id=run_id,
                    model=model_name,
                    split_seed=seed,
                )
                job_name = f"opt_panel_{model_name}_s{seed}_{run_id}"
                script = build_job_script(job_name=job_name, command=cmd, **job_params)
                job_id = submit_job(script, scheduler=scheduler, dry_run=dry_run_flag)

                if job_id:
                    submitted_jobs.append((job_name, job_id))
                    click.echo(f"  Submitted {job_name}: job_id={job_id}")
                elif dry_run_flag:
                    click.echo(f"  [DRY RUN] {job_name}")
                else:
                    click.echo(f"  {job_name}: Submission failed", err=True)
                    raise click.ClickException(
                        f"HPC job submission failed for {job_name}. "
                        "Check HPC configuration and job script."
                    )

            # Per-model aggregation job (runs after all seeds complete;
            # detects pre-computed seed results and skips to aggregation)
            agg_cmd = _build_panel_optimization_command(run_id=run_id, model=model_name)
            agg_job_name = f"opt_panel_{model_name}_agg_{run_id}"
            agg_script = build_job_script(job_name=agg_job_name, command=agg_cmd, **job_params)
            agg_job_id = submit_job(agg_script, scheduler=scheduler, dry_run=dry_run_flag)

            if agg_job_id:
                submitted_jobs.append((agg_job_name, agg_job_id))
                click.echo(f"  Submitted {agg_job_name}: job_id={agg_job_id}")
            elif dry_run_flag:
                click.echo(f"  [DRY RUN] {agg_job_name}")

        if dry_run_flag:
            click.echo("\n[DRY RUN] No jobs were actually submitted.")
        elif submitted_jobs:
            click.echo(f"\nSuccessfully submitted {len(submitted_jobs)} job(s)")
            click.echo(
                "Note: Aggregation jobs should be submitted after seed jobs complete.\n"
                "If using LSF/Slurm dependency chains, consider using "
                "'ced run-pipeline --hpc' for automatic orchestration."
            )
            click.echo(f"Monitor with: {scheduler.monitor_hint('opt_panel_*')}")

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
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise click.ClickException(
                    f"Failed to read run metadata from {metadata_file}: {e}. "
                    "The metadata file may be corrupt. Please specify --infile and --split-dir manually."
                ) from e

            # Use metadata values, but allow CLI overrides
            # Shared metadata format: top-level or under models/{model_name}
            first_model_name = next(iter(model_dirs))
            infile = kwargs.get("infile") or metadata.get("infile")
            split_dir = (
                kwargs.get("split_dir") or metadata.get("split_dir") or metadata.get("splits_dir")
            )
            if not infile or not split_dir:
                model_meta = metadata.get("models", {}).get(first_model_name, {})
                infile = infile or model_meta.get("infile")
                split_dir = split_dir or model_meta.get("split_dir")

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
                stability_threshold=kwargs.get("stability_threshold") or 0.90,
                start_size=kwargs.get("start_size"),
                min_size=kwargs.get("min_size") or 5,
                min_auroc_frac=kwargs.get("min_auroc_frac") or 0.50,
                cv_folds=kwargs.get("cv_folds") or 5,
                step_strategy=kwargs.get("step_strategy") or "fine",
                outdir=kwargs.get("outdir"),
                log_level=ctx.obj.get("log_level"),
                n_jobs=n_jobs,
                retune_n_trials=kwargs.get("retune_trials") or 60,
                corr_aware=kwargs.get("corr_aware", True),
                corr_threshold=kwargs.get("corr_threshold") or 0.80,
                corr_method=kwargs.get("corr_method") or "spearman",
                rfe_tune_spaces=config_params.get("rfe_tune_spaces"),
                require_significance=_resolve_cli_or_config(
                    kwargs.get("require_significance"),
                    significance_cfg.get("require_significance"),
                    False,
                ),
                significance_alpha=_resolve_cli_or_config(
                    kwargs.get("significance_alpha"),
                    significance_cfg.get("alpha"),
                    0.05,
                ),
                essentiality_corr_threshold=essentiality_cfg.get("corr_threshold"),
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
            require_significance=_resolve_cli_or_config(
                kwargs.get("require_significance"),
                significance_cfg.get("require_significance"),
                False,
            ),
            significance_alpha=_resolve_cli_or_config(
                kwargs.get("significance_alpha"),
                significance_cfg.get("alpha"),
                0.05,
            ),
            essentiality_corr_threshold=essentiality_cfg.get("corr_threshold"),
        )


@click.command("consensus-panel")
@config_option
@run_id_required_option(
    help_text="Run ID to process (e.g., 20260127_115115). Auto-discovers all models.",
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
    "--rra-method",
    type=click.Choice(["geometric_mean", "borda", "median"]),
    default=None,
    help="RRA aggregation method (default: geometric_mean)",
)
@click.option(
    "--ranking-signal",
    type=click.Choice(["oof_importance", "oof_shap"]),
    default=None,
    help="Per-model ranking signal for consensus (default: oof_importance)",
)
@click.option(
    "--shap-explicit-normalization/--no-shap-explicit-normalization",
    default=None,
    help=(
        "Apply explicit SHAP normalization for cross-model aggregation "
        "(only when --ranking-signal=oof_shap)"
    ),
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
    "--run-essentiality/--no-run-essentiality",
    default=None,
    help="Run drop-column essentiality validation on consensus panel (default: True)",
)
@click.option(
    "--essentiality-corr-threshold",
    type=float,
    default=None,
    help="Correlation threshold for clustering in essentiality validation (default: 0.75)",
)
@click.option(
    "--essentiality-refit-mode",
    type=click.Choice(["fixed", "retune", "fixed_retune"]),
    default=None,
    help=(
        "Refit strategy for essentiality: 'fixed' (clone, fast), "
        "'retune' (Optuna re-optimization), 'fixed_retune' (both side-by-side)"
    ),
)
@click.option(
    "--retune-n-trials",
    type=int,
    default=None,
    help="Optuna trials per cluster drop in retune modes (default: 20)",
)
@click.option(
    "--retune-inner-folds",
    type=int,
    default=None,
    help="Inner CV folds for retune's OptunaSearchCV (default: 3)",
)
@click.option(
    "--n-jobs",
    type=int,
    default=None,
    help="Number of parallel jobs for essentiality validation (default: 1, use -1 for all CPUs)",
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
            --corr-threshold 0.90

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

    from pathlib import Path

    from ced_ml.cli.consensus_panel import run_consensus_panel
    from ced_ml.cli.utils.config_merge import load_config_file, merge_config_with_cli
    from ced_ml.utils.paths import get_analysis_dir

    # Load config file if provided or auto-detect
    try:
        default_config = get_analysis_dir() / "configs" / "consensus_panel.yaml"
    except RuntimeError as e:
        import logging

        logging.getLogger(__name__).warning(f"Could not determine analysis directory: {e}")
        default_config = None
    config_path = config or (default_config if default_config and default_config.exists() else None)

    if config_path:
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Load full config for nested sections
    config_params = load_config_file(Path(config_path) if config_path else None)
    significance_cfg = config_params.get("significance", {})

    # Merge config with CLI args (CLI takes precedence)
    param_keys = [
        "run_id",
        "infile",
        "split_dir",
        "stability_threshold",
        "corr_threshold",
        "target_size",
        "rra_method",
        "ranking_signal",
        "shap_explicit_normalization",
        "outdir",
        "require_significance",
        "significance_alpha",
        "min_significant_models",
    ]
    kwargs.update(
        merge_config_with_cli(
            Path(config_path) if config_path else None, kwargs, param_keys, verbose=False
        )
    )

    # Extract essentiality config
    essentiality_cfg = config_params.get("essentiality", {})

    # Run consensus panel generation
    run_consensus_panel(
        run_id=kwargs["run_id"],
        infile=kwargs.get("infile"),
        split_dir=kwargs.get("split_dir"),
        stability_threshold=kwargs.get("stability_threshold") or 0.90,
        corr_threshold=kwargs.get("corr_threshold") or 0.85,
        target_size=kwargs.get("target_size") or 25,
        rra_method=kwargs.get("rra_method") or "geometric_mean",
        ranking_signal=kwargs.get("ranking_signal") or "oof_importance",
        shap_explicit_normalization=_resolve_cli_or_config(
            kwargs.get("shap_explicit_normalization"),
            config_params.get("shap_explicit_normalization"),
            True,
        ),
        outdir=kwargs.get("outdir"),
        log_level=ctx.obj.get("log_level"),
        require_significance=_resolve_cli_or_config(
            kwargs.get("require_significance"),
            significance_cfg.get("require_significance"),
            False,
        ),
        significance_alpha=_resolve_cli_or_config(
            kwargs.get("significance_alpha"),
            significance_cfg.get("alpha"),
            0.05,
        ),
        min_significant_models=_resolve_cli_or_config(
            kwargs.get("min_significant_models"),
            significance_cfg.get("min_significant_models"),
            2,
        ),
        run_essentiality=(
            kwargs.get("run_essentiality")
            if kwargs.get("run_essentiality") is not None
            else essentiality_cfg.get("enabled", True)
        ),
        essentiality_corr_threshold=kwargs.get("essentiality_corr_threshold")
        or essentiality_cfg.get("corr_threshold", 0.75),
        include_brier=essentiality_cfg.get("include_brier", True),
        include_pr_auc=essentiality_cfg.get("include_pr_auc", True),
        essentiality_refit_mode=(
            kwargs.get("essentiality_refit_mode") or essentiality_cfg.get("refit_mode", "fixed")
        ),
        essentiality_retune_n_trials=(
            kwargs.get("retune_n_trials") or essentiality_cfg.get("retune_n_trials", 20)
        ),
        essentiality_retune_inner_folds=(
            kwargs.get("retune_inner_folds") or essentiality_cfg.get("retune_inner_folds", 3)
        ),
        n_jobs=kwargs.get("n_jobs") or config_params.get("n_jobs_hpc", 1),
    )
