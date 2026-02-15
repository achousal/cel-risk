"""
Aggregation command for collecting and summarizing results across split seeds.
"""

from pathlib import Path

import click

from ced_ml.cli.options import results_dir_option, run_id_option
from ced_ml.cli.utils.validation import validate_mutually_exclusive


@click.command("aggregate-splits")
@results_dir_option
@run_id_option(
    required=False,
    help_text="Run ID for auto-detection (e.g., 20260127_115115, mutually exclusive with --results-dir)",
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

    try:
        validate_mutually_exclusive(
            "--results-dir",
            results_dir,
            "--run-id",
            run_id,
            "--results-dir and --run-id are mutually exclusive.\n"
            "Use --results-dir for explicit path OR --run-id for auto-detection.",
        )
    except ValueError as e:
        raise click.UsageError(str(e)) from e

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

            # Load SHAP plot flags from saved config when available
            shap_plot_flags = {}
            try:
                from ced_ml.config.loader import load_training_config

                config_path = run_path / "config.yaml"
                if config_path.exists():
                    cfg = load_training_config(str(config_path))
                    shap_plot_flags["plot_shap_summary"] = getattr(
                        cfg.output, "plot_shap_summary", True
                    )
                    shap_plot_flags["plot_shap_dependence"] = getattr(
                        cfg.output, "plot_shap_dependence", True
                    )
            except Exception:
                pass  # defaults to True via function signature

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
                model_kwargs.update(shap_plot_flags)

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

    # Try loading SHAP plot flags from saved config
    try:
        from ced_ml.config.loader import load_training_config

        results_path = Path(results_dir)
        for parent in [results_path] + list(results_path.parents):
            config_path = parent / "config.yaml"
            if config_path.exists():
                cfg = load_training_config(str(config_path))
                kwargs.setdefault(
                    "plot_shap_summary", getattr(cfg.output, "plot_shap_summary", True)
                )
                kwargs.setdefault(
                    "plot_shap_dependence", getattr(cfg.output, "plot_shap_dependence", True)
                )
                break
    except Exception:
        pass

    run_aggregate_splits(**kwargs, log_level=ctx.obj.get("log_level"))
