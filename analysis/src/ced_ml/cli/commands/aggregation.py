"""
Aggregation command for collecting and summarizing results across split seeds.
"""

from pathlib import Path

import click

from ced_ml.cli.options import config_option, results_dir_option, run_id_option
from ced_ml.cli.utils.validation import validate_mutually_exclusive


@click.command("aggregate-splits")
@config_option
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
    default=None,
    help="Fraction of splits a feature must appear in to be 'stable' (default: 0.75)",
)
@click.option(
    "--target-specificity",
    type=float,
    default=None,
    help="Target specificity for alpha threshold (default: 0.95)",
)
@click.option(
    "--plot-formats",
    multiple=True,
    default=(),
    help="Plot output formats (can be repeated, e.g., --plot-formats png --plot-formats pdf)",
)
@click.pass_context
def aggregate_splits(ctx, config, **kwargs):
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
    from ced_ml.cli.utils.config_merge import merge_config_with_cli
    from ced_ml.utils.paths import get_analysis_dir

    # Normalize click multi-option defaults for config merge.
    if kwargs.get("plot_formats") == ():
        kwargs["plot_formats"] = None

    # Load config file if provided or auto-detect.
    try:
        default_config = get_analysis_dir() / "configs" / "aggregate_config.yaml"
    except RuntimeError:
        default_config = Path("configs/aggregate_config.yaml")
    config_path = config or (default_config if default_config.exists() else None)
    if config_path:
        kwargs.update(
            merge_config_with_cli(
                Path(config_path),
                kwargs,
                [
                    "stability_threshold",
                    "target_specificity",
                    "plot_formats",
                    "n_boot",
                    "save_plots",
                    "plot_roc",
                    "plot_pr",
                    "plot_calibration",
                    "plot_risk_distribution",
                    "plot_dca",
                    "plot_oof_combined",
                    "plot_learning_curve",
                    "plot_shap_summary",
                    "plot_shap_dependence",
                    "plot_shap_heatmap",
                    "control_spec_targets",
                ],
                verbose=False,
            )
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

    # Fill defaults (CLI > config > defaults).
    kwargs["stability_threshold"] = (
        kwargs["stability_threshold"] if kwargs.get("stability_threshold") is not None else 0.75
    )
    kwargs["target_specificity"] = (
        kwargs["target_specificity"] if kwargs.get("target_specificity") is not None else 0.95
    )
    kwargs["n_boot"] = kwargs["n_boot"] if kwargs.get("n_boot") is not None else 500
    kwargs["save_plots"] = kwargs["save_plots"] if kwargs.get("save_plots") is not None else True
    kwargs["plot_roc"] = kwargs["plot_roc"] if kwargs.get("plot_roc") is not None else True
    kwargs["plot_pr"] = kwargs["plot_pr"] if kwargs.get("plot_pr") is not None else True
    kwargs["plot_calibration"] = (
        kwargs["plot_calibration"] if kwargs.get("plot_calibration") is not None else True
    )
    kwargs["plot_risk_distribution"] = (
        kwargs["plot_risk_distribution"]
        if kwargs.get("plot_risk_distribution") is not None
        else True
    )
    kwargs["plot_dca"] = kwargs["plot_dca"] if kwargs.get("plot_dca") is not None else True
    kwargs["plot_oof_combined"] = (
        kwargs["plot_oof_combined"] if kwargs.get("plot_oof_combined") is not None else True
    )
    kwargs["plot_learning_curve"] = (
        kwargs["plot_learning_curve"] if kwargs.get("plot_learning_curve") is not None else True
    )
    kwargs["plot_shap_summary"] = (
        kwargs["plot_shap_summary"] if kwargs.get("plot_shap_summary") is not None else True
    )
    kwargs["plot_shap_dependence"] = (
        kwargs["plot_shap_dependence"] if kwargs.get("plot_shap_dependence") is not None else True
    )
    kwargs["plot_shap_heatmap"] = (
        kwargs["plot_shap_heatmap"] if kwargs.get("plot_shap_heatmap") is not None else True
    )
    kwargs["control_spec_targets"] = (
        kwargs["control_spec_targets"]
        if kwargs.get("control_spec_targets") is not None
        else [0.90, 0.95, 0.99]
    )
    kwargs["plot_formats"] = kwargs["plot_formats"] or ["png"]
    if isinstance(kwargs["plot_formats"], tuple):
        kwargs["plot_formats"] = list(kwargs["plot_formats"])
    elif isinstance(kwargs["plot_formats"], str):
        kwargs["plot_formats"] = [kwargs["plot_formats"]]

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
                    shap_plot_flags["plot_shap_heatmap"] = getattr(
                        cfg.output, "plot_shap_heatmap", True
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
                kwargs.setdefault(
                    "plot_shap_heatmap", getattr(cfg.output, "plot_shap_heatmap", True)
                )
                break
    except Exception:
        pass

    run_aggregate_splits(**kwargs, log_level=ctx.obj.get("log_level"))
