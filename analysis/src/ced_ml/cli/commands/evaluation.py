"""
Evaluation commands for the CeD-ML CLI.

Provides:
  - eval-holdout: Evaluate trained model on holdout set
"""

import click

from ced_ml.cli.options import config_option


@click.command("eval-holdout")
@config_option
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=False,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--model-artifact",
    type=click.Path(exists=True),
    required=False,
    help="Path to trained model (.joblib file)",
)
@click.option(
    "--holdout-idx",
    type=click.Path(exists=True),
    required=False,
    help="Path to holdout indices CSV",
)
@click.option(
    "--outdir",
    type=click.Path(),
    required=False,
    help="Output directory for holdout evaluation results",
)
@click.option(
    "--compute-dca/--no-compute-dca",
    default=None,
    help="Compute decision curve analysis",
)
@click.pass_context
def eval_holdout(ctx, config, **kwargs):
    """Evaluate trained model on holdout set (run ONCE only)."""
    from pathlib import Path

    from ced_ml.cli.eval_holdout import run_eval_holdout
    from ced_ml.cli.utils.config_merge import merge_config_with_cli
    from ced_ml.utils.paths import get_analysis_dir

    # Load config file if provided or auto-detect default.
    try:
        default_config = get_analysis_dir() / "configs" / "holdout_config.yaml"
    except RuntimeError:
        default_config = Path("configs/holdout_config.yaml")
    config_path = config or (default_config if default_config.exists() else None)

    merged = kwargs.copy()
    if config_path:
        merged.update(
            merge_config_with_cli(
                Path(config_path),
                merged,
                [
                    "infile",
                    "holdout_idx",
                    "model_artifact",
                    "outdir",
                    "scenario",
                    "compute_dca",
                    "dca_threshold_min",
                    "dca_threshold_max",
                    "dca_threshold_step",
                    "dca_report_points",
                    "dca_use_target_prevalence",
                    "save_preds",
                    "toprisk_fracs",
                    "target_prevalence",
                    "clinical_threshold_points",
                ],
                verbose=False,
            )
        )

    # Required fields after config merge.
    missing = [
        key for key in ("infile", "model_artifact", "holdout_idx", "outdir") if not merged.get(key)
    ]
    if missing:
        missing_flags = ", ".join(f"--{k.replace('_', '-')}" for k in missing)
        raise click.UsageError(
            f"Missing required arguments after config merge: {missing_flags}. "
            "Provide via CLI or holdout config."
        )

    # Validate required input paths when values come from config.
    for key in ("infile", "model_artifact", "holdout_idx"):
        path = Path(str(merged[key])).expanduser()
        if not path.exists():
            raise click.UsageError(
                f"--{key.replace('_', '-')} path not found: {path}. "
                "Update holdout config or pass a valid CLI path."
            )

    # Convert list-valued YAML fields to comma-separated strings expected by evaluator.
    for key in ("toprisk_fracs", "dca_report_points", "clinical_threshold_points"):
        value = merged.get(key)
        if isinstance(value, list | tuple):
            merged[key] = ",".join(str(v) for v in value)

    run_eval_holdout(**merged)
