"""
Main CLI entry point for CeD-ML pipeline.

Provides subcommands:
  - ced save-splits: Generate train/val/test splits
  - ced train: Train ML models
  - ced train-ensemble: Train stacking ensemble meta-learner
  - ced aggregate-splits: Aggregate results across split seeds
  - ced eval-holdout: Evaluate on holdout set
  - ced optimize-panel: Optimize panel size via aggregated RFE
  - ced consensus-panel: Generate cross-model consensus panel
  - ced permutation-test: Test model significance via label permutation
  - ced run-pipeline: Run complete end-to-end workflow
  - ced config: Validate and inspect configuration files
  - ced convert-to-parquet: Convert CSV to Parquet format
"""

import click

from ced_ml import __version__
from ced_ml.cli.commands.aggregation import aggregate_splits
from ced_ml.cli.commands.config_tools import config_group
from ced_ml.cli.commands.data import convert_to_parquet, save_splits
from ced_ml.cli.commands.evaluation import eval_holdout
from ced_ml.cli.commands.orchestration import run_pipeline
from ced_ml.cli.commands.panel import consensus_panel, optimize_panel
from ced_ml.cli.commands.recipes import derive_recipes, generate_v0
from ced_ml.cli.commands.significance import permutation_test
from ced_ml.cli.commands.sweep import sweep
from ced_ml.cli.commands.training import train, train_ensemble
from ced_ml.cli.options import log_level_option


@click.group()
@click.version_option(version=__version__, prog_name="ced")
@log_level_option
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


# Register all commands
cli.add_command(save_splits)
cli.add_command(convert_to_parquet)
cli.add_command(config_group)
cli.add_command(train)
cli.add_command(train_ensemble)
cli.add_command(aggregate_splits)
cli.add_command(eval_holdout)
cli.add_command(optimize_panel)
cli.add_command(consensus_panel)
cli.add_command(permutation_test)
cli.add_command(run_pipeline)
cli.add_command(derive_recipes)
cli.add_command(generate_v0)
cli.add_command(sweep)


def main():
    """Entry point for console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
