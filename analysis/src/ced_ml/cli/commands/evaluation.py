"""
Evaluation commands for the CeD-ML CLI.

Provides:
  - eval-holdout: Evaluate trained model on holdout set
"""

import click


@click.command("eval-holdout")
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
