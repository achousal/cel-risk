"""
Significance testing commands.

Commands:
  - permutation-test: Test model significance via label permutation
"""

import click

from ced_ml.cli.options import run_id_required_option


@click.command("permutation-test")
@run_id_required_option(help_text="Run ID to test (e.g., 20260127_115115)")
@click.option(
    "--model",
    type=str,
    default=None,
    help="Specific model to test (default: all base models)",
)
@click.option(
    "--split-seed-start",
    type=int,
    default=0,
    help="First split seed to test (default: 0)",
)
@click.option(
    "--n-split-seeds",
    type=int,
    default=1,
    help="Number of consecutive seeds to test, starting from --split-seed-start (default: 1)",
)
@click.option(
    "--n-perms",
    type=int,
    default=200,
    help="Number of permutations (default: 200)",
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
    "--aggregate-only",
    is_flag=True,
    default=False,
    help="Only aggregate existing per-seed results (do not run permutations).",
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

    Examples:

        # Test all models locally with 4 parallel jobs
        ced permutation-test --run-id 20260127_115115 --n-jobs 4

        # Test specific model
        ced permutation-test --run-id 20260127_115115 --model LR_EN --n-perms 200

    Output (in results/run_{RUN_ID}/{MODEL}/significance/):
        - permutation_test_results.csv: Summary metrics
        - null_distribution.csv: Full null distributions

    Interpretation:
        p < 0.05: Strong evidence of generalization above chance
        p < 0.10: Marginal evidence
        p >= 0.10: No evidence above chance
    """
    from ced_ml.cli.permutation_test import run_permutation_test_cli
    from ced_ml.cli.utils.seed_parsing import parse_seed_range

    split_seed_start = kwargs["split_seed_start"]
    n_split_seeds = kwargs["n_split_seeds"]
    try:
        split_seeds = parse_seed_range(split_seed_start, n_split_seeds)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    run_permutation_test_cli(
        run_id=kwargs["run_id"],
        model=kwargs.get("model"),
        split_seeds=split_seeds,
        n_perms=kwargs["n_perms"],
        metric=kwargs["metric"],
        n_jobs=kwargs["n_jobs"],
        outdir=kwargs.get("outdir"),
        random_state=kwargs["random_state"],
        log_level=ctx.obj.get("log_level"),
        aggregate_only=kwargs.get("aggregate_only", False),
    )
