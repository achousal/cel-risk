"""
Significance testing commands.

Commands:
  - permutation-test: Test model significance via label permutation
"""

import click

from ced_ml.cli.options import config_option, run_id_required_option


@click.command("permutation-test")
@config_option
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
    default=None,
    help="First split seed to test (default from permutation config)",
)
@click.option(
    "--n-split-seeds",
    type=int,
    default=None,
    help="Number of consecutive seeds to test, starting from --split-seed-start (default from permutation config)",
)
@click.option(
    "--n-perms",
    type=int,
    default=None,
    help="Number of permutations (default from permutation config)",
)
@click.option(
    "--metric",
    type=str,
    default=None,
    help="Metric to use (default from permutation config, only auroc supported per ADR-007)",
)
@click.option(
    "--n-jobs",
    type=int,
    default=None,
    help="Parallel jobs for in-process parallelization (default from permutation config)",
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
    default=None,
    help="Random seed for reproducibility (default from permutation config)",
)
@click.option(
    "--aggregate-only",
    is_flag=True,
    default=False,
    help="Only aggregate existing per-seed results (do not run permutations).",
)
@click.pass_context
def permutation_test(ctx, config, **kwargs):
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
    from ced_ml.config.loader import load_permutation_config

    perm_cfg = load_permutation_config(config_file=config)
    split_seed_start = (
        kwargs["split_seed_start"]
        if kwargs.get("split_seed_start") is not None
        else perm_cfg.split_seed_start
    )
    n_split_seeds = (
        kwargs["n_split_seeds"]
        if kwargs.get("n_split_seeds") is not None
        else perm_cfg.n_split_seeds
    )
    n_perms = kwargs["n_perms"] if kwargs.get("n_perms") is not None else perm_cfg.n_perms
    metric = kwargs["metric"] if kwargs.get("metric") is not None else perm_cfg.metric
    n_jobs = kwargs["n_jobs"] if kwargs.get("n_jobs") is not None else perm_cfg.n_jobs
    random_state = (
        kwargs["random_state"] if kwargs.get("random_state") is not None else perm_cfg.random_state
    )
    outdir = kwargs.get("outdir") if kwargs.get("outdir") is not None else perm_cfg.outdir

    try:
        split_seeds = parse_seed_range(split_seed_start, n_split_seeds)
    except ValueError as e:
        raise click.UsageError(str(e)) from e

    run_permutation_test_cli(
        run_id=kwargs["run_id"],
        model=kwargs.get("model"),
        split_seeds=split_seeds,
        n_perms=n_perms,
        metric=metric,
        n_jobs=n_jobs,
        outdir=outdir,
        random_state=random_state,
        log_level=ctx.obj.get("log_level"),
        aggregate_only=kwargs.get("aggregate_only", False),
    )
