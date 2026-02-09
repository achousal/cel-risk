"""
Significance testing commands.

Commands:
  - permutation-test: Test model significance via label permutation
"""

import click

from ced_ml.cli.options import (
    dry_run_option,
    hpc_option,
    run_id_required_option,
)


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
    "--perm-index",
    type=int,
    default=None,
    help="Single permutation index for HPC job arrays (optional)",
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
@hpc_option
@click.option(
    "--hpc-config",
    type=click.Path(exists=True),
    help="HPC config file (default: configs/pipeline_hpc.yaml)",
)
@dry_run_option
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

    MODES:
        Local parallel: Runs all permutations locally with n_jobs parallelism
        HPC mode (--hpc): Submits job array to LSF/Slurm cluster

    Examples:

        # Test all models locally with 4 parallel jobs
        ced permutation-test --run-id 20260127_115115 --n-jobs 4

        # Test specific model
        ced permutation-test --run-id 20260127_115115 --model LR_EN --n-perms 200

        # Submit as HPC job array (recommended for large n_perms)
        ced permutation-test --run-id 20260127_115115 --model LR_EN --hpc

        # Preview HPC submission without executing
        ced permutation-test --run-id 20260127_115115 --model LR_EN --hpc --dry-run

    Output (in results/run_{RUN_ID}/{MODEL}/significance/):
        - permutation_test_results.csv: Summary metrics
        - null_distribution.csv: Full null distributions
        - perm_{i}.joblib: Individual permutation results (HPC mode)

    Interpretation:
        p < 0.05: Strong evidence of generalization above chance
        p < 0.10: Marginal evidence
        p >= 0.10: No evidence above chance
    """
    from ced_ml.cli.utils.seed_parsing import parse_seed_range

    # HPC mode: submit job array
    hpc_flag = kwargs.get("hpc", False)
    dry_run_flag = kwargs.get("dry_run", False)

    if hpc_flag:
        from ced_ml.cli.hpc import submit_permutation_test_jobs

        run_id = kwargs["run_id"]
        model = kwargs.get("model")
        n_perms = kwargs["n_perms"]
        split_seed_start = kwargs["split_seed_start"]
        n_split_seeds = kwargs["n_split_seeds"]
        try:
            split_seeds = parse_seed_range(split_seed_start, n_split_seeds)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

        if not model:
            raise click.UsageError(
                "HPC mode (--hpc) requires --model. Submit separate jobs for each model."
            )

        # Submit permutation test jobs
        submit_permutation_test_jobs(
            run_id=run_id,
            model=model,
            split_seeds=split_seeds,
            n_perms=n_perms,
            random_state=kwargs["random_state"],
            hpc_config_path=kwargs.get("hpc_config"),
            dry_run=dry_run_flag,
        )

        return

    # Local mode: run permutation test directly
    from ced_ml.cli.permutation_test import run_permutation_test_cli

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
        perm_index=kwargs.get("perm_index"),
        metric=kwargs["metric"],
        n_jobs=kwargs["n_jobs"],
        outdir=kwargs.get("outdir"),
        random_state=kwargs["random_state"],
        log_level=ctx.obj.get("log_level"),
    )
