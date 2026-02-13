"""
HPC job submission functions for CLI commands.

Provides high-level functions to submit training and permutation test jobs,
abstracting away the details of config loading, job script building, and submission.
"""

from pathlib import Path
from typing import Any

import click

from ced_ml.hpc import (
    build_job_script,
    detect_environment,
    get_scheduler,
    load_hpc_config,
    submit_job,
)
from ced_ml.hpc.common import (
    _build_permutation_test_command,
    _build_training_command,
)
from ced_ml.utils.paths import get_project_root


def load_hpc_config_with_fallback(hpc_config_path: str | None = None) -> Path:
    """
    Load HPC config file, searching in default locations if not provided.

    Args:
        hpc_config_path: Explicit path to HPC config (optional)

    Returns:
        Path to HPC config file

    Raises:
        click.UsageError: If no config file is found
    """
    if hpc_config_path:
        return Path(hpc_config_path)

    root = get_project_root()
    candidates = [
        root / "configs" / "pipeline_hpc.yaml",
        root / "analysis" / "configs" / "pipeline_hpc.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise click.UsageError(
        "HPC mode requires --hpc-config or configs/pipeline_hpc.yaml to exist. "
        "Searched: configs/ and analysis/configs/"
    )


def setup_hpc_environment(
    hpc_config_path: Path, log_subdir: str = "hpc", run_id: str | None = None
) -> tuple[dict[str, Any], Path]:
    """
    Set up HPC environment by loading config and creating log directory.

    The returned params dict includes a ``scheduler`` key suitable for passing
    through to :func:`build_job_script`.

    Args:
        hpc_config_path: Path to HPC config file
        log_subdir: Subdirectory under logs/hpc/ for job logs
        run_id: Optional run ID to include in log path

    Returns:
        Tuple of (job_params dict, log_dir path).
    """
    hpc_config = load_hpc_config(hpc_config_path)
    env_info = detect_environment(get_project_root())
    scheduler = get_scheduler(hpc_config.scheduler)

    # Build log directory
    root = get_project_root()
    if run_id:
        log_dir = root / "logs" / "hpc" / log_subdir / f"run_{run_id}"
    else:
        log_dir = root / "logs" / "hpc" / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build job parameters (scheduler included for build_job_script)
    default_resources = hpc_config.get_resources("default")
    job_params = {
        "scheduler": scheduler,
        "project": hpc_config.project,
        "env_activation": env_info.activation_cmd,
        "log_dir": log_dir,
        **default_resources,
    }

    return job_params, log_dir


def submit_train_jobs(
    seed_list: list[int],
    model: str,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    run_id: str,
    config_file: Path | None,
    hpc_config_path: str | None,
    dry_run: bool,
) -> list[tuple[int, str]]:
    """
    Submit training jobs to HPC cluster.

    Args:
        seed_list: List of split seeds to train
        model: Model name (e.g., "LR_EN", "RF")
        infile: Path to input data file
        split_dir: Path to split directory
        outdir: Path to output directory
        run_id: Run ID for grouping jobs
        config_file: Path to training config (optional)
        hpc_config_path: Path to HPC config (optional, will search defaults)
        dry_run: If True, preview jobs without submitting

    Returns:
        List of (seed, job_id) tuples for submitted jobs

    Raises:
        click.UsageError: If required files/directories are missing
        click.ClickException: If job submission fails
    """
    # Resolve HPC config
    hpc_config_path_resolved = load_hpc_config_with_fallback(hpc_config_path)

    # Setup environment
    bsub_params, log_dir = setup_hpc_environment(
        hpc_config_path_resolved, log_subdir="training", run_id=run_id
    )

    # Resolve config file
    if not config_file:
        root = get_project_root()
        default_config = root / "configs" / "training_config.yaml"
        if not default_config.exists():
            default_config = root / "analysis" / "configs" / "training_config.yaml"
        if not default_config.exists():
            raise click.UsageError(
                "HPC mode requires --config or configs/training_config.yaml to exist."
            )
        config_file = default_config

    # Display summary
    n_jobs = len(seed_list)
    click.echo(f"\nSubmitting {n_jobs} training job(s) to HPC:")
    click.echo(f"  Model: {model}")
    click.echo(f"  Seeds: {seed_list}")
    click.echo(f"  Run ID: {run_id}")
    click.echo(f"  Output: {outdir}")

    # Submit jobs
    submitted_jobs = []
    for seed in seed_list:
        job_name = f"CeD_{run_id}_{model}_s{seed}"

        # Build training command
        cmd_kwargs = {
            "infile": infile,
            "split_dir": split_dir,
            "outdir": outdir,
            "model": model,
            "split_seed": seed,
            "run_id": run_id,
            "config_file": config_file,
        }

        command = _build_training_command(**cmd_kwargs)

        script = build_job_script(
            job_name=job_name,
            command=command,
            **bsub_params,
        )

        scheduler = bsub_params["scheduler"]
        job_id = submit_job(script, scheduler=scheduler, dry_run=dry_run)

        if job_id:
            submitted_jobs.append((seed, job_id))
            click.echo(f"  Submitted seed {seed}: job_id={job_id}")
        elif dry_run:
            click.echo(f"  [DRY RUN] seed {seed}: {job_name}")
        else:
            click.echo(f"  seed {seed}: Submission failed", err=True)
            raise click.ClickException(
                f"HPC job submission failed for seed {seed}. "
                "Check HPC configuration and job script."
            )

    # Display post-submission message
    if dry_run:
        click.echo("\n[DRY RUN] No jobs were actually submitted.")
    elif submitted_jobs:
        scheduler = bsub_params["scheduler"]
        click.echo(f"\nSubmitted {len(submitted_jobs)} job(s).")
        click.echo(f"Monitor with: {scheduler.monitor_hint(f'CeD_{run_id}_{model}_*')}")

    return submitted_jobs


def submit_permutation_test_jobs(
    run_id: str,
    model: str,
    split_seeds: list[int],
    n_perms: int,
    random_state: int,
    hpc_config_path: str | None,
    dry_run: bool,
) -> list[tuple[int, str]]:
    """
    Submit permutation test job arrays to HPC cluster.

    Args:
        run_id: Run ID to test
        model: Model name to test
        split_seeds: List of split seeds to test
        n_perms: Number of permutations per seed
        random_state: Random seed for reproducibility
        hpc_config_path: Path to HPC config (optional, will search defaults)
        dry_run: If True, preview jobs without submitting

    Returns:
        List of (seed, job_id) tuples for submitted job arrays

    Raises:
        click.UsageError: If required config is missing
    """
    # Resolve HPC config
    hpc_config_path_resolved = load_hpc_config_with_fallback(hpc_config_path)

    # Setup environment
    bsub_params, log_dir = setup_hpc_environment(hpc_config_path_resolved, log_subdir="hpc")

    # Display summary
    click.echo(f"\nSubmitting permutation test job arrays for {model}:")
    click.echo(f"  Run ID: {run_id}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Split seeds: {split_seeds}")
    click.echo(f"  Permutations per seed: {n_perms}")
    click.echo(f"  Total permutations: {n_perms * len(split_seeds)}")

    # Submit job arrays
    submitted_ids = []
    for seed in split_seeds:
        cmd = _build_permutation_test_command(
            run_id=run_id,
            model=model,
            split_seed=seed,
            n_perms=n_perms,
            random_state=random_state,
        )

        job_name = f"perm_{model}_{run_id}_s{seed}[0-{n_perms - 1}]"

        script = build_job_script(
            job_name=job_name,
            command=cmd,
            **bsub_params,
        )

        scheduler = bsub_params["scheduler"]
        job_id = submit_job(script, scheduler=scheduler, dry_run=dry_run)

        if job_id:
            submitted_ids.append((seed, job_id))
            click.echo(f"  Seed {seed}: job_id={job_id}")
        elif dry_run:
            click.echo(f"  Seed {seed}: [DRY RUN] Job not submitted")

    # Display post-submission message
    if submitted_ids:
        scheduler = bsub_params["scheduler"]
        click.echo(f"\nMonitor with: {scheduler.monitor_hint(f'perm_{model}_{run_id}*')}")
        click.echo(
            f"After completion, run aggregation:\n"
            f"  ced permutation-test --run-id {run_id} --model {model}"
        )

    return submitted_ids
