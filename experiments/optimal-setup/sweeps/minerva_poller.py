"""Async job submission and polling for Minerva (SLURM).

Wraps the existing ced_ml.hpc infrastructure to provide a
submit-poll-result interface for sweep iterations.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of a submitted and completed job."""

    job_id: str
    status: str  # COMPLETED | FAILED | TIMEOUT | CANCELLED
    results_dir: Path
    wall_seconds: float | None


def submit_pipeline_job(
    pipeline_config: Path,
    training_config: Path,
    seeds: list[int],
    results_dir: Path,
    run_id: str,
    job_name: str = "sweep",
    dry_run: bool = False,
) -> str | None:
    """Submit a ced run-pipeline job to SLURM.

    Uses sbatch directly rather than the orchestrator infrastructure,
    since sweep iterations are standalone single-cell jobs.

    Returns
    -------
    job_id or None (if dry_run)
    """
    seeds_str = ",".join(str(s) for s in seeds)

    # Build the pipeline command
    cmd = (
        f"ced run-pipeline "
        f"--pipeline-config {pipeline_config} "
        f"--config {training_config} "
        f"--split-seeds {seeds_str} "
        f"--outdir {results_dir} "
        f"--run-id {run_id} "
        f"--log-level info"
    )

    # Build SLURM script
    script = _build_slurm_script(job_name=job_name, command=cmd)

    if dry_run:
        logger.info("[DRY RUN] Would submit:\n%s", script)
        return None

    result = subprocess.run(
        ["sbatch"],
        input=script,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("sbatch failed: %s", result.stderr)
        raise RuntimeError(f"sbatch submission failed: {result.stderr}")

    # Parse job ID from "Submitted batch job 12345"
    for line in result.stdout.splitlines():
        if "Submitted batch job" in line:
            job_id = line.strip().split()[-1]
            logger.info("Submitted sweep job %s: %s", job_name, job_id)
            return job_id

    raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")


def poll_job(
    job_id: str,
    poll_interval: int = 60,
    timeout_seconds: int = 14400,  # 4 hours default
) -> str:
    """Poll a SLURM job until completion.

    Parameters
    ----------
    job_id
        SLURM job ID.
    poll_interval
        Seconds between polls.
    timeout_seconds
        Maximum wait time before raising TimeoutError.

    Returns
    -------
    Final job status string (COMPLETED, FAILED, TIMEOUT, CANCELLED).
    """
    elapsed = 0
    while elapsed < timeout_seconds:
        status = _get_job_status(job_id)
        if status in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
            logger.info("Job %s finished with status: %s (after %ds)", job_id, status, elapsed)
            return status
        logger.debug("Job %s status: %s (elapsed: %ds)", job_id, status, elapsed)
        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(
        f"Job {job_id} did not complete within {timeout_seconds}s. "
        f"Last status: {_get_job_status(job_id)}"
    )


def submit_and_wait(
    pipeline_config: Path,
    training_config: Path,
    seeds: list[int],
    results_dir: Path,
    run_id: str,
    job_name: str = "sweep",
    poll_interval: int = 60,
    timeout_seconds: int = 14400,
    dry_run: bool = False,
) -> JobResult:
    """Submit a pipeline job and wait for completion.

    Combines submit + poll into a single blocking call.
    """
    job_id = submit_pipeline_job(
        pipeline_config=pipeline_config,
        training_config=training_config,
        seeds=seeds,
        results_dir=results_dir,
        run_id=run_id,
        job_name=job_name,
        dry_run=dry_run,
    )

    if dry_run:
        return JobResult(
            job_id="DRY_RUN", status="DRY_RUN",
            results_dir=results_dir, wall_seconds=None,
        )

    start = time.monotonic()
    status = poll_job(job_id, poll_interval=poll_interval, timeout_seconds=timeout_seconds)
    wall = time.monotonic() - start

    return JobResult(
        job_id=job_id,
        status=status,
        results_dir=results_dir,
        wall_seconds=wall,
    )


def _get_job_status(job_id: str) -> str:
    """Query SLURM for job status via sacct."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--noheader", "--parsable2", "-o", "State"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            # sacct can return multiple lines (job + job steps); take first
            return result.stdout.strip().splitlines()[0].split("|")[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to squeue for pending/running jobs
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "-o", "%T"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "UNKNOWN"


def _build_slurm_script(
    job_name: str,
    command: str,
    partition: str = "premium",
    cpus: int = 12,
    mem_gb: int = 32,
    walltime: str = "48:00:00",
) -> str:
    """Build a minimal SLURM batch script for one sweep iteration."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time={walltime}
#SBATCH --output=logs/sweeps/{job_name}_%j.out
#SBATCH --error=logs/sweeps/{job_name}_%j.err

set -euo pipefail

# Activate environment (assumes conda/mamba)
source ~/.bashrc
conda activate ced-ml

{command}
"""
