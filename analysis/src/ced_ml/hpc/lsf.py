"""LSF job submission utilities for HPC pipeline orchestration.

Provides functions to build and submit LSF (bsub) job scripts for running
the CeD-ML pipeline on HPC clusters with job dependency chains.

Parallelization strategy:
- Training: M × S parallel jobs (one per model per split)
- Post-processing: Single aggregation job (waits for all training)
- Panel optimization: M parallel jobs (one per model)
- Consensus: Single job (waits for aggregation + panels)

Example: 4 models × 10 splits = 40 training jobs running simultaneously
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from ced_ml.config.schema import HPCConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Environment detection result.

    Attributes:
        env_type: Type of Python environment ('venv', 'conda', etc.).
        activation_cmd: Shell command to activate the environment.
    """

    env_type: str
    activation_cmd: str


def detect_environment(base_dir: Path) -> EnvironmentInfo:
    """Detect venv at analysis/ subdirectory (for HPC mode).

    Args:
        base_dir: Base directory (cwd).

    Returns:
        EnvironmentInfo with env_type and activation_cmd.

    Raises:
        RuntimeError: If venv not found.
    """
    from ced_ml.utils.paths import get_project_root

    try:
        project_root = get_project_root()
    except Exception:
        if (base_dir / "analysis").exists():
            project_root = base_dir
        elif base_dir.name == "analysis":
            project_root = base_dir.parent
        else:
            project_root = base_dir

    venv_activate = project_root / "analysis" / "venv" / "bin" / "activate"
    if not venv_activate.exists():
        raise RuntimeError(f"venv not found at {venv_activate}. " f"Run: bash scripts/hpc_setup.sh")

    return EnvironmentInfo(
        env_type="venv",
        activation_cmd=f'source "{venv_activate}"',
    )


def load_hpc_config(config_path: Path) -> HPCConfig:
    """Load and validate HPC pipeline config.

    Args:
        config_path: Path to pipeline_hpc.yaml.

    Returns:
        Validated HPCConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required HPC fields are missing or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"HPC config not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    hpc_dict = raw_config.get("hpc", {})
    if not hpc_dict:
        raise ValueError(f"No 'hpc' section found in {config_path}")

    try:
        hpc_config = HPCConfig(**hpc_dict)
    except Exception as e:
        raise ValueError(f"Invalid HPC configuration in {config_path}: {e}") from e

    return hpc_config


def build_job_script(
    *,
    job_name: str,
    command: str,
    project: str,
    queue: str,
    cores: int,
    mem_per_core: int,
    walltime: str,
    env_activation: str,
    log_dir: Path,
    dependency: str | None = None,
) -> str:
    """Build an LSF job script.

    Args:
        job_name: LSF job name (-J).
        command: Shell command(s) to execute.
        project: HPC project allocation (-P).
        queue: Queue name (-q).
        cores: Number of cores (-n).
        mem_per_core: Memory per core in MB.
        walltime: Wall time limit as "HH:MM" (-W).
        env_activation: Bash command to activate Python environment.
        log_dir: Directory for log files.
        dependency: Optional LSF dependency expression for -w flag.

    Returns:
        Complete bash script string for bsub submission.

    Note:
        Job output is redirected to /dev/null because ced commands create
        their own log files in logs/training/, logs/ensemble/, etc.
        Only stderr is captured to {job_name}.%J.err for LSF errors.
    """
    log_err = log_dir / f"{job_name}.%J.err"

    dep_line = ""
    if dependency:
        dep_line = f'#BSUB -w "{dependency}"'

    script = f"""#!/bin/bash
#BSUB -P {project}
#BSUB -q {queue}
#BSUB -J {job_name}
#BSUB -n {cores}
#BSUB -W {walltime}
#BSUB -R "span[hosts=1] rusage[mem={mem_per_core}]"
#BSUB -oo /dev/null
#BSUB -eo {log_err}
{dep_line}

set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

{command}

# Clean up empty .err log if job succeeded
EXIT_CODE=$?
ERR_LOG="{log_dir}/{job_name}.$LSF_JOBID.err"
if [ $EXIT_CODE -eq 0 ] && [ -f "$ERR_LOG" ] && [ ! -s "$ERR_LOG" ]; then
    rm -f "$ERR_LOG"
fi

exit $EXIT_CODE
"""
    return script


def submit_job(script: str, dry_run: bool = False) -> str | None:
    """Submit an LSF job via bsub.

    Args:
        script: Job script content to submit via stdin.
        dry_run: If True, log the script but do not submit.

    Returns:
        Job ID string if submitted, None if dry_run or failure.

    Raises:
        RuntimeError: If bsub command is not available.
    """
    if dry_run:
        logger.info("[DRY RUN] Would submit job script:")
        for line in script.strip().split("\n"):
            if line.startswith("#BSUB") or line.startswith("stdbuf"):
                logger.info(f"  {line}")
        return None

    if not shutil.which("bsub"):
        raise RuntimeError("bsub command not found. LSF scheduler is required for --hpc mode.")

    result = subprocess.run(
        ["bsub"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        logger.error(f"bsub failed: {result.stderr}")
        return None

    match = re.search(r"Job <(\d+)>", result.stdout)
    if match:
        return match.group(1)

    logger.error(f"Could not parse job ID from bsub output: {result.stdout}")
    return None


def _build_training_command(
    *,
    config_file: Path,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    model: str,
    split_seed: int,
    run_id: str,
) -> str:
    """Build ced train command for a single (model, split_seed) pair.

    Each job trains ONE model on ONE split for maximum parallelization.
    Post-processing (aggregation, ensemble, panels, consensus) handled by
    separate downstream jobs with proper dependencies.
    """
    parts = [
        "ced train",
        f'--config "{config_file}"',
        f'--infile "{infile}"',
        f'--split-dir "{split_dir}"',
        f'--outdir "{outdir}"',
        f"--model {model}",
        f"--split-seed {split_seed}",
        f"--run-id {run_id}",
    ]
    return " \\\n  ".join(parts)


def _build_postprocessing_command(
    *,
    config_file: Path,
    run_id: str,
    outdir: Path,
    infile: Path,
    split_dir: Path,
    models: list[str],
    split_seeds: list[int],
    enable_ensemble: bool,
) -> str:
    """Build post-processing commands (aggregation and ensemble training only).

    Panel optimization and consensus are now handled by separate parallel jobs.

    Returns a multi-line bash script fragment that runs each step sequentially.
    """
    lines = [
        f'echo "Post-processing (aggregation + ensemble) for run {run_id}"',
        "",
    ]

    # Aggregate base models
    for model in models:
        lines.append(f'echo "Aggregating {model}..."')
        lines.append(f"ced aggregate-splits --run-id {run_id} --model {model}")
        lines.append("")

    # Train ensemble per seed
    if enable_ensemble:
        for seed in split_seeds:
            lines.append(f'echo "Training ensemble seed {seed}..."')
            lines.append(f"ced train-ensemble --run-id {run_id} --split-seed {seed}")
        lines.append("")
        lines.append('echo "Aggregating ENSEMBLE..."')
        lines.append(f"ced aggregate-splits --run-id {run_id} --model ENSEMBLE")
        lines.append("")

    lines.append(f'echo "Aggregation and ensemble training complete for run {run_id}"')

    return "\n".join(lines)


def _build_panel_optimization_command(
    *,
    run_id: str,
    model: str,
) -> str:
    """Build panel optimization command for a single model.

    Returns a bash command string for optimizing panel size via RFE.
    """
    return f"ced optimize-panel --run-id {run_id} --model {model}"


def _build_consensus_panel_command(
    *,
    run_id: str,
) -> str:
    """Build consensus panel command.

    Returns a bash command string for generating cross-model consensus panel.
    """
    return f"ced consensus-panel --run-id {run_id}"


def submit_hpc_pipeline(
    *,
    config_file: Path,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    models: list[str],
    split_seeds: list[int],
    run_id: str,
    enable_ensemble: bool,
    enable_consensus: bool,
    enable_optimize_panel: bool,
    hpc_config: HPCConfig,
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline with dependency chains.

    Job dependency architecture (OPTIMIZED FOR MAXIMUM PARALLELIZATION):
    1. Training jobs (M × S jobs: one per model per split, fully parallel)
       Example: 4 models × 10 splits = 40 parallel jobs
    2. Post-processing job (aggregation + ensemble, depends on ALL training jobs)
    3. Panel optimization jobs (M jobs: one per model, parallel, depends on post-processing)
    4. Consensus panel job (depends on post-processing + panel optimization)

    Performance: 4× speedup vs previous per-split parallelization (with 4 models)

    Args:
        config_file: Path to training config YAML.
        infile: Path to input data file.
        split_dir: Path to split indices directory.
        outdir: Path to results output directory.
        models: List of model names.
        split_seeds: List of split seeds.
        run_id: Shared run identifier.
        enable_ensemble: Enable ensemble training in post-processing.
        enable_consensus: Enable consensus panel generation.
        enable_optimize_panel: Enable parallel panel optimization jobs.
        hpc_config: HPCConfig schema instance.
        logs_dir: Directory for job logs.
        dry_run: Preview without submitting.
        pipeline_logger: Logger instance.

    Returns:
        Dict with run_id, training_jobs, postprocessing_job, panel_jobs, consensus_job, logs_dir.
    """
    base_dir = Path.cwd()

    # Detect environment
    env_info = detect_environment(base_dir)
    pipeline_logger.info(f"Python environment: {env_info.env_type}")

    # Create log directory
    run_logs_dir = logs_dir / "training" / f"run_{run_id}"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    # Get default resource config
    default_resources = hpc_config.get_resources("default")

    # Common bsub parameters (using default resources for now, can be customized per stage)
    bsub_params = {
        "project": hpc_config.project,
        "env_activation": env_info.activation_cmd,
        "log_dir": run_logs_dir,
        **default_resources,
    }

    # Submit training jobs (one per model per split for maximum parallelization)
    n_jobs = len(models) * len(split_seeds)
    pipeline_logger.info(
        f"Submitting {n_jobs} training jobs "
        f"({len(models)} models \u00d7 {len(split_seeds)} splits)..."
    )
    training_job_ids = []

    for model in models:
        for seed in split_seeds:
            job_name = f"CeD_{run_id}_{model}_s{seed}"

            command = _build_training_command(
                config_file=config_file.resolve(),
                infile=infile.resolve(),
                split_dir=split_dir.resolve(),
                outdir=outdir.resolve(),
                model=model,
                split_seed=seed,
                run_id=run_id,
            )

            script = build_job_script(
                job_name=job_name,
                command=command,
                **bsub_params,
            )

            job_id = submit_job(script, dry_run=dry_run)
            if job_id:
                pipeline_logger.info(f"  {model} seed {seed}: Job {job_id}")
                training_job_ids.append(job_id)
            elif dry_run:
                pipeline_logger.info(f"  [DRY RUN] {model} seed {seed}: {job_name}")
                training_job_ids.append(f"DRYRUN_{model}_{seed}")
            else:
                pipeline_logger.error(f"  {model} seed {seed}: Submission failed")

    # Submit post-processing job (aggregation + ensemble) with dependency on ALL training jobs
    pipeline_logger.info("Submitting post-processing job (aggregation + ensemble)...")
    post_job_name = f"CeD_{run_id}_post"
    dependency_expr = f"done(CeD_{run_id}_*_s*)"

    post_command = _build_postprocessing_command(
        config_file=config_file.resolve(),
        run_id=run_id,
        outdir=outdir.resolve(),
        infile=infile.resolve(),
        split_dir=split_dir.resolve(),
        models=models,
        split_seeds=split_seeds,
        enable_ensemble=enable_ensemble,
    )

    post_script = build_job_script(
        job_name=post_job_name,
        command=post_command,
        dependency=dependency_expr,
        **bsub_params,
    )

    post_job_id = submit_job(post_script, dry_run=dry_run)
    if post_job_id:
        pipeline_logger.info(f"  Post-processing: Job {post_job_id} (depends on training)")
    elif dry_run:
        pipeline_logger.info(f"  [DRY RUN] Post-processing: {post_job_name}")

    # Submit panel optimization jobs (one per model) with dependency on post-processing
    panel_job_ids = []
    if enable_optimize_panel:
        pipeline_logger.info(f"Submitting {len(models)} panel optimization jobs (parallel)...")
        for model in models:
            panel_job_name = f"CeD_{run_id}_panel_{model}"
            panel_dependency = f"done({post_job_name})"

            panel_command = _build_panel_optimization_command(
                run_id=run_id,
                model=model,
            )

            panel_script = build_job_script(
                job_name=panel_job_name,
                command=panel_command,
                dependency=panel_dependency,
                **bsub_params,
            )

            panel_job_id = submit_job(panel_script, dry_run=dry_run)
            if panel_job_id:
                pipeline_logger.info(f"  Panel optimization ({model}): Job {panel_job_id}")
                panel_job_ids.append(panel_job_id)
            elif dry_run:
                pipeline_logger.info(f"  [DRY RUN] Panel optimization ({model}): {panel_job_name}")
                panel_job_ids.append(f"DRYRUN_{panel_job_name}")
            else:
                pipeline_logger.error(f"  Panel optimization ({model}): Submission failed")

    # Submit consensus panel job with dependency on post-processing + all panel optimization jobs
    consensus_job_id = None
    if enable_consensus:
        pipeline_logger.info("Submitting consensus panel job...")
        consensus_job_name = f"CeD_{run_id}_consensus"

        # Consensus depends on post-processing (aggregation) AND panel optimization jobs
        # (if enabled)
        if enable_optimize_panel and panel_job_ids:
            # Wait for both post-processing and all panel optimization jobs
            consensus_dependency = f"done({post_job_name}) && done(CeD_{run_id}_panel_*)"
        else:
            # Only wait for post-processing (aggregation must be done)
            consensus_dependency = f"done({post_job_name})"

        consensus_command = _build_consensus_panel_command(run_id=run_id)

        consensus_script = build_job_script(
            job_name=consensus_job_name,
            command=consensus_command,
            dependency=consensus_dependency,
            **bsub_params,
        )

        consensus_job_id = submit_job(consensus_script, dry_run=dry_run)
        if consensus_job_id:
            pipeline_logger.info(f"  Consensus panel: Job {consensus_job_id}")
        elif dry_run:
            pipeline_logger.info(f"  [DRY RUN] Consensus panel: {consensus_job_name}")

    return {
        "run_id": run_id,
        "training_jobs": training_job_ids,
        "postprocessing_job": post_job_id or f"DRYRUN_{post_job_name}",
        "panel_optimization_jobs": panel_job_ids,
        "consensus_job": consensus_job_id
        or (f"DRYRUN_{consensus_job_name}" if enable_consensus else None),
        "logs_dir": run_logs_dir,
    }
