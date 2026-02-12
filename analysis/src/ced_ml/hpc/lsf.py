"""LSF job submission utilities for HPC pipeline orchestration.

Provides functions to build and submit LSF (bsub) job scripts for running
the CeD-ML pipeline on HPC clusters with job dependency chains.

Parallelization strategy:
- Training: M x S parallel jobs (one per model per split)
- Post-processing: Single aggregation job (waits for all training)
- Panel optimization: M x S seed jobs + M aggregation jobs
- Consensus: Single job (waits for panel aggregation)
- Permutation tests: M x S parallel jobs (one per model per seed)
- Permutation aggregation: M jobs (one per model, aggregates per-seed results)

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


def validate_identifier(value: str, name: str = "identifier") -> None:
    """Validate that a string is safe for shell interpolation.

    Args:
        value: String to validate
        name: Name of the parameter (for error messages)

    Raises:
        ValueError: If value contains unsafe characters
    """
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", value):
        raise ValueError(
            f"Invalid {name}: '{value}'\n"
            f"Must contain only alphanumeric characters, underscores, hyphens, and dots.\n"
            f"This prevents shell injection when interpolating into commands."
        )


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
    except (ValueError, RuntimeError, OSError):
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
        Both stdout and stderr are redirected to /dev/null because ced
        commands create their own structured log files in
        logs/run_{ID}/training/, etc.  Stderr from LSF jobs typically
        contains only library warnings (e.g. convergence) that are
        already captured in the ced logs when relevant.
    """
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
#BSUB -eo /dev/null
{dep_line}

set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

{command}
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
    # Validate identifiers before interpolating into shell commands
    validate_identifier(model, "model")
    validate_identifier(run_id, "run_id")
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

    Panel optimization and consensus are handled by separate parallel jobs.

    Returns a multi-line bash script fragment that runs each step sequentially.
    Aggregation steps are fatal (set -e); ensemble training failures are
    non-fatal so that one seed's failure does not abort subsequent seeds
    or the ENSEMBLE aggregation.
    """
    lines = [
        f'echo "Post-processing (aggregation + ensemble) for run {run_id}"',
        "",
    ]

    # Aggregate base models (fatal -- these must succeed)
    for model in models:
        lines.append(f'echo "Aggregating {model}..."')
        lines.append(f"ced aggregate-splits --run-id {run_id} --model {model}")
        lines.append("")

    # Train ensemble per seed (non-fatal: continue on failure so remaining
    # seeds and ENSEMBLE aggregation can still proceed)
    if enable_ensemble:
        lines.append("ENSEMBLE_FAILURES=0")
        for seed in split_seeds:
            lines.append(f'echo "Training ensemble seed {seed}..."')
            lines.append(
                f"ced train-ensemble --run-id {run_id} --split-seed {seed}"
                f" || {{ echo 'WARNING: ensemble seed {seed} failed'; ENSEMBLE_FAILURES=$((ENSEMBLE_FAILURES+1)); }}"
            )
        lines.append("")
        lines.append('echo "Aggregating ENSEMBLE..."')
        lines.append(
            f"ced aggregate-splits --run-id {run_id} --model ENSEMBLE"
            f' || echo "WARNING: ENSEMBLE aggregation failed (expected if all ensemble seeds failed)"'
        )
        lines.append("")
        lines.append(
            'if [ "$ENSEMBLE_FAILURES" -gt 0 ]; then'
            f' echo "WARNING: $ENSEMBLE_FAILURES/{len(split_seeds)} ensemble seeds failed";'
            " fi"
        )
        lines.append("")

    lines.append(f'echo "Aggregation and ensemble training complete for run {run_id}"')

    return "\n".join(lines)


def _build_panel_optimization_command(
    *,
    run_id: str,
    model: str,
    split_seed: int | None = None,
) -> str:
    """Build panel optimization command for a single model.

    Args:
        run_id: Run identifier.
        model: Model name.
        split_seed: If provided, run RFE for this single seed only.
            If None, run aggregation (detects pre-computed seed results).

    Returns:
        A bash command string for optimizing panel size via RFE.
    """
    # Validate identifiers before interpolating into shell commands
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    cmd = f"ced optimize-panel --run-id {run_id} --model {model}"
    if split_seed is not None:
        cmd += f" --split-seed {split_seed}"
    return cmd


def _build_consensus_panel_command(
    *,
    run_id: str,
) -> str:
    """Build consensus panel command.

    Returns a bash command string for generating cross-model consensus panel.
    """
    return f"ced consensus-panel --run-id {run_id}"


def _build_permutation_test_command(
    *,
    run_id: str,
    model: str,
    split_seed: int = 0,
    n_perms: int = 200,
    random_state: int = 42,
) -> str:
    """Build permutation test command for HPC job array.

    Uses $LSB_JOBINDEX for LSF or $SLURM_ARRAY_TASK_ID for Slurm.

    Args:
        run_id: Run identifier.
        model: Model name.
        split_seed: Split seed to use.
        n_perms: Total number of permutations (for reference, not used in command).
        random_state: Random seed for reproducibility.

    Returns:
        A bash command string that uses the job array index for --perm-index.
    """
    # Validate identifiers before interpolating into shell commands
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    # Support both LSF ($LSB_JOBINDEX) and Slurm ($SLURM_ARRAY_TASK_ID)
    cmd = f"""# Detect job array index (LSF or Slurm)
if [ -n "${{LSB_JOBINDEX:-}}" ]; then
    PERM_INDEX=$LSB_JOBINDEX
elif [ -n "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    PERM_INDEX=$SLURM_ARRAY_TASK_ID
else
    echo "Error: Not running in a job array context (no LSB_JOBINDEX or SLURM_ARRAY_TASK_ID)"
    exit 1
fi

ced permutation-test \\
  --run-id {run_id} \\
  --model {model} \\
  --split-seed-start {split_seed} \\
  --perm-index $PERM_INDEX \\
  --random-state {random_state}"""
    return cmd


def _build_permutation_test_full_command(
    *,
    run_id: str,
    model: str,
    split_seed: int = 0,
    n_perms: int = 200,
    n_jobs: int = -1,
    random_state: int = 42,
) -> str:
    """Build full permutation test command (all permutations in one job).

    Runs all permutations with internal parallelization via joblib.

    Args:
        run_id: Run identifier.
        model: Model name.
        split_seed: Split seed to use.
        n_perms: Number of permutations.
        n_jobs: Parallel jobs for internal parallelization (-1 = all cores).
        random_state: Random seed for reproducibility.

    Returns:
        A bash command string for running full permutation test.
    """
    # Validate identifiers before interpolating into shell commands
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    cmd = f"""ced permutation-test \\
  --run-id {run_id} \\
  --model {model} \\
  --split-seed-start {split_seed} \\
  --n-perms {n_perms} \\
  --n-jobs {n_jobs} \\
  --random-state {random_state}"""
    return cmd


def _build_permutation_aggregation_command(
    *,
    run_id: str,
    model: str,
) -> str:
    """Build permutation aggregation command (runs after job array completes).

    This command aggregates individual perm_*.joblib files into a pooled
    significance result.

    Args:
        run_id: Run identifier.
        model: Model name.

    Returns:
        A bash command string for aggregating permutation results.
    """
    # Validate identifiers before interpolating into shell commands
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    cmd = f"""echo "Aggregating permutation results for {model}..."
ced permutation-test \\
  --run-id {run_id} \\
  --model {model}"""
    return cmd


def _build_job_id_dependency(job_ids: list[str]) -> str:
    """Build a ``done(id1) && done(id2) && ...`` dependency expression.

    Uses numeric LSF job IDs instead of name-based wildcards so that
    dependencies remain valid even after matched jobs leave the active
    job table (which causes TERM_ORPHAN_SYSTEM with wildcard patterns).
    """
    if not job_ids:
        raise ValueError("job_ids must not be empty")
    return " && ".join(f"done({jid})" for jid in job_ids)


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
    enable_permutation_test: bool = False,
    permutation_n_perms: int = 200,
    permutation_n_jobs: int = -1,
    permutation_split_seeds: list[int] | None = None,
    hpc_config: HPCConfig,
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline with dependency chains.

    Job dependency architecture (OPTIMIZED FOR CORRECTNESS + PARALLELIZATION):
    1. Training jobs (M x S jobs: one per model per split, fully parallel)
       Example: 4 models x 10 splits = 40 parallel jobs
    2. Post-processing job (aggregation + ensemble, depends on ALL training)
    3. Permutation test jobs (M x S, depends on training - parallel w/ post)
    4. Permutation aggregation (M jobs, depends on perm tests - produces sig)
    5. Panel seed jobs (M x S, depends on post + perm agg for sig gating)
    6. Panel aggregation jobs (M jobs, depends on that model's seed jobs)
    7. Consensus panel (depends on post + panel agg + perm agg)

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
        enable_permutation_test: Enable permutation testing for significance.
        permutation_n_perms: Number of permutations (default: 200).
        permutation_n_jobs: Parallel jobs for permutation testing (-1 = all cores).
        hpc_config: HPCConfig schema instance.
        logs_dir: Directory for job logs.
        dry_run: Preview without submitting.
        pipeline_logger: Logger instance.

    Returns:
        Dict with run_id, training_jobs, postprocessing_job, panel_jobs,
        consensus_job, permutation_jobs, logs_dir.
    """
    base_dir = Path.cwd()

    # Detect environment
    env_info = detect_environment(base_dir)
    pipeline_logger.info(f"Python environment: {env_info.env_type}")

    # Create log directory (structure: logs/run_{ID}/training/)
    run_logs_dir = logs_dir / f"run_{run_id}" / "training"
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
    training_job_names = []
    training_job_ids: list[str] = []

    for model in models:
        for seed in split_seeds:
            job_name = f"CeD_{run_id}_{model}_s{seed}"
            training_job_names.append(job_name)

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
                training_job_ids.append(job_id)
                pipeline_logger.info(f"  {model} seed {seed}: Job {job_id}")
            elif dry_run:
                pipeline_logger.info(f"  [DRY RUN] {model} seed {seed}: {job_name}")
            else:
                pipeline_logger.error(f"  {model} seed {seed}: Submission failed")

    # Submit post-processing job (aggregation + ensemble) with dependency on all
    # training jobs.  Uses numeric job IDs so dependencies survive LSF job-table
    # purging (name-based wildcards cause TERM_ORPHAN_SYSTEM).
    pipeline_logger.info("Submitting post-processing job (aggregation + ensemble)...")
    post_job_name = f"CeD_{run_id}_post"
    dependency_expr = _build_job_id_dependency(training_job_ids) if training_job_ids else None

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
        pipeline_logger.info(
            f"  Post-processing: Job {post_job_id} (depends on training job names)"
        )
    elif dry_run:
        pipeline_logger.info(f"  [DRY RUN] Post-processing: {post_job_name}")

    # Submit permutation test jobs BEFORE panel optimization (significance gating)
    # Permutation jobs depend on training jobs (can run in parallel with post-processing)
    # Permutation aggregation produces aggregated_significance.csv needed by panel optimization
    permutation_job_names = []
    permutation_agg_names: list[str] = []
    perm_agg_ids_by_model: dict[str, str] = {}
    if enable_permutation_test:
        # Use dedicated permutation seeds if provided, else fall back to training seeds
        perm_seeds = permutation_split_seeds if permutation_split_seeds else split_seeds
        n_perm_jobs = len(models) * len(perm_seeds)
        pipeline_logger.info(
            f"Submitting {n_perm_jobs} permutation test jobs "
            f"({len(models)} models x {len(perm_seeds)} seeds, "
            f"seeds={perm_seeds[0]}..{perm_seeds[-1]})..."
        )

        # Track permutation job IDs by model for aggregation dependencies
        perm_ids_by_model: dict[str, list[str]] = {m: [] for m in models}

        for model in models:
            for seed in perm_seeds:
                perm_job_name = f"CeD_{run_id}_perm_{model}_s{seed}"
                permutation_job_names.append(perm_job_name)

                # Depend on post-processing (aggregation) so observed AUROC is
                # available and all training outputs exist.  Permutation tests
                # train from scratch with permuted labels -- they do not need
                # a training job for this specific seed.
                perm_dependency = (
                    f"done({post_job_id})" if post_job_id else f"done({post_job_name})"
                )

                perm_command = _build_permutation_test_full_command(
                    run_id=run_id,
                    model=model,
                    split_seed=seed,
                    n_perms=permutation_n_perms,
                    n_jobs=permutation_n_jobs,
                )

                perm_script = build_job_script(
                    job_name=perm_job_name,
                    command=perm_command,
                    dependency=perm_dependency,
                    **bsub_params,
                )

                perm_job_id = submit_job(perm_script, dry_run=dry_run)
                if perm_job_id:
                    perm_ids_by_model[model].append(perm_job_id)
                    pipeline_logger.info(f"  Permutation test ({model} s{seed}): Job {perm_job_id}")
                elif dry_run:
                    pipeline_logger.info(
                        f"  [DRY RUN] Permutation test ({model} s{seed}): {perm_job_name}"
                    )
                else:
                    pipeline_logger.error(
                        f"  Permutation test ({model} s{seed}): Submission failed"
                    )

        # Submit permutation aggregation jobs (one per model, depends on all
        # perm jobs for that model + post-processing)
        pipeline_logger.info(f"Submitting {len(models)} permutation aggregation jobs...")
        for model in models:
            agg_job_name = f"CeD_{run_id}_perm_{model}_agg"
            model_perm_ids = perm_ids_by_model[model]

            if model_perm_ids:
                # Depend on all permutation jobs for this model AND post-processing.
                # Post-processing produces pooled_val_metrics.csv which provides the
                # observed AUROC needed for computing empirical p-values.
                agg_dep_ids = list(model_perm_ids)
                if post_job_id:
                    agg_dep_ids.append(post_job_id)
                agg_dependency = _build_job_id_dependency(agg_dep_ids)

                agg_command = _build_permutation_aggregation_command(
                    run_id=run_id,
                    model=model,
                )

                agg_script = build_job_script(
                    job_name=agg_job_name,
                    command=agg_command,
                    dependency=agg_dependency,
                    **bsub_params,
                )

                agg_job_id = submit_job(agg_script, dry_run=dry_run)
                if agg_job_id:
                    pipeline_logger.info(f"  Permutation aggregation ({model}): Job {agg_job_id}")
                    perm_agg_ids_by_model[model] = agg_job_id
                elif dry_run:
                    pipeline_logger.info(
                        f"  [DRY RUN] Permutation aggregation ({model}): " f"{agg_job_name}"
                    )
                else:
                    pipeline_logger.error(f"  Permutation aggregation ({model}): Submission failed")

                permutation_agg_names.append(agg_job_name)

    # Submit panel optimization jobs: M x S seed jobs + M aggregation jobs
    # When permutation testing is enabled, panel optimization depends on permutation
    # aggregation completing first (significance gating requires aggregated_significance.csv)
    panel_job_ids = []
    panel_agg_ids: list[str] = []
    panel_agg_names: list[str] = []  # populated inside enable_optimize_panel block
    if enable_optimize_panel:
        n_panel_seed_jobs = len(models) * len(split_seeds)
        pipeline_logger.info(
            f"Submitting {n_panel_seed_jobs} panel seed jobs "
            f"({len(models)} models x {len(split_seeds)} seeds) + "
            f"{len(models)} aggregation jobs..."
        )

        # Phase 1: Per-seed RFE jobs (M x S, parallel, depend on post-processing)
        # When permutation testing is enabled, also depend on permutation aggregation
        # so that significance data is available for gating
        panel_seed_ids_by_model: dict[str, list[str]] = {m: [] for m in models}
        for model in models:
            for seed in split_seeds:
                seed_job_name = f"CeD_{run_id}_panel_{model}_s{seed}"
                # Base dependency: post-processing must complete
                seed_dep_ids: list[str] = []
                if post_job_id:
                    seed_dep_ids.append(post_job_id)
                # If permutation testing enabled, also depend on that model's perm aggregation
                if enable_permutation_test and model in perm_agg_ids_by_model:
                    seed_dep_ids.append(perm_agg_ids_by_model[model])
                seed_dependency = _build_job_id_dependency(seed_dep_ids) if seed_dep_ids else None

                seed_command = _build_panel_optimization_command(
                    run_id=run_id,
                    model=model,
                    split_seed=seed,
                )

                seed_script = build_job_script(
                    job_name=seed_job_name,
                    command=seed_command,
                    dependency=seed_dependency,
                    **bsub_params,
                )

                seed_job_id = submit_job(seed_script, dry_run=dry_run)
                if seed_job_id:
                    pipeline_logger.info(f"  Panel seed ({model} s{seed}): Job {seed_job_id}")
                    panel_seed_ids_by_model[model].append(seed_job_id)
                panel_job_ids.append(seed_job_name)
                if not seed_job_id and not dry_run:
                    pipeline_logger.error(f"  Panel seed ({model} s{seed}): Submission failed")
                elif dry_run:
                    pipeline_logger.info(
                        f"  [DRY RUN] Panel seed ({model} s{seed}): {seed_job_name}"
                    )

        # Phase 2: Per-model aggregation jobs (M, depend on all seed jobs for that model)
        for model in models:
            agg_job_name = f"CeD_{run_id}_panel_{model}_agg"
            model_seed_ids = panel_seed_ids_by_model[model]
            agg_dependency = _build_job_id_dependency(model_seed_ids) if model_seed_ids else None

            agg_command = _build_panel_optimization_command(
                run_id=run_id,
                model=model,
            )

            agg_script = build_job_script(
                job_name=agg_job_name,
                command=agg_command,
                dependency=agg_dependency,
                **bsub_params,
            )

            agg_job_id = submit_job(agg_script, dry_run=dry_run)
            if agg_job_id:
                pipeline_logger.info(f"  Panel aggregation ({model}): Job {agg_job_id}")
                panel_agg_ids.append(agg_job_id)
            panel_job_ids.append(agg_job_name)
            panel_agg_names.append(agg_job_name)
            if not agg_job_id and not dry_run:
                pipeline_logger.error(f"  Panel aggregation ({model}): Submission failed")
            elif dry_run:
                pipeline_logger.info(f"  [DRY RUN] Panel aggregation ({model}): {agg_job_name}")

    # Submit consensus panel job (depends on post + panel agg + perm agg)
    consensus_job_id = None
    if enable_consensus:
        pipeline_logger.info("Submitting consensus panel job...")
        consensus_job_name = f"CeD_{run_id}_consensus"

        # Consensus depends on:
        # - post-processing (aggregation)
        # - panel aggregation jobs (if enabled)
        # - permutation aggregation jobs (if enabled, for significance filtering)
        consensus_dep_ids: list[str] = []
        if post_job_id:
            consensus_dep_ids.append(post_job_id)
        if enable_optimize_panel:
            consensus_dep_ids.extend(panel_agg_ids)
        if enable_permutation_test:
            consensus_dep_ids.extend(perm_agg_ids_by_model.values())
        consensus_dependency = (
            _build_job_id_dependency(consensus_dep_ids) if consensus_dep_ids else None
        )

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
        "training_jobs": training_job_names,
        "postprocessing_job": post_job_id or f"DRYRUN_{post_job_name}",
        "panel_optimization_jobs": panel_job_ids,
        "consensus_job": consensus_job_id
        or (f"DRYRUN_{consensus_job_name}" if enable_consensus else None),
        "permutation_jobs": permutation_job_names,
        "permutation_aggregation_jobs": permutation_agg_names,
        "logs_dir": run_logs_dir,
    }
