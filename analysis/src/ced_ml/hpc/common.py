"""Scheduler-agnostic HPC utilities for pipeline orchestration.

Contains all shared logic: path helpers, environment detection, config loading,
command builders, bash helpers, orchestrator script generation, and the
top-level pipeline submission function.  Scheduler-specific behaviour is
delegated to a SchedulerBackend instance.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ced_ml.config.schema import HPCConfig
from ced_ml.utils.run_manifest import build_model_manifest_entry, ensure_run_manifest

if TYPE_CHECKING:
    from ced_ml.hpc.base import SchedulerBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation and path helpers
# ---------------------------------------------------------------------------


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


def _sentinel_dir(logs_dir: Path, run_id: str) -> Path:
    """Get run-specific sentinel directory."""
    validate_identifier(run_id, "run_id")
    return logs_dir / f"run_{run_id}" / "sentinels"


def _scripts_dir(logs_dir: Path, run_id: str) -> Path:
    """Get run-specific script directory."""
    validate_identifier(run_id, "run_id")
    return logs_dir / f"run_{run_id}" / "scripts"


def _sentinel_log_path(sentinel_dir: Path) -> Path:
    """Get the consolidated sentinel completion log path."""
    return sentinel_dir / "completed.log"


def _write_job_script(scripts_dir: Path, job_name: str, script: str) -> Path:
    """Write script to disk and make it executable."""
    validate_identifier(job_name, "job_name")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{job_name}.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o750)
    return script_path


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


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

    analysis_dir = project_root / "analysis"

    # Prefer the cluster-aware bootstrap script if it exists. It's sourced
    # in place of ``venv/bin/activate`` and handles module loads,
    # PYTHONPATH hygiene, and venv activation uniformly across wrappers.
    # Edit analysis/scripts/hpc_activate.sh to re-target a new cluster;
    # the library stays cluster-agnostic.
    hpc_activate = analysis_dir / "scripts" / "hpc_activate.sh"
    if hpc_activate.exists():
        return EnvironmentInfo(
            env_type="venv",
            activation_cmd=f'source "{hpc_activate.resolve()}"',
        )

    candidates = [
        analysis_dir / "venv" / "bin" / "activate",
        analysis_dir / ".venv" / "bin" / "activate",
    ]

    venv_activate = next((p for p in candidates if p.exists()), None)
    if venv_activate is None:
        # Fallback for environments that are already activated outside project-local venv.
        import os

        virtual_env_raw = os.environ.get("VIRTUAL_ENV")
        if virtual_env_raw:
            activate = Path(virtual_env_raw).expanduser() / "bin" / "activate"
            if activate.exists():
                return EnvironmentInfo(
                    env_type="venv",
                    activation_cmd=f'source "{activate}"',
                )

        searched = ", ".join(str(p) for p in candidates)
        raise RuntimeError(
            f"venv not found. Checked: {searched}. " "Run: bash analysis/scripts/hpc_setup.sh"
        )

    return EnvironmentInfo(
        env_type="venv",
        activation_cmd=f'source "{venv_activate}"',
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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

    from ced_ml.config.loader import load_yaml

    raw_config = load_yaml(config_path)

    hpc_dict = raw_config.get("hpc", {})
    if not hpc_dict:
        raise ValueError(f"No 'hpc' section found in {config_path}")

    try:
        hpc_config = HPCConfig(**hpc_dict)
    except Exception as e:
        raise ValueError(f"Invalid HPC configuration in {config_path}: {e}") from e

    return hpc_config


def _load_training_scenario(config_file: Path | None) -> str | None:
    """Best-effort scenario extraction from training config."""
    if config_file is None or not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None

    scenario = config.get("scenario")
    return str(scenario) if scenario else None


# ---------------------------------------------------------------------------
# Job script building (scheduler-aware)
# ---------------------------------------------------------------------------


def build_job_script(
    *,
    scheduler: SchedulerBackend,
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
    """Build a job script using scheduler-specific directives.

    Args:
        scheduler: Backend that provides directive formatting.
        job_name: Job name.
        command: Shell command(s) to execute.
        project: HPC project allocation.
        queue: Queue/partition name.
        cores: Number of cores.
        mem_per_core: Memory per core in MB.
        walltime: Wall time limit as "HH:MM" or "HH:MM:SS".
        env_activation: Bash command to activate Python environment.
        log_dir: Directory for log files.
        dependency: Optional dependency expression.

    Returns:
        Complete bash script string.

    Note:
        stdout and stderr are routed to log_dir/<job_name>.{out,err} so that
        job-level failures (env activation, missing commands) are visible.
    """
    directives = scheduler.build_directives(
        job_name=job_name,
        project=project,
        queue=queue,
        cores=cores,
        mem_per_core=mem_per_core,
        walltime=walltime,
        stdout_path=str(log_dir / f"{job_name}.out"),
        stderr_path=str(log_dir / f"{job_name}.err"),
        dependency=dependency,
    )
    header = "\n".join(directives)

    script = f"""#!/bin/bash
{header}

set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

{command}
"""
    return script


def submit_job(
    script: str,
    *,
    scheduler: SchedulerBackend,
    dry_run: bool = False,
) -> str | None:
    """Submit a job via the scheduler's submission command.

    Args:
        script: Job script content to submit via stdin.
        scheduler: Backend that provides the submission command and ID parser.
        dry_run: If True, log the script but do not submit.

    Returns:
        Job ID string if submitted, None if dry_run or failure.

    Raises:
        RuntimeError: If the submission command is not available.
    """
    cmd_name = scheduler.submit_command

    if dry_run:
        logger.info("[DRY RUN] Would submit job script:")
        for line in script.strip().split("\n"):
            if line.startswith("#") or line.startswith("stdbuf"):
                logger.info(f"  {line}")
        return None

    if not shutil.which(cmd_name):
        raise RuntimeError(
            f"{cmd_name} command not found. "
            f"{scheduler.name.upper()} scheduler is required for --hpc mode."
        )

    result = subprocess.run(
        [cmd_name],
        input=script,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        logger.error(f"{cmd_name} failed: {result.stderr}")
        return None

    job_id = scheduler.parse_job_id(result.stdout)
    if job_id:
        return job_id

    logger.error(f"Could not parse job ID from {cmd_name} output: {result.stdout}")
    return None


# ---------------------------------------------------------------------------
# Command builders (scheduler-agnostic -- they produce `ced ...` CLI strings)
# ---------------------------------------------------------------------------


def _build_training_command(
    *,
    config_file: Path,
    infile: Path,
    split_dir: Path,
    outdir: Path,
    model: str,
    split_seed: int,
    run_id: str,
    split_index: int = 0,
) -> str:
    """Build ced train command for a single (model, split_seed) pair.

    Each job trains ONE model on ONE split for maximum parallelization.
    Post-processing (aggregation, ensemble, panels, consensus) handled by
    separate downstream jobs with proper dependencies.
    """
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
        f"--split-index {split_index}",
        f"--run-id {run_id}",
    ]
    return " \\\n  ".join(parts)


def _build_aggregation_command(
    *,
    run_id: str,
    model: str,
) -> str:
    """Build aggregation command for a single model.

    Returns:
        A bash command string for aggregating a single model's split results.
    """
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    return f"ced aggregate-splits --run-id {run_id} --model {model}"


def _build_ensemble_training_command(
    *,
    config_file: Path,
    run_id: str,
    split_seed: int,
    split_index: int,
) -> str:
    """Build ensemble training command for a single split seed.

    Returns:
        A bash command string for training one ensemble seed.
    """
    validate_identifier(run_id, "run_id")
    return (
        f'ced train-ensemble --config "{config_file}" '
        f"--run-id {run_id} --split-seed {split_seed} --split-index {split_index}"
    )


def _build_ensemble_aggregation_command(
    *,
    run_id: str,
) -> str:
    """Build ensemble aggregation command.

    Returns:
        A bash command string for aggregating ENSEMBLE model results.
    """
    validate_identifier(run_id, "run_id")
    return f"ced aggregate-splits --run-id {run_id} --model ENSEMBLE"


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
        for split_index, seed in enumerate(split_seeds):
            lines.append(f'echo "Training ensemble seed {seed}..."')
            lines.append(
                f'ced train-ensemble --config "{config_file}" --run-id {run_id} --split-seed {seed} --split-index {split_index}'
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
    n_jobs: int | None = None,
) -> str:
    """Build panel optimization command for a single model.

    Args:
        run_id: Run identifier.
        model: Model name.
        split_seed: If provided, run RFE for this single seed only.
            If None, run aggregation (detects pre-computed seed results).
        n_jobs: Parallel jobs for essentiality validation (-1 = all cores).
            Only applied in aggregation mode (split_seed is None).

    Returns:
        A bash command string for optimizing panel size via RFE.
    """
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    cmd = f"ced optimize-panel --run-id {run_id} --model {model}"
    if split_seed is not None:
        cmd += f" --split-seed {split_seed}"
    elif n_jobs is not None and n_jobs != 1:
        cmd += f" --n-jobs {n_jobs}"
    return cmd


def _build_consensus_panel_command(
    *,
    run_id: str,
    n_jobs: int = -1,
) -> str:
    """Build consensus panel command.

    Args:
        run_id: Run identifier.
        n_jobs: Parallel jobs for essentiality validation (-1 = all cores).

    Returns a bash command string for generating cross-model consensus panel.
    """
    cmd = f"ced consensus-panel --run-id {run_id}"
    if n_jobs != 1:
        cmd += f" --n-jobs {n_jobs}"
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
    """Build permutation aggregation command for a single model.

    Produces a ``ced permutation-test --aggregate-only`` command that pools
    per-seed null distribution CSVs into ``aggregated_significance.csv``.

    Args:
        run_id: Run identifier.
        model: Model name to aggregate.

    Returns:
        A bash command string for aggregating per-seed permutation results.
    """
    validate_identifier(run_id, "run_id")
    validate_identifier(model, "model")
    return f"ced permutation-test --run-id {run_id} --model {model} --aggregate-only"


# ---------------------------------------------------------------------------
# Bash helpers
# ---------------------------------------------------------------------------


def _bash_array_literal(name: str, values: list[str]) -> str:
    """Render a bash array declaration."""
    if not values:
        return f"{name}=()"
    lines = [f"{name}=("]
    lines.extend(f'  "{value}"' for value in values)
    lines.append(")")
    return "\n".join(lines)


def _encode_command_b64(command: str) -> str:
    """Encode a command string for safe environment transport."""
    return base64.b64encode(command.encode("utf-8")).decode("ascii")


def _build_wrapper_script(env_activation: str) -> str:
    """Build a generic wrapper that decodes and executes command payloads.

    Sentinel write uses a two-layer strategy for NFS reliability:
    1. Per-job sentinel file (``$JOB_NAME.done``) -- file existence is more
       NFS-cache-reliable than grepping a shared append-only file.
    2. Consolidated ``completed.log`` append -- backward-compatible with
       orchestrator grep checks.
    Both writes are followed by ``sync`` to flush to the NFS server.
    The sentinel is written in an EXIT trap so that ``set -e`` cannot skip
    it -- the orchestrator always sees completion (success or failure) and
    ``check_upstream_failures`` handles EXIT/TERM detection separately via
    the scheduler's status commands.
    """
    return f"""#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

if [ -z "${{CED_JOB_COMMAND_B64:-}}" ]; then
    echo "[$(date '+%F %T')] FATAL: CED_JOB_COMMAND_B64 not set"
    exit 1
fi
if [ -z "${{CED_JOB_NAME:-}}" ]; then
    echo "[$(date '+%F %T')] FATAL: CED_JOB_NAME not set"
    exit 1
fi
if [ -z "${{CED_SENTINEL_DIR:-}}" ]; then
    echo "[$(date '+%F %T')] FATAL: CED_SENTINEL_DIR not set"
    exit 1
fi

# Write per-job sentinel file + append to consolidated log on exit.
# Per-job file gives NFS-reliable existence check; consolidated log is a fallback.
# sync flushes writes to the NFS server before the process exits.
_ced_rc=0
trap '
  touch "$CED_SENTINEL_DIR/${{CED_JOB_NAME}}.done"
  echo "${{CED_JOB_NAME}}" >> "$CED_SENTINEL_DIR/completed.log"
  sync
' EXIT

COMMAND=$(python - "$CED_JOB_COMMAND_B64" <<'PY'
import base64
import sys
print(base64.b64decode(sys.argv[1]).decode("utf-8"), end="")
PY
)

eval "$COMMAND" || _ced_rc=$?
exit "$_ced_rc"
"""


def _build_wrapped_command(
    *,
    command_b64: str,
    job_name: str,
    sentinel_dir: Path,
    wrapper_script_path: Path,
    project_root: Path | None = None,
) -> str:
    """Build a per-job command payload that runs through the shared wrapper."""
    lines = [
        f'export CED_JOB_COMMAND_B64="{command_b64}"',
        f'export CED_JOB_NAME="{job_name}"',
        f'export CED_SENTINEL_DIR="{sentinel_dir.resolve()}"',
    ]
    if project_root is not None:
        lines.append(f'export CED_PROJECT_ROOT="{project_root.resolve()}"')
    lines.append(f'"{wrapper_script_path.resolve()}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator bash functions (shared + scheduler-specific composition)
# ---------------------------------------------------------------------------


def _build_shared_orchestrator_bash_functions() -> str:
    """Scheduler-agnostic bash functions for the orchestrator."""
    return """# Check if a job has a sentinel via per-job file or consolidated log.
# Returns 0 if found, 1 if not. Refreshes NFS attribute cache first.
sentinel_exists() {
    local name="$1"
    # Per-job .done file (most NFS-reliable)
    if [ -f "$SENTINEL_DIR/${name}.done" ]; then
        return 0
    fi
    # Fallback: consolidated log
    if grep -qx "${name}" "$SENTINEL_DIR/completed.log" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Like sentinel_exists but forces NFS attribute cache refresh first.
sentinel_exists_refresh() {
    local name="$1"
    # Force NFS attribute cache invalidation by listing the directory
    ls "$SENTINEL_DIR/" > /dev/null 2>&1 || true
    sentinel_exists "$name"
}

# Retry sentinel check with exponential backoff (10s, 30s, 60s).
# Returns 0 if sentinel eventually found, 1 if not after all retries.
sentinel_wait_retry() {
    local name="$1"
    local delay
    for delay in 10 30 60; do
        echo "[$(date '+%F %T')] WARNING: upstream job ($name) no sentinel -- retrying in ${delay}s"
        sleep "$delay"
        if sentinel_exists_refresh "$name"; then
            echo "[$(date '+%F %T')] WARNING: upstream job ($name) sentinel appeared after retry"
            return 0
        fi
    done
    return 1
}

manifest_job_tsv() {
    local job_key="$1"
    python - "$MANIFEST_PATH" "$job_key" <<'PY'
import json
import sys

manifest_path, job_key = sys.argv[1], sys.argv[2]
with open(manifest_path, encoding="utf-8") as f:
    jobs = json.load(f)

job = jobs.get(job_key)
if job is None:
    raise SystemExit(1)

fields = [
    job["job_name"],
    str(job["queue"]),
    str(job["cores"]),
    str(job["mem_per_core"]),
    str(job["walltime"]),
    job["command_b64"],
]
print("\\t".join(fields), end="")
PY
}

barrier_wait() {
    local label="$1"; shift
    local timeout="$1"; shift
    local poll="$1"; shift
    local -a job_names=("$@")
    local total=${#job_names[@]}
    local elapsed=0

    printf '{"stage":"%s","status":"started","ts":"%s"}\\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
    echo "[$(date '+%F %T')] Waiting for $label ($total jobs, timeout=${timeout}s)..."

    if [ "$total" -eq 0 ]; then
        printf '{"stage":"%s","status":"done","ts":"%s"}\\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
        return 0
    fi

    while true; do
        if [ ${#UPSTREAM_IDS[@]} -gt 0 ]; then
            check_upstream_failures "${UPSTREAM_IDS[@]}"
        fi

        local missing=0
        local name
        for name in "${job_names[@]}"; do
            sentinel_exists "${name}" || missing=$((missing + 1))
        done

        if [ "$missing" -eq 0 ]; then
            echo "[$(date '+%F %T')] $label complete ($total/$total)."
            printf '{"stage":"%s","status":"done","ts":"%s"}\\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
            return 0
        fi

        if [ "$elapsed" -ge "$timeout" ]; then
            echo "[$(date '+%F %T')] TIMEOUT: $label after ${timeout}s ($((total - missing))/$total done)"
            echo "[$(date '+%F %T')] Missing jobs:"
            for name in "${job_names[@]}"; do
                sentinel_exists "${name}" || echo "  $name"
            done
            printf '{"stage":"%s","status":"timeout","ts":"%s"}\\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
            exit 1
        fi

        sleep "$poll"
        elapsed=$((elapsed + poll))
    done
}

submit_batch() {
    local chunk_size="$1"; shift
    local id_file="$1"; shift
    local -a job_keys=("$@")
    local i
    for ((i=0; i<${#job_keys[@]}; i++)); do
        submit_and_track "${job_keys[$i]}" "${job_keys[$i]}" "$id_file"
        if (( (i + 1) % chunk_size == 0 && i + 1 < ${#job_keys[@]} )); then
            echo "[$(date '+%F %T')] Submitted $((i+1))/${#job_keys[@]}, pausing..."
            sleep 2
        fi
    done
}"""


def _build_orchestrator_bash_functions(scheduler: SchedulerBackend) -> str:
    """Compose orchestrator bash functions from shared + scheduler-specific."""
    shared = _build_shared_orchestrator_bash_functions()
    submit_func = scheduler.build_orchestrator_submit_func()
    status_func = scheduler.build_orchestrator_status_func()
    return f"{shared}\n\n{submit_func}\n\n{status_func}"


# ---------------------------------------------------------------------------
# Orchestrator script generation
# ---------------------------------------------------------------------------


def _build_orchestrator_script(
    *,
    scheduler: SchedulerBackend,
    run_id: str,
    hpc_config: HPCConfig,
    sentinel_dir: Path,
    scripts_dir: Path,
    orchestrator_log: Path,
    orchestrator_job_name: str,
    manifest_path: Path,
    wrapper_script_path: Path,
    training_job_ids: list[str],
    training_job_names: list[str],
    agg_keys: list[str],
    agg_job_names: list[str],
    ensemble_keys: list[str],
    ensemble_job_names: list[str],
    ensemble_agg_key: str | None,
    ensemble_agg_job_name: str | None,
    perm_keys: list[str],
    perm_job_names: list[str],
    perm_agg_keys: list[str],
    perm_agg_job_names: list[str],
    panel_seed_keys: list[str],
    panel_seed_job_names: list[str],
    panel_agg_keys: list[str],
    panel_agg_job_names: list[str],
    consensus_key: str | None,
    consensus_job_name: str | None,
    expected_training_jobs: int,
    project_root: Path | None = None,
) -> str:
    """Build the barrier-orchestrator bash script.

    DAG (maximally parallel):
        training -> aggregation (per model, parallel)
        -> [ensemble (per seed) | perm tests | panel seeds] (all parallel)
        -> ensemble barrier -> ensemble_agg
        -> perm barrier -> perm_agg
        -> panel barrier -> panel_agg
        -> combined downstream barrier
        -> consensus -> done
    """
    validate_identifier(run_id, "run_id")
    orchestrator_cfg = hpc_config.orchestrator

    header_lines = scheduler.build_orchestrator_header(
        project=hpc_config.project,
        queue=hpc_config.queue,
        job_name=f"CeD_{run_id}_orchestrator",
        cores=orchestrator_cfg.cores,
        mem_per_core=orchestrator_cfg.mem_per_core,
        walltime=orchestrator_cfg.walltime,
        log_path=orchestrator_log,
    )

    lines: list[str] = [
        *header_lines,
        "",
        "set -euo pipefail",
        "",
        "export PYTHONUNBUFFERED=1",
        "export FORCE_COLOR=1",
        "",
        _build_orchestrator_bash_functions(scheduler),
        "",
        f'PROJECT="{hpc_config.project}"',
        f'MANIFEST_PATH="{manifest_path.resolve()}"',
        f'WRAPPER_SCRIPT="{wrapper_script_path.resolve()}"',
        f'SENTINEL_DIR="{sentinel_dir.resolve()}"',
        f'SCRIPTS_DIR="{scripts_dir.resolve()}"',
        *([f'CED_PROJECT_ROOT="{project_root.resolve()}"'] if project_root is not None else []),
        # Per-job stdout/stderr land here. Absolutely critical for
        # post-mortem debugging of silent downstream failures.
        f'LSF_OUTPUT_DIR="{(sentinel_dir.parent / "lsf_output").resolve()}"',
        'STATE_FILE="$SENTINEL_DIR/orchestrator_state.jsonl"',
        f"POLL_INTERVAL={orchestrator_cfg.poll_interval}",
        f"MAX_CHUNK={orchestrator_cfg.max_concurrent_submissions}",
        f"EXPECTED_TRAINING={expected_training_jobs}",
        "",
        _bash_array_literal("TRAINING_IDS", training_job_ids),
        _bash_array_literal("TRAINING_JOBS", training_job_names),
        "",
        "UPSTREAM_IDS=()",
        'mkdir -p "$SENTINEL_DIR" "$SCRIPTS_DIR" "$LSF_OUTPUT_DIR"',
        'touch "$STATE_FILE"',
        'touch "$SENTINEL_DIR/completed.log"',
        f"echo \"[$(date '+%F %T')] Orchestrator started for run {run_id}\"",
        "echo \"[$(date '+%F %T')] Training IDs submitted: ${#TRAINING_IDS[@]} (expected: $EXPECTED_TRAINING)\"",
        "",
        # Stage 1: Wait for training
        'UPSTREAM_IDS=("${TRAINING_IDS[@]}")',
        f'barrier_wait "training" {orchestrator_cfg.timeout_seconds("training")} "$POLL_INTERVAL" "${{TRAINING_JOBS[@]}}"',
        "",
    ]

    # Stage 2: Per-model aggregation (parallel)
    lines.extend(
        [
            _bash_array_literal("AGG_KEYS", agg_keys),
            _bash_array_literal("AGG_JOBS", agg_job_names),
            'AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/agg_ids.XXXXXX")',
            'submit_batch "$MAX_CHUNK" "$AGG_IDS_FILE" "${AGG_KEYS[@]}"',
            'mapfile -t UPSTREAM_IDS < "$AGG_IDS_FILE"',
            f'barrier_wait "aggregation" {orchestrator_cfg.timeout_seconds("aggregation")} "$POLL_INTERVAL" "${{AGG_JOBS[@]}}"',
            "",
        ]
    )

    # Stage 3: Submit all post-aggregation jobs simultaneously
    # These are independent: ensemble training, permutation tests, panel seeds
    has_ensemble = bool(ensemble_keys)
    has_perm = bool(perm_keys)
    has_panel = bool(panel_seed_keys)

    if has_ensemble:
        lines.extend(
            [
                _bash_array_literal("ENSEMBLE_KEYS", ensemble_keys),
                _bash_array_literal("ENSEMBLE_JOBS", ensemble_job_names),
                'ENSEMBLE_IDS_FILE=$(mktemp "$SENTINEL_DIR/ensemble_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$ENSEMBLE_IDS_FILE" "${ENSEMBLE_KEYS[@]}"',
                "",
            ]
        )

    if has_perm:
        lines.extend(
            [
                _bash_array_literal("PERM_KEYS", perm_keys),
                _bash_array_literal("PERM_JOBS", perm_job_names),
                'PERM_IDS_FILE=$(mktemp "$SENTINEL_DIR/perm_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PERM_IDS_FILE" "${PERM_KEYS[@]}"',
                "",
            ]
        )

    if has_panel:
        lines.extend(
            [
                _bash_array_literal("PANEL_SEED_KEYS", panel_seed_keys),
                _bash_array_literal("PANEL_SEED_JOBS", panel_seed_job_names),
                'PANEL_SEED_IDS_FILE=$(mktemp "$SENTINEL_DIR/panel_seed_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PANEL_SEED_IDS_FILE" "${PANEL_SEED_KEYS[@]}"',
                "",
            ]
        )

    # Stage 4: Wait for each chain and submit its downstream job(s).
    # Jobs from all chains are already running in parallel on HPC.
    # The orchestrator waits for each chain sequentially, but the actual
    # compute is fully overlapped.

    # Collect all "penultimate" job names for the combined downstream barrier
    downstream_job_names: list[str] = []

    if has_ensemble:
        lines.extend(
            [
                'mapfile -t UPSTREAM_IDS < "$ENSEMBLE_IDS_FILE"',
                f'barrier_wait "ensemble-training" {orchestrator_cfg.timeout_seconds("ensemble")} "$POLL_INTERVAL" "${{ENSEMBLE_JOBS[@]}}"',
                "",
            ]
        )
        if ensemble_agg_key and ensemble_agg_job_name:
            lines.extend(
                [
                    'ENSEMBLE_AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/ensemble_agg_ids.XXXXXX")',
                    f'submit_and_track "{ensemble_agg_key}" "ensemble-aggregation" "$ENSEMBLE_AGG_IDS_FILE"',
                    "",
                ]
            )
            downstream_job_names.append(ensemble_agg_job_name)

    if has_perm:
        lines.extend(
            [
                'mapfile -t UPSTREAM_IDS < "$PERM_IDS_FILE"',
                f'barrier_wait "permutation-tests" {orchestrator_cfg.timeout_seconds("perm")} "$POLL_INTERVAL" "${{PERM_JOBS[@]}}"',
                "",
            ]
        )
        if perm_agg_keys:
            lines.extend(
                [
                    _bash_array_literal("PERM_AGG_KEYS", perm_agg_keys),
                    _bash_array_literal("PERM_AGG_JOBS", perm_agg_job_names),
                    'PERM_AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/perm_agg_ids.XXXXXX")',
                    'submit_batch "$MAX_CHUNK" "$PERM_AGG_IDS_FILE" "${PERM_AGG_KEYS[@]}"',
                    "",
                ]
            )
            downstream_job_names.extend(perm_agg_job_names)

    if has_panel:
        lines.extend(
            [
                'mapfile -t UPSTREAM_IDS < "$PANEL_SEED_IDS_FILE"',
                f'barrier_wait "panel-seed" {orchestrator_cfg.timeout_seconds("panel")} "$POLL_INTERVAL" "${{PANEL_SEED_JOBS[@]}}"',
                "",
                _bash_array_literal("PANEL_AGG_KEYS", panel_agg_keys),
                _bash_array_literal("PANEL_AGG_JOBS", panel_agg_job_names),
                'PANEL_AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/panel_agg_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PANEL_AGG_IDS_FILE" "${PANEL_AGG_KEYS[@]}"',
                "",
            ]
        )
        downstream_job_names.extend(panel_agg_job_names)

    # Stage 5: Wait for all downstream aggregation jobs
    if downstream_job_names:
        lines.extend(
            [
                _bash_array_literal("DOWNSTREAM_JOBS", downstream_job_names),
                "UPSTREAM_IDS=()",
                # Collect IDs from whichever downstream stages exist
            ]
        )
        if has_ensemble and ensemble_agg_key:
            lines.append('mapfile -t _ids < "$ENSEMBLE_AGG_IDS_FILE"; UPSTREAM_IDS+=("${_ids[@]}")')
        if perm_agg_keys:
            lines.append('mapfile -t _ids < "$PERM_AGG_IDS_FILE"; UPSTREAM_IDS+=("${_ids[@]}")')
        if has_panel:
            lines.append('mapfile -t _ids < "$PANEL_AGG_IDS_FILE"; UPSTREAM_IDS+=("${_ids[@]}")')
        lines.extend(
            [
                f'barrier_wait "downstream-aggregation" {orchestrator_cfg.timeout_seconds("panel")} "$POLL_INTERVAL" "${{DOWNSTREAM_JOBS[@]}}"',
                "",
            ]
        )

    # Stage 6: Consensus
    if consensus_key and consensus_job_name:
        lines.extend(
            [
                'CONSENSUS_IDS_FILE=$(mktemp "$SENTINEL_DIR/consensus_ids.XXXXXX")',
                f'submit_and_track "{consensus_key}" "consensus" "$CONSENSUS_IDS_FILE"',
                'mapfile -t UPSTREAM_IDS < "$CONSENSUS_IDS_FILE"',
                _bash_array_literal("CONSENSUS_JOBS", [consensus_job_name]),
                f'barrier_wait "consensus" {orchestrator_cfg.timeout_seconds("consensus")} "$POLL_INTERVAL" "${{CONSENSUS_JOBS[@]}}"',
                "",
            ]
        )

    lines.extend(
        [
            f'echo "{orchestrator_job_name}" >> "$SENTINEL_DIR/completed.log"',
            'printf \'{"stage":"orchestrator","status":"done","ts":"%s"}\\n\' "$(date -u \'+%FT%TZ\')" >> "$STATE_FILE"',
            f"echo \"[$(date '+%F %T')] Orchestrator complete for run {run_id}\"",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline submission
# ---------------------------------------------------------------------------


def _submit_orchestrator_pipeline(
    *,
    scheduler: SchedulerBackend,
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
    configs: dict[str, str | Path] | None = None,
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline using a barrier-orchestrator job."""
    base_dir = Path.cwd()
    env_info = detect_environment(base_dir)
    pipeline_logger.info(f"Python environment: {env_info.env_type}")

    from ced_ml.utils.paths import get_project_root as _get_project_root

    try:
        _project_root: Path | None = _get_project_root()
    except RuntimeError:
        _project_root = None

    run_root = logs_dir / f"run_{run_id}"
    run_logs_dir = run_root / "training"
    sentinel_dir = _sentinel_dir(logs_dir, run_id)
    scripts_dir = _scripts_dir(logs_dir, run_id)
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize shared run manifest before any per-seed training jobs start.
    scenario = _load_training_scenario(config_file)
    run_metadata_path, metadata_changed = ensure_run_manifest(
        run_level_dir=outdir / f"run_{run_id}",
        run_id=run_id,
        infile=infile,
        split_dir=split_dir,
        model_entries={
            model: build_model_manifest_entry(
                scenario=scenario,
                infile=infile,
                split_dir=split_dir,
            )
            for model in models
        },
        configs=configs,
    )
    if metadata_changed:
        pipeline_logger.info(f"Initialized run metadata manifest: {run_metadata_path}")
    else:
        pipeline_logger.debug(f"Run metadata manifest already initialized: {run_metadata_path}")

    default_resources = hpc_config.get_resources("default")
    postprocessing_resources = hpc_config.get_resources("postprocessing")
    job_params = {
        "project": hpc_config.project,
        "env_activation": env_info.activation_cmd,
        "log_dir": run_logs_dir,
        **default_resources,
    }

    wrapper_job_name = f"CeD_{run_id}_job_wrapper"
    wrapper_script_path = _write_job_script(
        scripts_dir,
        wrapper_job_name,
        _build_wrapper_script(env_info.activation_cmd),
    )

    def _manifest_entry(
        *,
        job_name: str,
        command: str,
        resources: dict[str, str | int] | None = None,
    ) -> dict[str, str | int]:
        r = resources if resources is not None else default_resources
        return {
            "job_name": job_name,
            "queue": str(r["queue"]),
            "cores": int(r["cores"]),
            "mem_per_core": int(r["mem_per_core"]),
            "walltime": str(r["walltime"]),
            "command_b64": _encode_command_b64(command),
        }

    # Stage 1: training (submitted immediately)
    training_job_names: list[str] = []
    training_job_ids: list[str] = []
    expected_training_jobs = len(models) * len(split_seeds)
    pipeline_logger.info(
        f"Submitting {expected_training_jobs} training jobs "
        f"({len(models)} models x {len(split_seeds)} seeds)..."
    )
    pipeline_logger.info(f"Scripts dir: {scripts_dir}")
    pipeline_logger.info(f"Training logs: {run_logs_dir}")

    for model in models:
        for split_index, seed in enumerate(split_seeds):
            job_name = f"CeD_{run_id}_{model}_s{seed}"
            training_job_names.append(job_name)
            training_command = _build_training_command(
                config_file=config_file.resolve(),
                infile=infile.resolve(),
                split_dir=split_dir.resolve(),
                outdir=outdir.resolve(),
                model=model,
                split_seed=seed,
                run_id=run_id,
                split_index=split_index,
            )

            wrapped_command = _build_wrapped_command(
                command_b64=_encode_command_b64(training_command),
                job_name=job_name,
                sentinel_dir=sentinel_dir,
                wrapper_script_path=wrapper_script_path,
                project_root=_project_root,
            )
            submission_script = build_job_script(
                scheduler=scheduler,
                job_name=job_name,
                command=wrapped_command,
                **job_params,
            )

            _write_job_script(scripts_dir, job_name, submission_script)

            if dry_run:
                pipeline_logger.info(f"  [DRY RUN] Training ({model} s{seed}): {job_name}")
                continue

            job_id = submit_job(submission_script, scheduler=scheduler, dry_run=False)
            if not job_id:
                raise RuntimeError(f"Training job submission failed: {job_name}")
            training_job_ids.append(job_id)
            pipeline_logger.info(f"  Training ({model} s{seed}): Job {job_id}")

    if not dry_run and len(training_job_ids) != expected_training_jobs:
        raise RuntimeError(
            f"Expected {expected_training_jobs} training job IDs, got {len(training_job_ids)}"
        )

    # Downstream stages are represented in a single manifest
    manifest_jobs: dict[str, dict[str, str | int]] = {}

    # Stage 2: Per-model aggregation (parallel)
    agg_job_names: list[str] = []
    agg_keys: list[str] = []
    for model in models:
        agg_job = f"CeD_{run_id}_agg_{model}"
        agg_cmd = _build_aggregation_command(run_id=run_id, model=model)
        agg_key = f"agg_{model}"
        manifest_jobs[agg_key] = _manifest_entry(
            job_name=agg_job, command=agg_cmd, resources=postprocessing_resources
        )
        agg_job_names.append(agg_job)
        agg_keys.append(agg_key)

    # Stage 3a: Ensemble training (parallel per seed)
    ensemble_job_names: list[str] = []
    ensemble_keys: list[str] = []
    ensemble_agg_key: str | None = None
    ensemble_agg_job_name: str | None = None
    if enable_ensemble:
        for split_index, seed in enumerate(split_seeds):
            ens_job = f"CeD_{run_id}_ensemble_s{seed}"
            ens_cmd = _build_ensemble_training_command(
                config_file=config_file.resolve(),
                run_id=run_id,
                split_seed=seed,
                split_index=split_index,
            )
            ens_key = f"ensemble_s{seed}"
            manifest_jobs[ens_key] = _manifest_entry(job_name=ens_job, command=ens_cmd)
            ensemble_job_names.append(ens_job)
            ensemble_keys.append(ens_key)

        # Ensemble aggregation (single job after all ensemble seeds)
        ensemble_agg_job_name = f"CeD_{run_id}_ensemble_agg"
        ensemble_agg_cmd = _build_ensemble_aggregation_command(run_id=run_id)
        ensemble_agg_key = "ensemble_agg"
        manifest_jobs[ensemble_agg_key] = _manifest_entry(
            job_name=ensemble_agg_job_name,
            command=ensemble_agg_cmd,
            resources=postprocessing_resources,
        )

    # Stage 3b: Permutation tests (parallel per model x seed)
    permutation_job_names: list[str] = []
    perm_keys: list[str] = []
    perm_agg_names: list[str] = []
    perm_agg_keys: list[str] = []
    if enable_permutation_test:
        perm_seeds = permutation_split_seeds if permutation_split_seeds else split_seeds
        for model in models:
            for seed in perm_seeds:
                perm_job_name = f"CeD_{run_id}_perm_{model}_s{seed}"
                perm_command = _build_permutation_test_full_command(
                    run_id=run_id,
                    model=model,
                    split_seed=seed,
                    n_perms=permutation_n_perms,
                    n_jobs=permutation_n_jobs,
                )
                perm_key = f"perm_{model}_s{seed}"
                manifest_jobs[perm_key] = _manifest_entry(
                    job_name=perm_job_name,
                    command=perm_command,
                )
                permutation_job_names.append(perm_job_name)
                perm_keys.append(perm_key)

            # Per-model aggregation of per-seed null CSVs
            perm_agg_job = f"CeD_{run_id}_perm_{model}_agg"
            perm_agg_cmd = _build_permutation_aggregation_command(
                run_id=run_id,
                model=model,
            )
            perm_agg_key = f"perm_{model}_agg"
            manifest_jobs[perm_agg_key] = _manifest_entry(
                job_name=perm_agg_job,
                command=perm_agg_cmd,
                resources=postprocessing_resources,
            )
            perm_agg_names.append(perm_agg_job)
            perm_agg_keys.append(perm_agg_key)

    # Stage 3c: Panel optimization seeds (parallel per model x seed)
    panel_job_names: list[str] = []
    panel_agg_names: list[str] = []
    panel_seed_keys: list[str] = []
    panel_agg_keys: list[str] = []
    if enable_optimize_panel:
        for model in models:
            for seed in split_seeds:
                panel_seed_job = f"CeD_{run_id}_panel_{model}_s{seed}"
                panel_seed_cmd = _build_panel_optimization_command(
                    run_id=run_id,
                    model=model,
                    split_seed=seed,
                )
                panel_seed_key = f"panel_{model}_s{seed}"
                manifest_jobs[panel_seed_key] = _manifest_entry(
                    job_name=panel_seed_job,
                    command=panel_seed_cmd,
                )
                panel_job_names.append(panel_seed_job)
                panel_seed_keys.append(panel_seed_key)

            panel_agg_job = f"CeD_{run_id}_panel_{model}_agg"
            panel_agg_cmd = _build_panel_optimization_command(run_id=run_id, model=model, n_jobs=-1)
            panel_agg_key = f"panel_{model}_agg"
            manifest_jobs[panel_agg_key] = _manifest_entry(
                job_name=panel_agg_job,
                command=panel_agg_cmd,
                resources=postprocessing_resources,
            )
            panel_job_names.append(panel_agg_job)
            panel_agg_names.append(panel_agg_job)
            panel_agg_keys.append(panel_agg_key)

    consensus_job_name = f"CeD_{run_id}_consensus"
    consensus_key: str | None = None
    if enable_consensus:
        consensus_cmd = _build_consensus_panel_command(run_id=run_id, n_jobs=-1)
        consensus_key = "consensus"
        manifest_jobs[consensus_key] = _manifest_entry(
            job_name=consensus_job_name,
            command=consensus_cmd,
            resources=postprocessing_resources,
        )

    manifest_path = scripts_dir / "jobs_manifest.json"
    manifest_path.write_text(json.dumps(manifest_jobs, indent=2, sort_keys=True), encoding="utf-8")

    orchestrator_orch_name = f"CeD_{run_id}_orchestrator"
    orchestrator_log = run_root / "orchestrator.log"
    orchestrator_script = _build_orchestrator_script(
        scheduler=scheduler,
        run_id=run_id,
        hpc_config=hpc_config,
        sentinel_dir=sentinel_dir,
        scripts_dir=scripts_dir,
        orchestrator_log=orchestrator_log,
        orchestrator_job_name=orchestrator_orch_name,
        manifest_path=manifest_path,
        wrapper_script_path=wrapper_script_path,
        training_job_ids=training_job_ids,
        training_job_names=training_job_names,
        agg_keys=agg_keys,
        agg_job_names=agg_job_names,
        ensemble_keys=ensemble_keys,
        ensemble_job_names=ensemble_job_names,
        ensemble_agg_key=ensemble_agg_key,
        ensemble_agg_job_name=ensemble_agg_job_name,
        perm_keys=perm_keys,
        perm_job_names=permutation_job_names,
        perm_agg_keys=perm_agg_keys,
        perm_agg_job_names=perm_agg_names,
        panel_seed_keys=panel_seed_keys,
        panel_seed_job_names=[n for n in panel_job_names if n not in panel_agg_names],
        panel_agg_keys=panel_agg_keys,
        panel_agg_job_names=panel_agg_names,
        consensus_key=consensus_key,
        consensus_job_name=consensus_job_name if enable_consensus else None,
        expected_training_jobs=expected_training_jobs,
        project_root=_project_root,
    )
    _write_job_script(scripts_dir, orchestrator_orch_name, orchestrator_script)

    orchestrator_job_id = submit_job(orchestrator_script, scheduler=scheduler, dry_run=dry_run)
    if not dry_run and not orchestrator_job_id:
        raise RuntimeError("Orchestrator submission failed")

    orchestrator_job_display = orchestrator_job_id or f"DRYRUN_{orchestrator_orch_name}"
    pipeline_logger.info("-- HPC Pipeline Summary --")
    pipeline_logger.info("Mode:             orchestrator (parallel post-aggregation)")
    pipeline_logger.info(f"Run ID:           {run_id}")
    pipeline_logger.info(
        f"Training jobs:    {expected_training_jobs} ({len(models)} models x {len(split_seeds)} seeds)"
    )
    pipeline_logger.info(f"Aggregation jobs: {len(agg_keys)} (parallel per model)")
    if enable_ensemble:
        pipeline_logger.info(f"Ensemble jobs:    {len(ensemble_keys)} training + 1 aggregation")
    pipeline_logger.info(f"Orchestrator job: {orchestrator_job_display}")
    pipeline_logger.info(f"Sentinel dir:     {sentinel_dir}/")
    pipeline_logger.info(f"Scripts dir:      {scripts_dir}/")
    pipeline_logger.info(f"Orchestrator log: {orchestrator_log}")
    pipeline_logger.info(f"Manifest file:    {manifest_path}")
    pipeline_logger.info(f"Wrapper script:   {wrapper_script_path}")

    return {
        "run_id": run_id,
        "training_jobs": training_job_names,
        "aggregation_jobs": agg_job_names,
        "ensemble_jobs": ensemble_job_names,
        "ensemble_agg_job": ensemble_agg_job_name,
        "panel_optimization_jobs": panel_job_names,
        "consensus_job": f"ORCH_{consensus_job_name}" if enable_consensus else None,
        "permutation_jobs": permutation_job_names,
        "permutation_aggregation_jobs": perm_agg_names,
        "orchestrator_job": orchestrator_job_display,
        "logs_dir": run_logs_dir,
        "sentinel_dir": sentinel_dir,
        "scripts_dir": scripts_dir,
        "orchestrator_log": orchestrator_log,
        "panel_aggregation_jobs": panel_agg_names,
        "manifest_path": manifest_path,
        "wrapper_script": wrapper_script_path,
    }


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
    configs: dict[str, str | Path] | None = None,
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline using orchestrator barriers.

    Dispatches to the appropriate scheduler backend based on hpc_config.scheduler.
    """
    from ced_ml.hpc.base import get_scheduler

    scheduler = get_scheduler(hpc_config.scheduler)
    pipeline_logger.info(f"Scheduler backend: {scheduler.name}")

    return _submit_orchestrator_pipeline(
        scheduler=scheduler,
        config_file=config_file,
        infile=infile,
        split_dir=split_dir,
        outdir=outdir,
        models=models,
        split_seeds=split_seeds,
        run_id=run_id,
        enable_ensemble=enable_ensemble,
        enable_consensus=enable_consensus,
        enable_optimize_panel=enable_optimize_panel,
        enable_permutation_test=enable_permutation_test,
        permutation_n_perms=permutation_n_perms,
        permutation_n_jobs=permutation_n_jobs,
        permutation_split_seeds=permutation_split_seeds,
        hpc_config=hpc_config,
        configs=configs,
        logs_dir=logs_dir,
        dry_run=dry_run,
        pipeline_logger=pipeline_logger,
    )
