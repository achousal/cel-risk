"""LSF job submission utilities for HPC pipeline orchestration.

Provides functions to build and submit LSF (bsub) job scripts for running
the CeD-ML pipeline on HPC clusters.

Parallelization strategy:
- Training: M x S parallel jobs (one per model per split)
- Post-processing: Single aggregation job
- Panel optimization: M x S seed jobs + M aggregation jobs
- Consensus: Single job
- Permutation tests: M x S parallel jobs (one per model per seed)
- Permutation aggregation: M jobs (one per model, aggregates per-seed results)

Pipeline sequencing is coordinated by a lightweight orchestrator job that:
1. Waits on sentinel files for stage completion.
2. Polls upstream job status via bjobs/bhist for fail-fast behavior.
3. Submits downstream stage scripts only after barriers are satisfied.
"""

import base64
import json
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


def _sentinel_dir(logs_dir: Path, run_id: str) -> Path:
    """Get run-specific sentinel directory."""
    validate_identifier(run_id, "run_id")
    return logs_dir / f"run_{run_id}" / "sentinels"


def _scripts_dir(logs_dir: Path, run_id: str) -> Path:
    """Get run-specific script directory."""
    validate_identifier(run_id, "run_id")
    return logs_dir / f"run_{run_id}" / "scripts"


def _sentinel_path(sentinel_dir: Path, job_name: str) -> Path:
    """Get sentinel file path for a job name."""
    validate_identifier(job_name, "job_name")
    return sentinel_dir / f"{job_name}.done"


def _append_sentinel_touch(script: str, sentinel_path: Path) -> str:
    """Append a sentinel touch command to a job script."""
    return script.rstrip() + f'\n\ntouch "{sentinel_path}"\n'


def _write_job_script(scripts_dir: Path, job_name: str, script: str) -> Path:
    """Write script to disk and make it executable."""
    validate_identifier(job_name, "job_name")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{job_name}.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o750)
    return script_path


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
#BSUB -R "rusage[mem={mem_per_core}] span[hosts=1]"
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
    """Build a generic wrapper that decodes and executes command payloads."""
    return f"""#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

{env_activation}

if [ -z "${{CED_JOB_COMMAND_B64:-}}" ]; then
    echo "[$(date '+%F %T')] FATAL: CED_JOB_COMMAND_B64 not set"
    exit 1
fi
if [ -z "${{CED_SENTINEL_PATH:-}}" ]; then
    echo "[$(date '+%F %T')] FATAL: CED_SENTINEL_PATH not set"
    exit 1
fi

COMMAND=$(python - "$CED_JOB_COMMAND_B64" <<'PY'
import base64
import sys
print(base64.b64decode(sys.argv[1]).decode("utf-8"), end="")
PY
)

eval "$COMMAND"
touch "$CED_SENTINEL_PATH"
"""


def _build_wrapped_command(
    *,
    command_b64: str,
    sentinel_path: Path,
    wrapper_script_path: Path,
) -> str:
    """Build a per-job command payload that runs through the shared wrapper."""
    return "\n".join(
        [
            f'export CED_JOB_COMMAND_B64="{command_b64}"',
            f'export CED_SENTINEL_PATH="{sentinel_path.resolve()}"',
            f'"{wrapper_script_path.resolve()}"',
        ]
    )


def _build_orchestrator_bash_functions() -> str:
    """Bash helpers used by the orchestrator script."""
    return """manifest_job_tsv() {
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
    job["sentinel"],
]
print("\t".join(fields), end="")
PY
}

submit_and_track() {
    local job_key="$1"
    local label="$2"
    local id_file="$3"

    local job_tsv
    if ! job_tsv=$(manifest_job_tsv "$job_key"); then
        echo "[$(date '+%F %T')] FATAL: no manifest entry for $job_key"
        exit 1
    fi

    local job_name queue cores mem_per_core walltime command_b64 sentinel
    IFS=$'\t' read -r job_name queue cores mem_per_core walltime command_b64 sentinel <<< "$job_tsv"

    local job_script
    job_script=$(cat <<EOF
#!/bin/bash
#BSUB -P $PROJECT
#BSUB -q $queue
#BSUB -J $job_name
#BSUB -n $cores
#BSUB -W $walltime
#BSUB -R "rusage[mem=$mem_per_core] span[hosts=1]"
#BSUB -oo /dev/null
#BSUB -eo /dev/null

set -euo pipefail
export CED_JOB_COMMAND_B64="$command_b64"
export CED_SENTINEL_PATH="$sentinel"
"$WRAPPER_SCRIPT"
EOF
)

    local output
    output=$(echo "$job_script" | bsub 2>&1)
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] FATAL: bsub failed for $label (rc=$rc): $output"
        exit 1
    fi

    local job_id
    job_id=$(echo "$output" | sed -n 's/.*Job <\\([0-9]*\\)>.*/\\1/p')
    if [ -z "$job_id" ]; then
        echo "[$(date '+%F %T')] FATAL: cannot parse job ID for $label: $output"
        exit 1
    fi

    echo "[$(date '+%F %T')] Submitted $label: Job $job_id"
    echo "$job_id" >> "$id_file"
}

check_upstream_failures() {
    local -a job_ids=("$@")
    local jid
    for jid in "${job_ids[@]}"; do
        [ -z "$jid" ] && continue

        local raw
        raw=$(bjobs -noheader -o "stat" "$jid" 2>&1 || true)
        local stat
        stat=$(echo "$raw" | awk 'NF {print $1; exit}')

        if [ "$stat" = "EXIT" ] || [ "$stat" = "TERM" ]; then
            local jname
            jname=$(bjobs -noheader -o "job_name" "$jid" 2>/dev/null | awk 'NF {print $1; exit}')
            echo "[$(date '+%F %T')] FATAL: upstream job $jid ($jname) $stat (bjobs)"
            exit 1
        fi

        if [ -z "$stat" ] || echo "$raw" | grep -qi "not found"; then
            local hist_exit
            hist_exit=$(bhist -l "$jid" 2>/dev/null | awk '/Completed <exit>|Exited with exit code/ {print; exit}')
            if [ -n "$hist_exit" ]; then
                echo "[$(date '+%F %T')] FATAL: upstream job $jid EXIT (bhist): $hist_exit"
                exit 1
            fi
            local hist_term
            hist_term=$(bhist -l "$jid" 2>/dev/null | awk '/TERM/ {print; exit}')
            if [ -n "$hist_term" ]; then
                echo "[$(date '+%F %T')] FATAL: upstream job $jid TERM (bhist): $hist_term"
                exit 1
            fi
        fi
    done
}

barrier_wait() {
    local label="$1"; shift
    local timeout="$1"; shift
    local poll="$1"; shift
    local -a sentinels=("$@")
    local total=${#sentinels[@]}
    local elapsed=0

    echo "[$(date '+%F %T')] Waiting for $label ($total sentinels, timeout=${timeout}s)..."

    if [ "$total" -eq 0 ]; then
        printf '{"stage":"%s","status":"done","ts":"%s"}\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
        return 0
    fi

    while true; do
        if [ ${#UPSTREAM_IDS[@]} -gt 0 ]; then
            check_upstream_failures "${UPSTREAM_IDS[@]}"
        fi

        local missing=0
        local sentinel
        for sentinel in "${sentinels[@]}"; do
            [ ! -f "$sentinel" ] && missing=$((missing + 1))
        done

        if [ "$missing" -eq 0 ]; then
            echo "[$(date '+%F %T')] $label complete ($total/$total)."
            printf '{"stage":"%s","status":"done","ts":"%s"}\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
            return 0
        fi

        if [ "$elapsed" -ge "$timeout" ]; then
            echo "[$(date '+%F %T')] TIMEOUT: $label after ${timeout}s ($((total - missing))/$total done)"
            echo "[$(date '+%F %T')] Missing sentinels:"
            for sentinel in "${sentinels[@]}"; do
                [ ! -f "$sentinel" ] && echo "  $sentinel"
            done
            printf '{"stage":"%s","status":"timeout","ts":"%s"}\n' "$label" "$(date -u '+%FT%TZ')" >> "$STATE_FILE"
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


def _build_orchestrator_script(
    *,
    run_id: str,
    hpc_config: HPCConfig,
    sentinel_dir: Path,
    scripts_dir: Path,
    orchestrator_log: Path,
    orchestrator_sentinel: Path,
    manifest_path: Path,
    wrapper_script_path: Path,
    training_job_ids: list[str],
    training_sentinels: list[Path],
    post_key: str,
    post_sentinel: Path,
    perm_keys: list[str],
    perm_sentinels: list[Path],
    perm_agg_keys: list[str],
    perm_agg_sentinels: list[Path],
    panel_seed_keys: list[str],
    panel_seed_sentinels: list[Path],
    panel_agg_keys: list[str],
    panel_agg_sentinels: list[Path],
    consensus_key: str | None,
    consensus_sentinel: Path | None,
    expected_training_jobs: int,
) -> str:
    """Build the barrier-orchestrator bash script."""
    validate_identifier(run_id, "run_id")
    orchestrator_cfg = hpc_config.orchestrator

    training_sentinel_values = [str(p.resolve()) for p in training_sentinels]
    perm_sentinel_values = [str(p.resolve()) for p in perm_sentinels]
    perm_agg_sentinel_values = [str(p.resolve()) for p in perm_agg_sentinels]
    panel_seed_sentinel_values = [str(p.resolve()) for p in panel_seed_sentinels]
    panel_agg_sentinel_values = [str(p.resolve()) for p in panel_agg_sentinels]

    lines: list[str] = [
        "#!/bin/bash",
        f"#BSUB -P {hpc_config.project}",
        f"#BSUB -q {hpc_config.queue}",
        f"#BSUB -J CeD_{run_id}_orchestrator",
        f"#BSUB -n {orchestrator_cfg.cores}",
        f"#BSUB -W {orchestrator_cfg.walltime}",
        f'#BSUB -R "rusage[mem={orchestrator_cfg.mem_per_core}] span[hosts=1]"',
        f"#BSUB -oo {orchestrator_log.resolve()}",
        f"#BSUB -eo {orchestrator_log.resolve()}",
        "",
        "set -euo pipefail",
        "",
        "export PYTHONUNBUFFERED=1",
        "export FORCE_COLOR=1",
        "",
        _build_orchestrator_bash_functions(),
        "",
        f'PROJECT="{hpc_config.project}"',
        f'MANIFEST_PATH="{manifest_path.resolve()}"',
        f'WRAPPER_SCRIPT="{wrapper_script_path.resolve()}"',
        f'SENTINEL_DIR="{sentinel_dir.resolve()}"',
        f'SCRIPTS_DIR="{scripts_dir.resolve()}"',
        'STATE_FILE="$SENTINEL_DIR/orchestrator_state.jsonl"',
        f"POLL_INTERVAL={orchestrator_cfg.poll_interval}",
        f"MAX_CHUNK={orchestrator_cfg.max_concurrent_submissions}",
        f"EXPECTED_TRAINING={expected_training_jobs}",
        "",
        _bash_array_literal("TRAINING_IDS", training_job_ids),
        _bash_array_literal("TRAINING_SENTINELS", training_sentinel_values),
        "",
        "UPSTREAM_IDS=()",
        'mkdir -p "$SENTINEL_DIR" "$SCRIPTS_DIR"',
        'touch "$STATE_FILE"',
        f"echo \"[$(date '+%F %T')] Orchestrator started for run {run_id}\"",
        "echo \"[$(date '+%F %T')] Training IDs submitted: ${#TRAINING_IDS[@]} (expected: $EXPECTED_TRAINING)\"",
        "",
        'UPSTREAM_IDS=("${TRAINING_IDS[@]}")',
        f'barrier_wait "training" {orchestrator_cfg.training_timeout} "$POLL_INTERVAL" "${{TRAINING_SENTINELS[@]}}"',
        "",
        'POST_IDS_FILE=$(mktemp "$SENTINEL_DIR/post_ids.XXXXXX")',
        f'submit_and_track "{post_key}" "post-processing" "$POST_IDS_FILE"',
        'mapfile -t UPSTREAM_IDS < "$POST_IDS_FILE"',
        _bash_array_literal("POST_SENTINELS", [str(post_sentinel.resolve())]),
        f'barrier_wait "post-processing" {orchestrator_cfg.post_timeout} "$POLL_INTERVAL" "${{POST_SENTINELS[@]}}"',
        "",
    ]

    if perm_keys:
        lines.extend(
            [
                _bash_array_literal("PERM_KEYS", perm_keys),
                _bash_array_literal("PERM_SENTINELS", perm_sentinel_values),
                'PERM_IDS_FILE=$(mktemp "$SENTINEL_DIR/perm_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PERM_IDS_FILE" "${PERM_KEYS[@]}"',
                'mapfile -t UPSTREAM_IDS < "$PERM_IDS_FILE"',
                f'barrier_wait "permutation-tests" {orchestrator_cfg.perm_timeout} "$POLL_INTERVAL" "${{PERM_SENTINELS[@]}}"',
                "",
                _bash_array_literal("PERM_AGG_KEYS", perm_agg_keys),
                _bash_array_literal("PERM_AGG_SENTINELS", perm_agg_sentinel_values),
                'PERM_AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/perm_agg_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PERM_AGG_IDS_FILE" "${PERM_AGG_KEYS[@]}"',
                'mapfile -t UPSTREAM_IDS < "$PERM_AGG_IDS_FILE"',
                f'barrier_wait "permutation-aggregation" {orchestrator_cfg.perm_timeout} "$POLL_INTERVAL" "${{PERM_AGG_SENTINELS[@]}}"',
                "",
            ]
        )

    if panel_seed_keys:
        lines.extend(
            [
                _bash_array_literal("PANEL_SEED_KEYS", panel_seed_keys),
                _bash_array_literal("PANEL_SEED_SENTINELS", panel_seed_sentinel_values),
                'PANEL_SEED_IDS_FILE=$(mktemp "$SENTINEL_DIR/panel_seed_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PANEL_SEED_IDS_FILE" "${PANEL_SEED_KEYS[@]}"',
                'mapfile -t UPSTREAM_IDS < "$PANEL_SEED_IDS_FILE"',
                f'barrier_wait "panel-seed" {orchestrator_cfg.panel_timeout} "$POLL_INTERVAL" "${{PANEL_SEED_SENTINELS[@]}}"',
                "",
                _bash_array_literal("PANEL_AGG_KEYS", panel_agg_keys),
                _bash_array_literal("PANEL_AGG_SENTINELS", panel_agg_sentinel_values),
                'PANEL_AGG_IDS_FILE=$(mktemp "$SENTINEL_DIR/panel_agg_ids.XXXXXX")',
                'submit_batch "$MAX_CHUNK" "$PANEL_AGG_IDS_FILE" "${PANEL_AGG_KEYS[@]}"',
                'mapfile -t UPSTREAM_IDS < "$PANEL_AGG_IDS_FILE"',
                f'barrier_wait "panel-aggregation" {orchestrator_cfg.panel_timeout} "$POLL_INTERVAL" "${{PANEL_AGG_SENTINELS[@]}}"',
                "",
            ]
        )

    if consensus_key and consensus_sentinel:
        lines.extend(
            [
                'CONSENSUS_IDS_FILE=$(mktemp "$SENTINEL_DIR/consensus_ids.XXXXXX")',
                f'submit_and_track "{consensus_key}" "consensus" "$CONSENSUS_IDS_FILE"',
                'mapfile -t UPSTREAM_IDS < "$CONSENSUS_IDS_FILE"',
                _bash_array_literal("CONSENSUS_SENTINELS", [str(consensus_sentinel.resolve())]),
                f'barrier_wait "consensus" {orchestrator_cfg.consensus_timeout} "$POLL_INTERVAL" "${{CONSENSUS_SENTINELS[@]}}"',
                "",
            ]
        )

    lines.extend(
        [
            f'touch "{orchestrator_sentinel.resolve()}"',
            'printf \'{"stage":"orchestrator","status":"done","ts":"%s"}\\n\' "$(date -u \'+%FT%TZ\')" >> "$STATE_FILE"',
            f"echo \"[$(date '+%F %T')] Orchestrator complete for run {run_id}\"",
            "",
        ]
    )

    return "\n".join(lines)


def _submit_orchestrator_pipeline(
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
    """Submit complete HPC pipeline using a barrier-orchestrator job."""
    base_dir = Path.cwd()
    env_info = detect_environment(base_dir)
    pipeline_logger.info(f"Python environment: {env_info.env_type}")

    run_root = logs_dir / f"run_{run_id}"
    run_logs_dir = run_root / "training"
    sentinel_dir = _sentinel_dir(logs_dir, run_id)
    scripts_dir = _scripts_dir(logs_dir, run_id)
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    default_resources = hpc_config.get_resources("default")
    bsub_params = {
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
        sentinel: Path,
    ) -> dict[str, str | int]:
        return {
            "job_name": job_name,
            "queue": str(default_resources["queue"]),
            "cores": int(default_resources["cores"]),
            "mem_per_core": int(default_resources["mem_per_core"]),
            "walltime": str(default_resources["walltime"]),
            "command_b64": _encode_command_b64(command),
            "sentinel": str(sentinel.resolve()),
        }

    # Stage 1: training (submitted immediately)
    training_job_names: list[str] = []
    training_job_ids: list[str] = []
    training_sentinels: list[Path] = []
    expected_training_jobs = len(models) * len(split_seeds)
    pipeline_logger.info(
        f"Submitting {expected_training_jobs} training jobs "
        f"({len(models)} models x {len(split_seeds)} seeds)..."
    )

    for model in models:
        for seed in split_seeds:
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
            )
            sentinel = _sentinel_path(sentinel_dir, job_name)
            training_sentinels.append(sentinel)

            wrapped_command = _build_wrapped_command(
                command_b64=_encode_command_b64(training_command),
                sentinel_path=sentinel,
                wrapper_script_path=wrapper_script_path,
            )
            submission_script = build_job_script(
                job_name=job_name,
                command=wrapped_command,
                **bsub_params,
            )

            if dry_run:
                pipeline_logger.info(f"  [DRY RUN] Training ({model} s{seed}): {job_name}")
                continue

            job_id = submit_job(submission_script, dry_run=False)
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

    post_job_name = f"CeD_{run_id}_post"
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
    post_sentinel = _sentinel_path(sentinel_dir, post_job_name)
    post_key = "post"
    manifest_jobs[post_key] = _manifest_entry(
        job_name=post_job_name,
        command=post_command,
        sentinel=post_sentinel,
    )

    permutation_job_names: list[str] = []
    permutation_agg_names: list[str] = []
    perm_keys: list[str] = []
    perm_sentinels: list[Path] = []
    perm_agg_keys: list[str] = []
    perm_agg_sentinels: list[Path] = []
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
                perm_sentinel = _sentinel_path(sentinel_dir, perm_job_name)
                manifest_jobs[perm_key] = _manifest_entry(
                    job_name=perm_job_name,
                    command=perm_command,
                    sentinel=perm_sentinel,
                )
                permutation_job_names.append(perm_job_name)
                perm_keys.append(perm_key)
                perm_sentinels.append(perm_sentinel)

            perm_agg_job_name = f"CeD_{run_id}_perm_{model}_agg"
            perm_agg_command = _build_permutation_aggregation_command(run_id=run_id, model=model)
            perm_agg_key = f"perm_{model}_agg"
            perm_agg_sentinel = _sentinel_path(sentinel_dir, perm_agg_job_name)
            manifest_jobs[perm_agg_key] = _manifest_entry(
                job_name=perm_agg_job_name,
                command=perm_agg_command,
                sentinel=perm_agg_sentinel,
            )
            permutation_agg_names.append(perm_agg_job_name)
            perm_agg_keys.append(perm_agg_key)
            perm_agg_sentinels.append(perm_agg_sentinel)

    panel_job_names: list[str] = []
    panel_agg_names: list[str] = []
    panel_seed_keys: list[str] = []
    panel_seed_sentinels: list[Path] = []
    panel_agg_keys: list[str] = []
    panel_agg_sentinels: list[Path] = []
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
                panel_seed_sentinel = _sentinel_path(sentinel_dir, panel_seed_job)
                manifest_jobs[panel_seed_key] = _manifest_entry(
                    job_name=panel_seed_job,
                    command=panel_seed_cmd,
                    sentinel=panel_seed_sentinel,
                )
                panel_job_names.append(panel_seed_job)
                panel_seed_keys.append(panel_seed_key)
                panel_seed_sentinels.append(panel_seed_sentinel)

            panel_agg_job = f"CeD_{run_id}_panel_{model}_agg"
            panel_agg_cmd = _build_panel_optimization_command(run_id=run_id, model=model)
            panel_agg_key = f"panel_{model}_agg"
            panel_agg_sentinel = _sentinel_path(sentinel_dir, panel_agg_job)
            manifest_jobs[panel_agg_key] = _manifest_entry(
                job_name=panel_agg_job,
                command=panel_agg_cmd,
                sentinel=panel_agg_sentinel,
            )
            panel_job_names.append(panel_agg_job)
            panel_agg_names.append(panel_agg_job)
            panel_agg_keys.append(panel_agg_key)
            panel_agg_sentinels.append(panel_agg_sentinel)

    consensus_job_name = f"CeD_{run_id}_consensus"
    consensus_key: str | None = None
    consensus_sentinel: Path | None = None
    if enable_consensus:
        consensus_cmd = _build_consensus_panel_command(run_id=run_id)
        consensus_key = "consensus"
        consensus_sentinel = _sentinel_path(sentinel_dir, consensus_job_name)
        manifest_jobs[consensus_key] = _manifest_entry(
            job_name=consensus_job_name,
            command=consensus_cmd,
            sentinel=consensus_sentinel,
        )

    manifest_path = scripts_dir / "jobs_manifest.json"
    manifest_path.write_text(json.dumps(manifest_jobs, indent=2, sort_keys=True), encoding="utf-8")

    orchestrator_job_name = f"CeD_{run_id}_orchestrator"
    orchestrator_log = run_root / "orchestrator.log"
    orchestrator_sentinel = _sentinel_path(sentinel_dir, orchestrator_job_name)
    orchestrator_script = _build_orchestrator_script(
        run_id=run_id,
        hpc_config=hpc_config,
        sentinel_dir=sentinel_dir,
        scripts_dir=scripts_dir,
        orchestrator_log=orchestrator_log,
        orchestrator_sentinel=orchestrator_sentinel,
        manifest_path=manifest_path,
        wrapper_script_path=wrapper_script_path,
        training_job_ids=training_job_ids,
        training_sentinels=training_sentinels,
        post_key=post_key,
        post_sentinel=post_sentinel,
        perm_keys=perm_keys,
        perm_sentinels=perm_sentinels,
        perm_agg_keys=perm_agg_keys,
        perm_agg_sentinels=perm_agg_sentinels,
        panel_seed_keys=panel_seed_keys,
        panel_seed_sentinels=panel_seed_sentinels,
        panel_agg_keys=panel_agg_keys,
        panel_agg_sentinels=panel_agg_sentinels,
        consensus_key=consensus_key,
        consensus_sentinel=consensus_sentinel,
        expected_training_jobs=expected_training_jobs,
    )
    _write_job_script(scripts_dir, orchestrator_job_name, orchestrator_script)

    orchestrator_job_id = submit_job(orchestrator_script, dry_run=dry_run)
    if not dry_run and not orchestrator_job_id:
        raise RuntimeError("Orchestrator submission failed")

    orchestrator_job_display = orchestrator_job_id or f"DRYRUN_{orchestrator_job_name}"
    pipeline_logger.info("-- HPC Pipeline Summary --")
    pipeline_logger.info("Mode:             orchestrator")
    pipeline_logger.info(f"Run ID:           {run_id}")
    pipeline_logger.info(
        f"Training jobs:    {expected_training_jobs} ({len(models)} models x {len(split_seeds)} seeds)"
    )
    pipeline_logger.info(f"Orchestrator job: {orchestrator_job_display}")
    pipeline_logger.info(f"Sentinel dir:     {sentinel_dir}/")
    pipeline_logger.info(f"Scripts dir:      {scripts_dir}/")
    pipeline_logger.info(f"Orchestrator log: {orchestrator_log}")
    pipeline_logger.info(f"Manifest file:    {manifest_path}")
    pipeline_logger.info(f"Wrapper script:   {wrapper_script_path}")

    return {
        "run_id": run_id,
        "training_jobs": training_job_names,
        "postprocessing_job": f"ORCH_{post_job_name}",
        "panel_optimization_jobs": panel_job_names,
        "consensus_job": f"ORCH_{consensus_job_name}" if enable_consensus else None,
        "permutation_jobs": permutation_job_names,
        "permutation_aggregation_jobs": permutation_agg_names,
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
    logs_dir: Path,
    dry_run: bool,
    pipeline_logger: logging.Logger,
) -> dict:
    """Submit complete HPC pipeline using orchestrator barriers."""
    return _submit_orchestrator_pipeline(
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
        logs_dir=logs_dir,
        dry_run=dry_run,
        pipeline_logger=pipeline_logger,
    )
