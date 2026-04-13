"""Slurm scheduler backend implementation.

Implements SchedulerBackend for Slurm (sbatch/sacct/squeue).
"""

import re
from pathlib import Path

from ced_ml.hpc.base import SchedulerBackend, register_scheduler


@register_scheduler("slurm")
class SlurmScheduler(SchedulerBackend):
    """Slurm workload manager scheduler backend."""

    @property
    def name(self) -> str:
        return "slurm"

    @property
    def submit_command(self) -> str:
        return "sbatch"

    def build_directives(
        self,
        *,
        job_name: str,
        project: str,
        queue: str,
        cores: int,
        mem_per_core: int,
        walltime: str,
        stdout_path: str = "/dev/null",
        stderr_path: str = "/dev/null",
        dependency: str | None = None,
    ) -> list[str]:
        walltime_slurm = _normalise_walltime(walltime)
        directives = [
            f"#SBATCH --account={project}",
            f"#SBATCH --partition={queue}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --cpus-per-task={cores}",
            f"#SBATCH --time={walltime_slurm}",
            f"#SBATCH --mem-per-cpu={mem_per_core}M",
            f"#SBATCH --output={stdout_path}",
            f"#SBATCH --error={stderr_path}",
        ]
        if dependency:
            directives.append(f"#SBATCH --dependency={dependency}")
        return directives

    def parse_job_id(self, stdout: str) -> str | None:
        match = re.search(r"Submitted batch job (\d+)", stdout)
        return match.group(1) if match else None

    def build_orchestrator_submit_func(self) -> str:
        return """submit_and_track() {
    local job_key="$1"
    local label="$2"
    local id_file="$3"

    local job_tsv
    if ! job_tsv=$(manifest_job_tsv "$job_key"); then
        echo "[$(date '+%F %T')] FATAL: no manifest entry for $job_key"
        exit 1
    fi

    local job_name queue cores mem_per_core walltime command_b64
    IFS=$'\\t' read -r job_name queue cores mem_per_core walltime command_b64 <<< "$job_tsv"

    local walltime_slurm
    walltime_slurm=$(echo "$walltime" | awk -F: '{if(NF==2) printf "%s:%s:00",$1,$2; else print}')

    local job_script
    local sbatch_directive="#SBATCH"
    job_script=$(cat <<EOF
#!/bin/bash
${sbatch_directive} --account=$PROJECT
${sbatch_directive} --partition=$queue
${sbatch_directive} --job-name=$job_name
${sbatch_directive} --cpus-per-task=$cores
${sbatch_directive} --time=$walltime_slurm
${sbatch_directive} --mem-per-cpu=${mem_per_core}M
${sbatch_directive} --output=/dev/null
${sbatch_directive} --error=/dev/null

set -euo pipefail
export CED_JOB_COMMAND_B64="$command_b64"
export CED_JOB_NAME="$job_name"
export CED_SENTINEL_DIR="$SENTINEL_DIR"
"$WRAPPER_SCRIPT"
EOF
)

    local output
    output=$(echo "$job_script" | sbatch 2>&1)
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] FATAL: sbatch failed for $label (rc=$rc): $output"
        exit 1
    fi

    local job_id
    job_id=$(echo "$output" | sed -n 's/.*Submitted batch job \\([0-9]*\\).*/\\1/p')
    if [ -z "$job_id" ]; then
        echo "[$(date '+%F %T')] FATAL: cannot parse job ID for $label: $output"
        exit 1
    fi

    echo "[$(date '+%F %T')] Submitted $label: Job $job_id"
    echo "$job_id" >> "$id_file"
}"""

    def build_orchestrator_status_func(self) -> str:
        # Scheduler status is ALWAYS authoritative when available. The
        # sentinel is only consulted as a fallback when sacct/squeue
        # can't report a state. See lsf.py:build_orchestrator_status_func
        # for the full rationale.
        return """check_upstream_failures() {
    local -a job_ids=("$@")
    local jid
    for jid in "${job_ids[@]}"; do
        [ -z "$jid" ] && continue

        local raw
        raw=$(sacct -j "$jid" --noheader --parsable2 -o State 2>/dev/null | head -1 || true)
        local stat
        stat=$(echo "$raw" | awk -F'|' '{print $1}')

        if [ "$stat" = "FAILED" ] || [ "$stat" = "CANCELLED" ] || [ "$stat" = "TIMEOUT" ] || [ "$stat" = "NODE_FAIL" ]; then
            local jname
            jname=$(sacct -j "$jid" --noheader --parsable2 -o JobName 2>/dev/null | head -1)
            echo "[$(date '+%F %T')] FATAL: upstream job $jid ($jname) $stat (sacct)"
            exit 1
        fi

        if [ -z "$stat" ]; then
            local sq_stat
            sq_stat=$(squeue -j "$jid" --noheader -o "%T" 2>/dev/null | head -1)
            if [ -z "$sq_stat" ]; then
                echo "[$(date '+%F %T')] WARNING: upstream job $jid not found in sacct or squeue"
            fi
        fi
    done
}"""

    def build_orchestrator_header(
        self,
        *,
        project: str,
        queue: str,
        job_name: str,
        cores: int,
        mem_per_core: int,
        walltime: str,
        log_path: Path,
    ) -> list[str]:
        walltime_slurm = _normalise_walltime(walltime)
        return [
            "#!/bin/bash",
            f"#SBATCH --account={project}",
            f"#SBATCH --partition={queue}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --cpus-per-task={cores}",
            f"#SBATCH --time={walltime_slurm}",
            f"#SBATCH --mem-per-cpu={mem_per_core}M",
            f"#SBATCH --output={log_path.resolve()}",
            f"#SBATCH --error={log_path.resolve()}",
        ]

    def monitor_hint(self, job_name_pattern: str) -> str:
        return f"squeue -u $USER --name='{job_name_pattern}'"


def _normalise_walltime(walltime: str) -> str:
    """Convert HH:MM to HH:MM:SS if needed (Slurm expects HH:MM:SS)."""
    parts = walltime.split(":")
    if len(parts) == 2:
        return f"{walltime}:00"
    return walltime
