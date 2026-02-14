"""LSF scheduler backend implementation.

Implements SchedulerBackend for IBM Spectrum LSF (bsub/bjobs/bhist).
Also re-exports common symbols for backward compatibility so that
existing ``from ced_ml.hpc.lsf import X`` imports continue to work.
"""

import re
from pathlib import Path

from ced_ml.hpc.base import SchedulerBackend, register_scheduler

# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# ---------------------------------------------------------------------------
# All shared code now lives in common.py.  Re-export here so callers that
# still do ``from ced_ml.hpc.lsf import build_job_script`` keep working.
from ced_ml.hpc.common import (  # noqa: F401 -- re-exports
    EnvironmentInfo,
    _bash_array_literal,
    _build_consensus_panel_command,
    _build_orchestrator_bash_functions,
    _build_orchestrator_script,
    _build_panel_optimization_command,
    _build_permutation_aggregation_command,
    _build_permutation_test_full_command,
    _build_postprocessing_command,
    _build_training_command,
    _build_wrapped_command,
    _build_wrapper_script,
    _encode_command_b64,
    _load_training_scenario,
    _scripts_dir,
    _sentinel_dir,
    _sentinel_log_path,
    _write_job_script,
    build_job_script,
    detect_environment,
    load_hpc_config,
    submit_hpc_pipeline,
    submit_job,
    validate_identifier,
)

# ---------------------------------------------------------------------------
# LSF Backend
# ---------------------------------------------------------------------------


@register_scheduler("lsf")
class LSFScheduler(SchedulerBackend):
    """IBM Spectrum LSF scheduler backend."""

    @property
    def name(self) -> str:
        return "lsf"

    @property
    def submit_command(self) -> str:
        return "bsub"

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
        directives = [
            f"#BSUB -P {project}",
            f"#BSUB -q {queue}",
            f"#BSUB -J {job_name}",
            f"#BSUB -n {cores}",
            f"#BSUB -W {walltime}",
            f'#BSUB -R "rusage[mem={mem_per_core}] span[hosts=1]"',
            f"#BSUB -oo {stdout_path}",
            f"#BSUB -eo {stderr_path}",
        ]
        if dependency:
            directives.append(f'#BSUB -w "{dependency}"')
        return directives

    def parse_job_id(self, stdout: str) -> str | None:
        match = re.search(r"Job <(\d+)>", stdout)
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

    local job_script
    local bsub_directive="#BSUB"
    job_script=$(cat <<EOF
#!/bin/bash
${bsub_directive} -P $PROJECT
${bsub_directive} -q $queue
${bsub_directive} -J $job_name
${bsub_directive} -n $cores
${bsub_directive} -W $walltime
${bsub_directive} -R "rusage[mem=$mem_per_core] span[hosts=1]"
${bsub_directive} -oo /dev/null
${bsub_directive} -eo /dev/null

set -euo pipefail
export CED_JOB_COMMAND_B64="$command_b64"
export CED_JOB_NAME="$job_name"
export CED_SENTINEL_DIR="$SENTINEL_DIR"
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
}"""

    def build_orchestrator_status_func(self) -> str:
        return """check_upstream_failures() {
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
            hist_exit=$(bhist -l "$jid" 2>/dev/null | awk '/Completed <exit>|Exited with exit code/ {print; exit}' || true)
            if [ -n "$hist_exit" ]; then
                echo "[$(date '+%F %T')] FATAL: upstream job $jid EXIT (bhist): $hist_exit"
                exit 1
            fi
            local hist_term
            hist_term=$(bhist -l "$jid" 2>/dev/null | awk '/TERM/ {print; exit}' || true)
            if [ -n "$hist_term" ]; then
                echo "[$(date '+%F %T')] FATAL: upstream job $jid TERM (bhist): $hist_term"
                exit 1
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
        return [
            "#!/bin/bash",
            f"#BSUB -P {project}",
            f"#BSUB -q {queue}",
            f"#BSUB -J {job_name}",
            f"#BSUB -n {cores}",
            f"#BSUB -W {walltime}",
            f'#BSUB -R "rusage[mem={mem_per_core}] span[hosts=1]"',
            f"#BSUB -oo {log_path.resolve()}",
            f"#BSUB -eo {log_path.resolve()}",
        ]

    def monitor_hint(self, job_name_pattern: str) -> str:
        return f"bjobs -w | grep '{job_name_pattern}'"
