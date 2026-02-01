#!/bin/bash
# -----------------------------------------------------------------------
# HPC runner for 2x2x2 factorial experiment (LSF / bsub)
#
# Two-step workflow:
#   1. Tune baseline hyperparams (single short job)
#   2. Run full factorial experiment (single job, ~160 runs)
#
# Usage:
#   # Step 1 -- tune baseline
#   bash run_factorial_hpc.sh tune
#
#   # Step 2 -- run experiment (after step 1 finishes)
#   bash run_factorial_hpc.sh run
#
#   # Both steps chained (step 2 waits for step 1)
#   bash run_factorial_hpc.sh all
#
#   # Override resources
#   N_SEEDS=20 QUEUE=long WALLTIME=48:00 bash run_factorial_hpc.sh run
# -----------------------------------------------------------------------
set -euo pipefail

# -- Configuration (override via environment) --
PROJ_ROOT="${PROJ_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DATA_PATH="${DATA_PATH:-${PROJ_ROOT}/data/Celiac_dataset_proteomics_w_demo.parquet}"
PANEL_PATH="${PANEL_PATH:-${PROJ_ROOT}/data/fixed_panel.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/results/factorial_2x2x2}"
N_SEEDS="${N_SEEDS:-10}"
PROJECT_ACCT="${PROJECT_ACCT:-acc_Chipuk_Laboratory}"
QUEUE="${QUEUE:-premium}"
WALLTIME="${WALLTIME:-24:00}"
MEM_MB="${MEM_MB:-8000}"
N_CPUS="${N_CPUS:-8}"
CONDA_ENV="${CONDA_ENV:-}"
LOG_DIR="${OUTPUT_DIR}/logs"

SCRIPT="${PROJ_ROOT}/analysis/docs/investigations/run_factorial_2x2x2.py"

# -- Helpers --
die() { echo "ERROR: $*" >&2; exit 1; }

check_prereqs() {
    command -v bsub >/dev/null 2>&1 || die "bsub not found -- are you on an LSF cluster?"
    [ -f "$DATA_PATH" ] || die "Data file not found: $DATA_PATH"
    [ -f "$PANEL_PATH" ] || die "Panel file not found: $PANEL_PATH"
    [ -f "$SCRIPT" ] || die "Script not found: $SCRIPT"
}

activate_cmd() {
    if [ -n "$CONDA_ENV" ]; then
        echo "conda activate $CONDA_ENV"
    elif [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "source ${VIRTUAL_ENV}/bin/activate"
    else
        echo "# no env activation configured"
    fi
}

mkdir_logs() {
    mkdir -p "$LOG_DIR"
}

# -- Job submission --
submit_tune() {
    mkdir_logs
    local job_name="fac_tune"
    local log_file="${LOG_DIR}/${job_name}_%J.log"

    bsub \
        -J "$job_name" \
        -P "$PROJECT_ACCT" \
        -q "$QUEUE" \
        -W "$WALLTIME" \
        -n "$N_CPUS" \
        -M "$MEM_MB" \
        -o "$log_file" \
        -e "$log_file" \
        <<EOF
#!/bin/bash
set -euo pipefail

# HPC metadata for reproducibility
echo "Job ID: \${LSB_JOBID}"
echo "Host: \$(hostname)"
echo "Start: \$(date)"
echo "Working directory: \$(pwd)"

$(activate_cmd)

python "$SCRIPT" \\
    --data-path "$DATA_PATH" \\
    --panel-path "$PANEL_PATH" \\
    --output-dir "$OUTPUT_DIR" \\
    --tune-baseline

echo "Baseline tuning complete."
echo "End: \$(date)"
EOF

    echo "Submitted tuning job: $job_name"
    echo "Log: $log_file"
}

submit_run() {
    local depend_on="${1:-}"
    mkdir_logs
    local job_name="fac_run"
    local log_file="${LOG_DIR}/${job_name}_%J.log"
    local hp_path="${OUTPUT_DIR}/frozen_hyperparams.yaml"

    local depend_flag=""
    if [ -n "$depend_on" ]; then
        depend_flag="-w done($depend_on)"
    fi

    # shellcheck disable=SC2086
    bsub \
        -J "$job_name" \
        -P "$PROJECT_ACCT" \
        -q "$QUEUE" \
        -W "$WALLTIME" \
        -n "$N_CPUS" \
        -M "$MEM_MB" \
        -o "$log_file" \
        -e "$log_file" \
        $depend_flag \
        <<EOF
#!/bin/bash
set -euo pipefail

# HPC metadata for reproducibility
echo "Job ID: \${LSB_JOBID}"
echo "Host: \$(hostname)"
echo "Start: \$(date)"
echo "Working directory: \$(pwd)"

$(activate_cmd)

python "$SCRIPT" \\
    --data-path "$DATA_PATH" \\
    --panel-path "$PANEL_PATH" \\
    --output-dir "$OUTPUT_DIR" \\
    --hyperparams-path "$hp_path" \\
    --n-seeds "$N_SEEDS"

echo "Factorial experiment complete. Results: ${OUTPUT_DIR}/factorial_results.csv"
echo "End: \$(date)"
EOF

    echo "Submitted experiment job: $job_name${depend_on:+ (depends on $depend_on)}"
    echo "Log: $log_file"
}

# -- Main --
main() {
    local mode="${1:-}"

    case "$mode" in
        tune)
            check_prereqs
            submit_tune
            ;;
        run)
            check_prereqs
            submit_run
            ;;
        all)
            check_prereqs
            submit_tune
            submit_run "fac_tune"
            echo "Both jobs submitted. Step 2 will wait for step 1."
            ;;
        *)
            echo "Usage: bash run_factorial_hpc.sh {tune|run|all}"
            echo ""
            echo "  tune  -- Submit baseline hyperparameter tuning job"
            echo "  run   -- Submit factorial experiment job"
            echo "  all   -- Submit both (chained: run waits for tune)"
            echo ""
            echo "Environment overrides:"
            echo "  N_SEEDS=20 QUEUE=long WALLTIME=48:00 CONDA_ENV=ced-ml bash run_factorial_hpc.sh all"
            echo ""
            echo "Current defaults (from pipeline_hpc.yaml):"
            echo "  PROJECT_ACCT=$PROJECT_ACCT"
            echo "  QUEUE=$QUEUE"
            echo "  WALLTIME=$WALLTIME"
            echo "  N_CPUS=$N_CPUS"
            echo "  MEM_MB=$MEM_MB"
            exit 1
            ;;
    esac
}

main "$@"
