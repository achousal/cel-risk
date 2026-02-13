#!/usr/bin/env bash
#
# Collect orchestrator and LSF context for a failed HPC run.
#
# Usage:
#   bash analysis/scripts/hpc_debug_collect_context.sh <RUN_ID> [FAILED_JOB_ID]
#
# Example:
#   bash analysis/scripts/hpc_debug_collect_context.sh 20260212_214538 229031925

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <RUN_ID> [FAILED_JOB_ID]" >&2
    exit 1
fi

RUN_ID="$1"
FAILED_JOB_ID="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/logs/run_${RUN_ID}}"
SENTINEL_DIR="${RUN_ROOT}/sentinels"
SCRIPTS_DIR="${RUN_ROOT}/scripts"

if [[ ! -d "${RUN_ROOT}" ]]; then
    echo "Run directory not found: ${RUN_ROOT}" >&2
    echo "If logs are on shared storage, set RUN_ROOT explicitly." >&2
    echo "Example: RUN_ROOT=/sc/arion/projects/Chipuk_Laboratory/chousa01/CeliacRisks/logs/run_${RUN_ID} $0 ${RUN_ID} ${FAILED_JOB_ID}" >&2
    exit 1
fi

DEBUG_DIR="${RUN_ROOT}/debug"
mkdir -p "${DEBUG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="${DEBUG_DIR}/orchestrator_context_${RUN_ID}_${TS}.log"

exec > >(tee -a "${REPORT}") 2>&1

section() {
    printf "\n===== %s =====\n" "$1"
}

extract_training_jobs() {
    local orch_script="$1"
    awk '
        /^TRAINING_JOBS=\(/ { in_block=1; next }
        in_block && /^\)/ { in_block=0; exit }
        in_block {
            gsub(/^  "/, "", $0)
            gsub(/"$/, "", $0)
            if (length($0) > 0) print $0
        }
    ' "${orch_script}"
}

section "Host Context"
date
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "pwd:  $(pwd)"

section "Run Paths"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SENTINEL_DIR=${SENTINEL_DIR}"
echo "SCRIPTS_DIR=${SCRIPTS_DIR}"
ls -lah "${RUN_ROOT}" || true
ls -lah "${SENTINEL_DIR}" || true
ls -lah "${SCRIPTS_DIR}" || true

section "Orchestrator Log (last 200 lines)"
if [[ -f "${RUN_ROOT}/orchestrator.log" ]]; then
    tail -n 200 "${RUN_ROOT}/orchestrator.log"
else
    echo "missing: ${RUN_ROOT}/orchestrator.log"
fi

section "Orchestrator State"
if [[ -f "${SENTINEL_DIR}/orchestrator_state.jsonl" ]]; then
    cat "${SENTINEL_DIR}/orchestrator_state.jsonl"
else
    echo "missing: ${SENTINEL_DIR}/orchestrator_state.jsonl"
fi

section "Completion Sentinel Summary"
if [[ -f "${SENTINEL_DIR}/completed.log" ]]; then
    total_lines="$(wc -l < "${SENTINEL_DIR}/completed.log" | tr -d ' ')"
    unique_lines="$(sort -u "${SENTINEL_DIR}/completed.log" | wc -l | tr -d ' ')"
    echo "completed.log total lines : ${total_lines}"
    echo "completed.log unique jobs : ${unique_lines}"
    echo ""
    echo "Top duplicates (if any):"
    sort "${SENTINEL_DIR}/completed.log" | uniq -c | sort -nr | head -n 20
else
    echo "missing: ${SENTINEL_DIR}/completed.log"
fi

ORCH_SCRIPT="${SCRIPTS_DIR}/CeD_${RUN_ID}_orchestrator.sh"
if [[ -f "${ORCH_SCRIPT}" && -f "${SENTINEL_DIR}/completed.log" ]]; then
    section "Training Barrier Coverage"
    mapfile -t training_jobs < <(extract_training_jobs "${ORCH_SCRIPT}")
    echo "TRAINING_JOBS entries in orchestrator: ${#training_jobs[@]}"
    missing_count=0
    for jname in "${training_jobs[@]}"; do
        if ! grep -qx "${jname}" "${SENTINEL_DIR}/completed.log"; then
            missing_count=$((missing_count + 1))
        fi
    done
    echo "TRAINING_JOBS missing from completed.log: ${missing_count}"
    if [[ "${missing_count}" -gt 0 ]]; then
        echo "Missing jobs:"
        for jname in "${training_jobs[@]}"; do
            if ! grep -qx "${jname}" "${SENTINEL_DIR}/completed.log"; then
                echo "  ${jname}"
            fi
        done
    fi
fi

if [[ -n "${FAILED_JOB_ID}" ]]; then
    section "Failed Job Diagnostics (${FAILED_JOB_ID})"
    echo "-- bjobs -l ${FAILED_JOB_ID}"
    bjobs -l "${FAILED_JOB_ID}" || true
    echo ""
    echo "-- bhist -l ${FAILED_JOB_ID}"
    bhist -l "${FAILED_JOB_ID}" || true
    echo ""
    if command -v bacct >/dev/null 2>&1; then
        echo "-- bacct -l ${FAILED_JOB_ID}"
        bacct -l "${FAILED_JOB_ID}" || true
    fi

    FAILED_JOB_NAME="$(bhist -l "${FAILED_JOB_ID}" 2>/dev/null | sed -n 's/.*Job Name <\([^>]*\)>.*/\1/p' | head -n 1 || true)"
    if [[ -z "${FAILED_JOB_NAME}" ]]; then
        FAILED_JOB_NAME="$(bjobs -noheader -o "job_name" "${FAILED_JOB_ID}" 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    fi

    if [[ -n "${FAILED_JOB_NAME}" ]]; then
        echo ""
        echo "Derived failed job name: ${FAILED_JOB_NAME}"
        if [[ -f "${SENTINEL_DIR}/completed.log" ]]; then
            if grep -qx "${FAILED_JOB_NAME}" "${SENTINEL_DIR}/completed.log"; then
                echo "completed.log contains failed job name (expected with EXIT trap)."
            else
                echo "completed.log does NOT contain failed job name."
            fi
        fi
    fi
fi

section "Saved Report"
echo "${REPORT}"
