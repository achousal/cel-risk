#!/usr/bin/env bash
#
# Replay a failed training job with logs enabled (not /dev/null).
#
# Usage:
#   bash analysis/scripts/hpc_debug_replay_training_job.sh <RUN_ID> <JOB_NAME> [submit|local]
#
# Example:
#   bash analysis/scripts/hpc_debug_replay_training_job.sh \
#     20260212_214538 CeD_20260212_214538_LR_EN_s119 submit
#
# Optional env overrides:
#   INFILE, SPLIT_DIR, OUTDIR, CONFIG_FILE
#   PROJECT, QUEUE, CORES, MEM_PER_CORE, WALLTIME

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <RUN_ID> <JOB_NAME> [submit|local]" >&2
    exit 1
fi

RUN_ID="$1"
JOB_NAME="$2"
MODE="${3:-submit}"

if [[ "${MODE}" != "submit" && "${MODE}" != "local" ]]; then
    echo "MODE must be 'submit' or 'local' (got: ${MODE})" >&2
    exit 1
fi

if [[ ! "${JOB_NAME}" =~ ^CeD_[0-9]{8}_[0-9]{6}_(.+)_s([0-9]+)$ ]]; then
    echo "JOB_NAME format not recognized: ${JOB_NAME}" >&2
    echo "Expected pattern: CeD_<run_id>_<MODEL>_s<SEED>" >&2
    exit 1
fi

MODEL="${BASH_REMATCH[1]}"
SPLIT_SEED="${BASH_REMATCH[2]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIPELINE_HPC_CONFIG_DEFAULT="${PROJECT_ROOT}/analysis/configs/pipeline_hpc.yaml"
CONFIG_FILE_DEFAULT="${PROJECT_ROOT}/analysis/configs/training_config.yaml"

PIPELINE_HPC_CONFIG="${PIPELINE_HPC_CONFIG:-${PIPELINE_HPC_CONFIG_DEFAULT}}"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_FILE_DEFAULT}}"

INFILE_DEFAULT=""
SPLIT_DIR_DEFAULT=""
OUTDIR_DEFAULT=""

RUN_METADATA_PATH="$(find "${PROJECT_ROOT}" -maxdepth 8 -type f -path "*/run_${RUN_ID}/run_metadata.json" | head -n 1 || true)"
if [[ -n "${RUN_METADATA_PATH}" ]]; then
    eval "$(
        python - "${RUN_METADATA_PATH}" <<'PY'
import json
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
with open(path, encoding="utf-8") as f:
    payload = json.load(f)

infile = payload.get("infile", "")
split_dir = payload.get("split_dir", "")
outdir = str(path.parent.parent)

print(f'INFILE_DEFAULT={shlex.quote(infile)}')
print(f'SPLIT_DIR_DEFAULT={shlex.quote(split_dir)}')
print(f'OUTDIR_DEFAULT={shlex.quote(outdir)}')
PY
    )"
fi

INFILE="${INFILE:-${INFILE_DEFAULT}}"
SPLIT_DIR="${SPLIT_DIR:-${SPLIT_DIR_DEFAULT}}"
OUTDIR="${OUTDIR:-${OUTDIR_DEFAULT}}"

if [[ -z "${INFILE}" || -z "${SPLIT_DIR}" || -z "${OUTDIR}" ]]; then
    echo "Could not infer INFILE/SPLIT_DIR/OUTDIR automatically." >&2
    echo "Set env vars and rerun, for example:" >&2
    echo "  INFILE=/path/to/data.parquet SPLIT_DIR=/path/to/splits OUTDIR=/path/to/results \\" >&2
    echo "  $0 ${RUN_ID} ${JOB_NAME} ${MODE}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Training config not found: ${CONFIG_FILE}" >&2
    exit 1
fi

if [[ -f "${PIPELINE_HPC_CONFIG}" ]]; then
    eval "$(
        awk '
            $1 == "project:" { print "PROJECT_DEFAULT=" $2 }
            $1 == "queue:"   { print "QUEUE_DEFAULT=" $2 }
            $1 == "walltime:" { gsub(/"/, "", $2); print "WALLTIME_DEFAULT=" $2 }
            $1 == "cores:" && !seen_cores { print "CORES_DEFAULT=" $2; seen_cores=1 }
            $1 == "mem_per_core:" && !seen_mem { print "MEM_DEFAULT=" $2; seen_mem=1 }
        ' "${PIPELINE_HPC_CONFIG}"
    )"
fi

PROJECT="${PROJECT:-${PROJECT_DEFAULT:-acc_Chipuk_Laboratory}}"
QUEUE="${QUEUE:-${QUEUE_DEFAULT:-premium}}"
WALLTIME="${WALLTIME:-${WALLTIME_DEFAULT:-48:00}}"
CORES="${CORES:-${CORES_DEFAULT:-12}}"
MEM_PER_CORE="${MEM_PER_CORE:-${MEM_DEFAULT:-8000}}"

RUN_LOG_DIR="${PROJECT_ROOT}/logs/run_${RUN_ID}"
DEBUG_DIR="${RUN_LOG_DIR}/debug"
mkdir -p "${DEBUG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
DEBUG_JOB_NAME="${JOB_NAME}_debug"
DEBUG_LOG="${DEBUG_DIR}/${DEBUG_JOB_NAME}_${TS}.log"

TRAIN_CMD=(
    ced
    --log-level
    debug
    train
    --config "${CONFIG_FILE}"
    --infile "${INFILE}"
    --split-dir "${SPLIT_DIR}"
    --outdir "${OUTDIR}"
    --model "${MODEL}"
    --split-seed "${SPLIT_SEED}"
    --run-id "${RUN_ID}"
)

printf -v TRAIN_CMD_STR '%q ' "${TRAIN_CMD[@]}"

echo "Replay parameters:"
echo "  RUN_ID:      ${RUN_ID}"
echo "  JOB_NAME:    ${JOB_NAME}"
echo "  MODEL:       ${MODEL}"
echo "  SPLIT_SEED:  ${SPLIT_SEED}"
echo "  INFILE:      ${INFILE}"
echo "  SPLIT_DIR:   ${SPLIT_DIR}"
echo "  OUTDIR:      ${OUTDIR}"
echo "  CONFIG_FILE: ${CONFIG_FILE}"
echo "  MODE:        ${MODE}"
echo "  DEBUG_LOG:   ${DEBUG_LOG}"
echo ""
echo "Command:"
echo "  ${TRAIN_CMD_STR}"
echo ""

if [[ "${MODE}" == "local" ]]; then
    set -o pipefail
    cd "${PROJECT_ROOT}"
    source "${PROJECT_ROOT}/analysis/venv/bin/activate"
    "${TRAIN_CMD[@]}" 2>&1 | tee "${DEBUG_LOG}"
    exit $?
fi

read -r -d '' JOB_SCRIPT <<EOF || true
#!/bin/bash
#BSUB -P ${PROJECT}
#BSUB -q ${QUEUE}
#BSUB -J ${DEBUG_JOB_NAME}
#BSUB -n ${CORES}
#BSUB -W ${WALLTIME}
#BSUB -R "rusage[mem=${MEM_PER_CORE}] span[hosts=1]"
#BSUB -oo ${DEBUG_LOG}
#BSUB -eo ${DEBUG_LOG}

set -euo pipefail
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/analysis/venv/bin/activate"
${TRAIN_CMD_STR}
EOF

SUBMIT_OUTPUT="$(echo "${JOB_SCRIPT}" | bsub 2>&1)"
echo "${SUBMIT_OUTPUT}"
JOB_ID="$(echo "${SUBMIT_OUTPUT}" | sed -n 's/.*Job <\([0-9]*\)>.*/\1/p')"
if [[ -n "${JOB_ID}" ]]; then
    echo ""
    echo "Submitted debug replay job: ${JOB_ID}"
    echo "Monitor: bjobs -l ${JOB_ID}"
    echo "Log:     tail -f ${DEBUG_LOG}"
fi
