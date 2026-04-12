#!/bin/bash
# Submit incident validation (LinSVM_cal) pipeline to Minerva HPC — sequential mode
# Usage: bash scripts/submit_incident_validation_svm.sh [--penalty l1|l2] [--smoke]
#
# Runs both penalties by default. Pass --penalty l1 or --penalty l2 for a single variant.

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/scripts/run_svm.py"
DATAFILE="${BASEDIR}/data/Celiac_dataset_proteomics_w_demo.parquet"
LOGDIR="${BASEDIR}/logs/incident-validation"
RESULTS_ROOT="${BASEDIR}/results/incident-validation/linsvm_cal"

# Pre-create log + per-penalty output dirs (Python also does this; fail early at submit time)
mkdir -p "${LOGDIR}" "${RESULTS_ROOT}/l1" "${RESULTS_ROOT}/l2"

SMOKE_FLAG=""
WALLTIME="48:00"
MEM="16000"
CORES=8

# Parse args
PENALTIES=("l1" "l2")
for arg in "$@"; do
    case "${arg}" in
        --smoke) SMOKE_FLAG="--smoke"; WALLTIME="02:00"; MEM="8000"; CORES=4 ;;
        --penalty) shift; PENALTIES=("$1") ;;
        l1|l2) PENALTIES=("${arg}") ;;
    esac
done

for PENALTY in "${PENALTIES[@]}"; do
    OUTDIR="${BASEDIR}/results/incident-validation/linsvm_cal/${PENALTY}"
    JOB_NAME="CeD_iv_svm_${PENALTY}"
    [[ -n "${SMOKE_FLAG}" ]] && OUTDIR="${OUTDIR}_smoke" && JOB_NAME="${JOB_NAME}_smoke"

    echo "=== Submitting LinSVM_cal penalty=${PENALTY} ==="
    echo "  Output: ${OUTDIR}"

    CMD="python ${SCRIPT} \
        --penalty ${PENALTY} \
        --data-path ${DATAFILE} \
        --output-dir ${OUTDIR} \
        ${SMOKE_FLAG}"

    bsub -P ${PROJECT} -q ${QUEUE} -W "${WALLTIME}" \
        -n ${CORES} -R "rusage[mem=${MEM}] span[hosts=1]" \
        -J "${JOB_NAME}" \
        -oo "${LOGDIR}/${JOB_NAME}_%J.stdout" \
        -eo "${LOGDIR}/${JOB_NAME}_%J.stderr" \
        bash -c "cd ${BASEDIR} && module load anaconda3/2024.06 && source analysis/venv/bin/activate && ${CMD}"

    echo "  Submitted: ${JOB_NAME}. Monitor: bjobs -w | grep ${JOB_NAME}"
done
