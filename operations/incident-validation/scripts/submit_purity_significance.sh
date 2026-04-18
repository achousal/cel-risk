#!/bin/bash
# Submit purity panel significance test to Minerva HPC.
#
# Reads saturation_all_models.csv, estimates prevalence from the parquet,
# runs bootstrap z-test + BH-FDR, outputs purity_significance.csv and figure.
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/saturation_all_models.csv
#
# Usage:
#   bash operations/incident-validation/scripts/submit_purity_significance.sh

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_purity_significance.py"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

JOB_NAME="CeD_purity_sig"

JOB_ID=$(bsub \
    -P ${PROJECT} \
    -q ${QUEUE} \
    -W "01:00" \
    -n 2 \
    -R "rusage[mem=8000] span[hosts=1]" \
    -J "${JOB_NAME}" \
    -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
    -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --out ${OUTDIR} --prevalence 0.003351" \
    | grep -oP '\d+')

echo "Submitted: ${JOB_NAME} → job ${JOB_ID}"
echo "Logs:  ${LOGDIR}/${JOB_NAME}_${JOB_ID}.{stdout,stderr}"
echo "Out:   ${OUTDIR}/purity_significance.csv"
echo "       ${OUTDIR}/fig_purity_significance.{pdf,png}"
echo ""
echo "Monitor: bjobs -J ${JOB_NAME}"
echo "Kill:    bkill ${JOB_ID}"
