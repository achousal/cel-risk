#!/bin/bash
# Submit panel size selection (z-test vs peak AUPRC) to Minerva HPC.
#
# For each (ordering, model): finds smallest N non-inferior to peak AUPRC
# using one-sided z-test with SE recovered from bootstrap 95% CI.
# BH-FDR applied across panel sizes within each (model, ordering).
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/saturation_all_models.csv
#
# Usage:
#   bash operations/incident-validation/scripts/submit_panel_size_selection.sh
#   bash operations/incident-validation/scripts/submit_panel_size_selection.sh --alpha 0.01

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_panel_size_selection.py"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

EXTRA_ARGS="$*"

JOB_NAME="CeD_panel_select"

JOB_ID=$(bsub \
    -P ${PROJECT} \
    -q ${QUEUE} \
    -W "00:30" \
    -n 2 \
    -R "rusage[mem=8000] span[hosts=1]" \
    -J "${JOB_NAME}" \
    -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
    -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --out ${OUTDIR} ${EXTRA_ARGS}" \
    | grep -oP '\d+')

echo "Submitted: ${JOB_NAME} → job ${JOB_ID}"
echo "Out:   ${OUTDIR}/panel_size_selection.csv"
echo "       ${OUTDIR}/fig_panel_size_selection.{pdf,png}"
echo ""
echo "Monitor: bjobs -J ${JOB_NAME}"
echo "Kill:    bkill ${JOB_ID}"
