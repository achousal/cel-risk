#!/bin/bash
# Submit purity panel significance test to Minerva HPC.
#
# Reads saturation_all_models.csv, runs bootstrap z-test + BH-FDR,
# and applies the three-criterion size rule (same as the main pipeline)
# to find the Pareto-optimal panel size.
#
# Parameters are read from operations/incident-validation/manifest.yaml
# (purity_significance section). Override via CLI if needed.
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/saturation_all_models.csv
#
# Usage:
#   bash operations/incident-validation/scripts/submit_purity_significance.sh
#   bash operations/incident-validation/scripts/submit_purity_significance.sh --delta 0.01

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_purity_significance.py"
MANIFEST="${BASEDIR}/operations/incident-validation/manifest.yaml"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

# Pass through any extra CLI args (e.g. --delta 0.01 --min-criteria 3)
EXTRA_ARGS="$*"

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
    bash -c "${ACTIVATE} && python ${SCRIPT} --config ${MANIFEST} --out ${OUTDIR} ${EXTRA_ARGS}" \
    | grep -oP '\d+')

echo "Submitted: ${JOB_NAME} → job ${JOB_ID}"
echo "Config: ${MANIFEST}"
echo "Logs:  ${LOGDIR}/${JOB_NAME}_${JOB_ID}.{stdout,stderr}"
echo "Out:   ${OUTDIR}/purity_significance.csv"
echo "       ${OUTDIR}/purity_significance_audit.json"
echo "       ${OUTDIR}/fig_purity_significance.{pdf,png}"
echo ""
echo "Monitor: bjobs -J ${JOB_NAME}"
echo "Kill:    bkill ${JOB_ID}"
