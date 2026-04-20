#!/bin/bash
# Submit purity-ranked saturation comparison to Minerva HPC.
#
# Evaluates AUPRC vs panel size for purity-ranked vs stability-ranked protein
# orderings across LR_EN, SVM_L1, and SVM_L2 (incident_only, model-best weight).
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/prevalent_noise_scores.csv  (compute_prevalent_noise_score.py)
#   - operations/incident-validation/analysis/out/saturation_results.csv       (existing baseline)
#
# Usage:
#   bash operations/incident-validation/scripts/submit_purity_saturation.sh
#   bash operations/incident-validation/scripts/submit_purity_saturation.sh --smoke

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_purity_saturation.py"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

SMOKE=0
WALL="24:00"
MEM="24000"
CORES=4

for arg in "$@"; do
    case "${arg}" in
        --smoke) SMOKE=1; WALL="00:30"; MEM="8000" ;;
    esac
done

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

SMOKE_ARGS=""
if [[ ${SMOKE} -eq 1 ]]; then
    SMOKE_ARGS="--panel-sizes 5 10 28"
fi

JOB_NAME="CeD_purity_sat"
[[ ${SMOKE} -eq 1 ]] && JOB_NAME="${JOB_NAME}_smoke"

JOB_ID=$(bsub \
    -P ${PROJECT} \
    -q ${QUEUE} \
    -W "${WALL}" \
    -n ${CORES} \
    -R "rusage[mem=${MEM}] span[hosts=1]" \
    -J "${JOB_NAME}" \
    -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
    -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --out ${OUTDIR} ${SMOKE_ARGS}" \
    | grep -oP '\d+')

echo "Submitted: ${JOB_NAME} → job ${JOB_ID}"
echo "Logs:  ${LOGDIR}/${JOB_NAME}_${JOB_ID}.{stdout,stderr}"
echo "Out:   ${OUTDIR}/saturation_all_models.csv"
echo "       ${OUTDIR}/features_purity.csv  features_stability.csv"
echo "       ${OUTDIR}/fig_purity_saturation.{pdf,png}"
echo ""
echo "Monitor: bjobs -J ${JOB_NAME}"
echo "Kill:    bkill ${JOB_ID}"
