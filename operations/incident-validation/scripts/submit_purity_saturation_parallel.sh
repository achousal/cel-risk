#!/bin/bash
# Submit purity-ranked saturation comparison as parallel jobs on Minerva HPC.
#
# Fans out 6 parallel run jobs (3 models x 2 orderings), then one aggregate job.
#
# Job dependency chain:
#   6 run jobs (parallel)  ->  aggregate
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/prevalent_noise_scores.csv
#   - results/incident-validation/<model>/feature_coefficients.csv (stability rank)
#
# Usage:
#   bash operations/incident-validation/scripts/submit_purity_saturation_parallel.sh
#   bash operations/incident-validation/scripts/submit_purity_saturation_parallel.sh --smoke

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_purity_saturation.py"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

SMOKE=0
RUN_WALL="12:00"
AGG_WALL="01:00"
RUN_MEM="16000"
AGG_MEM="8000"
RUN_CORES=4
AGG_CORES=2

for arg in "$@"; do
    case "${arg}" in
        --smoke) SMOKE=1; RUN_WALL="00:30"; AGG_WALL="00:15"; RUN_MEM="8000"; AGG_MEM="4000" ;;
    esac
done

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

SMOKE_ARGS=""
[[ ${SMOKE} -eq 1 ]] && SMOKE_ARGS="--panel-sizes 5 10 28"

PREFIX="CeD_purity_sat"
[[ ${SMOKE} -eq 1 ]] && PREFIX="${PREFIX}_smoke"

MODELS=("LR_EN" "SVM_L1" "SVM_L2")
ORDERINGS=("purity" "stability")

echo "=== Submitting purity saturation (parallel) ==="
echo "  Jobs: ${#MODELS[@]} models x ${#ORDERINGS[@]} orderings = $(( ${#MODELS[@]} * ${#ORDERINGS[@]} )) run + 1 aggregate"
echo ""

RUN_JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    for ORDERING in "${ORDERINGS[@]}"; do
        JOB_NAME="${PREFIX}_${MODEL}_${ORDERING}"
        JOB_ID=$(bsub \
            -P ${PROJECT} \
            -q ${QUEUE} \
            -W "${RUN_WALL}" \
            -n ${RUN_CORES} \
            -R "rusage[mem=${RUN_MEM}] span[hosts=1]" \
            -J "${JOB_NAME}" \
            -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
            -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
            bash -c "${ACTIVATE} && python ${SCRIPT} --phase run --model ${MODEL} --ordering ${ORDERING} --out ${OUTDIR} ${SMOKE_ARGS}" \
            | grep -oP '\d+')
        RUN_JOB_IDS+=("${JOB_ID}")
        echo "  ${MODEL} x ${ORDERING}: job ${JOB_ID}"
    done
done

DEP_EXPR=$(printf " && done(%s)" "${RUN_JOB_IDS[@]}")
DEP_EXPR="${DEP_EXPR:4}"

AGG_JOB_NAME="${PREFIX}_agg"
AGG_JOB_ID=$(bsub \
    -P ${PROJECT} \
    -q ${QUEUE} \
    -W "${AGG_WALL}" \
    -n ${AGG_CORES} \
    -R "rusage[mem=${AGG_MEM}] span[hosts=1]" \
    -J "${AGG_JOB_NAME}" \
    -w "${DEP_EXPR}" \
    -o "${LOGDIR}/${AGG_JOB_NAME}_%J.stdout" \
    -e "${LOGDIR}/${AGG_JOB_NAME}_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --phase aggregate --out ${OUTDIR} ${SMOKE_ARGS}" \
    | grep -oP '\d+')

echo ""
echo "  Aggregate: job ${AGG_JOB_ID} (depends on: ${RUN_JOB_IDS[*]})"
echo ""
echo "Outputs:"
echo "  ${OUTDIR}/saturation_all_models.csv"
echo "  ${OUTDIR}/fig_purity_saturation.{pdf,png}"
echo ""
echo "Monitor: bjobs -w | grep ${PREFIX}"
echo "Kill:    bkill ${RUN_JOB_IDS[*]} ${AGG_JOB_ID}"
