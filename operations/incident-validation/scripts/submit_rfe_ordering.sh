#!/bin/bash
# Submit RFE ordering extension to the purity saturation comparison.
#
# Dependency chain:
#   rank_job  →  3 rfe run jobs (parallel)  →  aggregate
#
# Assumes purity and stability partial CSVs already exist from a prior run of
# submit_purity_saturation_parallel.sh. The aggregate phase reads all
# purity_sat_*.csv files and overwrites saturation_all_models.csv.
#
# Prerequisites:
#   - operations/incident-validation/analysis/out/prevalent_noise_scores.csv
#   - operations/incident-validation/analysis/out/purity_sat_*_purity.csv
#   - operations/incident-validation/analysis/out/purity_sat_*_stability.csv
#
# Usage:
#   bash operations/incident-validation/scripts/submit_rfe_ordering.sh
#   bash operations/incident-validation/scripts/submit_rfe_ordering.sh --smoke

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
RANK_SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_rfe_ranking.py"
SAT_SCRIPT="${BASEDIR}/operations/incident-validation/analysis/compute_purity_saturation.py"
LOGDIR="${BASEDIR}/logs/incident-validation"
OUTDIR="${BASEDIR}/operations/incident-validation/analysis/out"

mkdir -p "${LOGDIR}" "${OUTDIR}"

SMOKE=0
RANK_WALL="02:00"
RUN_WALL="12:00"
AGG_WALL="01:00"
RANK_MEM="16000"
RUN_MEM="16000"
AGG_MEM="8000"
RANK_CORES=2
RUN_CORES=4
AGG_CORES=2

for arg in "$@"; do
    case "${arg}" in
        --smoke)
            SMOKE=1
            RANK_WALL="00:30"; RUN_WALL="00:30"; AGG_WALL="00:15"
            RANK_MEM="8000"; RUN_MEM="8000"; AGG_MEM="4000"
            ;;
    esac
done

VENV_SITE="${BASEDIR}/analysis/venv/lib/python3.12/site-packages"
ACTIVATE="cd ${BASEDIR} && unset PYTHONPATH && module load python/3.12.5 && source analysis/venv/bin/activate && export PYTHONPATH=${VENV_SITE}"

SMOKE_ARGS=""
[[ ${SMOKE} -eq 1 ]] && SMOKE_ARGS="--panel-sizes 5 10 28"

PREFIX="CeD_rfe_sat"
[[ ${SMOKE} -eq 1 ]] && PREFIX="${PREFIX}_smoke"

MODELS=("LR_EN" "SVM_L1" "SVM_L2")

echo "=== Submitting RFE ordering extension ==="
echo ""

# --- Step 1: RFE ranking ---
RANK_JOB_NAME="${PREFIX}_rank"
RANK_JOB_ID=$(bsub \
    -P ${PROJECT} \
    -q ${QUEUE} \
    -W "${RANK_WALL}" \
    -n ${RANK_CORES} \
    -R "rusage[mem=${RANK_MEM}] span[hosts=1]" \
    -J "${RANK_JOB_NAME}" \
    -o "${LOGDIR}/${RANK_JOB_NAME}_%J.stdout" \
    -e "${LOGDIR}/${RANK_JOB_NAME}_%J.stderr" \
    bash -c "${ACTIVATE} && python ${RANK_SCRIPT} --out ${OUTDIR}" \
    | grep -oP '\d+')

echo "  Rank job: ${RANK_JOB_NAME} → job ${RANK_JOB_ID}"

# --- Step 2: RFE run jobs (depend on rank) ---
RUN_JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    JOB_NAME="${PREFIX}_${MODEL}_rfe"
    JOB_ID=$(bsub \
        -P ${PROJECT} \
        -q ${QUEUE} \
        -W "${RUN_WALL}" \
        -n ${RUN_CORES} \
        -R "rusage[mem=${RUN_MEM}] span[hosts=1]" \
        -J "${JOB_NAME}" \
        -w "done(${RANK_JOB_ID})" \
        -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
        -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
        bash -c "${ACTIVATE} && python ${SAT_SCRIPT} --phase run --model ${MODEL} --ordering rfe --out ${OUTDIR} ${SMOKE_ARGS}" \
        | grep -oP '\d+')
    RUN_JOB_IDS+=("${JOB_ID}")
    echo "  ${MODEL} x rfe: job ${JOB_ID} (depends on rank job ${RANK_JOB_ID})"
done

# --- Step 3: Aggregate (depends on all rfe run jobs) ---
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
    bash -c "${ACTIVATE} && python ${SAT_SCRIPT} --phase aggregate --out ${OUTDIR} ${SMOKE_ARGS}" \
    | grep -oP '\d+')

echo ""
echo "  Aggregate: job ${AGG_JOB_ID} (depends on: ${RUN_JOB_IDS[*]})"
echo ""
echo "Outputs:"
echo "  ${OUTDIR}/rfe_protein_ranking.csv"
echo "  ${OUTDIR}/purity_sat_*_rfe.csv"
echo "  ${OUTDIR}/saturation_all_models.csv    (overwritten with all 3 orderings)"
echo "  ${OUTDIR}/features_rfe.csv"
echo "  ${OUTDIR}/fig_purity_saturation.{pdf,png}"
echo ""
echo "Monitor: bjobs -w | grep ${PREFIX}"
echo "Kill:    bkill ${RANK_JOB_ID} ${RUN_JOB_IDS[*]} ${AGG_JOB_ID}"
