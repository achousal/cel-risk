#!/bin/bash
# Submit incident validation (LinSVM_cal) as parallel jobs on Minerva HPC
#
# Submits one 14-job chain per penalty (l1, l2) by default.
# Feature selection is shared — l2 chain reuses l1 feature artifacts.
#
# Job dependency chain (per penalty):
#   feat  →  12 CV combos (parallel)  →  aggregate
#
# Usage:
#   bash scripts/submit_incident_validation_svm_parallel.sh [--penalty l1|l2] [--smoke]

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/experiments/optimal-setup/incident-validation-svm/scripts/run_incident_validation_svm.py"
DATAFILE="${BASEDIR}/data/Celiac_dataset_proteomics_w_demo.parquet"
LOGDIR="${BASEDIR}/logs"

mkdir -p "${LOGDIR}"

SMOKE_FLAG=""
FEAT_WALL="48:00"
CV_WALL="48:00"
AGG_WALL="02:00"
FEAT_MEM="8000"
CV_MEM="8000"
AGG_MEM="4000"
FEAT_CORES=4
CV_CORES=2
AGG_CORES=2

PENALTIES=("l1" "l2")
for arg in "$@"; do
    case "${arg}" in
        --smoke)
            SMOKE_FLAG="--smoke"
            FEAT_WALL="00:30"; CV_WALL="00:30"; AGG_WALL="00:15"
            CV_CORES=4; FEAT_CORES=4
            ;;
        l1|l2) PENALTIES=("${arg}") ;;
    esac
done

STRATEGIES=("incident_only" "incident_prevalent" "prevalent_only")
WEIGHTS=("none" "balanced" "sqrt" "log")
ACTIVATE="cd ${BASEDIR} && module load anaconda3/2024.06 && source analysis/venv/bin/activate"

for PENALTY in "${PENALTIES[@]}"; do
    OUTDIR="${BASEDIR}/results/incident_validation_svm_${PENALTY}"
    PREFIX="CeD_iv_svm_${PENALTY}"
    [[ -n "${SMOKE_FLAG}" ]] && OUTDIR="${OUTDIR}_smoke" && PREFIX="${PREFIX}_smoke"

    COMMON_ARGS="--penalty ${PENALTY} --data-path ${DATAFILE} --output-dir ${OUTDIR} ${SMOKE_FLAG}"

    echo "=== Submitting parallel LinSVM_cal penalty=${PENALTY} ==="
    echo "  Output: ${OUTDIR}"
    echo "  Combos: ${#STRATEGIES[@]} × ${#WEIGHTS[@]} = $(( ${#STRATEGIES[@]} * ${#WEIGHTS[@]} )) jobs"
    echo ""

    # --- Job 0: Feature selection ---
    FEAT_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W "${FEAT_WALL}" \
        -n ${FEAT_CORES} -R "rusage[mem=${FEAT_MEM}] span[hosts=1]" \
        -J "${PREFIX}_feat" \
        -o "${LOGDIR}/${PREFIX}_feat_%J.stdout" \
        -e "${LOGDIR}/${PREFIX}_feat_%J.stderr" \
        bash -c "${ACTIVATE} && python ${SCRIPT} --phase features ${COMMON_ARGS}" \
        | grep -oP '\d+')
    echo "  Feature selection: job ${FEAT_JOB}"

    # --- Jobs 1-12: CV combos ---
    CV_JOBS=()
    for strat in "${STRATEGIES[@]}"; do
        for wt in "${WEIGHTS[@]}"; do
            JOB_NAME="${PREFIX}_${strat}_${wt}"
            JOB_ID=$(bsub -P ${PROJECT} -q ${QUEUE} -W "${CV_WALL}" \
                -n ${CV_CORES} -R "rusage[mem=${CV_MEM}] span[hosts=1]" \
                -J "${JOB_NAME}" \
                -w "done(${FEAT_JOB})" \
                -o "${LOGDIR}/${JOB_NAME}_%J.stdout" \
                -e "${LOGDIR}/${JOB_NAME}_%J.stderr" \
                bash -c "${ACTIVATE} && python ${SCRIPT} --phase cv --strategy ${strat} --weight-scheme ${wt} ${COMMON_ARGS}" \
                | grep -oP '\d+')
            CV_JOBS+=("${JOB_ID}")
            echo "  ${strat} + ${wt}: job ${JOB_ID}"
        done
    done

    DEP_EXPR=$(printf " && done(%s)" "${CV_JOBS[@]}")
    DEP_EXPR="${DEP_EXPR:4}"

    # --- Job 13: Aggregate ---
    AGG_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W "${AGG_WALL}" \
        -n ${AGG_CORES} -R "rusage[mem=${AGG_MEM}] span[hosts=1]" \
        -J "${PREFIX}_agg" \
        -w "${DEP_EXPR}" \
        -o "${LOGDIR}/${PREFIX}_agg_%J.stdout" \
        -e "${LOGDIR}/${PREFIX}_agg_%J.stderr" \
        bash -c "${ACTIVATE} && python ${SCRIPT} --phase aggregate ${COMMON_ARGS}" \
        | grep -oP '\d+')
    echo "  Aggregation: job ${AGG_JOB}"
    echo ""
    echo "  Kill all penalty=${PENALTY}: bkill ${FEAT_JOB} ${CV_JOBS[*]} ${AGG_JOB}"
    echo ""
done

echo "Monitor all: bjobs -w | grep CeD_iv_svm"
