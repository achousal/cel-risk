#!/bin/bash
# Submit unified incident validation as parallel jobs on Minerva HPC
#
# Runs one 14-job chain per model. Feature selection runs once, then
# 12 CV combos (3 strategies x 4 weights) run in parallel, then aggregate.
#
# Job dependency chain (per model):
#   feat  ->  12 CV combos (parallel)  ->  aggregate
#
# Usage:
#   bash scripts/submit_incident_validation_parallel.sh --model LR_EN
#   bash scripts/submit_incident_validation_parallel.sh --model SVM_L1
#   bash scripts/submit_incident_validation_parallel.sh --model SVM_L2
#   bash scripts/submit_incident_validation_parallel.sh --model all        # all 3 models
#   bash scripts/submit_incident_validation_parallel.sh --model SVM_L1 --smoke

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
SCRIPT="${BASEDIR}/operations/incident-validation/scripts/run_lr.py"
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

MODELS=()
for arg in "$@"; do
    case "${arg}" in
        --smoke)
            SMOKE_FLAG="--smoke"
            FEAT_WALL="00:30"; CV_WALL="00:30"; AGG_WALL="00:15"
            CV_CORES=4; FEAT_CORES=4
            ;;
        --model) ;; # consumed with next arg below
        LR_EN|SVM_L1|SVM_L2) MODELS+=("${arg}") ;;
        all) MODELS=("LR_EN" "SVM_L1" "SVM_L2") ;;
    esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Usage: $0 --model {LR_EN|SVM_L1|SVM_L2|all} [--smoke]"
    exit 1
fi

STRATEGIES=("incident_only" "incident_prevalent" "prevalent_only")
WEIGHTS=("none" "balanced" "sqrt" "log")
ACTIVATE="cd ${BASEDIR} && module load anaconda3/2024.06 && source analysis/venv/bin/activate"

for MODEL in "${MODELS[@]}"; do
    PREFIX="CeD_iv_${MODEL}"
    [[ -n "${SMOKE_FLAG}" ]] && PREFIX="${PREFIX}_smoke"

    COMMON_ARGS="--model ${MODEL} --data-path ${DATAFILE} ${SMOKE_FLAG}"

    echo "=== Submitting ${MODEL} ==="
    echo "  Combos: ${#STRATEGIES[@]} x ${#WEIGHTS[@]} = $(( ${#STRATEGIES[@]} * ${#WEIGHTS[@]} )) jobs"
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
    echo "  Kill all ${MODEL}: bkill ${FEAT_JOB} ${CV_JOBS[*]} ${AGG_JOB}"
    echo ""
done

echo "Monitor all: bjobs -w | grep CeD_iv_"
