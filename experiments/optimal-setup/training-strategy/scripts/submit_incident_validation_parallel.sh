#!/bin/bash
# Submit incident validation pipeline as parallel jobs (Option A: 12 combos)
#
# Job dependency chain:
#   Job 0 (features)  →  Jobs 1-12 (CV combos, parallel)  →  Job 13 (aggregate)
#
# Usage:
#   bash scripts/submit_incident_validation_parallel.sh [--smoke]
#
# Full run:  ~3 hrs wall (feature sel ~2h + longest combo ~3h)
# Smoke:     ~15 min

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
ANALYSISDIR="${BASEDIR}/analysis"
DATAFILE="${BASEDIR}/data/Celiac_dataset_proteomics_w_demo.parquet"
LOGDIR="${BASEDIR}/logs"
SCRIPT="${ANALYSISDIR}/scripts/run_incident_validation.py"
POSTPROCESS="${ANALYSISDIR}/scripts/postprocess_incident_validation.py"

mkdir -p "${LOGDIR}"

SMOKE_FLAG=""
OUTDIR="${BASEDIR}/results/incident_validation"
FEAT_WALL="48:00"
CV_WALL="48:00"
AGG_WALL="02:00"
FEAT_MEM="8000"
CV_MEM="8000"
AGG_MEM="4000"
FEAT_CORES=4
CV_CORES=2
AGG_CORES=2
PREFIX="CeD_iv"

if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke"
    OUTDIR="${BASEDIR}/results/incident_validation_smoke_parallel"
    FEAT_WALL="00:30"
    CV_WALL="00:30"
    AGG_WALL="00:15"
    FEAT_MEM="8000"
    CV_MEM="8000"
    AGG_MEM="4000"
    FEAT_CORES=4
    CV_CORES=4
    AGG_CORES=2
    PREFIX="CeD_iv_smoke"
    echo "=== SMOKE TEST MODE ==="
fi

ACTIVATE="cd ${ANALYSISDIR} && module load anaconda3/2024.06 && source venv/bin/activate"
COMMON_ARGS="--data-path ${DATAFILE} --output-dir ${OUTDIR} ${SMOKE_FLAG}"

STRATEGIES=("incident_only" "incident_prevalent" "prevalent_only")
WEIGHTS=("none" "balanced" "sqrt" "log")

echo "=== Submitting parallel incident validation ==="
echo "  Output: ${OUTDIR}"
echo "  Combos: ${#STRATEGIES[@]} strategies × ${#WEIGHTS[@]} weights = $(( ${#STRATEGIES[@]} * ${#WEIGHTS[@]} )) jobs"
echo ""

# --- Job 0: Feature selection ---
echo "Submitting feature selection..."
FEAT_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W "${FEAT_WALL}" \
    -n ${FEAT_CORES} -R "rusage[mem=${FEAT_MEM}] span[hosts=1]" \
    -J "${PREFIX}_feat" \
    -o "${LOGDIR}/${PREFIX}_feat_%J.stdout" \
    -e "${LOGDIR}/${PREFIX}_feat_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --phase features ${COMMON_ARGS}" \
    | grep -oP '\d+')

echo "  Feature selection: job ${FEAT_JOB}"

# --- Jobs 1-12: CV combos (depend on feature selection) ---
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
        echo "  ${strat} + ${wt}: job ${JOB_ID} (depends on ${FEAT_JOB})"
    done
done

# Build dependency expression: done(job1) && done(job2) && ...
DEP_EXPR=$(printf " && done(%s)" "${CV_JOBS[@]}")
DEP_EXPR="${DEP_EXPR:4}"  # strip leading " && "

# --- Job 13: Aggregate + final refit + postprocess ---
echo ""
echo "Submitting aggregation..."
AGG_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W "${AGG_WALL}" \
    -n ${AGG_CORES} -R "rusage[mem=${AGG_MEM}] span[hosts=1]" \
    -J "${PREFIX}_agg" \
    -w "${DEP_EXPR}" \
    -o "${LOGDIR}/${PREFIX}_agg_%J.stdout" \
    -e "${LOGDIR}/${PREFIX}_agg_%J.stderr" \
    bash -c "${ACTIVATE} && python ${SCRIPT} --phase aggregate ${COMMON_ARGS} && python ${POSTPROCESS} --results-dir ${OUTDIR}" \
    | grep -oP '\d+')

echo "  Aggregation + postprocess: job ${AGG_JOB} (depends on all CV jobs)"

echo ""
echo "=== Submitted 14 jobs ==="
echo "  Feature selection: ${FEAT_JOB}"
echo "  CV combos (12):   ${CV_JOBS[*]}"
echo "  Aggregation:      ${AGG_JOB}"
echo ""
echo "Monitor: bjobs -w | grep ${PREFIX}"
echo "Kill all: bkill ${FEAT_JOB} ${CV_JOBS[*]} ${AGG_JOB}"
