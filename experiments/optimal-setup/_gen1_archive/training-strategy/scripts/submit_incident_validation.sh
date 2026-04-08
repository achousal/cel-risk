#!/bin/bash
# Submit incident validation pipeline to Minerva HPC
# Usage: bash scripts/submit_incident_validation.sh [--smoke]
#
# Full run: ~2-4 hrs (100 bootstrap × 80 Optuna trials × 12 combos)
# Smoke:    ~10-15 min

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
BASEDIR="/sc/arion/projects/vascbrain/andres/cel-risk"
ANALYSISDIR="${BASEDIR}/analysis"
DATAFILE="${BASEDIR}/data/Celiac_dataset_proteomics_w_demo.parquet"
LOGDIR="${BASEDIR}/logs"

mkdir -p "${LOGDIR}"

SMOKE_FLAG=""
OUTDIR="${BASEDIR}/results/incident_validation"
WALLTIME="48:00"
MEM="16000"
CORES=8
JOB_NAME="CeD_incident_val"

if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE_FLAG="--smoke"
    OUTDIR="${BASEDIR}/results/incident_validation_smoke"
    WALLTIME="02:00"
    MEM="8000"
    CORES=4
    JOB_NAME="CeD_incident_val_smoke"
    echo "=== SMOKE TEST MODE ==="
fi

echo "=== Submitting incident validation pipeline ==="
echo "  Output: ${OUTDIR}"
echo "  Wall time: ${WALLTIME}"
echo "  Cores: ${CORES}, Memory: ${MEM}MB"

CMD="python ${ANALYSISDIR}/scripts/run_incident_validation.py \
    --data-path ${DATAFILE} \
    --output-dir ${OUTDIR} \
    ${SMOKE_FLAG}"

bsub -P ${PROJECT} -q ${QUEUE} -W "${WALLTIME}" \
    -n ${CORES} -R "rusage[mem=${MEM}] span[hosts=1]" \
    -J "${JOB_NAME}" \
    -oo "${LOGDIR}/incident_val_%J.stdout" \
    -eo "${LOGDIR}/incident_val_%J.stderr" \
    bash -c "cd ${ANALYSISDIR} && module load anaconda3/2024.06 && source venv/bin/activate && ${CMD}"

echo "Submitted. Monitor with: bjobs -w | grep ${JOB_NAME}"
