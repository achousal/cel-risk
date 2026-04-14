#!/usr/bin/env bash
# Submit 16-cell panel x model matrix on Minerva (LSF).
# Run from project root on Minerva:
#   bash operations/panel_model_matrix/scripts/submit_minerva.sh
#
# 4 panels x 4 models, holdout ds5, IncidentOnly, top-20 fixed panels.
# Each (panel, model) becomes one ced run-pipeline invocation wrapped in bsub.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PANELS=(LinSVM_cal LR_EN RF XGBoost)
MODELS=(LinSVM_cal LR_EN RF XGBoost)
RESULTS_ROOT="results/panel_model_matrix"
LOGS_ROOT="logs/panel_model_matrix"

mkdir -p "$RESULTS_ROOT" "$LOGS_ROOT"

WALL="48:00"
CORES=12
MEM=8000
QUEUE="premium"
PROJECT="acc_vascbrain"

for PANEL in "${PANELS[@]}"; do
  CFG="operations/panel_model_matrix/configs/pipeline_hpc_holdout_ds5_${PANEL}_top20.yaml"
  for MODEL in "${MODELS[@]}"; do
    RUN_ID="pmm_p${PANEL}_m${MODEL}"
    OUTDIR="$RESULTS_ROOT/$RUN_ID"
    LOGDIR="$LOGS_ROOT/$RUN_ID"
    mkdir -p "$OUTDIR" "$LOGDIR"

    JOB_NAME="CeD_pmm_p${PANEL}_m${MODEL}"
    BSUB_CMD=(bsub
      -J "$JOB_NAME"
      -P "$PROJECT"
      -q "$QUEUE"
      -n "$CORES"
      -R "span[hosts=1]"
      -R "rusage[mem=${MEM}]"
      -W "$WALL"
      -o "$LOGDIR/%J.out"
      -e "$LOGDIR/%J.err")

    CMD="cd $PROJECT_ROOT && module load python/3.12.5 && source analysis/.venv/bin/activate && unset PYTHONPATH && ced run-pipeline \
      --pipeline-config $CFG \
      --models $MODEL \
      --experiment panel_model_matrix \
      --run-id $RUN_ID \
      --outdir $OUTDIR"

    echo "[submit] $JOB_NAME"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "  ${BSUB_CMD[*]} \"$CMD\""
    else
      "${BSUB_CMD[@]}" "$CMD"
    fi
  done
done

echo "Submitted 16 cells. Monitor with: bjobs -w | grep CeD_pmm_"
