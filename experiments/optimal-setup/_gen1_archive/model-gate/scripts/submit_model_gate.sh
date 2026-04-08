#!/bin/bash
#BSUB -P acc_vascbrain
#BSUB -q premium
#BSUB -J CeD_model_gate
#BSUB -n 12
#BSUB -W 48:00
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -oo results/model_gate/model_gate_%J.stdout
#BSUB -eo results/model_gate/model_gate_%J.stderr

# Phase A: Model Gate
# 4 models x 3 strategies x 4 weights = 48 combos
# 5-fold CV with Optuna inner tuning each
#
# Usage:
#   cd cel-risk
#   bsub < experiments/optimal-setup/model-gate/scripts/submit_model_gate.sh

set -euo pipefail

cd "$(dirname "$0")/../../../.."  # cel-risk root

mkdir -p results/model_gate

echo "=== Phase A: Model Gate ==="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "Working dir: $(pwd)"

source analysis/.venv/bin/activate

python experiments/optimal-setup/model-gate/scripts/run_model_gate.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --output-dir results/model_gate

echo "=== Done: $(date -Iseconds) ==="
