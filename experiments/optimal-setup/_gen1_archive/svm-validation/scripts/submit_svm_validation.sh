#!/bin/bash
#BSUB -P acc_vascbrain
#BSUB -q premium
#BSUB -J CeD_svm_validation
#BSUB -n 8
#BSUB -W 48:00
#BSUB -R "rusage[mem=16000] span[hosts=1]"
#BSUB -oo results/svm_validation/svm_validation_%J.stdout
#BSUB -eo results/svm_validation/svm_validation_%J.stderr

# SVM Incident Validation
# Replicates incident validation with LinearSVC + CalibratedClassifierCV.
# 12 strategy-weight combinations, 5-fold CV with Optuna inner tuning.
#
# Usage:
#   cd cel-risk
#   bsub < experiments/optimal-setup/svm-validation/scripts/submit_svm_validation.sh
#
# Smoke test (local):
#   cd cel-risk
#   source analysis/.venv/bin/activate
#   python experiments/optimal-setup/svm-validation/scripts/run_svm_validation.py --smoke

set -euo pipefail

cd "$(dirname "$0")/../../../.."  # cel-risk root

mkdir -p results/svm_validation

echo "=== SVM Incident Validation ==="
echo "Host: $(hostname)"
echo "Date: $(date -Iseconds)"
echo "Working dir: $(pwd)"

# Activate environment
source analysis/.venv/bin/activate

python experiments/optimal-setup/svm-validation/scripts/run_svm_validation.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --output-dir results/svm_validation

echo "=== Done: $(date -Iseconds) ==="
