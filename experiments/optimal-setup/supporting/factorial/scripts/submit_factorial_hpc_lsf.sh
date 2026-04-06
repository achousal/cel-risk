#!/bin/bash
#BSUB -J factorial_2x2x2
#BSUB -P acc_vascbrain
#BSUB -q premium
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R rusage[mem=8000]
#BSUB -R span[hosts=1]
#BSUB -o logs/factorial_%J.out
#BSUB -e logs/factorial_%J.err

# HPC LSF submission script for 2x2x2 factorial experiment
# Matches settings from analysis/configs/pipeline_hpc.yaml
#
# Usage:
#   bsub < experiments/optimal-setup/supporting/factorial/submit_factorial_hpc_lsf.sh

set -euo pipefail

# Environment setup (adjust module names for your HPC)
module purge
module load python/3.10
source ~/venvs/ced_ml/bin/activate

# Logging
mkdir -p logs
echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $LSB_JOBID"
echo "Running on node: $(hostname)"
echo "Project: $LSB_PROJECT_NAME"
echo "Queue: $LSB_QUEUE"
echo "CPUs allocated: $LSB_DJOB_NUMPROC"
echo "Memory per CPU: 8000 MB"
echo "=========================================="

# Paths (auto-detect repo root from script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_PATH="${REPO_ROOT}/data/Celiac_dataset_proteomics_w_demo.parquet"
PANEL_PATH="${REPO_ROOT}/data/fixed_panel.csv"
OUTPUT_DIR="${REPO_ROOT}/results/factorial_2x2x2"
HYPERPARAMS="${OUTPUT_DIR}/cell_hyperparams.yaml"

echo "Repository root: $REPO_ROOT"
cd "$REPO_ROOT"

# Validate inputs
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$PANEL_PATH" ]; then
    echo "ERROR: Panel file not found: $PANEL_PATH"
    exit 1
fi

# Step 1: Tune hyperparameters for all cells (if not already done)
echo ""
echo "=========================================="
echo "STEP 1: Hyperparameter Tuning"
echo "=========================================="
if [ ! -f "$HYPERPARAMS" ]; then
    echo "Tuning cell-specific hyperparameters (this may take 2-3 hours)..."
    python experiments/optimal-setup/supporting/factorial/run_factorial_2x2x2.py \
        --data-path "$DATA_PATH" \
        --panel-path "$PANEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --tune-cells \
        --n-trials 50 \
        --models LR_EN XGBoost

    echo "Hyperparameter tuning complete at $(date)"
else
    echo "Using existing hyperparameters from:"
    echo "  $HYPERPARAMS"
fi

# Step 2: Run full factorial experiment in parallel
echo ""
echo "=========================================="
echo "STEP 2: Factorial Experiment"
echo "=========================================="
N_CPUS=${LSB_DJOB_NUMPROC:-8}
echo "Parallelizing across $N_CPUS workers..."
echo "Configuration:"
echo "  Seeds: 10"
echo "  Cells: 8 (2×2×2 factorial)"
echo "  Models: LR_EN, XGBoost"
echo "  Total jobs: 160"

python experiments/optimal-setup/supporting/factorial/run_factorial_2x2x2.py \
    --data-path "$DATA_PATH" \
    --panel-path "$PANEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --hyperparams-path "$HYPERPARAMS" \
    --n-seeds 10 \
    --models LR_EN XGBoost \
    --n-jobs "$N_CPUS"

# Final summary
echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
echo "Outputs:"
echo "  Results: ${OUTPUT_DIR}/factorial_results.csv"
echo "  Feature importances: ${OUTPUT_DIR}/feature_importances.csv"
echo "  Hyperparameters: ${OUTPUT_DIR}/cell_hyperparams.yaml"
echo "  Test indices: ${OUTPUT_DIR}/test_indices.csv"
echo ""
echo "Next steps:"
echo "  1. Analyze results:"
echo "     python experiments/optimal-setup/supporting/factorial/analyze_factorial_2x2x2.py \\"
echo "       --results ${OUTPUT_DIR}/factorial_results.csv \\"
echo "       --output ${OUTPUT_DIR}/analysis"
echo "=========================================="
