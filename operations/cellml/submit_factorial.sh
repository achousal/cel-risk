#!/usr/bin/env bash
# submit_factorial.sh — SLURM array submission for factorial cells
#
# Supports two-phase warm-start workflow:
#   Phase 1 (scout):  PHASE=scout sbatch submit_factorial.sh <scout_manifest.csv>
#   Phase 2 (main):   sbatch submit_factorial.sh <cell_manifest.csv>
#
# Single-phase (no warm-start):
#   sbatch submit_factorial.sh <cell_manifest.csv>
#
# After scout completes, extract params before submitting main:
#   python extract_scout_params.py --storage-dir <dir> --output scout_top_params.json
#   Then re-run config_gen.py with warm_start_params set in manifest.yaml

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MANIFEST="${1:?Usage: sbatch submit_factorial.sh <cell_manifest.csv>}"
SEEDS_START="${SEEDS_START:-100}"
SEEDS_END="${SEEDS_END:-129}"
PHASE="${PHASE:-main}"

# Count cells (subtract 1 for header)
N_CELLS=$(( $(wc -l < "$MANIFEST") - 1 ))
if [[ "$N_CELLS" -le 0 ]]; then
    echo "ERROR: No cells found in $MANIFEST"
    exit 1
fi
echo "Phase: $PHASE | Submitting $N_CELLS cells (seeds $SEEDS_START-$SEEDS_END)"

# ---------------------------------------------------------------------------
# SLURM directives
# ---------------------------------------------------------------------------
#SBATCH --job-name=factorial_${PHASE}
#SBATCH --array=1-${N_CELLS}
#SBATCH --output=logs/factorial_${PHASE}_%A_%a.out
#SBATCH --error=logs/factorial_${PHASE}_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --partition=premium

# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------
CELL_ID=${SLURM_ARRAY_TASK_ID}

# Parse cell manifest by column NAME (handles both factorial and V0 formats)
HEADER=$(head -1 "$MANIFEST")

# Find column indices from header
col_idx() {
    echo "$HEADER" | awk -F',' -v col="$1" '{for(i=1;i<=NF;i++) if($i==col) print i}'
}

COL_CELL_ID=$(col_idx "cell_id")
COL_RECIPE=$(col_idx "recipe_id")
COL_MODEL=$(col_idx "model")
COL_PIPELINE=$(col_idx "pipeline_config")

if [[ -z "$COL_CELL_ID" || -z "$COL_MODEL" || -z "$COL_PIPELINE" ]]; then
    echo "ERROR: Manifest missing required columns (cell_id, model, pipeline_config)"
    echo "Header: $HEADER"
    exit 1
fi

ROW=$(awk -F',' -v id="$CELL_ID" -v col="$COL_CELL_ID" 'NR>1 && $col==id' "$MANIFEST")
if [[ -z "$ROW" ]]; then
    echo "ERROR: Cell ID $CELL_ID not found in $MANIFEST"
    exit 1
fi

PIPELINE_CONFIG=$(echo "$ROW" | awk -F',' -v col="$COL_PIPELINE" '{print $col}')
RECIPE_ID=$(echo "$ROW" | awk -F',' -v col="$COL_RECIPE" '{print $col}')
MODEL=$(echo "$ROW" | awk -F',' -v col="$COL_MODEL" '{print $col}')

echo "Cell $CELL_ID: recipe=$RECIPE_ID model=$MODEL phase=$PHASE"
echo "Pipeline config: $PIPELINE_CONFIG"

# Run pipeline for each seed
for SEED in $(seq "$SEEDS_START" "$SEEDS_END"); do
    echo "--- Seed $SEED ---"
    ced run-pipeline \
        --pipeline-config "$PIPELINE_CONFIG" \
        --split-seeds "$SEED" \
        --log-level info
done

echo "Cell $CELL_ID complete."
