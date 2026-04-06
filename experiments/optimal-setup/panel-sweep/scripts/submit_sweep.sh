#!/usr/bin/env bash
# submit_sweep.sh -- Submit panel saturation sweep to Minerva HPC
#
# Usage:
#   bash submit_sweep.sh                    # submit all 62 new runs
#   bash submit_sweep.sh --order rra        # submit only RRA order (20 runs)
#   bash submit_sweep.sh --dry-run          # preview without submitting
#   bash submit_sweep.sh --batch 10         # submit 10 at a time with pause
#
# Each run submits 4 models × 10 seeds = 40 training jobs via ced orchestrator.
# LSF handles scheduling. Orchestrators are lightweight (1 core, 2GB).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ANALYSIS_DIR="$PROJECT_ROOT/analysis"
PANELS_DIR="$SCRIPT_DIR/panels"
CONFIGS_DIR="$ANALYSIS_DIR/configs"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_DIR="$SCRIPT_DIR/submission_logs"

# Defaults
DRY_RUN=false
ORDER_FILTER=""
BATCH_SIZE=0  # 0 = no batching
SLEEP_BETWEEN=30  # seconds between batches

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --order) ORDER_FILTER="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --sleep) SLEEP_BETWEEN="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "$ANALYSIS_DIR"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Verify ced is available
if ! command -v ced &>/dev/null; then
    echo "ERROR: ced CLI not found. Activate venv first."
    exit 1
fi

mkdir -p "$LOG_DIR"

# RRA rank order proteins
PROTEINS=(
    tgm2_resid cpa2_resid itgb7_resid gip_resid cxcl9_resid
    cd160_resid muc2_resid nos2_resid fabp6_resid agr2_resid
    reg3a_resid mln_resid ccl25_resid pafah1b3_resid tnfrsf8_resid
    tigit_resid cxcl11_resid ckmt1a_ckmt1b_resid acy3_resid hla_a_resid
    xcl1_resid nell2_resid pof1b_resid ppp1r14d_resid ada2_resid
)

# Define addition orders (indices into PROTEINS array, 0-based)
# Order: importance (cxcl9 before gip)
IMPORTANCE_ORDER=(0 1 2 4 5 6 3 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
# Order: pathway (mucosal, immune, metabolic, extended immune, extended GI)
PATHWAY_ORDER=(0 6 2 4 5 1 3 7 16 15 14 12 20 19 8 9 10 11 13 18 21 22 23 24 17)

get_proteins_for_order() {
    local order=$1
    local size=$2
    local result=()

    if [ "$order" = "rra" ]; then
        for ((i=0; i<size; i++)); do
            result+=("${PROTEINS[$i]}")
        done
    elif [ "$order" = "importance" ]; then
        for ((i=0; i<size; i++)); do
            result+=("${PROTEINS[${IMPORTANCE_ORDER[$i]}]}")
        done
    elif [ "$order" = "pathway" ]; then
        for ((i=0; i<size; i++)); do
            result+=("${PROTEINS[${PATHWAY_ORDER[$i]}]}")
        done
    fi
    echo "${result[*]}"
}

# Track submissions
SUBMITTED=0
SKIPPED=0
FAILED=0

submit_run() {
    local order=$1
    local size=$2
    local run_id="sweep_${order}_${size}p"
    local panel_csv="$PANELS_DIR/${order}_${size}p.csv"
    local training_cfg="$SCRIPT_DIR/configs/training_${order}_${size}p.yaml"
    local result_dir="$RESULTS_DIR/$run_id"

    # Skip if results already exist (all 4 models aggregated)
    local complete=true
    for model in LR_EN LinSVM_cal RF XGBoost; do
        if [ ! -f "$result_dir/$model/aggregated/metrics/pooled_test_metrics.csv" ]; then
            complete=false
            break
        fi
    done
    if $complete; then
        echo "  SKIP: $run_id (already complete)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Skip reusable existing runs (4p rra = run_phase3_holdout_4protein, 7p rra = run_phase3_holdout)
    if [ "$order" = "rra" ] && [ "$size" -eq 4 ]; then
        echo "  SKIP: $run_id (reuse run_phase3_holdout_4protein)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi
    if [ "$order" = "rra" ] && [ "$size" -eq 7 ]; then
        echo "  SKIP: $run_id (reuse run_phase3_holdout)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Verify panel CSV exists
    if [ ! -f "$panel_csv" ]; then
        echo "  ERROR: Missing panel CSV: $panel_csv"
        FAILED=$((FAILED + 1))
        return 1
    fi

    # Build command: use --config for training config, explicit flags for everything else
    local cmd="ced run-pipeline"
    cmd+=" --hpc"
    cmd+=" --hpc-config configs/pipeline_hpc.yaml"
    cmd+=" --config $training_cfg"
    cmd+=" --split-seeds 200,201,202,203,204,205,206,207,208,209"
    cmd+=" --run-id $run_id"
    cmd+=" --no-optimize-panel"
    cmd+=" --no-consensus"
    cmd+=" --no-permutation-test"

    if $DRY_RUN; then
        echo "  DRY: $cmd"
        SUBMITTED=$((SUBMITTED + 1))
        return 0
    fi

    echo "  SUBMIT: $run_id (${size}p, $order order)"
    $cmd >> "$LOG_DIR/${run_id}.log" 2>&1 &
    local pid=$!
    echo "$pid $run_id" >> "$LOG_DIR/pids.txt"
    ((SUBMITTED++))
}

# Main submission loop
ORDERS=("rra" "importance" "pathway")
if [ -n "$ORDER_FILTER" ]; then
    ORDERS=("$ORDER_FILTER")
fi

echo "================================================================"
echo "PANEL SATURATION SWEEP -- HPC SUBMISSION"
echo "================================================================"
echo "Orders: ${ORDERS[*]}"
echo "Panel sizes: 4-25"
echo "Models: LR_EN, LinSVM_cal, RF, XGBoost + ENSEMBLE"
echo "Seeds: 200-209"
echo "Dry run: $DRY_RUN"
echo "================================================================"
echo ""

: > "$LOG_DIR/pids.txt"  # clear pid tracking
BATCH_COUNT=0

for order in "${ORDERS[@]}"; do
    echo "--- Order: $order ---"
    for size in $(seq 4 25); do
        submit_run "$order" "$size"

        if [ "$BATCH_SIZE" -gt 0 ]; then
            BATCH_COUNT=$((BATCH_COUNT + 1))
            if [ "$BATCH_COUNT" -ge "$BATCH_SIZE" ]; then
                echo ""
                echo "  [Batch of $BATCH_SIZE submitted, sleeping ${SLEEP_BETWEEN}s...]"
                sleep "$SLEEP_BETWEEN"
                BATCH_COUNT=0
            fi
        fi
    done
    echo ""
done

echo "================================================================"
echo "SUBMISSION SUMMARY"
echo "================================================================"
echo "  Submitted: $SUBMITTED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "  Logs:      $LOG_DIR/"
if ! $DRY_RUN; then
    echo "  PIDs:      $LOG_DIR/pids.txt"
    echo ""
    echo "Monitor with:"
    echo "  bjobs -w | grep sweep    # LSF job status"
    echo "  tail -f $LOG_DIR/*.log   # orchestrator logs"
fi
echo "================================================================"
