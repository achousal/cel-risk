#!/usr/bin/env bash

##########################################################################################
# Consolidated Factorial Experiment Runner (HPC Parallel Execution)
##########################################################################################
#
# Full-featured HPC runner for factorial experiments testing prevalent fractions and
# case:control ratios with:
# - Parallel job submission (all runs execute simultaneously on HPC cluster)
# - Multiple random seeds for robust statistics
# - Fixed 100-protein panel (eliminates FS variability)
# - Frozen hyperparameter/calibration config
# - Statistical analysis (paired t-tests, Bonferroni, Cohen's d)
# - Comprehensive logging and progress tracking
#
# Usage:
#   bash run_factorial_experiment.sh --project acc_YourProject [options]
#
# Presets:
#   --quick                             Quick test (1 seed, 2 models)
#   --overnight                         Overnight run (5 seeds, 2 models, 6 configs)
#   --full                              Full experiment (10 seeds, 4 models, 6 configs)
#
# Options:
#   --prevalent-fracs FRAC1,FRAC2,...   Prevalent sampling fractions (default: 0.5,1.0)
#   --case-control-ratios RATIO1,...    Case:control ratios (default: 1,5,10)
#   --models MODEL1,MODEL2,...          Models to train (default: LR_EN,RF)
#   --split-seeds SEED1,SEED2,...       Random seeds (default: 0,1,2,3,4)
#   --skip-training                     Skip retraining (analyze existing)
#   --skip-splits                       Skip split generation (use existing)
#   --skip-panel                        Skip panel generation (use existing)
#   --force-panel                       Force panel regeneration
#   --dry-run                           Preview without executing
#   --project PROJECT                   HPC project allocation
#   --queue QUEUE                       HPC queue (default: premium)
#   --walltime TIME                     Job walltime (default: 12:00)
#   --cores N                           Cores per job (default: 4)
#   --mem MB                            Memory per job in MB (default: 8000)
#   --help                              Show this message
#
# Examples:
#   # Quick test (12 parallel jobs: 6 configs × 1 seed × 2 models)
#   bash run_factorial_experiment.sh --quick --project acc_MyProject
#
#   # Overnight run (60 parallel jobs: 6 configs × 5 seeds × 2 models)
#   bash run_factorial_experiment.sh --overnight --project acc_MyProject
#
#   # Full experiment (240 parallel jobs: 6 configs × 10 seeds × 4 models)
#   bash run_factorial_experiment.sh --full --project acc_MyProject
#
#   # Custom HPC configuration
#   bash run_factorial_experiment.sh \
#     --prevalent-fracs 0.5,1.0 \
#     --case-control-ratios 1,5,10 \
#     --models LR_EN,RF,XGBoost \
#     --split-seeds 0,1,2,3,4,5,6,7,8,9 \
#     --project acc_MyProject \
#     --queue premium \
#     --walltime 24:00 \
#     --cores 8 \
#     --skip-splits
#
#   # Re-run analysis only (after jobs complete)
#   bash run_factorial_experiment.sh --skip-training --skip-splits --skip-panel
#
##########################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../../../results"
INVEST_RESULTS_DIR="$SCRIPT_DIR/../../../results/investigations"
SPLITS_BASE_DIR="$ANALYSIS_DIR/../splits_experiments"
LOG_DIR="$ANALYSIS_DIR/../logs/experiments"
PANEL_FILE="$ANALYSIS_DIR/../data/fixed_panel.csv"
FROZEN_CONFIG="$SCRIPT_DIR/training_config_frozen.yaml"

# Defaults
PREVALENT_FRACS=(0.5 1.0)
CASE_CONTROL_RATIOS=(1 5 10)
MODELS=("LR_EN" "RF")
SPLIT_SEEDS=(0 1 2 3 4)
SKIP_TRAINING=false
SKIP_SPLITS=false
SKIP_PANEL=false
FORCE_PANEL=false
DRY_RUN=false
PRESET=""

# HPC defaults
PROJECT="${PROJECT:-acc_Chipuk_Laboratory}"
QUEUE="${QUEUE:-premium}"
WALLTIME="${WALLTIME:-48:00}"
CORES="${CORES:-8}"
MEM="${MEM:-8000}"

# Experiment tracking
EXPERIMENT_ID=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT_DIR="$INVEST_RESULTS_DIR/experiment_${EXPERIMENT_ID}"
EXPERIMENT_LOG="$LOG_DIR/experiment_${EXPERIMENT_ID}.log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Helper functions
print_header() {
    echo ""
    echo "################################################################################################"
    echo "  $1"
    echo "################################################################################################"
    echo ""
}

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[i]${NC} $1"
}

show_help() {
    head -60 "$0" | tail -n +4
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            PRESET="quick"
            SPLIT_SEEDS=(0)
            MODELS=("LR_EN" "RF")
            shift
            ;;
        --overnight)
            PRESET="overnight"
            SPLIT_SEEDS=(0 1 2 3 4)
            MODELS=("LR_EN" "RF")
            shift
            ;;
        --full)
            PRESET="full"
            SPLIT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
            MODELS=("LR_EN" "RF" "XGBoost" "LinSVM_cal")
            shift
            ;;
        --prevalent-fracs)
            IFS=',' read -ra PREVALENT_FRACS <<< "$2"
            shift 2
            ;;
        --case-control-ratios)
            IFS=',' read -ra CASE_CONTROL_RATIOS <<< "$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --split-seeds)
            IFS=',' read -ra SPLIT_SEEDS <<< "$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-splits)
            SKIP_SPLITS=true
            shift
            ;;
        --skip-panel)
            SKIP_PANEL=true
            shift
            ;;
        --force-panel)
            FORCE_PANEL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --walltime)
            WALLTIME="$2"
            shift 2
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect all output to log file AND terminal
exec > >(tee -a "$EXPERIMENT_LOG") 2>&1

##########################################################################################
# Print Header
##########################################################################################

print_header "FACTORIAL EXPERIMENT RUNNER"

print_status "Experiment ID: $EXPERIMENT_ID"
if [ -n "$PRESET" ]; then
    print_info "Running preset: $PRESET"
fi
print_status "Start time: $(date)"
print_status "Log file: $EXPERIMENT_LOG"
echo ""

##########################################################################################
# Pre-flight Checks
##########################################################################################

print_header "PRE-FLIGHT CHECKS"

# Check data file
DATA_FILE="$ANALYSIS_DIR/../data/Celiac_dataset_proteomics_w_demo.parquet"
if [ ! -f "$DATA_FILE" ]; then
    print_error "Data file not found: $DATA_FILE"
    exit 1
fi
print_success "Data file found ($(du -h "$DATA_FILE" | cut -f1))"

# Check environment
cd "$ANALYSIS_DIR"
if ! python -c "import ced_ml; print('OK')" > /dev/null 2>&1; then
    print_error "ced_ml package not installed"
    echo "Run: cd analysis && pip install -e ."
    exit 1
fi
print_success "ced_ml package available"

# Check frozen config
if [ ! -f "$FROZEN_CONFIG" ]; then
    print_error "Frozen config not found: $FROZEN_CONFIG"
    exit 1
fi
print_success "Frozen config found"

# Check panel
if [ -f "$PANEL_FILE" ]; then
    PANEL_SIZE=$(wc -l < "$PANEL_FILE" | tr -d ' ')
    print_success "Panel file found: $PANEL_SIZE proteins"
elif [ "$SKIP_PANEL" = true ]; then
    print_error "Panel file not found and --skip-panel specified"
    exit 1
fi

echo ""

##########################################################################################
# Configuration Summary
##########################################################################################

print_header "EXPERIMENT CONFIGURATION"

echo "Design:"
echo "  Prevalent fractions:    ${PREVALENT_FRACS[@]}"
echo "  Case:control ratios:    ${CASE_CONTROL_RATIOS[@]}"
echo "  Models:                 ${MODELS[@]}"
echo "  Random seeds:           ${SPLIT_SEEDS[@]}"
echo ""

echo "Files:"
echo "  Fixed panel:            $PANEL_FILE"
echo "  Frozen config:          $FROZEN_CONFIG"
echo "  Data file:              $DATA_FILE"
echo ""

# Calculate total configs and runs
TOTAL_CONFIGS=0
TOTAL_RUNS=0
for pf in "${PREVALENT_FRACS[@]}"; do
    for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
        TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
        for seed in "${SPLIT_SEEDS[@]}"; do
            for model in "${MODELS[@]}"; do
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
            done
        done
    done
done

echo "Execution:"
echo "  Total configurations:   $TOTAL_CONFIGS"
echo "  Seeds per config:       ${#SPLIT_SEEDS[@]}"
echo "  Models per seed:        ${#MODELS[@]}"
echo "  Total runs:             $TOTAL_RUNS"
echo ""

echo "HPC Resources (per job):"
echo "  Project:                $PROJECT"
echo "  Queue:                  $QUEUE"
echo "  Walltime:               $WALLTIME"
echo "  Cores:                  $CORES"
echo "  Memory:                 ${MEM}MB"
echo ""

# Estimate duration (parallel execution)
MINUTES_PER_RUN=60  # Conservative estimate for single job
echo "Time estimate (parallel execution):"
echo "  Est. per job:           ~$MINUTES_PER_RUN min"
echo "  Total jobs:             $TOTAL_RUNS (running in parallel)"
echo "  Expected completion:    ~$MINUTES_PER_RUN min (all parallel)"
echo ""

echo "Flags:"
echo "  Skip training:          $SKIP_TRAINING"
echo "  Skip splits:            $SKIP_SPLITS"
echo "  Skip panel:             $SKIP_PANEL"
echo "  Force panel:            $FORCE_PANEL"
echo "  Dry run:                $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - no actual execution"
    echo ""
fi

##########################################################################################
# PHASE 0: Generate Fixed Panel
##########################################################################################

if [ "$SKIP_PANEL" = false ] || [ "$FORCE_PANEL" = true ]; then
    print_header "PHASE 0: Fixed Panel Generation"

    if [ -f "$PANEL_FILE" ] && [ "$FORCE_PANEL" = false ]; then
        print_warning "Panel file exists: $PANEL_FILE ($(wc -l < "$PANEL_FILE" | tr -d ' ') proteins)"
        print_status "Using existing panel (use --force-panel to regenerate)"
        SKIP_PANEL=true
    fi

    if [ "$SKIP_PANEL" = false ] || [ "$FORCE_PANEL" = true ]; then
        print_status "Generating panel (Mann-Whitney screening + k-best selection)..."

        if [ "$DRY_RUN" = false ]; then
            PANEL_LOG="$LOG_DIR/panel_generation_${EXPERIMENT_ID}.log"
            if (cd "$ANALYSIS_DIR" && python docs/investigations/generate_fixed_panel.py \
                --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
                --outfile ../data/fixed_panel.csv \
                --final-k 25) > "$PANEL_LOG" 2>&1; then
                PANEL_SIZE=$(wc -l < "$PANEL_FILE" | tr -d ' ')
                print_success "Panel generated: $PANEL_SIZE proteins"
                print_info "Panel log: $PANEL_LOG"
            else
                print_error "Panel generation FAILED"
                print_error "Check log: $PANEL_LOG"
                exit 1
            fi
        else
            print_status "[DRY RUN] Would generate panel"
        fi
    fi

    echo ""
fi

##########################################################################################
# PHASE 1: Generate Splits for Each Configuration
##########################################################################################

if [ "$SKIP_SPLITS" = false ]; then
    print_header "PHASE 1: Split Generation"

    CONFIG_ID=0
    SPLIT_FAILURES=0

    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_ID=$((CONFIG_ID + 1))
            PROGRESS="[$CONFIG_ID/$TOTAL_CONFIGS]"

            print_status "$PROGRESS Generating splits: prevalent_frac=$pf, case_control=$ccr"

            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"
            mkdir -p "$CONFIG_SPLITS_DIR"

            TEMP_CONFIG="$SCRIPT_DIR/splits_config_experiment_${pf}_${ccr}.yaml"

            cat > "$TEMP_CONFIG" << EOF
mode: development
scenarios:
  - IncidentPlusPrevalent

val_size: 0.25
test_size: 0.25
holdout_size: 0.30

n_splits: ${#SPLIT_SEEDS[@]}
seed_start: ${SPLIT_SEEDS[0]}

train_control_per_case: $ccr
prevalent_sampling_frac: $pf

split_metadata:
  experiment_id: $EXPERIMENT_ID
  prevalent_frac: $pf
  case_control_ratio: $ccr
EOF

            if [ "$DRY_RUN" = false ]; then
                SPLIT_LOG="$LOG_DIR/splits_${pf}_${ccr}_${EXPERIMENT_ID}.log"
                if (cd "$ANALYSIS_DIR" && ced save-splits \
                    --config "$TEMP_CONFIG" \
                    --infile "$DATA_FILE" \
                    --outdir "$CONFIG_SPLITS_DIR") > "$SPLIT_LOG" 2>&1; then
                    print_success "$PROGRESS Splits saved to: ${CONFIG_SPLITS_DIR##*/}"
                else
                    print_error "$PROGRESS Split generation FAILED"
                    SPLIT_FAILURES=$((SPLIT_FAILURES + 1))
                fi
            else
                print_status "[DRY RUN] Would generate splits to: ${CONFIG_SPLITS_DIR##*/}"
            fi

            # Clean up temp config
            rm -f "$TEMP_CONFIG"
        done
    done

    if [ $SPLIT_FAILURES -gt 0 ]; then
        print_error "Split generation failed for $SPLIT_FAILURES configurations"
        exit 1
    fi

    echo ""
fi

##########################################################################################
# PHASE 2: Submit Parallel HPC Training Jobs
##########################################################################################

if [ "$SKIP_TRAINING" = false ]; then
    print_header "PHASE 2: Submit Parallel HPC Training Jobs"

    # Check HPC project allocation
    if [ "$PROJECT" == "YOUR_PROJECT_ALLOCATION" ]; then
        print_error "HPC project not set. Use --project or set PROJECT environment variable"
        exit 1
    fi

    # Check virtual environment
    VENV_PATH="$ANALYSIS_DIR/venv/bin/activate"
    if [ ! -f "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        print_error "Run: cd analysis && bash scripts/hpc_setup.sh"
        exit 1
    fi

    RUN_ID=0
    SUBMITTED_JOBS=()

    for pf in "${PREVALENT_FRACS[@]}"; do
        for ccr in "${CASE_CONTROL_RATIOS[@]}"; do
            CONFIG_SPLITS_DIR="$SPLITS_BASE_DIR/${pf}_${ccr}"

            for seed in "${SPLIT_SEEDS[@]}"; do
                # Check if split files exist (CSV format)
                TRAIN_IDX_FILE="$CONFIG_SPLITS_DIR/train_idx_IncidentPlusPrevalent_seed${seed}.csv"
                if [ ! -f "$TRAIN_IDX_FILE" ]; then
                    print_error "Split files not found for seed $seed in: $CONFIG_SPLITS_DIR"
                    continue
                fi

                for model in "${MODELS[@]}"; do
                    RUN_ID=$((RUN_ID + 1))
                    PROGRESS="[$RUN_ID/$TOTAL_RUNS]"

                    JOB_NAME="FACTORIAL_${model}_pf${pf}_ccr${ccr}_seed${seed}"
                    RESULTS_SUBDIR="$INVEST_RESULTS_DIR/${pf}_${ccr}"
                    mkdir -p "$RESULTS_SUBDIR"

                    if [ "$DRY_RUN" = false ]; then
                        print_status "$PROGRESS Submitting: $model, prevalent=$pf, ccr=$ccr, seed=$seed"

                        LOG_ERR="$LOG_DIR/${JOB_NAME}.%J.err"
                        LIVE_LOG="$LOG_DIR/${JOB_NAME}.%J.live.log"

                        BSUB_OUT=$(bsub \
                            -P "$PROJECT" \
                            -q "$QUEUE" \
                            -J "$JOB_NAME" \
                            -n $CORES \
                            -W "$WALLTIME" \
                            -R "span[hosts=1] rusage[mem=$MEM]" \
                            -oo /dev/null \
                            -eo "$LOG_ERR" \
                            <<EOF
#!/bin/bash
set -euo pipefail

# Force unbuffered output and colors for live logging
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

source "$VENV_PATH"

# Stream to both LSF logs and live log with line buffering
stdbuf -oL -eL ced train \
  --config "$FROZEN_CONFIG" \
  --model "$model" \
  --infile "$DATA_FILE" \
  --split-dir "$CONFIG_SPLITS_DIR" \
  --split-seed "$seed" \
  --scenario IncidentPlusPrevalent \
  --outdir "$RESULTS_SUBDIR" \
  --fixed-panel ../data/fixed_panel.csv \
  2>&1 | tee -a "$LIVE_LOG"

exit \${PIPESTATUS[0]}
EOF
                        )

                        JOB_ID=$(echo "$BSUB_OUT" | grep -oE 'Job <[0-9]+>' | head -n1 | tr -cd '0-9')

                        if [ -n "$JOB_ID" ]; then
                            print_success "$PROGRESS Job submitted: $JOB_ID"
                            SUBMITTED_JOBS+=("$JOB_ID")
                        else
                            print_error "$PROGRESS Submission failed"
                            echo "$BSUB_OUT"
                        fi
                    else
                        print_status "[DRY RUN] Would submit: $model (seed=$seed, pf=$pf, ccr=$ccr)"
                        SUBMITTED_JOBS+=("DRYRUN_${RUN_ID}")
                    fi
                done
            done
        done
    done

    echo ""
    print_status "Submitted ${#SUBMITTED_JOBS[@]} parallel training jobs"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        print_info "Monitor jobs with: bjobs -w | grep FACTORIAL_"
        print_info "Live logs: tail -f $LOG_DIR/FACTORIAL_*.live.log"
        print_info "Error logs: cat $LOG_DIR/FACTORIAL_*.err"
    fi

    echo ""
fi

##########################################################################################
# PHASE 3: Post-Processing Instructions
##########################################################################################

print_header "PHASE 3: Post-Processing Instructions"

if [ "$DRY_RUN" = false ] && [ "$SKIP_TRAINING" = false ]; then
    print_warning "Training jobs submitted to HPC cluster"
    print_warning "Statistical analysis MUST be run AFTER all jobs complete"
    echo ""
    print_info "Monitor job completion with:"
    echo "  bjobs -w | grep FACTORIAL_"
    echo ""
    print_info "When ALL jobs are DONE, run statistical analysis:"
    echo "  python $SCRIPT_DIR/analyze_factorial_results.py \\"
    echo "    --results-dir $INVEST_RESULTS_DIR \\"
    echo "    --output-dir $EXPERIMENT_OUTPUT_DIR \\"
    echo "    --experiment-id $EXPERIMENT_ID"
    echo ""
elif [ "$SKIP_TRAINING" = true ]; then
    print_status "Training skipped - running statistical analysis now..."

    if [ "$DRY_RUN" = false ]; then
        ANALYSIS_LOG="$LOG_DIR/analysis_${EXPERIMENT_ID}.log"
        mkdir -p "$EXPERIMENT_OUTPUT_DIR"

        if python "$SCRIPT_DIR/analyze_factorial_results.py" \
            --results-dir "$INVEST_RESULTS_DIR" \
            --output-dir "$EXPERIMENT_OUTPUT_DIR" \
            --experiment-id "$EXPERIMENT_ID" > "$ANALYSIS_LOG" 2>&1; then
            print_success "Statistical analysis complete"
            print_info "Results saved to: ${EXPERIMENT_OUTPUT_DIR##*/}"
        else
            print_error "Statistical analysis FAILED"
            print_error "Check log: $ANALYSIS_LOG"
        fi
    else
        print_status "[DRY RUN] Would run statistical analysis"
    fi
fi

echo ""

##########################################################################################
# Summary
##########################################################################################

print_header "JOB SUBMISSION COMPLETE"

END_TIME=$(date)
print_status "End time: $END_TIME"

if [ "$DRY_RUN" = false ]; then
    echo ""
    print_info "Experiment ID: $EXPERIMENT_ID"
    print_info "Full log: $EXPERIMENT_LOG"
    echo ""

    if [ "$SKIP_TRAINING" = false ]; then
        print_info "Submitted ${#SUBMITTED_JOBS[@]} parallel training jobs to HPC"
        echo ""
        print_status "Next steps:"
        echo "  1. Monitor jobs: bjobs -w | grep FACTORIAL_"
        echo "  2. Check live logs: tail -f $LOG_DIR/FACTORIAL_*.live.log"
        echo "  3. When all jobs complete, run statistical analysis:"
        echo "     python $SCRIPT_DIR/analyze_factorial_results.py \\"
        echo "       --results-dir $INVEST_RESULTS_DIR \\"
        echo "       --output-dir $EXPERIMENT_OUTPUT_DIR \\"
        echo "       --experiment-id $EXPERIMENT_ID"
    fi

    if [ -f "$EXPERIMENT_OUTPUT_DIR/summary.md" ]; then
        echo ""
        print_status "Quick findings preview:"
        echo ""
        head -n 60 "$EXPERIMENT_OUTPUT_DIR/summary.md"
        echo ""
        print_info "Full summary: $EXPERIMENT_OUTPUT_DIR/summary.md"
    fi

    if [ -d "$EXPERIMENT_OUTPUT_DIR" ]; then
        echo ""
        print_status "Generated files:"
        ls -lh "$EXPERIMENT_OUTPUT_DIR"/*.{csv,md,json} 2>/dev/null || true
    fi
else
    print_info "Dry run complete - no jobs submitted"
fi

echo ""
print_success "Script completed successfully"
exit 0
