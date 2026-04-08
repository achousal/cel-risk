#!/bin/bash
# Submit Phase 2 (validation) and/or Phase 3 (holdout) for 4-protein panel
# Usage:
#   bash submit_4protein_phases.sh           # all phases (2 -> 3 -> holdout eval)
#   bash submit_4protein_phases.sh --phase 2 # Phase 2 only
#   bash submit_4protein_phases.sh --phase 3 # Phase 3 + holdout eval only

set -euo pipefail

# --- Parse args ---
PHASE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$PHASE" != "all" && "$PHASE" != "2" && "$PHASE" != "3" ]]; then
    echo "Error: --phase must be 2, 3, or omitted (all)"
    exit 1
fi

# --- Config ---
PROJECT="acc_vascbrain"
QUEUE="premium"
WALLTIME="48:00"
CORES=4
MEM=8000

cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis

ACTIVATE="module load anaconda3/2024.06 && source venv/bin/activate"

MODELS=("LR_EN" "LinSVM_cal" "RF" "XGBoost")
SEEDS=(200 201 202 203 204 205 206 207 208 209)

P2_RUN_ID="phase2_val_4protein"
P2_CONFIG="configs/pipeline_hpc_val_4protein.yaml"
P2_JOB_NAME="CeD_p2_4prot"

P3_RUN_ID="phase3_holdout_4protein"
P3_CONFIG="configs/pipeline_hpc_holdout_4protein.yaml"
P3_JOB_NAME="CeD_p3_4prot"

# ======================================================================
# Phase 2: Validation (4-protein panel, 10 seeds x 4 models)
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "2" ]]; then
    echo "=== Phase 2: 4-protein validation ==="

    P2_CMD="ced run-pipeline --pipeline-config ${P2_CONFIG} --run-id ${P2_RUN_ID}"

    P2_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W ${WALLTIME} \
        -n ${CORES} -R "rusage[mem=${MEM}] span[hosts=1]" \
        -J "${P2_JOB_NAME}" \
        -oo "../logs/${P2_RUN_ID}_%J.stdout" \
        -eo "../logs/${P2_RUN_ID}_%J.stderr" \
        bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && ${ACTIVATE} && ${P2_CMD}" 2>&1 | grep -oP 'Job <\K\d+')

    echo "SUBMITTED: ${P2_JOB_NAME} -> Job ${P2_JOB}"
fi

# ======================================================================
# Phase 3: Holdout training (4-protein panel)
# ======================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "3" ]]; then
    echo ""
    echo "=== Phase 3: 4-protein holdout training ==="

    P3_CMD="ced run-pipeline --pipeline-config ${P3_CONFIG} --run-id ${P3_RUN_ID}"

    # If running all phases, depend on Phase 2; otherwise submit independently
    DEP_FLAG=""
    if [[ "$PHASE" == "all" ]]; then
        DEP_FLAG="-w done(${P2_JOB_NAME})"
    fi

    P3_JOB=$(bsub -P ${PROJECT} -q ${QUEUE} -W ${WALLTIME} \
        -n ${CORES} -R "rusage[mem=${MEM}] span[hosts=1]" \
        -J "${P3_JOB_NAME}" \
        ${DEP_FLAG} \
        -oo "../logs/${P3_RUN_ID}_%J.stdout" \
        -eo "../logs/${P3_RUN_ID}_%J.stderr" \
        bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && ${ACTIVATE} && ${P3_CMD}" 2>&1 | grep -oP 'Job <\K\d+')

    if [[ "$PHASE" == "all" ]]; then
        echo "SUBMITTED: ${P3_JOB_NAME} -> Job ${P3_JOB} (depends on Phase 2)"
    else
        echo "SUBMITTED: ${P3_JOB_NAME} -> Job ${P3_JOB}"
    fi

    # --- Holdout evaluation (depends on Phase 3 training) ---
    echo ""
    echo "=== Phase 3: holdout evaluation ==="

    INFILE="../data/Celiac_dataset_proteomics_w_demo.parquet"
    HOLDOUT_IDX="../splits/HOLDOUT_idx_IncidentPlusPrevalent.csv"
    ALL_MODELS=("LR_EN" "LinSVM_cal" "RF" "XGBoost" "ENSEMBLE")

    for model in "${ALL_MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ARTIFACT="../results/run_${P3_RUN_ID}/${model}/splits/split_seed${seed}/core/${model}__final_model.joblib"
            OUTDIR="../results/run_${P3_RUN_ID}/${model}/holdout/split_seed${seed}"

            JOB_NAME="CeD_ho_4p_${model}_s${seed}"
            CMD="ced eval-holdout --infile ${INFILE} --holdout-idx ${HOLDOUT_IDX} --model-artifact ${ARTIFACT} --outdir ${OUTDIR} --compute-dca"

            bsub -P ${PROJECT} -q ${QUEUE} -W "02:00" \
                -n 4 -R "rusage[mem=8000] span[hosts=1]" \
                -J "${JOB_NAME}" \
                -w "done(${P3_JOB_NAME})" \
                -oo "../logs/holdout_4p_${model}_s${seed}_%J.stdout" \
                -eo "../logs/holdout_4p_${model}_s${seed}_%J.stderr" \
                bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && ${ACTIVATE} && ${CMD}"

            echo "SUBMITTED: ${JOB_NAME} (depends on Phase 3 training)"
        done
    done
fi

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: bjobs -w | grep CeD_"
