#!/bin/bash
# Evaluate trained models on holdout set
# Run after Phase 3 training completes
# Usage: bash scripts/submit_holdout_eval.sh <RUN_ID>

set -euo pipefail

RUN_ID="${1:?Usage: submit_holdout_eval.sh <RUN_ID>}"
PROJECT="acc_vascbrain"
QUEUE="premium"

cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis

MODELS=("LR_EN" "LinSVM_cal" "RF" "XGBoost" "ENSEMBLE")
SEEDS=(200 201 202 203 204 205 206 207 208 209)
INFILE="../data/Celiac_dataset_proteomics_w_demo.parquet"
HOLDOUT_IDX="../splits/holdout_indices_IncidentPlusPrevalent.csv"

echo "=== Submitting holdout evaluation for ${RUN_ID} ==="

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ARTIFACT="../results/run_${RUN_ID}/${model}/splits/split_seed${seed}/core/${model}__final_model.joblib"
        OUTDIR="../results/run_${RUN_ID}/${model}/holdout/split_seed${seed}"

        # Check artifact exists
        if [ ! -f "${ARTIFACT}" ]; then
            echo "SKIP: ${model} seed ${seed} -- no model artifact"
            continue
        fi

        JOB_NAME="CeD_holdout_${model}_s${seed}"
        CMD="ced eval-holdout --infile ${INFILE} --holdout-idx ${HOLDOUT_IDX} --model-artifact ${ARTIFACT} --outdir ${OUTDIR} --compute-dca"

        bsub -P ${PROJECT} -q ${QUEUE} -W "02:00" \
            -n 4 -R "rusage[mem=8000] span[hosts=1]" \
            -J "${JOB_NAME}" \
            -oo "../logs/holdout_${model}_s${seed}_%J.stdout" \
            -eo "../logs/holdout_${model}_s${seed}_%J.stderr" \
            bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && module load anaconda3/2024.06 && source venv/bin/activate && ${CMD}"

        echo "SUBMITTED: ${JOB_NAME}"
    done
done

echo ""
echo "=== All holdout eval jobs submitted ==="
echo "Monitor with: bjobs -w | grep CeD_holdout"
