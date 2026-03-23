#!/bin/bash
# Submit permutation tests for Phase 2 validation run
# Run ID: phase2_val_consensus
# Models: LR_EN, LinSVM_cal, RF, XGBoost
# Seeds: 200-209 (10 splits)
# Perms: 300 per seed

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
WALLTIME="24:00"
CORES=12
MEM=8000
RUN_ID="phase2_val_consensus"
N_PERMS=300
N_JOBS=-1  # use all cores
RANDOM_STATE=42

cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis

# Activate environment
module load anaconda3/2024.06
source /sc/arion/projects/vascbrain/andres/cel-risk/analysis/venv/bin/activate

MODELS=("LR_EN" "LinSVM_cal" "RF" "XGBoost")
SEEDS=(200 201 202 203 204 205 206 207 208 209)

echo "=== Submitting permutation tests for ${RUN_ID} ==="
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Perms per seed: ${N_PERMS}"
echo ""

# Track job names for aggregation dependencies
declare -A MODEL_PERM_JOBS

for model in "${MODELS[@]}"; do
    perm_job_names=()
    for seed in "${SEEDS[@]}"; do
        JOB_NAME="CeD_perm_${model}_s${seed}"

        # Check if this seed already has results (idempotent)
        SIG_DIR="../results/run_${RUN_ID}/${model}/significance"
        if [ -f "${SIG_DIR}/null_distribution_seed${seed}.csv" ]; then
            echo "SKIP: ${JOB_NAME} -- already completed"
            continue
        fi

        CMD="ced permutation-test --run-id ${RUN_ID} --model ${model} --split-seed-start ${seed} --n-split-seeds 1 --n-perms ${N_PERMS} --n-jobs ${N_JOBS} --random-state ${RANDOM_STATE}"

        JOB_ID=$(bsub -P ${PROJECT} -q ${QUEUE} -W ${WALLTIME} \
            -n ${CORES} -R "rusage[mem=${MEM}] span[hosts=1]" \
            -J "${JOB_NAME}" \
            -oo "../logs/perm_${model}_s${seed}_%J.stdout" \
            -eo "../logs/perm_${model}_s${seed}_%J.stderr" \
            bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && module load anaconda3/2024.06 && source venv/bin/activate && ${CMD}" 2>&1 | grep -oP 'Job <\K\d+')

        echo "SUBMITTED: ${JOB_NAME} -> Job ${JOB_ID}"
        perm_job_names+=("${JOB_NAME}")
    done

    # Submit aggregation job dependent on all perm jobs for this model
    if [ ${#perm_job_names[@]} -gt 0 ]; then
        DEP_EXPR=$(printf "done(%s) && " "${perm_job_names[@]}")
        DEP_EXPR="${DEP_EXPR% && }"  # remove trailing &&

        AGG_NAME="CeD_perm_${model}_agg"
        AGG_CMD="ced permutation-test --run-id ${RUN_ID} --model ${model} --aggregate-only"

        AGG_ID=$(bsub -P ${PROJECT} -q ${QUEUE} -W "01:00" \
            -n 1 -R "rusage[mem=4000]" \
            -J "${AGG_NAME}" \
            -w "${DEP_EXPR}" \
            -oo "../logs/perm_${model}_agg_%J.stdout" \
            -eo "../logs/perm_${model}_agg_%J.stderr" \
            bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && module load anaconda3/2024.06 && source venv/bin/activate && ${AGG_CMD}" 2>&1 | grep -oP 'Job <\K\d+')

        echo "SUBMITTED: ${AGG_NAME} -> Job ${AGG_ID} (depends on ${#perm_job_names[@]} perm jobs)"
    fi
    echo ""
done

echo "=== All permutation jobs submitted ==="
echo "Monitor with: bjobs -w | grep CeD_perm"
