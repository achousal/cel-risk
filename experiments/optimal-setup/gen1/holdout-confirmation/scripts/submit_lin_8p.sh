#!/bin/bash
# Submit lin-goal 8p holdout evaluation on Minerva
# Single model (LinSVM_cal), 10 seeds, fixed 8-protein panel
#
# Usage:
#   bash experiments/optimal-setup/holdout-confirmation/scripts/submit_lin_8p.sh
#
# Expected wall time: ~50 min per seed (50 Optuna trials x ~1 min)
# Total: 10 seeds x 1 model = 10 jobs

set -euo pipefail

PROJECT="acc_vascbrain"
QUEUE="premium"
RUN_ID="lin_8p_holdout"
PIPELINE_CONFIG="experiments/optimal-setup/holdout-confirmation/configs/pipeline_lin_8p.yaml"

cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis

SEEDS=(200 201 202 203 204 205 206 207 208 209)

echo "=== Submitting lin-goal 8p holdout: LinSVM_cal, 10 seeds ==="
echo "Pipeline config: ${PIPELINE_CONFIG}"
echo "Run ID: ${RUN_ID}"
echo ""

for seed in "${SEEDS[@]}"; do
    JOB_NAME="CeD_lin8p_s${seed}"

    bsub -P ${PROJECT} -q ${QUEUE} -W "48:00" \
        -n 4 -R "rusage[mem=8000] span[hosts=1]" \
        -J "${JOB_NAME}" \
        -oo "../logs/${RUN_ID}_s${seed}_%J.stdout" \
        -eo "../logs/${RUN_ID}_s${seed}_%J.stderr" \
        bash -c "cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis && \
                 module load anaconda3/2024.06 && \
                 source venv/bin/activate && \
                 ced train --config ${PIPELINE_CONFIG} \
                     --model LinSVM_cal \
                     --seed ${seed} \
                     --run-id ${RUN_ID}"

    echo "SUBMITTED: ${JOB_NAME}"
done

echo ""
echo "=== All 10 jobs submitted ==="
echo "Monitor: bjobs -w | grep CeD_lin8p"
echo "Results: ../results/run_${RUN_ID}/LinSVM_cal/"
echo ""
echo "After completion, run holdout eval:"
echo "  bash experiments/optimal-setup/holdout-confirmation/scripts/submit_holdout_eval.sh ${RUN_ID}"
