# Holdout Confirmation

## Goal

Evaluate the locked configuration on the holdout split without reopening panel-selection decisions.

## Scope

This stage is the final acceptance gate after:

1. Core panel selection.
2. Sweep-based configuration choice.
3. Training-strategy selection.

## Execution assets

- `configs/pipeline_hpc_holdout.yaml`
- `configs/training_config_holdout.yaml`
- `configs/splits_config_holdout.yaml`
- `configs/holdout_config.yaml`
- `scripts/eval_ensemble_holdout.py`
- `scripts/submit_holdout_eval.sh`
