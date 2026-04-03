# 4-Protein Panel Validation Experiment

## Motivation

RRA universe sensitivity analysis (1M permutations, BH correction at N=2920)
shows only 4/7 proteins survive the most conservative correction:
**tgm2, cpa2, itgb7, gip**.

This experiment runs Phase 2 validation with the 4-protein locked panel
to compare head-to-head against the 7-protein results.

## Question

Does the 7-protein panel outperform the 4-protein panel, or do
cxcl9/cd160/muc2 add noise rather than signal?

## Design

- Same seeds as Phase 2 (200-209)
- Same 4 models + ENSEMBLE
- Only the panel differs: 4 proteins vs 7
- Comparison metrics: AUROC, PR-AUC, Brier, calibration slope

## Commands (run on Minerva)

```bash
# 1. Sync code changes to Minerva (the significance.py + CLI changes)
# From local:
rsync -avz analysis/src/ced_ml/ minerva:/path/to/cel-risk/analysis/src/ced_ml/
rsync -avz analysis/configs/*4protein* minerva:/path/to/cel-risk/analysis/configs/

# 2. On Minerva, generate splits if needed (same seeds as Phase 2)
cd /sc/arion/projects/vascbrain/andres/cel-risk/analysis
ced save-splits --config configs/splits_config_val_consensus.yaml

# 3. Run the 4-protein validation pipeline
ced run-pipeline \
    --pipeline-config configs/pipeline_hpc_val_4protein.yaml \
    --run-id phase2_val_4protein

# 4. After completion, generate comparison plots
ced aggregate-splits --run-id phase2_val_4protein --model LR_EN
ced aggregate-splits --run-id phase2_val_4protein --model LinSVM_cal
ced aggregate-splits --run-id phase2_val_4protein --model RF
ced aggregate-splits --run-id phase2_val_4protein --model XGBoost
ced train-ensemble --run-id phase2_val_4protein
ced aggregate-splits --run-id phase2_val_4protein --model ENSEMBLE

# 5. Sync results back to local
# From local:
rsync -avz minerva:/path/to/cel-risk/results/run_phase2_val_4protein/ \
    results/run_phase2_val_4protein/
```

## Expected output

`results/run_phase2_val_4protein/` with same structure as `run_phase2_val_consensus/`

## Comparison analysis

After syncing results, compare:
```
Phase 2 (7-protein): results/run_phase2_val_consensus/
Phase 2 (4-protein): results/run_phase2_val_4protein/
```

Key question: is the AUROC delta (7-protein minus 4-protein) > 0 and
significant, or within CI overlap?

## References

- Bourgon et al. 2010 PNAS 107:9546 (independent filtering)
- Zehetmayer & Posch 2012 BMC Bioinformatics 13:81 (two-stage FDR)
- experiments/rra_universe_sensitivity.py (sensitivity analysis)
