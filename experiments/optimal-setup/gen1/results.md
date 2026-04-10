# Optimal Setup Results

## Current synthesis

- Core truth panel: 4 proteins after conservative BH correction over the full 2920-protein universe: `tgm2`, `cpa2`, `itgb7`, `gip`.
- Configuration optimization: the saturation sweep supports a broader operating panel around 8-10 proteins, with pathway ordering and `LinSVM_cal` as the preferred operating point.
- Training strategy: incident-only validation remains the decision point for cohort construction and class-weighting choices.
- Final gate: holdout evaluation remains the endpoint for accepting the locked setup.

## Artifact map

- Phase 1 / panel discovery: `results/run_20260217_194153/`, `results/run_20260317_131842/`
- Sweep outputs: `results/experiments/`
- Incident strategy validation: `results/incident_validation/`
- Consensus validation: `results/run_phase2_val_consensus/`, `results/run_phase2_val_consensus_100t/`
- 4-protein comparison: `results/run_phase2_val_4protein/`
- Holdout confirmation: `results/run_phase3_holdout/`, `results/run_phase3_holdout_4protein/`

See each subdirectory `results.md` for the local decision record.
