# Gen 1 Archive

Frozen experiments from the sequential optimization phase (Feb-Apr 2026). These produced the initial decisions that the Gen 2 factorial either confirms or revises.

**Do not run these scripts.** Many have hardcoded HPC paths, stale config references, and panel path mismatches. They are preserved for provenance and scientific rationale only.

## What's here

| Directory | What it decided | Key artifact |
|---|---|---|
| `panel-selection/` | 4-protein BH core (tgm2, cpa2, itgb7, gip) | `results/run_20260317_131842/consensus/` |
| `panel-sweep/` | 8-10p operating range, pathway order, LinSVM_cal | `results/compiled_results_aggregated.csv` |
| `training-strategy/` | Incident-only + log weights | `results/incident_validation/` |
| `holdout-confirmation/` | Gen 1 holdout validation | `results/run_phase3_holdout/` |
| `model-gate/` | (never run — superseded by factorial V2) | N/A |
| `svm-validation/` | (never run — superseded by factorial MS_* recipes) | N/A |
| `supporting/` | Sensitivity analyses (2x2x2, consensus, permutation, methodology) | Various |
| `narrative/` | Gen 1 figure scripts (3 figs) | `narrative/out/` |
| `claude_viz/` | Visualization scripts | Not generated |
| `codex_viz/` | Publication figure specs | Not generated |
| `analysis-narrative.md` | Historical decision chain narrative | N/A |
| `results.md` | Gen 1 artifact map | N/A |

## Relationship to Gen 2

Gen 1 produced the inputs that Gen 2 consumes:
- Trunk T1 (consensus): `rra_significance_corrected.csv` from panel-selection
- Trunk T2 (incident): `feature_consistency.csv` from training-strategy
- Sweep data: `compiled_results_aggregated.csv` from panel-sweep
- OOF/RFE importance files: from `results/run_20260217_194153/`

These artifacts live in `results/` (not archived) and are referenced by `analysis/configs/manifest.yaml`.
