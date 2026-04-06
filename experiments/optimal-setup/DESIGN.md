# Optimal Setup

## Goal

Converge on the final celiac disease proteomics analysis setup by separating:

1. Cornerstone analyses that changed the chosen configuration.
2. Supporting analyses that validated or stress-tested those decisions.

## Cornerstone analyses

- `panel-selection/`: establish the core truth set of proteins.
- `panel-sweep/`: optimize panel size, ordering strategy, and model choice.
- `training-strategy/`: determine who to train on and how to weight classes.
- `holdout-confirmation/`: confirm the locked setup on holdout data.

## Supporting analyses

- `supporting/methodology-audit/`: empirical checks for consensus weighting, stacking calibration, and prevalence adjustment.
- `supporting/factorial/`: interaction study for sample size, imbalance, and prevalent-case inclusion.
- `supporting/4protein-comparison/`: head-to-head confirmation of the locked 4-protein core against the broader panel.
- `supporting/phase2-consensus/`: fresh-seed validation of the consensus panel.
- `supporting/permutation-testing/`: statistical significance follow-up.

## Decision flow

`panel-selection` -> `panel-sweep` -> `training-strategy` -> `holdout-confirmation`

Supporting studies feed into interpretation, not the main decision chain.
