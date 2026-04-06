# Panel Sweep Results

## Outcome

- The sweep is the main configuration-optimization experiment.
- Accepted decision: pathway order, `LinSVM_cal`, and an operating panel around 10 proteins.
- The 4-protein BH-corrected core remains the truth anchor; the sweep determines how much to extend that core for the final predictive setup.

## Artifacts

- Sweep manifests and configs in this directory.
- Decision report: `sweep-analysis.md`
- Result artifacts: `results/experiments/`

## Decision role

This experiment decides panel size, protein addition order, and preferred model class.
