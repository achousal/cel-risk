---
gate: v0-strategy
project: celiac
observed_at: "TODO: populate from results/<namespace>/run_metadata.json once V0 namespace is identified"
---

# V0 Observation (retrospective; facts only)

No reasoning. Metrics pulled from documented sources in `MASTER_PLAN.md` and
`DESIGN.md`. Where specific numbers are not in those docs, a TODO points to
the results artifact that would produce them.

## Gate identity

- Axes evaluated: training strategy {IncidentOnly, IncidentPlusPrevalent,
  PrevalentOnly} × prevalent_frac {0.25, 0.5, 1.0} × control_ratio {1.0, 2.0,
  5.0} × model {LR_EN, LinSVM_cal, RF, XGBoost} × recipe {R1_sig (4p),
  R1_plateau (8p)}
- Cells per `MASTER_PLAN.md`: 120 cells total (5 strategies × 3 control ratios
  × 4 models × 2 recipes). 20 selection seeds per cell (100-119).
- Submission metadata per `MASTER_PLAN.md`: job IDs 237012328-237012447,
  status as of 2026-04-12 "V0 gate submitted. Vanillamax discovery (10 seeds)
  running."

## Per-cell metrics

TODO: pull from `results/<v0-namespace>/metrics/*.json`. Expected per cell:

- AUROC (mean and SE over 20 seeds)
- PR-AUC / AUPRC (mean and SE)
- Brier score + reliability decomposition (mean and SE)
- Wall-clock runtime
- Bootstrap CIs (1000 resamples over outer folds) per SCHEMA metric-specific
  rule

## Anchor points documented elsewhere

Per `MASTER_PLAN.md` Gen 1 inputs table: Gen 1 incident-only + log weights
baseline reported **AUPRC 0.215, AUROC 0.867**. This is the Gen 1 operating
point the V0 gate revisits. TODO: confirm whether the V0 run reproduced this
baseline within Direction tolerance.

## Aggregation

TODO: per-strategy summary (marginal over control_ratio, model, recipe).

TODO: per-(strategy, control_ratio) joint summary (marginal over model,
recipe) — this is the table the Dominance claim is evaluated against.

TODO: per-model breakdown to test Dominance criterion (Direction must hold
independently on each model axis).

## Known gaps in this retrospective record

- V0 namespace in `results/` is not unambiguously identified. Candidate
  directories under `results/` include `run_phase3_holdout_incident_only_ds5/`
  and `run_phase2_val_incident_only_ds5/` but these look like Gen 1 holdout
  artifacts, not the full V0 factorial. The V0 factorial may write to
  `results/cellml/v0_gate/` once `scripts/submit_experiment.sh --experiment v0_gate`
  completes compilation.
- Compilation tool: `scripts/compile_factorial.py --optuna-storage-dir <dir>`. TODO:
  run this once the V0 namespace is identified and paste the compiled table
  here (or link to `results/<v0-namespace>/factorial_compiled.csv`).
- Per `SCHEMA.md`, observation.md is normally emitted by the `cellml-reduce`
  CLI. That tool has not yet been implemented; this file is a hand-written
  stub with the expected structure.
