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

- Axes evaluated (per rb-v0.2.0 `protocols/v0-strategy.md`): training
  strategy {IncidentOnly, IncidentPlusPrevalent, PrevalentOnly} ×
  prevalent_frac {0.25, 0.5, 1.0} × imbalance_probe {none, downsample_5,
  cw_log} × model {LR_EN, LinSVM_cal, RF, XGBoost} × recipe {R1_sig (4p),
  R1_plateau (8p)}
- Cells per `MASTER_PLAN.md`: 120 cells total (5 strategies × 3 imbalance
  probes × 4 models × 2 recipes). 20 selection seeds per cell (100-119).
  Note: the total cell count is unchanged from rb-v0.1.0 (same 120); the
  factor decomposition changed from `control_ratio ∈ {1, 2, 5}` (numeric
  levels within the downsample family) to `imbalance_probe ∈ {none,
  downsample_5, cw_log}` (categorical probes across 3 families).
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
point the V0 gate revisits. Under rb-v0.2.0 this operating point maps to
`(IncidentOnly, imbalance_probe = downsample_5)` — i.e., the downsample
family probed at the 5:1 representative level. TODO: confirm whether the
V0 run reproduced this baseline within Direction tolerance at the same
probe point.

## Aggregation

TODO: per-strategy summary (marginal over imbalance_probe, model, recipe).

TODO: per-(strategy, imbalance_probe) joint summary (marginal over model,
recipe) — this is the table the Dominance claim is evaluated against under
rb-v0.2.0. The family-level Direction test compares the three probe
representatives (`none`, `downsample_5`, `cw_log`) as family proxies. This
replaces the rb-v0.1.0 per-(strategy, control_ratio) joint summary.

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
- Per rb-v0.2.0, existing Gen 1 artifacts were produced under a
  `control_ratio = 5` (numeric level) framing. They map cleanly to the
  `downsample_5` probe under the new family-level framing, but artifacts
  at other probe levels (`none`, `cw_log`) may or may not have been
  produced by the V0 submission. TODO: verify the V0 submission included
  all three probe levels once the namespace is identified.
