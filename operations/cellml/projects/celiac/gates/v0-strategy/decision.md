---
gate: v0-strategy
project: celiac
observed_at: "TODO: populate with UTC timestamp when V0 reduction is finalized"
author: retrospective
---

# V0 Decision (retrospective reconstruction)

Reconstructed from ADR-002, ADR-003, and `MASTER_PLAN.md`. The actual lock
was made during Gen 1 / pre-CellML planning; V0 was structured to CONFIRM
the Gen 1 operating point rather than discover it.

## Predictions that held

Inferred from the decision to lock incident-only with control_ratio=5 per
ADR-002 and ADR-003:

- **P1** IncidentOnly > {IncidentPlusPrevalent, PrevalentOnly} on AUROC.
  Held under Direction (retrospectively inferred). ADR-002 records that
  prevalent cases are restricted to TRAIN at 50% sampling and VAL/TEST
  remain incident-only to preserve prospective screening semantics.
- **P2** control_ratio=5 delivers Equivalence-or-better vs control_ratio=2
  with ~60x compute savings. Held under Equivalence (retrospectively
  inferred). ADR-003 explicitly argues 1:5 preserves discrimination while
  1:2 is ~2x slower for similar AUROC.
- **P3** The joint winner (IncidentOnly, control_ratio=5) is stable across
  LR_EN, LinSVM_cal, RF, XGBoost. TODO: confirm by reading the actual
  per-model breakdown in `results/<v0-namespace>/` once identified.

## Predictions that failed

None identified at retrospective time. This is a known weakness of
reconstructing decisions post hoc — only claims that WERE true are visible
in the locked configuration; claims that failed leave weaker traces unless
logged explicitly at decision time.

TODO: if `results/<v0-namespace>/` contains per-cell metrics that contradict
P1-P3, add them here rather than to the ledger.

## Actual claim type (per rubric)

**Dominance** — the joint `(IncidentOnly, control_ratio=5)` winner is
reported to hold across all 4 models, implying the Direction criterion
holds independently on each model axis. Source of the claim: `MASTER_PLAN.md`
"V0: Training Strategy Gate" decision rule ("If the strategy+control winner
is the same across all 4 models, lock both"). The Dominance claim has NOT
been re-verified against the actual bootstrap CIs in this retrospective
pass — flagging as a data-computation TODO below.

## Locks passed forward

| Axis | Locked value | Source ADR | Carried into |
|---|---|---|---|
| `training_strategy` | IncidentOnly | ADR-002 | V1 (structural prior; not a factor) |
| `prevalent_train_frac` | N/A (incident-only locks prevalent handling out) | ADR-002 | V1 (no longer variable) |
| `train_control_per_case` | 5 | ADR-003 | V1 (structural prior; V3 downsampling axis tests 1.0/2.0/5.0 as a training-dynamics factor, separate from the control-ratio that generates the splits) |
| Split scheme | 50/25/25 stratified | ADR-001 | All downstream gates (not revisited) |

Note on the control-ratio carrying: ADR-003 sets the split-generation
control ratio to 5. `MASTER_PLAN.md` notes that V5 (pre-restructure) was
supposed to revisit 1.0/2.0/5.0; the April 2026 restructure (per
DECISION_TREE_AUDIT) folded weighting × downsampling into a joint V3 grid,
so the downsampling levels tested at V3 now operate on top of the locked
5:1 split generation. This is a structural subtlety worth carrying into
the V1 protocol document.

## Predictions for v1-recipe

At the V1 factorial under locked strategy, the following predictions are
passed forward (to be formalized in `gates/v1-recipe/ledger.md`):

- R1 (consensus ordering) will not dominate R2 (stream-balanced ordering)
  under any Direction criterion on AUROC — Gen 1 observed ~5% variance
  from ordering (per `MASTER_PLAN.md` Gen 1 artifacts table), consistent
  with Equivalence.
- Model-specific recipes (MS_*) at their plateau sizes will not dominate
  shared recipes (R1/R2) at equivalent sizes — Direction criterion will
  fail, and parsimony toward shared (4p sig or plateau) will apply.
- The 4-protein BH core (tgm2, cpa2, itgb7, gip) remains in the winning
  panel under any V1 partition — it is robust to partition changes per
  `MASTER_PLAN.md` Gen 1 caveat table.

These V1 predictions are documentation; they live in the V1 ledger once
that gate opens.
