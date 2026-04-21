---
gate: v0-strategy
project: celiac
observed_at: "TODO: populate with UTC timestamp when V0 reduction is finalized"
author: retrospective
---

# V0 Decision (retrospective reconstruction)

Reconstructed from ADR-002, ADR-003, and `MASTER_PLAN.md`. The actual lock
was made during Gen 1 / pre-CellML planning; V0 was structured to CONFIRM
the Gen 1 operating point rather than discover it. This decision has been
re-interpreted under the rb-v0.2.0 axis redefinition — the Gen 1
control_ratio=5 value is now read as family-level evidence for
`imbalance_family = downsample` at V0, with level refinement deferred to V3.

## Predictions that held

Inferred from the decision to lock incident-only with
`imbalance_family = downsample` (Gen 1 operated at the `downsample_5` probe
point) per ADR-002 and ADR-003:

- **P1** IncidentOnly > {IncidentPlusPrevalent, PrevalentOnly} on AUROC.
  Held under Direction (retrospectively inferred). ADR-002 records that
  prevalent cases are restricted to TRAIN at 50% sampling and VAL/TEST
  remain incident-only to preserve prospective screening semantics.
- **P2** `imbalance_family = downsample` delivers Direction-or-Equivalence
  over `none` and `weight` family representatives on AUROC. Held under
  Direction vs `none` and Equivalence-or-better vs `cw_log`
  (retrospectively inferred). The Gen 1 empirical evidence was gathered
  at `control_ratio = 5` (now re-interpreted as the `downsample_5` probe
  within the downsample family); level-specific claims (ratio 5 vs 2 vs 1)
  are NOT adjudicated at V0 under rb-v0.2.0 — they are V3's scope.
- **P3** The joint winner (IncidentOnly, `imbalance_family = downsample`)
  is stable across LR_EN, LinSVM_cal, RF, XGBoost. TODO: confirm by
  reading the actual per-model breakdown in `results/<v0-namespace>/` once
  identified.

## Predictions that failed

None identified at retrospective time. This is a known weakness of
reconstructing decisions post hoc — only claims that WERE true are visible
in the locked configuration; claims that failed leave weaker traces unless
logged explicitly at decision time.

TODO: if `results/<v0-namespace>/` contains per-cell metrics that contradict
P1-P3, add them here rather than to the ledger.

## Actual claim type (per rubric)

**Dominance** — the joint `(IncidentOnly, imbalance_family = downsample)`
winner is reported to hold across all 4 models, implying the Direction
criterion holds independently on each model axis. Source of the claim:
`MASTER_PLAN.md` "V0: Training Strategy Gate" decision rule ("If the
strategy+imbalance winner is the same across all 4 models, lock both").
The Dominance claim has NOT been re-verified against the actual bootstrap
CIs in this retrospective pass — flagging as a data-computation TODO below.

## Locks passed forward

| Axis | Locked value | Source ADR | Carried into |
|---|---|---|---|
| `training_strategy` | IncidentOnly | ADR-002 | V1 (structural prior; not a factor) |
| `prevalent_train_frac` | N/A (incident-only locks prevalent handling out) | ADR-002 | V1 (no longer variable) |
| `imbalance_family` | downsample (retrospectively inferred from Gen 1 baseline; Gen 1 operated at the `downsample_5` probe point) | ADR-003 (re-interpreted under rb-v0.2.0) | V3 (V3 refines within-family level; V1 inherits family lock as a structural prior) |
| Split scheme | 50/25/25 stratified | ADR-001 | All downstream gates (not revisited) |

> **Note on the within-family level.** ADR-003's `control_ratio = 5` value
> from Gen 1 is now treated as the PROBED level at V0 (i.e., the
> `downsample_5` probe representing the downsample family), NOT a final
> lock on the numeric ratio. V3 refines the level within the downsample
> family (e.g., 1 vs 2 vs 5) and carries the refined level forward; V0's
> output is the FAMILY, not the level. This separation — family at V0,
> level at V3 — is the rb-v0.2.0 two-stage decision per the new
> [[condensates/imbalance-two-stage-decision]] condensate.

Note on the imbalance-family carrying: the rb-v0.2.0 rewrite replaces the
multiplicative-composition reading of rb-v0.1.0 (where V0's
`control_ratio` and V3's `downsampling` axis nested multiplicatively). V0
now probes the FAMILY at a representative level; V3 supersedes the level
once the family is locked. Within-family level refinement is V3's native
scope and is not pre-committed at V0. The
[[condensates/nested-downsampling-composition]] multiplicative-composition
claim is boundary-conditioned to the V0-Inconclusive fallback branch under
rb-v0.2.0 and does NOT apply on the happy-path V0 lock.

## Predictions for v1-recipe

At the V1 factorial under locked strategy and locked `imbalance_family`,
the following predictions are passed forward (to be formalized in
`gates/v1-recipe/ledger.md`):

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
- V1 inherits `imbalance_family = downsample` as a structural prior
  (family locked); within-family level refinement is NOT re-opened at V1
  (it is V3's native scope). V1 cells operate at the Gen 1 probe level
  (`downsample_5`) as a placeholder until V3 refines it.

These V1 predictions are documentation; they live in the V1 ledger once
that gate opens.
