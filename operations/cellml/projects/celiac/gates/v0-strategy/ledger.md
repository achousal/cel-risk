---
gate: v0-strategy
project: celiac
rulebook_snapshot: "rb-v0.1.0"
dataset_fingerprint: "sha256:8c02e33cc693edfb361a4d901113cd3d5d79c8c43193919440305d84c278c0e9"
created: "2026-04-20"
author: retrospective
---

> **Retrospective notice.** This ledger was reconstructed AFTER the gate ran.
> Prediction-style falsifier criteria were not formally stated at the time;
> they are reverse-engineered from `DESIGN.md` and `DECISION_TREE_AUDIT.md`
> for format-validation purposes only. Use for format-validation, not for
> claims about what was predicted ex ante.

## Hypothesis

V0 asked two coupled methodological questions before committing ~1,566 cells
of main-factorial compute:

1. **Training strategy.** Which case partitioning best supports prospective
   screening performance: incident-only, incident + prevalent (prevalent
   restricted to TRAIN at fraction f in {0.25, 0.5, 1.0}), or prevalent-only?
   Prospective screening requires VAL/TEST to remain incident-only (ADR-002)
   regardless of the TRAIN partition chosen.
2. **Control ratio.** At what case:control downsampling ratio does incident
   discrimination remain stable while holding compute cost tractable? Levels
   tested: 1.0, 2.0, 5.0.

Hypothesis (implicit, reconstructed): the Gen 1 operating point
(`IncidentOnly`, `control_ratio=5`) generalizes across four models (LR_EN,
LinSVM_cal, RF, XGBoost) and two representative recipes (R1_sig at 4p,
R1_plateau at 8p), and therefore can be locked before V1 rather than
promoted to a main-factorial factor.

## Search-space restriction

Gate axes explored at V0 (per `MASTER_PLAN.md` "V0: Training Strategy Gate"):

| Axis | Levels | Rationale |
|---|---|---|
| Training strategy | IncidentOnly, IncidentPlusPrevalent, PrevalentOnly | 3 discrete cases; IncidentPlusPrevalent expands on `prevalent_frac` below |
| Prevalent fraction | 0.25, 0.5, 1.0 (IncidentPlusPrevalent only) | Brackets Gen 1 default (0.5) per DECISION_TREE_AUDIT.md P3 action |
| Control ratio | 1.0, 2.0, 5.0 | ADR-003 default is 5; V0 confirms the tradeoff holds |
| Model | LR_EN, LinSVM_cal, RF, XGBoost | Whole model axis crossed to test robustness |
| Recipe | R1_sig (4p), R1_plateau (8p) | 2 representative shared recipes to avoid confounding strategy with panel |

Cell count: 5 strategies x 3 control ratios x 4 models x 2 recipes = 120
cells, 20 selection seeds (100-119) per cell.

**Axes explicitly deferred** (not part of V0): recipe composition beyond the
two representatives, calibration, weighting, downsampling-beyond-control-ratio,
seed-split confirmation. These advance to V1-V5.

## Cited rulebook entries

Slugs resolved against the rulebook at `rb-v0.1.0` (see rulebook-snapshot).

- `[[condensates/downsample-preserves-discrimination-cuts-compute]]` — claim
  that `control_ratio=5` preserves discrimination (AUROC) while reducing
  compute ~60x. Underwrites the V0 control-ratio axis.
- `[[condensates/prevalent-restricted-to-train]]` — claim that prevalent
  cases belong in TRAIN only (VAL/TEST remain incident-only per ADR-002).
  Underwrites why V0 treats `prevalent_train_frac` as a bounded axis
  ({0.25, 0.5, 1.0}) rather than allowing injection into VAL/TEST.
  <!-- TODO: the ledger originally wanted a "prevalent-train-frac-bias"
  condensate narrowly about the 0.5-default bias; no such standalone
  condensate exists in rb-v0.0.0-unfinalized. The current slug covers the
  TRAIN-only restriction and acknowledges the fraction-testing edge, but
  the specific 0.5-prior-bias claim is not cleanly isolated. Candidate
  rulebook addition after ADR-002 migration. -->
- `[[condensates/perm-validity-full-pipeline]]` — exists in the rulebook
  (`rulebook/condensates/perm-validity-full-pipeline.md`). Referenced
  because any V0 significance claim would require pipeline-level permutation
  validity, though V0 did not formally invoke permutation testing at lock
  time.
- `[[equations/perm-test-pvalue]]` — permutation test p-value with +1
  correction (`rulebook/equations/perm-test-pvalue.md`). Referenced as the
  pre-declared p-value computation if permutation testing is re-invoked
  retrospectively on V0 outputs.

## Falsifier criteria

Retroactively applying the fixed rubric from `rulebook/SCHEMA.md`:

| Claim under test | Rubric claim type | Pre-registered criterion |
|---|---|---|
| IncidentOnly > {IncidentPlusPrevalent, PrevalentOnly} on AUROC | **Direction** | \|ΔAUROC\| >= 0.02 AND 95% bootstrap CI (1000 resamples over seeds 100-119) excludes 0, holding **independently across all 4 models** |
| IncidentOnly ≈ IncidentPlusPrevalent at f=0.5 on AUROC | **Equivalence** | \|ΔAUROC\| < 0.01 AND 95% bootstrap CI ⊂ [-0.02, 0.02], on all 4 models |
| control_ratio=5 ≈ control_ratio=2 at locked strategy | **Equivalence** | same as above |
| Any axis fails both Direction and Equivalence | **Inconclusive** | Promote to main-factorial factor per `MASTER_PLAN.md` V0 fallback |

**Dominance claim** (the actual lock test): IncidentOnly × control_ratio=5
**dominates** across all 4 models. Direction criterion must hold on **each
model axis independently** (per SCHEMA rubric definition of Dominance).

## Predictions with criteria

Retrospective, inferred from `DESIGN.md` vanillamax section and ADR-002/003:

- **P1** IncidentOnly wins. Rationale: VAL/TEST are incident-only, so
  training distribution aligned with evaluation distribution avoids
  distribution shift. Criterion: Direction (ΔAUROC >= 0.02, CI excludes 0)
  on at least 3 of 4 models; Equivalence or better on the 4th.
- **P2** control_ratio=5 >= control_ratio=2 on AUROC; Equivalence or
  better vs control_ratio=1.0. Rationale: ADR-003 argues 5:1 preserves
  discrimination while cutting compute ~60x; 2:1 expected similar AUROC
  at higher cost.
- **P3** The joint winner (strategy, control_ratio) is the same across
  all 4 models. If not, V0 fails and the two axes promote to full factors
  in V1.

## Risks & fallbacks

- **Fallback A** — if strategy winner varies across models, promote
  `training_strategy` to a V1 factor. Cells grow from 1,566 to 4,698.
- **Fallback B** — if control_ratio winner varies across models, promote
  it to V1 as well.
- **Fallback C** — if the Dominance criterion is met on AUROC but violated
  on any model's Brier reliability, flag a cross-metric tension and defer
  to V4 (calibration).
- **Known risk** — V0 was submitted with 20 selection seeds (100-119)
  only. The 10-seed confirmation set (120-129) is reserved for V5 and was
  not used at V0. Any V0-era claim about generalization stability uses
  within-selection-seed bootstraps, not held-out seed confirmation.
