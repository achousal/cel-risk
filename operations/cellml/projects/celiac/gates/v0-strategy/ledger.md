---
gate: v0-strategy
project: celiac
rulebook_snapshot: "rb-v0.2.0"
dataset_fingerprint: "sha256:8c02e33cc693edfb361a4d901113cd3d5d79c8c43193919440305d84c278c0e9"
created: "2026-04-20"
author: retrospective
---

> **Retrospective notice.** This ledger was reconstructed AFTER the gate ran.
> Prediction-style falsifier criteria were not formally stated at the time;
> they are reverse-engineered from `DESIGN.md` and `DECISION_TREE_AUDIT.md`
> for format-validation purposes only. Use for format-validation, not for
> claims about what was predicted ex ante.
>
> **Axis redefinition notice (rb-v0.2.0).** V0's imbalance-handling axis has
> been redefined from `control_ratio ∈ {1, 2, 5}` (numeric level within the
> downsample family) to `imbalance_probe ∈ {none, downsample_5, cw_log}`
> (categorical probe across families). The V0 lock is now `imbalance_family`
> (categorical: `none` | `downsample` | `weight`), not a numeric ratio. See
> `tensions.md` for the delta and `rulebook/CHANGELOG.md rb-v0.2.0` for
> rationale. This retrospective has been updated to reflect Strategy C;
> prior revisions under rb-v0.1.0 are preserved via git history.

## Hypothesis

V0 asks two coupled methodological questions before committing
main-factorial compute:

1. **Training strategy.** Which case partitioning best supports prospective
   screening performance: incident-only, incident + prevalent (prevalent
   restricted to TRAIN at fraction f in {0.25, 0.5, 1.0}), or prevalent-only?
   Prospective screening requires VAL/TEST to remain incident-only (ADR-002)
   regardless of the TRAIN partition chosen.
2. **Imbalance family.** Which imbalance-handling family best preserves
   discrimination on the incident-only TEST? V0 probes three representative
   points — `none` (no correction), `downsample_5` (downsample family at
   Gen 1's representative 5:1 level), and `cw_log` (weight family at the
   log-frequency representative level) — and locks the FAMILY
   (`imbalance_family ∈ {none, downsample, weight}`), not a numeric level.
   Level refinement within the locked family is deferred to V3.

Hypothesis (retrospective, reconstructed): Gen 1 baseline operated at
`downsample_5` (i.e., within the `downsample` family at the representative
5:1 level). V0 tests whether this family dominates `none` and `cw_log`
(weight family representative) on AUROC across all four models (LR_EN,
LinSVM_cal, RF, XGBoost) and two representative recipes (R1_sig at 4p,
R1_plateau at 8p), and therefore whether `imbalance_family = downsample`
can be locked before V1 rather than promoted to a main-factorial factor.

## Search-space restriction

Gate axes explored at V0 (per the rb-v0.2.0 `protocols/v0-strategy.md`
rewrite):

| Axis | Levels | Rationale |
|---|---|---|
| Training strategy | IncidentOnly, IncidentPlusPrevalent, PrevalentOnly | 3 discrete cases; IncidentPlusPrevalent expands on `prevalent_frac` below |
| Prevalent fraction | 0.25, 0.5, 1.0 (IncidentPlusPrevalent only) | Brackets Gen 1 default (0.5) per DECISION_TREE_AUDIT.md P3 action |
| Imbalance probe | none, downsample_5, cw_log | 3 categorical probes across 3 families; V0 locks FAMILY (none / downsample / weight), V3 refines level within family |
| Model | LR_EN, LinSVM_cal, RF, XGBoost | Whole model axis crossed to test robustness |
| Recipe | R1_sig (4p), R1_plateau (8p) | 2 representative shared recipes to avoid confounding strategy with panel |

Cell count: 5 strategies × 3 imbalance probes × 4 models × 2 recipes = 120
cells, 20 selection seeds (100-119) per cell. (Count is unchanged from the
pre-rb-v0.2.0 design; the factor decomposition changed but the grid size
matches.)

**Axes explicitly deferred** (not part of V0): recipe composition beyond the
two representatives, calibration, within-family imbalance level refinement
(V3), panel size beyond the two representatives, seed-split confirmation.
These advance to V1-V5.

## Cited rulebook entries

Slugs resolved against the rulebook at `rb-v0.2.0` (see rulebook-snapshot).

- `[[condensates/imbalance-two-stage-decision]]` — new meta-condensate at
  rb-v0.2.0 justifying the family-then-level cascade: V0 locks the
  imbalance family across three probes (none / downsample / weight); V3
  refines the level within the locked family. Replaces the single-level
  lock semantics of rb-v0.1.0 on the imbalance axis.
  <!-- TODO: verify slug after parallel author merges the new condensate -->
- `[[condensates/downsample-preserves-discrimination-cuts-compute]]` — claim
  that the downsample family preserves discrimination (AUROC) while
  reducing compute. Underwrites why `downsample_5` is a defensible probe
  point for the downsample family (and why Gen 1 picked it).
- `[[condensates/imbalance-utility-equal-weight]]` — claim that class
  weighting (`cw_log` family) yields utility comparable to downsampling
  at equal effective cost on the celiac cohort. Underwrites `cw_log` as
  the weight-family probe point.
- `[[condensates/prevalent-restricted-to-train]]` — claim that prevalent
  cases belong in TRAIN only (VAL/TEST remain incident-only per ADR-002).
  Underwrites why V0 treats `prevalent_train_frac` as a bounded axis
  ({0.25, 0.5, 1.0}) rather than allowing injection into VAL/TEST.
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

Retroactively applying the fixed rubric from `rulebook/SCHEMA.md`, adapted
to the rb-v0.2.0 family-level axis:

| Claim under test | Rubric claim type | Pre-registered criterion |
|---|---|---|
| IncidentOnly > {IncidentPlusPrevalent, PrevalentOnly} on AUROC | **Direction** | \|ΔAUROC\| >= 0.02 AND 95% bootstrap CI (1000 resamples over seeds 100-119) excludes 0, holding **independently across all 4 models** |
| IncidentOnly ≈ IncidentPlusPrevalent at f=0.5 on AUROC | **Equivalence** | \|ΔAUROC\| < 0.01 AND 95% bootstrap CI ⊂ [-0.02, 0.02], on all 4 models |
| imbalance_family=downsample > {none, weight} at locked strategy on AUROC | **Direction** (per family) | \|ΔAUROC\| >= 0.02 AND CI excludes 0, comparing family representative probes |
| downsample ≈ weight at locked strategy | **Equivalence** | \|ΔAUROC\| < 0.01 AND 95% CI ⊂ [-0.02, 0.02], on all 4 models |
| Any axis fails both Direction and Equivalence | **Inconclusive** | Promote to main-factorial factor per `MASTER_PLAN.md` V0 fallback |

**Dominance claim** (the actual lock test): IncidentOnly ×
`imbalance_family = downsample` **dominates** across all 4 models.
Direction criterion for the family comparison uses the probe points
(`downsample_5` vs `none` vs `cw_log`) as family representatives; Direction
must hold on **each model axis independently** (per SCHEMA rubric
definition of Dominance). Level-specific claims about `control_ratio = 5`
(e.g., "ratio 5 dominates ratio 2") are OUT OF SCOPE at V0 under rb-v0.2.0
— those are V3 questions, deferred to within-family refinement.

## Predictions with criteria

Retrospective, inferred from `DESIGN.md` vanillamax section and ADR-002/003:

- **P1** IncidentOnly wins. Rationale: VAL/TEST are incident-only, so
  training distribution aligned with evaluation distribution avoids
  distribution shift. Criterion: Direction (ΔAUROC >= 0.02, CI excludes 0)
  on at least 3 of 4 models; Equivalence or better on the 4th.
- **P2** `imbalance_family = downsample` dominates `imbalance_family = none`
  and `imbalance_family = weight` at the Gen 1 operating point (represented
  by probe `downsample_5`). Rationale: Gen 1 baseline operated within the
  downsample family and achieved AUROC 0.867; `none` is expected to
  under-perform due to prevalence-driven learning degeneracy; `cw_log`
  (weight representative) is expected to approximate downsample on AUROC
  but the family-level Direction test is the question V0 answers. Criterion:
  Direction on AUROC (\|ΔAUROC\| >= 0.02, CI excludes 0) on at least 3 of
  4 models; Equivalence between downsample and weight would defeat
  Dominance and promote the family question to V3. Note: V0 does NOT
  adjudicate within-family level (e.g., downsample_1 vs downsample_2 vs
  downsample_5); that is a V3 question.
- **P3** The joint winner (strategy, imbalance_family) is the same across
  all 4 models. Criterion: Dominance — Direction on each axis holds
  independently on each model. If not, V0 fails and the two axes promote
  to full factors in V1/V3 respectively.

## Risks & fallbacks

- **Fallback A** — if strategy winner varies across models, promote
  `training_strategy` to a V1 factor. Cells grow from 1,566 to 4,698.
- **Fallback B** — if `imbalance_family` winner varies across models (or
  if downsample ≈ weight returns Equivalence on all four models), the
  family lock fails and the imbalance axis is promoted to V3 as a full
  3-family factorial. Within-family level refinement (V3's native scope)
  then absorbs the question.
- **Fallback C** — if the Dominance criterion is met on AUROC but violated
  on any model's Brier reliability, flag a cross-metric tension and defer
  to V4 (calibration).
- **Known risk** — V0 was submitted with 20 selection seeds (100-119)
  only. The 10-seed confirmation set (120-129) is reserved for V5 and was
  not used at V0. Any V0-era claim about generalization stability uses
  within-selection-seed bootstraps, not held-out seed confirmation.
- **Axis-redefinition risk** — this ledger was originally written against
  rb-v0.1.0 `control_ratio` semantics. Gen 1's empirical evidence for
  `control_ratio = 5` is now interpreted as family-level evidence for
  `imbalance_family = downsample` at V0; the specific level (1 vs 2 vs 5)
  is deferred to V3 and is not a V0 claim. See `tensions.md`.
