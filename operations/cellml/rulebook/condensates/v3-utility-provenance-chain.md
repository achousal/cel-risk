---
type: condensate
depends_on:
  - "[[condensates/nested-downsampling-composition]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/auprc-for-imbalance-handling]]"
  - "[[condensates/imbalance-utility-equal-weight]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/stratified-split-proportions]]"
applies_to:
  - "V3 imbalance gate utility evaluation"
  - "gates whose scoring metric stacks on top of an upstream split-generation ratio AND an upstream training-dynamics ratio AND a post-hoc calibration layer"
  - "any rulebook claim that combines AUPRC and Brier-REL into a joint utility on low-prevalence cohorts (pi < 0.05)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "V3 utility U = 0.5*AUPRC_norm + 0.5*(1 - REL_norm) is only well-defined when AUPRC and REL are both computed on the prevalence-adjusted OOF-posthoc prediction substrate; the 4 V3-area condensates ([[condensates/nested-downsampling-composition]], [[condensates/downsample-requires-prevalence-adjustment]], [[condensates/auprc-for-imbalance-handling]], [[condensates/imbalance-utility-equal-weight]]) each presuppose this substrate without naming it as a shared invariant"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §V3; v3-imbalance protocol §2.3; round A gap surfaced during rulebook audit 2026-04-20"
falsifier: |
  If on >=3 cohorts, V3 utility computed on raw within-training scores vs
  prevalence-adjusted OOF-posthoc scores stays within Equivalence band
  (|Delta U| < 0.01, 95% CI subset [-0.02, 0.02]), this invariant is weakened.
  If on >=5 cohorts with non-overlapping cohorts, retire in favor of the
  simpler rule (raw within-training scores suffice at V3).
---

# V3 utility metrics must be computed on prevalence-adjusted OOF-posthoc predictions, never raw within-training scores

## Claim

V3's joint utility function

$$U_{(w, d)} = 0.5 \cdot \widetilde{\mathrm{AUPRC}}_{(w, d)} + 0.5 \cdot \left(1 - \widetilde{\mathrm{REL}}_{(w, d)}\right)$$

is only well-defined on a specific prediction substrate: **prevalence-adjusted
OOF-posthoc predictions**, aggregated across the 20 selection seeds. Both the
AUPRC half and the REL half of the utility MUST be computed on this substrate;
raw within-training scores are forbidden as inputs to V3 utility evaluation.
This meta-condensate makes explicit an invariant that the 4 V3-area condensates
([[condensates/nested-downsampling-composition]],
[[condensates/downsample-requires-prevalence-adjustment]],
[[condensates/auprc-for-imbalance-handling]],
[[condensates/imbalance-utility-equal-weight]]) each presuppose but do not
individually name: V3 utility is valid only on the adjusted OOF substrate
because that is the substrate on which each of those four claims was
constructed.

## Mechanism

Three substrate layers stack sequentially and each downstream stage
presupposes the prior's correction has been applied. Computing utility on raw
within-training scores skips all three layers and delivers a number that none
of the four V3-area condensates predicts or bounds.

**Layer 1 — V0 split-generation substrate.** Per
[[equations/stratified-split-proportions]] and
[[condensates/nested-downsampling-composition]], V0's locked
`train_control_per_case` (ratio `r_V0`) fixes the per-seed TRAIN/VAL/TEST
allocation under stratified 50/25/25 split. This layer determines **which
samples** are available for V3's training-dynamics operations; V3 does not
revisit the split.

**Layer 2 — V3 training-dynamics substrate.** Per
[[condensates/nested-downsampling-composition]], V3's `downsampling` axis
(ratio `r_V3`) composes multiplicatively with `r_V0` inside the V0-restricted
TRAIN partition, producing an effective training ratio `r_eff = r_V0 * r_V3`
and effective training prevalence `pi_train_eff = 1 / (1 + r_eff)`. The
trained model fits under this shifted marginal — its raw output estimates
`P(Y=1 | X, pi_train_eff)`, not `P(Y=1 | X, pi_population)`.

**Layer 3 — OOF posthoc calibration substrate.** Per
[[condensates/downsample-requires-prevalence-adjustment]], the raw layer-2
output must be passed through a Bayes label-shift correction using
`pi_train_eff` (from layer 2) and the population prevalence `pi` (measured
pre-split from the un-downsampled cohort) before any probability-dependent
metric (Brier, ECE, AUPRC on prevalence-weighted bins) is computed. Per
[[condensates/calib-per-fold-leakage]], this correction is applied to
**OOF** predictions aggregated across the 20 selection seeds — not to
per-fold VAL predictions — so the calibration fit is held out from the
hyperparameter-selection signal.

The two utility components consume this layer-3 substrate:

- **AUPRC** per [[condensates/auprc-for-imbalance-handling]] must be computed
  on prevalence-adjusted predictions, because AUPRC's precision denominator
  is prevalence-sensitive. On raw layer-2 scores, AUPRC bakes in a training-
  prevalence artifact rather than the genuine imbalance-handling signal, and
  the two effects cannot be separated post-hoc.
- **REL** per [[equations/brier-decomp]] and
  [[condensates/imbalance-utility-equal-weight]] must be computed on the
  same adjusted OOF substrate. Computing REL on raw layer-2 scores produces
  a reliability number that reflects the training-prevalence shift, not the
  model's calibration quality at the population prevalence — precisely the
  quantity V3 needs to adjudicate on.

Feeding layer-2 output (raw within-training scores) into either AUPRC or REL
produces a utility number that the four V3-area condensates neither predict
nor bound. The claim that V3 utility adjudicates between imbalance-handling
configurations **depends on each downstream stage presupposing the prior's
correction applied**; skipping layer 3 dissolves the utility's interpretability.

## Actionable rule

1. **[[protocols/v3-imbalance]] §2.3 MUST compute utility on the adjusted
   OOF substrate.** The protocol text must cite this meta-condensate
   alongside the four component condensates when declaring the utility
   function. Raw within-training score substrate is forbidden.
2. **`cellml-reduce` MUST produce the adjusted OOF substrate** before V3's
   `observation.md` is written. The reduce step applies the Bayes label-
   shift correction using `pi_train_eff` per cell (from layer 2) and the
   population `pi` (from the dataset fingerprint), and aggregates OOF
   predictions across the 20 selection seeds. Raw within-training scores
   may be logged for provenance but MUST NOT enter the utility computation.
3. **V3's `observation.md` MUST report the substrate used.** The
   provenance block names the substrate as `prevalence_adjusted_oof_posthoc`
   (or equivalent) and records: `pi_train_eff` per cell, `pi_population`
   from the fingerprint, and the calibration strategy (`oof_posthoc`). Any
   `observation.md` whose utility computation does not cite this substrate
   is a rulebook violation and is flagged by the tension detector.
4. **V3's `ledger.md` MUST cite this meta-condensate** alongside the four
   component condensates when declaring the utility function. The citation
   is the mechanical enforcement that the chain is intact at ledger entry.

## Boundary conditions

- **Does NOT apply when V0 `control_ratio = 1.0` AND V3 `downsampling = 1.0`**
  simultaneously. In that degenerate case, `r_eff = 1.0` and
  `pi_train_eff = 0.5`, which collapses to the population prevalence iff the
  cohort itself has `pi = 0.5`. On low-prevalence cohorts (the intended V3
  regime, `pi < 0.05`) this boundary case cannot arise because V0 does not
  admit `r_V0 = 1.0` on such cohorts (V0's parsimony floor is
  `control_ratio = 5.0` per [[protocols/v0-strategy]]). More precisely, the
  adjustment is a no-op **only** when `pi_train_eff = pi_population`; on
  balanced or near-balanced cohorts, the raw and adjusted substrates are
  numerically indistinguishable.
- **Does NOT apply to V1 or V2 gates**, whose primary metrics (AUROC) are
  prevalence-invariant by construction per
  [[condensates/auprc-for-imbalance-handling]]. The utility-provenance
  chain is V3-specific because V3 is the first gate whose metric is
  prevalence-sensitive.
- **Does NOT apply to V4 calibration gates**, whose axes adjudicate on REL
  after V3 has locked the imbalance configuration. V4 inherits V3's adjusted
  OOF substrate as input but does not re-adjudicate on utility.
- **Does NOT apply to informational gates (V6 ensemble)** — V6 reports
  ensemble verdicts against the V5-locked baseline and does not use V3's
  joint utility.
- **Weakens if the population prevalence `pi` is itself uncertain** beyond
  the V3 Equivalence band (0.01 on utility). On the celiac cohort,
  `SE(pi) ~ 0.003%` per [[condensates/downsample-requires-prevalence-adjustment]]
  boundary condition, so this is negligible. On smaller cohorts
  (`n_case < 50`) the propagated uncertainty may become material and the
  adjusted substrate itself carries additional noise.

## Evidence

| Component condensate | Role in the chain | Without this component | Source |
|---|---|---|---|
| [[condensates/nested-downsampling-composition]] | Defines `pi_train_eff = 1 / (1 + r_V0 * r_V3)` from upstream locks | Layer 2 to layer 3 handoff is undefined; the adjustment magnitude cannot be computed | DESIGN.md §V3; v3-imbalance T-V3.1 |
| [[condensates/downsample-requires-prevalence-adjustment]] | Mandates Bayes label-shift correction between layer 2 and layer 3 | Raw layer-2 scores enter utility; the shift factor (~49x on celiac) pollutes REL and AUPRC | ADR-003; v3-imbalance §2.3 |
| [[condensates/auprc-for-imbalance-handling]] | Specifies AUPRC as the discrimination metric on layer-3 substrate | Utility discrimination component is not well-defined for V3's axes | DESIGN.md §V3; v3-imbalance T-V3.3 |
| [[condensates/imbalance-utility-equal-weight]] | Specifies 0.5/0.5 weighting of AUPRC and REL on layer-3 substrate | Utility scale-alignment is not fixed; the weighted combination is unidentified | DECISION_TREE_AUDIT.md §2.3; v3-imbalance T-V3.4 |

The four component condensates, taken together, fully determine the V3
utility function ONLY on the prevalence-adjusted OOF-posthoc substrate. The
meta-condensate's contribution is to name this substrate as a shared
invariant — without it, each component condensate's claim is conditional on
an unstated assumption.

## Related

- [[protocols/v3-imbalance]] §2.3 — utility function that depends on this chain
- [[condensates/nested-downsampling-composition]] — layer 2 substrate definition
- [[condensates/downsample-requires-prevalence-adjustment]] — layer 2 to layer 3 transform
- [[condensates/auprc-for-imbalance-handling]] — layer 3 discrimination component
- [[condensates/imbalance-utility-equal-weight]] — layer 3 weighting rule
- [[condensates/calib-per-fold-leakage]] — motivates OOF (not per-fold) posthoc calibration in layer 3
- [[equations/brier-decomp]] — REL half of utility
- [[equations/stratified-split-proportions]] — layer 1 substrate definition (V0 stratified 50/25/25)
- DESIGN.md §V3 — canonical source for the utility function and the three-layer stacking logic
