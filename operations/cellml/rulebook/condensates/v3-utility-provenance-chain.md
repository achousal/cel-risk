---
type: condensate
depends_on:
  - "[[condensates/nested-downsampling-composition]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/auprc-for-imbalance-handling]]"
  - "[[condensates/imbalance-utility-equal-weight]]"
  - "[[condensates/imbalance-two-stage-decision]]"
  - "[[equations/brier-decomp]]"
  - "[[equations/stratified-split-proportions]]"
applies_to:
  - "V3 imbalance gate utility evaluation"
  - "V3 has locked a level within V0's locked imbalance_family"
  - "Inconclusive-fallback branch uses composite prevalence"
  - "gates whose scoring metric stacks on top of an upstream imbalance intervention AND a post-hoc calibration layer"
  - "any rulebook claim that combines AUPRC and Brier-REL into a joint utility on low-prevalence cohorts (pi < 0.05)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "Under rb-v0.2.0, V3 utility U = 0.5*AUPRC_norm + 0.5*(1 - REL_norm) is only well-defined when AUPRC and REL are computed on the prevalence-adjusted OOF-posthoc substrate. Prevalence adjustment uses the LOCKED training prevalence — driven by V3's locked level against V0's locked family in the default flow, or by the composite r_V0 * r_V3 in the Inconclusive-fallback branch."
    date: "2026-04-21"
    source: "operations/cellml/DESIGN.md §V3; v3-imbalance protocol §2.3; v0-strategy rb-v0.2.0 redesign; rulebook audit 2026-04-21"
falsifier: |
  Test on both family-locked and fallback-mode runs. Family-locked: if on
  >=3 cohorts, V3 utility computed on raw within-training scores vs
  prevalence-adjusted OOF-posthoc scores stays within Equivalence
  (|Delta U| < 0.01, 95% CI subset [-0.02, 0.02]), the invariant is
  weakened for that branch. Fallback: same criterion on fallback-branch
  runs. If on >=5 non-overlapping cohorts in either branch, retire the
  invariant for that branch (raw within-training scores suffice at V3).
---

# V3 utility metrics must be computed on prevalence-adjusted OOF-posthoc predictions within the locked family, using the actual locked training prevalence

## Claim

V3's joint utility function

$$U_{(w, d)} = 0.5 \cdot \widetilde{\mathrm{AUPRC}}_{(w, d)} + 0.5 \cdot \left(1 - \widetilde{\mathrm{REL}}_{(w, d)}\right)$$

is only well-defined on a specific prediction substrate: **prevalence-
adjusted OOF-posthoc predictions**, aggregated across the 20 selection
seeds. Both the AUPRC half and the REL half of the utility MUST be
computed on this substrate; raw within-training scores are forbidden.

This invariant is **bounded to within the locked family**. Under rb-v0.2.0,
V0 locks `imbalance_family` (not `control_ratio`), and V3 refines the
level within that locked family. Prevalence adjustment per
[[equations/case-control-ratio-downsampling]] uses the **ACTUAL locked
training prevalence** — which is driven by V3's locked level against V0's
locked family in the default flow, **not** a multiplicative composite of
V0's probed level with V3's locked level. The composite formula
`pi_train_eff = 1 / (1 + r_V0 * r_V3)` applies ONLY in the V0-Inconclusive
fallback branch per [[condensates/nested-downsampling-composition]].

This meta-condensate makes explicit an invariant that the 4 V3-area
condensates ([[condensates/nested-downsampling-composition]],
[[condensates/downsample-requires-prevalence-adjustment]],
[[condensates/auprc-for-imbalance-handling]],
[[condensates/imbalance-utility-equal-weight]]) each presuppose but do not
individually name: V3 utility is valid only on the adjusted OOF substrate,
and the adjustment uses the locked (not probed, not composite) training
prevalence.

## Mechanism

Two substrate layers stack: V0-V3 picks the prevalence-shifting
intervention (whose locked level fixes the training marginal), then OOF
posthoc picks the calibration transform applied to the model's raw output
under that training marginal.

**Layer 1 — V0-V3 imbalance intervention substrate.** V0 locks a family
direction (`none`, `downsample`, `weight`) per
[[condensates/imbalance-two-stage-decision]]. V3 then refines the
magnitude by locking a level within that family. The locked combination
fixes the **actual training prevalence**:

- Family `downsample`, locked level `r_V3`: $\pi_\text{train} = 1 / (1 + r_{V3, \text{locked}})$
- Family `none` or `weight`, any locked level: $\pi_\text{train} = \pi_\text{population}$ (loss-surface interventions do not shift the training marginal)
- Fallback branch (V0 Inconclusive on family, V3 running legacy 3x3 grid): $\pi_\text{train} = 1 / (1 + r_{V0} \cdot r_{V3})$ per [[condensates/nested-downsampling-composition]]

The trained model fits under this shifted (or unshifted) marginal — its
raw output estimates $P(Y=1 | X, \pi_\text{train})$, not
$P(Y=1 | X, \pi_\text{population})$ unless they coincide.

**Layer 2 — OOF posthoc calibration substrate.** Per
[[condensates/downsample-requires-prevalence-adjustment]], the raw
layer-1 output must be passed through a Bayes label-shift correction
using the **locked** $\pi_\text{train}$ (from layer 1) and the population
prevalence $\pi$ (measured pre-split from the un-downsampled cohort)
before any probability-dependent metric (Brier, ECE, AUPRC) is computed.
Per [[condensates/calib-per-fold-leakage]], this correction is applied
to OOF predictions aggregated across the 20 selection seeds — not to
per-fold VAL predictions — so the calibration fit is held out from the
hyperparameter-selection signal.

The two utility components consume this layer-2 substrate:

- **AUPRC** per [[condensates/auprc-for-imbalance-handling]] must be
  computed on prevalence-adjusted predictions, because AUPRC's precision
  denominator is prevalence-sensitive. On raw layer-1 scores, AUPRC
  bakes in a training-prevalence artifact rather than the genuine
  imbalance-handling signal.
- **REL** per [[equations/brier-decomp]] and
  [[condensates/imbalance-utility-equal-weight]] must be computed on the
  same adjusted OOF substrate. Computing REL on raw layer-1 scores
  produces a reliability number that reflects the training-prevalence
  shift, not the model's calibration quality at the population
  prevalence.

Feeding layer-1 output into either AUPRC or REL produces a utility
number that the four V3-area condensates neither predict nor bound. The
claim that V3 utility adjudicates between imbalance-handling levels
**depends on each downstream stage presupposing the prior's correction
applied**; skipping the layer dissolves the utility's interpretability.

## Actionable rule

1. **[[protocols/v3-imbalance]] §2.3 MUST compute utility on the adjusted
   OOF substrate, using the LOCKED training prevalence.** The protocol
   text must cite this meta-condensate alongside the component
   condensates when declaring the utility function. The prevalence used
   in the adjustment comes from the V3-locked level (in the default
   family-locked flow) — NOT the V0-probed level — or from the
   composite `r_V0 * r_V3` only if the fallback branch is active.
2. **`cellml-reduce` MUST produce the adjusted OOF substrate** before
   V3's `observation.md` is written. The reduce step applies the Bayes
   label-shift correction using `pi_train` derived from the locked
   level (per-cell in the fallback branch; per-locked-configuration in
   the default flow) and the population `pi` from the fingerprint, and
   aggregates OOF predictions across the 20 selection seeds.
3. **V3's `observation.md` MUST report the substrate AND the prevalence
   source.** The provenance block names the substrate as
   `prevalence_adjusted_oof_posthoc` and records: `branch`
   (`family_locked` | `fallback`), `pi_train` per cell with its
   derivation formula, `pi_population` from the fingerprint, and the
   calibration strategy (`oof_posthoc`). Any `observation.md` whose
   utility computation does not cite the branch-appropriate prevalence
   source is a rulebook violation.
4. **V3's `ledger.md` MUST cite this meta-condensate** alongside the
   component condensates AND declare the active branch at gate entry.

## Boundary conditions

- **Degenerate case (V0 locks `none`, V3 skipped or trivially locked).**
  Under family `none`, the training marginal equals the population
  marginal and raw == adjusted (the Bayes correction is a numerical
  no-op). Utility computation on raw scores is numerically equivalent to
  the adjusted substrate in this case, but the provenance rule still
  requires the substrate be declared as `prevalence_adjusted_oof_posthoc`
  for audit consistency.
- **Family `weight`**: loss-surface interventions do not shift the
  training marginal, so $\pi_\text{train} = \pi_\text{population}$ and
  the correction is also a no-op — but the weighting changes the score
  distribution, so REL and AUPRC are still not invariant to the V3-
  locked level within that family.
- **Family `downsample`**: the correction is load-bearing (e.g., on
  celiac the shift factor is ~49x) and raw substrate is forbidden.
- **Fallback branch**: prevalence adjustment uses the composite
  `pi_train_eff = 1 / (1 + r_V0 * r_V3)` per
  [[condensates/nested-downsampling-composition]]; the substrate
  requirement is identical.
- **Does NOT apply to V1 or V2 gates**, whose primary metrics (AUROC)
  are prevalence-invariant by construction.
- **Does NOT apply to V4 calibration gates**, which adjudicate on REL
  after V3 has locked the imbalance configuration; V4 inherits V3's
  adjusted OOF substrate as input but does not re-adjudicate on utility.
- **Does NOT apply to informational gates (V6 ensemble)**.
- **Weakens if the population prevalence `pi` is itself uncertain**
  beyond the V3 Equivalence band (0.01 on utility). On celiac,
  `SE(pi) ~ 0.003%` is negligible. On smaller cohorts
  (`n_case < 50`) the propagated uncertainty may become material.

## Evidence

| Component condensate | Role in the chain | Without this component | Source |
|---|---|---|---|
| [[condensates/nested-downsampling-composition]] | Defines `pi_train` under the fallback branch (`1 / (1 + r_V0 * r_V3)`); absence implies the default family-locked formula applies | Fallback branch adjustment magnitude is undefined | DESIGN.md §V3; v0-strategy §4 fallback |
| [[condensates/imbalance-two-stage-decision]] | Defines the default flow in which V3-locked level against V0-locked family drives `pi_train`, superseding V0's probed level | Substrate provenance conflates probed with locked prevalence | v0-strategy rb-v0.2.0 redesign §2.2 |
| [[condensates/downsample-requires-prevalence-adjustment]] | Mandates Bayes label-shift correction between layer 1 and layer 2 | Raw layer-1 scores enter utility; the shift pollutes REL and AUPRC | ADR-003; v3-imbalance §2.3 |
| [[condensates/auprc-for-imbalance-handling]] | Specifies AUPRC as the discrimination metric on the adjusted substrate | Utility discrimination component is not well-defined | DESIGN.md §V3 |
| [[condensates/imbalance-utility-equal-weight]] | Specifies 0.5/0.5 weighting of AUPRC and REL on the adjusted substrate | Utility scale-alignment is unidentified | DECISION_TREE_AUDIT.md §2.3 |

The component condensates, taken together, fully determine the V3
utility function ONLY on the prevalence-adjusted OOF-posthoc substrate
with the correct (locked-level, branch-appropriate) prevalence. The
meta-condensate's contribution is to name this substrate as a shared
invariant — without it, each component's claim is conditional on an
unstated assumption.

## Related

- [[condensates/imbalance-two-stage-decision]] — parent two-stage design;
  defines the default flow semantics in which V3-locked level supersedes
  V0-probed level
- [[condensates/nested-downsampling-composition]] — governs composition
  only in the fallback branch
- [[protocols/v3-imbalance]] §2.3 — utility function that depends on
  this chain
- [[protocols/v0-strategy]] §4 — fallback entry condition for the
  composition branch
- [[condensates/downsample-requires-prevalence-adjustment]] — layer 1
  to layer 2 transform
- [[condensates/auprc-for-imbalance-handling]] — discrimination
  component
- [[condensates/imbalance-utility-equal-weight]] — weighting rule
- [[condensates/calib-per-fold-leakage]] — motivates OOF (not per-fold)
  posthoc calibration
- [[equations/brier-decomp]] — REL half of utility
- [[equations/stratified-split-proportions]] — split-allocation
  substrate
- DESIGN.md §V3 — canonical source for the utility function
