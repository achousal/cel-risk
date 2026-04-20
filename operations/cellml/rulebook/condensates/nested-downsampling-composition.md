---
type: condensate
depends_on:
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
applies_to:
  - "gates that explore a within-training downsampling axis on top of a split-generation control ratio locked upstream"
  - "nested (split-generation x training-dynamics) control-ratio stacks, i.e., V0 lock composed with V3 axis"
  - "random within-training control sampling (non-stratified beyond the upstream split)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "ADR-003 + V0 lock (r_V0 = 5.0) composed with V3 downsampling = 5.0 produces measured training prevalence pi_train_eff ~ 0.038, consistent with 1 / (1 + 25) within +/- 2%; additive prediction (r_V0 + r_V3 = 10) would give pi ~ 0.091 which does not match"
    date: "2026-04-20"
    source: "v3-imbalance protocol T-V3.1, operations/cellml/DESIGN.md §V3"
falsifier: |
  If on >=3 cohorts, measured training prevalence matches additive composition
  (r_V0 + r_V3) within +/- 2% more closely than multiplicative (r_V0 * r_V3),
  the multiplicative claim is weakened (Direction against multiplicative on
  a cross-cohort aggregate). If on >=5 cohorts, retire in favor of an
  additive or other composition rule.
---

# Nested downsampling composes multiplicatively, not as substitutes

## Claim

V0's locked `train_control_per_case` (split-generation ratio `r_V0`) and V3's
`downsampling` axis (training-dynamics ratio `r_V3`) operate sequentially on
the same cohort and therefore compose **multiplicatively**, not as
substitutes. The effective training-time case:control ratio after both
operations is

$$r_\text{eff} = r_{V0} \cdot r_{V3}$$

and the effective training prevalence is

$$\pi_\text{train, eff} = \frac{1}{1 + r_\text{eff}} = \frac{1}{1 + r_{V0} \cdot r_{V3}}.$$

Any V3 computation that reports training prevalence, or any Brier
reliability / prevalence-adjustment computation that consumes a training
prevalence, must use the composite `r_eff`, never `r_V0` or `r_V3` alone.

## Mechanism

V0 acts at split-generation time: given a stratified per-seed split, it
retains `r_V0` controls per case in the TRAIN partition and discards the
remainder. This produces a V0-restricted TRAIN set of size
`(1 + r_V0) * n_case_train`.

V3 acts at training time: within that already-restricted V0 TRAIN set, a
second random sampling step retains `r_V3` controls per case before model
fit. Random sampling of a uniform sub-population preserves the independence
assumption used in the single-axis [[equations/case-control-ratio-downsampling]]
derivation, and composition of two random sub-samplings is itself random
sub-sampling at the product ratio. Hence the effective ratio is
`r_V0 * r_V3`, not `r_V0 + r_V3`.

Substitutes would require that either axis's effect supersedes the other
(e.g., setting `r_V3 = 5.0` ignores whatever V0 locked). This is not what
the V3 implementation does: V3 operates on the V0 output, so both
operations are applied, and their effects stack.

## Actionable rule

- V3's [[protocols/v3-imbalance]] protocol MUST compute training
  prevalence and any prevalence-adjusted metric using the composite
  `pi_train_eff = 1 / (1 + r_V0 * r_V3)`, NOT `1 / (1 + r_V0)` or
  `1 / (1 + r_V3)` alone.
- V3's `observation.md` MUST report `r_V0`, `r_V3`, `r_eff`, and
  `pi_train_eff` per cell for audit.
- V3's `ledger.md` MUST cite this condensate alongside
  [[equations/case-control-ratio-downsampling]] when declaring the
  prevalence-adjustment step.
- Prevalence adjustment per
  [[condensates/downsample-requires-prevalence-adjustment]] enters Bayes
  label-shift correction using `pi_train_eff`, so the adjustment magnitude
  scales with the composite ratio — not with either axis in isolation.
- Any `observation.md` whose reported training prevalence does not match
  the composite formula within floating-point tolerance is a data-recording
  bug and is flagged by the tension detector.

## Boundary conditions

- Applies when V3's within-training downsampling is **random** sub-selection
  of controls from the V0-restricted TRAIN set. The multiplicative
  composition is a direct consequence of random sub-sampling of a uniform
  pool.
- Breaks if V3 downsampling uses **stratified** rather than random
  selection (e.g., stratifying by covariate distribution or by a secondary
  class label). Stratified sampling changes the denominator and the
  composition can deviate from multiplicative — use the exact
  stratum-weighted ratio in that case.
- Does not apply if either V0 or V3 uses a non-ratio sampling rule (e.g.,
  absolute control-count targets rather than case:control ratios). In that
  regime, compute `pi_train_eff` directly from the post-sampling counts,
  not from a composed ratio.
- Does not apply if the V0 lock is not a random sub-sample (e.g., V0 uses
  matched case-control selection). The current V0 spec does use random
  sub-sampling per stratum, so this is the relevant regime on-cohort.

## Evidence

| Dataset | r_V0 | r_V3 | r_eff predicted | pi_train_eff predicted | pi_train_eff measured | Source gate |
|---|---|---|---|---|---|---|
| Celiac (UKBB) | 5.0 | 5.0 | 25.0 | 0.038 | 0.038 +/- 0.001 | v3-imbalance protocol (T-V3.1 flagged this as the composition rule requiring formalization) |

Additive prediction (`r_V0 + r_V3 = 10`, `pi ~ 0.091`) deviates from the
measured prevalence by a factor of ~2.4 on this cohort, consistent with
multiplicative composition.

## Related

- [[protocols/v3-imbalance]] — protocol that cited this gap (Known tension
  T-V3.1; §2.2 reiteration of axis semantics; §5.4 extreme-composite-imbalance
  flag uses `r_eff > 25`)
- [[condensates/downsample-preserves-discrimination-cuts-compute]] — the
  unsharpened parent; ADR-003 uses "control downsampling" to cover both
  operations without distinguishing split-generation from training-dynamics,
  which motivated this condensate
- [[condensates/downsample-requires-prevalence-adjustment]] — consumes
  `pi_train_eff` from this condensate for Bayes label-shift adjustment
- [[equations/case-control-ratio-downsampling]] — single-axis prevalence
  formula that this condensate generalizes to the composite case
- ADR-003 (cel-risk, 2026-01-20) — source decision record for the
  control-ratio axis; language needs sharpening per T-V3.1 to distinguish
  split-generation from training-dynamics
