---
type: condensate
depends_on:
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
  - "[[condensates/imbalance-two-stage-decision]]"
applies_to:
  - "V0 returns Inconclusive on imbalance_family"
  - "V3 operates in fallback mode running old 3x3 grid"
  - "random within-training control sampling (non-stratified beyond the upstream split)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v3-imbalance
    delta: "Under rb-v0.2.0 fallback branch (V0 Inconclusive on imbalance_family -> V3 runs full 3x3 grid): composed r_V0 * r_V3 yields measured training prevalence pi_train_eff ~ 0.038 at r_V0=5 / r_V3=5, consistent with 1 / (1 + 25) within +/- 2%; additive prediction (r_V0 + r_V3 = 10) gives pi ~ 0.091 which does not match"
    date: "2026-04-21"
    source: "v0-strategy protocol §4 fallback; v3-imbalance protocol fallback branch; operations/cellml/DESIGN.md §V3"
falsifier: |
  On V3-fallback-mode runs only (V0 returned Inconclusive on imbalance_family
  AND V3 is executing the legacy 3x3 grid): if on >=3 cohorts, measured
  training prevalence matches additive composition (r_V0 + r_V3) within
  +/- 2% more closely than multiplicative (r_V0 * r_V3), the multiplicative
  claim is weakened (Direction against multiplicative on a cross-cohort
  aggregate). If on >=5 cohorts, retire in favor of an additive or other
  composition rule. Family-locked flow runs are OUT OF SCOPE for this
  falsifier — composition does not apply there.
---

# Nested downsampling composes multiplicatively within the V0-Inconclusive fallback branch; outside that branch V3 supersedes V0's probed level rather than composing with it

## Claim

Under rb-v0.2.0, V0 locks `imbalance_family` (categorical: `none` |
`downsample` | `weight`) rather than `control_ratio`. Composition of V0's
probed ratio with V3's training-time ratio is NOT the default flow: the
default is **family-locked**, in which V3 refines the level within V0's
locked family and V3's locked level REPLACES V0's probed level. Composition
applies only inside the **fallback branch** activated when V0 returns
Inconclusive on `imbalance_family` and V3 falls back to running the legacy
3x3 (weighting x downsampling) grid against the cohort without a family
lock. Within that fallback branch, V0's probed `control_ratio` and V3's
`downsampling` axis operate sequentially on the same cohort and therefore
compose **multiplicatively**:

$$r_\text{eff} = r_{V0} \cdot r_{V3}$$
$$\pi_\text{train, eff} = \frac{1}{1 + r_{V0} \cdot r_{V3}}$$

Any V3 computation that reports training prevalence or consumes a training
prevalence (prevalence-adjustment, Brier REL) must use the composite
`r_eff` in fallback mode; in family-locked mode, the training prevalence
is driven by V3's locked level alone against V0's locked family — not by
any composition.

## Mechanism

**Family-locked flow (default, no composition).** V0 probes one
representative level per family (per
[[condensates/imbalance-two-stage-decision]]) and locks the winning family.
V3 then runs a level-grid *within* that family. Because V0 did not lock a
level (only a family direction), V3's locked level is the single
prevalence-shifting intervention applied at training time — there is no
"probed r_V0 plus locked r_V3" stack to compose. Composition is trivial:
$\pi_\text{train, eff} = 1 / (1 + r_{V3, \text{locked}})$ if the locked
family is `downsample`, or it equals the population prevalence if the
locked family is `none` or `weight` (the latter shifts the loss surface
without shifting the training marginal).

**Fallback flow (V0 Inconclusive -> V3 runs legacy 3x3 grid).** When V0
cannot lock a family, the pipeline reverts to the pre-rb-v0.2.0 behavior:
V0's probed `control_ratio` (e.g., `r_V0 = 5.0`) is carried forward as a
split-generation sub-sampling step, and V3's `downsampling` axis performs
a second random sub-sampling step inside the V0-restricted TRAIN set.
Random sub-sampling of a uniform sub-population preserves the independence
assumption used in the single-axis
[[equations/case-control-ratio-downsampling]] derivation, and composition
of two random sub-samplings is itself random sub-sampling at the product
ratio. Hence the effective ratio is $r_{V0} \cdot r_{V3}$, not
$r_{V0} + r_{V3}$, within this fallback branch only.

The critical distinction: family-locked flow has no "probed level +
locked level" to combine, because V0 did not lock a level. Fallback flow
preserves the old semantics for audit continuity and because legacy
observation logs used multiplicative composition.

## Actionable rule

- **Family-locked flow (default): do NOT apply multiplicative composition.**
  V3's `observation.md` and [[protocols/v3-imbalance]] compute
  `pi_train_eff = 1 / (1 + r_{V3, \text{locked}})` under a `downsample`
  family lock; `pi_train_eff = pi_population` under a `none` or `weight`
  family lock. Any prevalence-adjustment per
  [[condensates/downsample-requires-prevalence-adjustment]] uses this
  single-axis `pi_train_eff`.
- **Fallback flow only: apply multiplicative composition.** When
  [[protocols/v3-imbalance]] enters the V0-Inconclusive fallback branch,
  V3's `observation.md` MUST compute
  `pi_train_eff = 1 / (1 + r_{V0} \cdot r_{V3})` and report
  `r_V0, r_V3, r_eff, pi_train_eff` per cell.
- **Ledger provenance.** V3's `ledger.md` MUST declare which branch is
  active (`family_locked` | `fallback`) at gate entry, and cite this
  condensate only when `fallback` is active. Family-locked ledgers cite
  [[condensates/imbalance-two-stage-decision]] instead.
- **Tension detector.** Any `observation.md` whose reported training
  prevalence does not match the branch-appropriate formula within
  floating-point tolerance is a data-recording bug and is flagged.

## Boundary conditions

- **Family-locked flow: NO composition.** V0 locked a family, V3 refines
  the level within it. The locked level IS the intervention; there is no
  upstream ratio to compose with.
- **Fallback flow: composition applies.** V0 returned Inconclusive on
  `imbalance_family`, V3 runs the legacy 3x3 weighting x downsampling
  grid. Both V0's probed ratio and V3's axis ratio apply sequentially;
  multiplicative composition holds.
- **Requires random sub-sampling.** In the fallback branch, composition
  assumes V3's within-training downsampling is **random** sub-selection
  of controls from the V0-restricted TRAIN set. Stratified selection
  (e.g., stratifying by covariate distribution) changes the denominator
  and the composition can deviate from multiplicative — use the exact
  stratum-weighted ratio in that case.
- **Does not apply to non-ratio sampling rules.** If either V0 or V3
  uses absolute control-count targets rather than case:control ratios,
  compute `pi_train_eff` directly from the post-sampling counts, not
  from a composed ratio.

## Evidence

| Dataset | Branch | r_V0 | r_V3 | r_eff predicted | pi_train_eff predicted | pi_train_eff measured | Source gate |
|---|---|---|---|---|---|---|---|
| Celiac (UKBB) | fallback | 5.0 | 5.0 | 25.0 | 0.038 | 0.038 +/- 0.001 | v3-imbalance protocol (fallback branch, rb-v0.2.0) |

Additive prediction (`r_V0 + r_V3 = 10`, `pi ~ 0.091`) deviates from the
measured prevalence by a factor of ~2.4 on this cohort in the fallback
branch, consistent with multiplicative composition within that branch.
Family-locked evidence is not yet collected (rb-v0.2.0 design-era).

## Related

- [[condensates/imbalance-two-stage-decision]] — parent claim that
  motivates the family-locked (default) vs fallback (composition)
  branching; this condensate is the narrow successor that governs
  composition semantics within the fallback branch only
- [[condensates/v3-utility-provenance-chain]] — shared invariant; uses
  the branch-appropriate `pi_train_eff` from this condensate
- [[protocols/v0-strategy]] §4 — fallback section; defines when V0
  returns Inconclusive on `imbalance_family` and hands off to V3
  fallback
- [[protocols/v3-imbalance]] — fallback branch; the only flow where
  multiplicative composition applies under rb-v0.2.0
- [[condensates/downsample-preserves-discrimination-cuts-compute]] —
  the unsharpened parent; pre-rb-v0.2.0 used "control downsampling" to
  cover both operations without distinguishing split-generation from
  training-dynamics
- [[condensates/downsample-requires-prevalence-adjustment]] — consumes
  `pi_train_eff` from this condensate (fallback) or from V3's locked
  level alone (family-locked) for Bayes label-shift adjustment
- [[equations/case-control-ratio-downsampling]] — single-axis prevalence
  formula that this condensate generalizes to the composite case within
  the fallback branch
- ADR-003 (cel-risk, 2026-01-20) — original control-ratio decision;
  superseded by rb-v0.2.0 family-locking design for the default flow
