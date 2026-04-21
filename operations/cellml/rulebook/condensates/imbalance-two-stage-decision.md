---
type: condensate
depends_on:
  - "[[equations/case-control-ratio-downsampling]]"
  - "[[condensates/parsimony-tiebreaker-when-equivalence]]"
  - "[[condensates/downsample-preserves-discrimination-cuts-compute]]"
  - "[[condensates/downsample-requires-prevalence-adjustment]]"
applies_to:
  - "Validation trees with imbalance handling as an axis"
  - "Cohorts where native prevalence < 0.1"
  - "Pipelines testing >1 imbalance strategy"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "design-level: 3-probe family gate (120 cells) vs 7-alternative full-imbalance gate (280 cells)"
    date: "2026-04-21"
    source: "v0-strategy.md §2.2 post-Strategy-C revision"
falsifier: |
  If on >=3 cohorts, single-gate imbalance resolution (testing all levels of
  all families simultaneously) and two-stage family-then-level resolution
  produce locked configurations within Equivalence (|ΔAUROC_final| < 0.01,
  95% bootstrap CI subset of [-0.02, 0.02]), the two-stage advantage is
  weakened. At >=5 cohorts, retire and collapse back to single-gate.
---

# Imbalance handling is a two-stage decision: family at V0 (direction), level at V3 (magnitude); committing to both simultaneously at a single gate is premature optimization

## Claim

Imbalance handling decomposes into two orthogonal axes: **direction**
(which family — `none`, `downsample`, `weight`) and **magnitude** (level
within the chosen family — e.g., `r_V3 in {1, 2, 5}` within `downsample`,
or `{sqrt, log}` within `weight`). These axes should be resolved at
**separate gates**: V0 locks the family at the lowest compute point using
one representative probe per family; V3 refines the level conditional on
V0's family lock. Committing to both simultaneously at a single gate
(enumerating every family x level cell) is premature optimization — it
pays compute to explore levels inside families that the direction
evidence would have eliminated.

## Mechanism

**Information cascades through gates.** V0 operates at the lowest cell
count in the pipeline (3 probes x seeds x splits ~ 120 cells on celiac).
Testing direction at V0 means selecting one representative level per
family — a single level chosen to be a fair mid-range stand-in for the
family's plausible operating point (e.g., `r = 5` for `downsample`, which
the ADR-003 default reflects; `weight = sqrt` for `weight`; no-op for
`none`). V0's AUROC evidence across these 3 probes picks the winning
family via the fixed falsifier rubric.

V3 then refines magnitude **conditional on V0's answer**. V3's grid
becomes one-dimensional: levels within the locked family only (e.g., 3
cells of `r_V3 in {1, 2, 5}` under `downsample`; 2 cells of
`{sqrt, log}` under `weight`; 1 trivial cell under `none`). Total
V0+V3 cells in the two-stage flow are substantially fewer than the
single-gate full-factorial, and crucially the V3 cells that execute are
always inside the winning family — no compute wasted on losing families.

**Single-gate alternative (full family x level factorial at one stage).**
Enumerates every combination upfront. On celiac this is ~7 alternatives
(3 downsample levels + 2 weight levels + 1 none + 1 probed baseline
~= 7 cells x seeds x splits ~ 280 cells at V0). The single-gate
formulation pays compute for levels inside losing families: if the
direction evidence eventually eliminates `weight` as a family, the
per-level exploration inside `weight` was wasted work that V0's
representative probe would have avoided.

The two-stage design is an information-efficiency optimization that
exploits the orthogonality: family direction is (approximately)
separable from within-family level, so conditioning the expensive level
search on the cheap family probe reduces total compute without
sacrificing the locked configuration's quality.

## Actionable rule

1. **V0 MUST probe imbalance with exactly one representative level per
   family.** [[protocols/v0-strategy]] locks `imbalance_family`
   (categorical: `none` | `downsample` | `weight`), NOT `control_ratio`.
   The representative probe levels are declared in the protocol and
   MUST NOT be cited as the "locked level" — they are direction probes
   only.
2. **V3 MUST branch on V0's family lock.** [[protocols/v3-imbalance]]
   reads V0's `locks: [imbalance_family]` and restricts its level grid
   to levels within that family. Level search outside the locked
   family is forbidden in the default flow.
3. **Fallback escape hatch.** If V0 returns Inconclusive on
   `imbalance_family` (neither Direction nor Equivalence met per
   SCHEMA §Fixed falsifier rubric), V3 falls back to the legacy full
   3x3 (weighting x downsampling) grid. This preserves the old
   single-gate behavior as a last-resort path when family direction
   cannot be adjudicated. The fallback branch is governed by
   [[condensates/nested-downsampling-composition]] (composition
   semantics apply only inside fallback).
4. **Ledger provenance.** V0's `ledger.md` MUST record which three
   levels it probes and declare them as representative. V3's
   `ledger.md` MUST declare which branch is active (`family_locked` |
   `fallback`) based on V0's observation.

## Boundary conditions

- **Does NOT apply when only one imbalance family is methodologically
  viable.** If the cohort's properties rule out `weight` (e.g., no
  loss-weighting hook available in the modeling library for the
  selected models) or `downsample` (e.g., prevalence is already
  balanced and sub-sampling further would shrink the training set
  below a minimum viable size), then single-stage level adjudication
  within the only viable family is correct and the two-stage rule
  does not apply.
- **Does NOT apply when level-within-family interacts with family
  choice.** Hybrid regimes like balanced+downsample, or
  weight-then-downsample cascades, break the orthogonality
  assumption — the optimal level is not separable from the family.
  In that regime, single-gate full-factorial is the correct design
  and the two-stage rule is inapplicable.
- **Requires V0's probe levels to be reasonably representative.** If
  V0's chosen probe level within a family is a poor stand-in for the
  family's best level (e.g., probing `r = 5` when the family's best
  level is `r = 1`, and the AUROC gap between `r = 5` and `r = 1`
  within `downsample` is larger than the cross-family AUROC gap), V0
  can lock the wrong family. Representative probe choice is a
  prerequisite condensate-level assumption.
- **When Equivalence holds on family at V0**, parsimony tiebreaker per
  [[condensates/parsimony-tiebreaker-when-equivalence]] applies
  (`none` ≺ `weight` ≺ `downsample` under the locally declared
  family parsimony order).

## Evidence

| Dataset | n | Gate | Phenomenon | Source |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | v0-strategy | Design-level: Strategy C (2026-04-21 revision) switches V0 from locking `control_ratio` (7-alternative full-imbalance gate, ~280 cells) to locking `imbalance_family` (3-probe family gate, ~120 cells). V3 refines level conditional on V0's family. Measured outcome pending — rb-v0.2.0 is a design-era condensate. | v0-strategy.md §2.2 post-Strategy-C revision 2026-04-21 |
| TODO | TODO | TODO | Second-cohort confirmation needed before promotion to `established` | TODO |

Promotion to `established` requires at least 3 dataset confirmations on
non-overlapping cohorts per SCHEMA §Rulebook updates. Current status is
`provisional` on design-era evidence only.

## Related

- [[protocols/v0-strategy]] — locks `imbalance_family` at the 3-probe
  gate under rb-v0.2.0
- [[protocols/v3-imbalance]] — refines level within V0's locked
  family; falls back to 3x3 grid on V0 Inconclusive
- [[condensates/nested-downsampling-composition]] — governs composition
  semantics in the fallback branch only
- [[condensates/v3-utility-provenance-chain]] — consumes the
  branch-appropriate prevalence from this two-stage design
- [[condensates/parsimony-tiebreaker-when-equivalence]] — applies when
  V0 family gate returns Equivalence, with a family-specific
  parsimony order
- [[condensates/downsample-preserves-discrimination-cuts-compute]] —
  grounds why `downsample` is a viable family candidate at all
- [[condensates/downsample-requires-prevalence-adjustment]] — required
  follow-up whenever the locked family is `downsample` (or the
  fallback branch runs `r_V0 * r_V3` composition)
- [[equations/case-control-ratio-downsampling]] — single-axis
  prevalence math consumed by the downstream adjustment
