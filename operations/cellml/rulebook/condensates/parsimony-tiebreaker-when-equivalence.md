---
type: condensate
depends_on:
  - "[[condensates/calib-parsimony-order]]"
applies_to:
  - "any factorial axis where the fixed falsifier rubric returns Equivalence between a lower-complexity and higher-complexity option"
  - "axes with a declared, locally-grounded parsimony order (V1 panel size, V2 model complexity, V3 weighting, V3 downsampling, V4 calibrator)"
  - "discrimination-type gate decisions (V1 AUROC, V2 AUROC/REL, V3 utility, V4 REL) where the rubric claim type is Equivalence rather than Direction"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v4-calibration
    delta: "logistic_intercept < beta < isotonic parsimony tiebreaker locked via ADR-008 and formalized in [[condensates/calib-parsimony-order]]; analogous axis-specific parsimony orders are codified in v1-recipe §3.2 (panel size), v2-model §3.3 (model complexity), v3-imbalance §3.4 (weighting, downsampling) and are each invoked only when the rubric returns Equivalence on that axis"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §Factorial Factors Parsimony Ordering; ADR-008; DECISION_TREE_AUDIT.md §1.1 / §5 P2 resolution"
falsifier: |
  Direction claim: if on at least 3 datasets the higher-complexity option
  beats the lower-complexity option on a distinct-but-correlated metric
  (e.g., calibration Brier reliability while AUROC is Equivalence-tied;
  or TEST-set discrimination while VAL-set discrimination is Equivalence-
  tied) with the Direction criterion from the fixed falsifier rubric
  (|Delta| >= axis-specific threshold AND 95% bootstrap CI excludes 0),
  this meta-condensate is WEAKENED for that axis-metric pair. Observing
  this inversion on at least 5 datasets with non-overlapping cohorts -> retire
  the meta-claim for that axis-metric pair (the axis-specific parsimony
  order remains in force until its own condensate is retired).
---

# When the fixed falsifier rubric returns Equivalence on a factorial axis, the lock goes to the lower-complexity option per that axis's declared parsimony order

## Claim

The cel-risk factorial resolves gate-level ties on multiple axes (panel size
at V1, model family at V2, weighting and downsampling at V3, calibrator
family at V4) using axis-specific parsimony orderings. This meta-condensate
states the shared rule: **whenever the SCHEMA.md §Fixed falsifier rubric
returns Equivalence (|Delta| < epsilon AND 95% bootstrap CI inside the
Equivalence band) between two options on a factorial axis, the gate MUST
lock the lower-complexity option per that axis's declared parsimony order**.
This is a parent claim over the axis-specific parsimony condensates
([[condensates/calib-parsimony-order]] being the canonical instance); it
does NOT replace the axis-specific orders, which remain locally declared
and locally falsifiable.

## Mechanism

Parsimony functions as a **bias-variance prior for generalization when
discrimination evidence is tied**. When two options are statistically
indistinguishable on the gate's primary metric (rubric Equivalence on
AUROC, REL, AUPRC, or the V3 utility), the observed tie is not null
evidence: it is evidence that any extra capacity afforded by the
higher-complexity option is not paying for itself within the selection-
seed sample. Out of sample, the extra capacity trades off against variance
— more parameters fit more noise — so the simpler option has lower
expected TEST-set metric deterioration. Van Calster et al. (2019) and
Niculescu-Mizil & Caruana (2005) document this bias-variance tradeoff in
clinical prediction; [[condensates/calib-parsimony-order]] instantiates it
for the V4 calibrator axis.

The meta-claim strengthens when the tie is observed on a paired bootstrap
over outer folds (per SCHEMA.md metric rules), because paired-bootstrap
Equivalence is a tighter statistical statement than point-estimate
closeness. The meta-claim weakens when the tie is driven by measurement
noise rather than true equivalence (see boundary conditions §4).

## Actionable rule

1. **Gate protocols cite this condensate when applying tiebreakers.** V1,
   V2, V3, V4 ledgers that invoke an Equivalence-triggered parsimony lock
   MUST cite both this meta-condensate and the axis-specific parsimony
   condensate (or DESIGN.md §Parsimony Ordering where no axis-specific
   condensate exists yet).
2. **Each axis declares its parsimony order locally.** Axis-specific
   parsimony orders live in their own condensates or protocol sections;
   this meta-condensate does not enumerate them. The canonical local
   declarations at 2026-04-20 are:
   - V1 panel size: smaller `p` wins on Equivalence (see [[protocols/v1-recipe]] §3.2).
   - V2 model complexity: LR_EN ≺ LinSVM_cal ≺ RF ≺ XGBoost (see [[protocols/v2-model]] §3.3).
   - V3 weighting: `none` ≺ `sqrt` ≺ `log` (see [[protocols/v3-imbalance]] §3.4).
   - V3 downsampling: `1.0` ≺ `2.0` ≺ `5.0` (see [[protocols/v3-imbalance]] §3.4).
   - V4 calibrator: `logistic_intercept` ≺ `beta` ≺ `isotonic` (see [[condensates/calib-parsimony-order]]).
3. **Claim type must be named explicitly.** When a gate invokes this
   meta-condensate, `decision.md` under `## Actual claim type (per rubric)`
   must record `Equivalence` (not `Direction`, not `Inconclusive`) and
   name the axis-specific parsimony order invoked.
4. **No silent parsimony locks.** A gate that locks a lower-complexity
   option without an Equivalence claim from the rubric is NOT a parsimony
   tiebreaker and this meta-condensate does not apply — the gate is
   either a Direction lock (which does not need parsimony) or an
   Inconclusive fallback (which is governed by the protocol's fallback
   section, not this meta-claim).

## Boundary conditions

- **Does NOT apply to informational gates.** V6 (ensemble) is
  informational per DESIGN.md §Design Constraint: Single-Model Primary
  and does not lock factorial axes; parsimony tiebreakers are not
  invoked at V6 because no lock is issued there.
- **Does NOT apply when Equivalence is driven by measurement noise
  rather than true equivalence.** If the paired-bootstrap CI half-width
  is larger than the axis-specific Equivalence band's upper bound
  (signaling under-powered estimation rather than genuine closeness),
  the correct gate outcome is Inconclusive, not Equivalence. The
  protocol's fallback (widen trials, extend seeds, or escalate to human
  review) takes precedence over the parsimony tiebreaker.
- **Does NOT generalize across axis-metric pairs.** The meta-claim is
  evaluated per (axis, metric) pair. Observing that parsimony holds for
  V4 calibrator on REL does not transfer evidence to V2 model on AUROC;
  each axis-metric pair accumulates its own confirmations and
  potentially its own weakening evidence.
- **Does NOT apply when the rubric claim is Direction or Dominance.** By
  construction, Direction and Dominance resolve the axis without needing
  a tiebreaker. Parsimony is only consulted when the rubric cannot
  adjudicate on primary-metric magnitude and CI.
- **Applies only with a locally declared parsimony order.** An axis
  without a declared order cannot invoke this meta-condensate; the gate
  must either declare the order at ledger entry or forward the
  Equivalence outcome as Inconclusive for protocol fallback.

## Evidence

| Dataset | n | Gate | Phenomenon | Source |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | v4-calibration | Locks `logistic_intercept` under Equivalence via the calibrator parsimony order; ADR-008 codifies the order and [[condensates/calib-parsimony-order]] formalizes it | ADR-008; [[condensates/calib-parsimony-order]] 2026-01-22 |
| Celiac (UKBB) | 43,810 | v2-model / v3-imbalance / v1-recipe | Axis-specific parsimony orders are declared in the protocols (LR_EN ≺ LinSVM_cal ≺ RF ≺ XGBoost; `none` ≺ `sqrt` ≺ `log`; `1.0` ≺ `2.0` ≺ `5.0`; smaller `p` wins) but lack dedicated condensates — this meta-claim is the parent that binds them under the shared Equivalence-triggered rule | DESIGN.md §Factorial Factors Parsimony Ordering; DECISION_TREE_AUDIT.md §1.1 / §5 P2 resolution 2026-04-08 |

## Related

- [[condensates/calib-parsimony-order]] — axis-specific instantiation for V4 calibrator (first confirmed instance of this meta-claim)
- [[protocols/v1-recipe]] §3.2 — panel-size parsimony tiebreaker
- [[protocols/v2-model]] §3.3 — model-complexity parsimony tiebreaker
- [[protocols/v3-imbalance]] §3.4 — weighting and downsampling parsimony tiebreakers
- [[protocols/v4-calibration]] §3.2 — calibrator parsimony tiebreaker (top-down decision tree)
- DESIGN.md §Factorial Factors Parsimony Ordering — canonical source for the five axis-specific orders
- ADR-008 — canonical source for the V4 calibrator parsimony ordering (first axis promoted to a condensate)
- DECISION_TREE_AUDIT.md §1.1 / §5 P2 resolution (2026-04-08) — codifies the parsimony tiebreaker across V2 / V3
