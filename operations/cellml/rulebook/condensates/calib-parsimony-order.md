---
type: condensate
depends_on: ["[[equations/platt-scaling]]", "[[equations/beta-calib]]", "[[equations/isotonic-calib]]", "[[equations/brier-decomp]]", "[[condensates/parsimony-tiebreaker-when-equivalence]]"]
applies_to:
  - "calibration strategy selection when multiple candidates tie on reliability"
  - "small-to-medium calibration sets (n_cal_positives < 500)"
  - "V4 calibration gate in the cel-risk factorial"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v4-calibration
    delta: "parsimony ordering logistic_intercept (1 param) < beta (3 params) < isotonic (~n_cal params) is enforced as tiebreaker when reliability CIs overlap"
    date: "2026-01-22"
    source: "ADR-008, operations/cellml/DESIGN.md §Factorial Factors Parsimony Ordering, DESIGN.md §V4"
falsifier: |
  Direction claim: when the Equivalence criterion holds on REL between a
  simpler and a more complex calibrator, selecting the simpler calibrator
  yields equal-or-better TEST-set REL with 95% bootstrap CI crossing 0. If
  the more complex calibrator beats the simpler one (Direction, |Delta REL|
  >= 0.02) on TEST despite Equivalence on VAL, the parsimony ordering is
  weakened. Observing this inversion across at least 3 datasets with
  n_cal_positives > 100 -> retire.
---

# Calibrator selection obeys the parsimony order logistic_intercept then beta then isotonic

## Claim

When two calibration strategies produce reliability estimates that fall inside the Equivalence band of the fixed falsifier rubric ($|\Delta\text{REL}| < 0.01$ AND 95% bootstrap CI $\subset [-0.02, 0.02]$), the V4 gate MUST prefer the **simpler** calibrator. The complexity ordering is

$$\text{logistic\_intercept} \;(\text{1 param}) \;\prec\; \text{beta} \;(\text{3 params}) \;\prec\; \text{isotonic} \;(\sim n_\text{cal} \text{ params})$$

This is a specific instance of the broader parsimony tiebreaker used across the factorial (see DESIGN.md §Factorial Factors Parsimony Ordering), justified by the fact that simpler calibrators have lower variance out-of-sample when in-sample REL is statistically indistinguishable.

## Mechanism

The three calibrator families are **strictly nested** in expressive power:
- `logistic_intercept` fixes slope $A = 1$ and fits intercept only ([[equations/platt-scaling]], one-parameter variant).
- `beta` generalizes to a 3-parameter monotone sigmoid ([[equations/beta-calib]]); when $a = b = 0$, beta reduces to logistic_intercept.
- `isotonic` is the non-parametric monotone calibrator ([[equations/isotonic-calib]]); in the large-$n$ limit it dominates any parametric monotone form in-sample.

Nesting implies that a more complex calibrator can **only** reduce in-sample REL relative to a simpler one (more parameters fit more noise). If the observed $\Delta\text{REL}$ on VAL is within the Equivalence band despite the higher-complexity calibrator having strictly more freedom, the extra freedom was not productive on this cohort — it fit noise. Out of sample, the simpler calibrator has lower variance and thus lower expected TEST REL.

Niculescu-Mizil & Caruana (2005) and Van Calster et al. (2019) both document this bias-variance tradeoff in calibration: with $n_\text{cal\_positives} \lesssim 100$, isotonic typically overfits; with $n_\text{cal\_positives}$ in the thousands, isotonic starts to pay off.

## Actionable rule

- V4 gate comparison: bootstrap CI on $\Delta\text{REL}$ between each pair of candidate calibrators over 1000 outer-fold resamples.
- Decision tree:
    1. If `isotonic` beats `beta` by Direction criterion (|Delta REL| >= 0.02 AND CI excludes 0) -> pick isotonic.
    2. Else if `beta` beats `logistic_intercept` by Direction -> pick beta.
    3. Else (Equivalence or Inconclusive for all pairwise comparisons) -> pick `logistic_intercept` (parsimony tiebreaker).
- Also applies across other axes: logistic_intercept < beta < isotonic is ONE of five parsimony orderings in the factorial (see DESIGN.md §Parsimony Ordering). The tiebreaker logic is shared.
- `none` is a valid fourth option not covered by this ordering — use it when the base model's reliability is already within the Equivalence band of logistic_intercept AND the extra transformation step is an undesirable deployment complexity.

## Boundary conditions

- Applies at the V4 gate specifically, where calibration is applied post-hoc after all training-time decisions (V1–V3) are locked.
- Does NOT apply when the base model is non-monotonically miscalibrated. In that case isotonic is the only family in this list that can fit the shape, and parsimony is irrelevant. Such cohorts should be flagged in the ledger and escalated.
- Requires a valid calibration set under [[condensates/calib-per-fold-leakage]] — i.e., `oof_posthoc` predictions. Per-fold calibration's optimism bias can inflate the apparent advantage of complex calibrators, which would push the decision wrongly toward isotonic.
- Applies only when $n_\text{cal\_positives} \gtrsim 30$; below that, even logistic_intercept's single parameter has high variance, and the V4 gate should return Inconclusive and defer to the clinical default.

## Evidence

| Dataset | n_cal_positives | Phenomenon | Source gate |
|---|---|---|---|
| Celiac (UKBB) | 148 (selection) / 37 (confirmation VAL slice) | ADR-008 locks the parsimony ordering as the V4 tiebreaker; empirical comparison is the factorial's V4 gate | V4 calibration |

## Related

- [[condensates/parsimony-tiebreaker-when-equivalence]] — parent meta-condensate; this calibrator-axis ordering is the canonical instance of the shared Equivalence-triggered parsimony rule, and every V4 lock under Equivalence cites both condensates
- [[equations/platt-scaling]], [[equations/beta-calib]], [[equations/isotonic-calib]] — the three families in the ordering
- [[equations/brier-decomp]] — the REL metric that enters the comparison
- [[condensates/calib-per-fold-leakage]] — complements this condensate; addresses which SLICE the calibrator is fit on
- ADR-008 (cel-risk) — decision record this condensate formalizes
<!-- TODO: verify slug exists after batch merge --> - [[protocols/v4-calibration]]
