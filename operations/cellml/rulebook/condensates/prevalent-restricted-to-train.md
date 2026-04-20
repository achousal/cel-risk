---
type: condensate
depends_on: ["[[equations/stratified-split-proportions]]"]
applies_to:
  - "cohorts with distinct incident (pre-diagnosis) and prevalent (post-diagnosis) case subpopulations"
  - "tasks framed as prospective screening (evaluation must reflect the prediction-from-pre-diagnosis-samples setting)"
  - "small incident-case cohorts where training signal is sample-limited"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "adds 75 prevalent positives to TRAIN (+51% over 148 incident-only) while VAL/TEST remain incident-only for prospective evaluation"
    date: "2026-01-20"
    source: "ADR-002, docs/adr/ADR-002-prevalent-train-only.md"
falsifier: |
  Direction: if a pipeline trained with prevalent cases added to TRAIN at
  fraction ~0.5 outperforms the incident-only TRAIN baseline by |Δ AUROC| ≥ 0.02
  on the incident-only TEST split with 95% bootstrap CI excluding 0, the
  condensate is confirmed as Direction (prevalent augmentation helps). If the
  incident-only baseline and the prevalent-augmented model differ by |Δ| < 0.01
  with 95% CI ⊂ [−0.02, 0.02] on TEST, the condensate is weakened to
  Equivalence — prevalent cases did not improve prospective prediction and
  should be dropped from TRAIN to simplify the recipe.
---

# Prevalent (post-diagnosis) cases belong in TRAIN only when held-out splits must preserve a prospective evaluation

## Claim

When a cohort contains both incident cases (biomarkers collected before diagnosis, i.e., truly prospective) and prevalent cases (biomarkers collected after diagnosis, retrospective), the two subpopulations are drawn from measurably different distributions. Using prevalent cases in VAL or TEST contaminates the evaluation by mixing retrospective signal into a held-out split that is supposed to estimate forward-looking screening performance. Using them in TRAIN at a capped sampling fraction (ADR-002 locks 50%) is defensible: it augments the positive-class training signal without leaking the retrospective distribution into either tuning or evaluation surfaces.

## Mechanism

Biomarkers in prevalent cases reflect established disease, post-treatment biology, or survivorship bias. They are systematically different from incident-case biomarkers drawn before diagnosis. Mixing them into VAL or TEST shifts the distribution of the positive class in the held-out splits away from the clinically meaningful "prospective screening" distribution. TRAIN can absorb the augmentation because the model is being exposed to the positive decision boundary from multiple angles — the train-vs-holdout distribution shift becomes an empirical question about transfer, not an evaluation artifact.

See [[equations/stratified-split-proportions]]: the stratified split rule operates on the incident label alone. Prevalent cases are injected into TRAIN after stratification, breaking the symmetry that made VAL and TEST identically distributed to the population. That asymmetry is the point — VAL/TEST mirror the prospective target, TRAIN does not need to.

## Actionable rule

- The V0 gate locks the prevalent-to-TRAIN injection with `prevalent_train_frac` ∈ {0.0, 0.5, 1.0}. The 0.5 choice (ADR-002 default) is a prior, not a lock — the factorial should test it.
- VAL and TEST splits MUST remain incident-only. Hook-enforced by `add_prevalent_to_train` (prevalent IDs are filtered out of non-TRAIN splits by construction).
- If a new cohort lacks a meaningful incident/prevalent distinction (e.g., only prospective sampling), this condensate does not apply and the axis should be dropped from the factorial rather than defaulted.

## Boundary conditions

- Applies when incident and prevalent case distributions are measurably different (typically true for immune / metabolic / cardiovascular outcomes where biomarkers respond to diagnosis or treatment).
- Does NOT apply when biomarkers are temporally stable relative to disease onset (e.g., fixed germline features). In that regime, prevalent and incident draw from the same distribution and the TRAIN-only restriction is unnecessary.
- The 50% sampling fraction is an untested prior. It trades signal gain (more positives) against distribution shift (more retrospective bias in TRAIN). The factorial V0 axis should measure AUROC at `prevalent_train_frac` ∈ {0.0, 0.5, 1.0} to choose empirically.

## Evidence

| Dataset | n_incident | n_prevalent | Sampling frac | Observed effect | Source gate |
|---|---|---|---|---|---|
| Celiac (UKBB) | 148 | 150 | 0.5 → +75 TRAIN positives | No incident-only baseline measured yet (ADR-002 asserts structure, not delta) | v0-strategy 2026-01-20 |

## Evidence gaps

ADR-002 asserts a 50% sampling fraction as "balanced signal vs shift" but provides no empirical justification — no AUROC delta measured between `prevalent_train_frac=0.0`, `0.5`, and `1.0`. This is a candidate tension: ADR-002's 50% choice is a methodological prior with no data behind it on this cohort. The V0 factorial gate is the place to resolve it; a downstream observation that `0.5` is dominated by `0.0` or `1.0` would trigger a condensate revision. Flag as "prior, not measurement."

## Related

- [[equations/stratified-split-proportions]] — split mechanics that this rule operates within
- [[condensates/three-way-split-prevents-threshold-leakage]] — the held-out split principle this rule protects
- [[condensates/downsample-preserves-discrimination-cuts-compute]] — downsampling is applied after this rule injects prevalent cases
<!-- TODO: verify slugs exist after batch 2/3 merge -->
- ADR-002 (cel-risk, 2026-01-20) — source decision record
