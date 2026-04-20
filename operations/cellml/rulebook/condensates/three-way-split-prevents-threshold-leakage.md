---
type: condensate
depends_on: ["[[equations/stratified-split-proportions]]"]
applies_to:
  - "binary outcome prediction with a decision threshold"
  - "calibrated probability models with post-hoc threshold selection"
  - "datasets where held-out evaluation is meant to be unbiased (no hyperparameter or threshold reuse)"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "TEST used as threshold-selection surface in 2-way splits → downstream TEST AUROC/Brier become optimistic estimates of generalization"
    date: "2026-01-20"
    source: "ADR-001, docs/adr/ADR-001-split-strategy.md"
falsifier: |
  Direction: if TEST metrics (AUROC, Brier) from a 2-way (TRAIN / TEST) pipeline
  with threshold tuned on TEST versus a 3-way (TRAIN / VAL / TEST) pipeline with
  threshold tuned on VAL differ by |Δ| ≥ 0.02 with a 95% bootstrap CI excluding
  0 across at least 3 datasets, this condensate is confirmed as Direction. If
  |Δ| < 0.01 and 95% CI ⊂ [−0.02, 0.02] on those same datasets, the condensate
  is weakened to Equivalence and should be retired.
---

# A three-way TRAIN/VAL/TEST split is required whenever any threshold, cutoff, or calibrator fit must be kept out of the final evaluation surface

## Claim

Two-way splits (TRAIN + TEST) force every post-training decision — calibration intercepts, decision thresholds, operating-point cutoffs — to be fit on TEST, which then stops being an unbiased evaluation surface. Introducing a VAL split absorbs all post-training parameter fits and preserves TEST as a held-out estimator of true generalization. On small-n cohorts (fewer than ~200 positives), a 50/25/25 stratified split is the minimum-viable three-way allocation: TRAIN gets the largest share for nested CV tuning, while VAL and TEST each retain enough samples to estimate a threshold and a held-out metric with acceptable variance.

## Mechanism

Any parameter fit on TEST — threshold $\tau$, calibration slope, operating point — induces selection bias when that same TEST is then used to report the final metric. The bias is not random: it systematically improves the reported score. VAL absorbs this fit, so TEST sees the fully-specified model with no further degrees of freedom.

See [[equations/stratified-split-proportions]]: the 50/25/25 allocation preserves per-stratum prevalence across all three splits, so VAL and TEST are each representative of the population and can each play their distinct role (threshold surface vs evaluation surface) without class-composition artifacts.

## Actionable rule

- Any gate that selects an operating threshold, calibration intercept, or cutoff MUST use a dedicated VAL split distinct from TEST.
- The V0 gate locks `splits_strategy` to the 50/25/25 stratified form. Deviations below 40% for TRAIN, 20% for VAL, or 20% for TEST require documented justification and a re-run of prior locks.
- Stratification variable MUST match the primary outcome label (for celiac: `incident_CeD`).
- Schema validators (`SplitsConfig.validate_split_sizes`) must reject configurations with fewer than three split labels when any downstream step fits a threshold.

## Boundary conditions

- Applies when ANY parameter is selected post-training. Pure pre-registered models with no post-hoc fitting can tolerate a two-way split (but forfeit the ability to recalibrate).
- Does NOT apply to repeated-cross-validation-only pipelines with no single held-out test — those use nested CV for unbiased estimation and do not need a separate VAL.
- Below roughly 30 positives per split, the variance on threshold selection becomes large enough that a 60/20/20 or even a nested-CV-only protocol may outperform 50/25/25. The celiac cohort is at the lower edge of the regime where 3-way is feasible ($n_\text{case}=148$ yields 37 positive TEST cases).

## Evidence

| Dataset | n_case | n_control | Observed effect | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 148 | 43,662 | 3-way locked at 50/25/25 per ADR-001; no comparison 2-way run measured yet — decision driven by Steyerberg (2019) recommendation | v0-strategy 2026-01-20 |

## Evidence gaps

ADR-001 cites Steyerberg (2019) as the methodological basis but does NOT include a head-to-head measurement on celiac of the 2-way vs 3-way split's effect on TEST-metric bias. This condensate is asserted on methodological grounds, not on empirical delta. A candidate V0 gate measurement would re-run a pipeline subset under a 2-way split and record the optimism bias directly. Flag as "methodological assertion, not yet empirically confirmed on this cohort" — candidate tension if a later dataset shows Equivalence.

## Related

- [[equations/stratified-split-proportions]] — math of proportional allocation
- [[condensates/prevalent-restricted-to-train]] — uses the same VAL/TEST held-out principle
- [[condensates/downsample-preserves-discrimination-cuts-compute]] — downsampling operates within the splits this rule produces
<!-- TODO: verify slugs exist after batch 2/3 merge -->
- ADR-001 (cel-risk, 2026-01-20) — source decision record
