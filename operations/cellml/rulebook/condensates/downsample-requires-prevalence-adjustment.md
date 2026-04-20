---
type: condensate
depends_on: ["[[equations/case-control-ratio-downsampling]]"]
applies_to:
  - "probability outputs reported as absolute risk (not pure ranking)"
  - "calibration metrics (Brier, ECE, reliability) computed on held-out splits"
  - "decision thresholds expressed in terms of population-scale risk cutoffs"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "raw downsampled-model output gives P(case|x, π_train=0.167); population prevalence π ≈ 0.00338, so unadjusted outputs overstate risk by roughly 49×"
    date: "2026-01-20"
    source: "ADR-003, docs/adr/ADR-003-control-downsampling.md"
falsifier: |
  Direction: if the Brier / ECE of a downsampled-but-prevalence-adjusted model
  on a population-representative TEST split exceeds the Brier / ECE of a
  no-downsampling baseline by |Δ| ≥ 0.02 with 95% bootstrap CI excluding 0,
  the adjustment is insufficient and this condensate is retired in favor of
  no-downsampling. If the adjusted Brier matches the baseline (|Δ| < 0.01,
  95% CI ⊂ [−0.02, 0.02]), the condensate holds as Equivalence.
---

# Downsampling-induced training prevalence shift requires an explicit Bayes prevalence correction before probability outputs or calibration metrics can be trusted

## Claim

Downsampling controls changes the training prevalence from the population rate $\pi$ to the downsampled rate $\pi_\text{train} = 1 / (1 + r)$. Models fit under this shifted marginal estimate $P(Y=1 | X, \pi_\text{train})$, not $P(Y=1 | X, \pi)$. Feeding the raw output to downstream calibration, decision thresholding, or absolute-risk reporting produces overstated risk (by a factor of roughly $\pi_\text{train} / \pi$ in the low-score regime). A Bayes-rule prevalence correction applied after model output and before calibration restores population-scale probabilities — and this correction is non-optional whenever the pipeline reports anything but pure score rankings.

## Mechanism

The label-shift correction (Elkan 2001; Saerens et al. 2002) rescales the odds by the ratio of marginal class priors:

$$\text{odds}_\text{pop}(x) = \text{odds}_\text{train}(x) \cdot \frac{\pi}{1 - \pi} \cdot \frac{1 - \pi_\text{train}}{\pi_\text{train}}$$

The correction assumes $P(X | Y)$ is class-conditionally stationary — only the class marginal changed. Under random downsampling of controls, this holds by construction: each retained control was sampled i.i.d. from the population control distribution.

See [[equations/case-control-ratio-downsampling]] for the full probability transform. Discrimination (AUROC, ranking) is unaffected because it is monotonic in the score; calibration is rescaled linearly in log-odds space.

## Actionable rule

- Any pipeline that reports probabilities OR computes calibration-based metrics (Brier, ECE, reliability curves) MUST apply the prevalence correction between raw model output and downstream use.
- The correction MUST be applied using the population prevalence $\pi$ measured on the un-downsampled cohort (before the 50/25/25 split), not the per-split prevalence.
- The V0 gate observation.md MUST log both the raw-prevalence metric and the prevalence-adjusted metric. Silent substitution of one for the other is a provenance failure.
- Calibration fits (logistic intercept, beta, isotonic) fit on VAL MUST operate on prevalence-adjusted scores, not raw downsampled-model scores — otherwise the calibrator absorbs the prevalence shift into its slope and masks the issue.

## Boundary conditions

- Applies when training prevalence differs from the evaluation/deployment prevalence. Both controlled downsampling (this ADR) and natural distribution shift between train and deployment require the same correction.
- Does NOT apply when only AUROC / ranking metrics are reported — those are invariant to prevalence and the correction is a no-op for score ordering.
- Does NOT apply when $P(X | Y)$ is NOT class-conditionally stationary (covariate shift dominates). In that regime the correction is unreliable and a full domain-adaptation step is required.
- The correction assumes $\pi$ is known exactly. If the population prevalence is itself estimated from the same cohort, uncertainty on $\pi$ propagates into the corrected probabilities — the celiac cohort has $n_\text{case}=148$, so the standard error on $\pi$ is $\sqrt{\pi(1-\pi)/n} \approx 0.003\%$, negligible. On smaller cohorts this becomes material.

## Evidence

| Dataset | π | π_train at r=5 | Magnitude of correction at low-score regime | Source |
|---|---|---|---|---|
| Celiac (UKBB) | 0.00338 | 0.1667 | $\pi_\text{train} / \pi \approx 49\times$ overstatement without correction | ADR-003, 2026-01-20 |

## Evidence gaps

ADR-003 notes "requires prevalence adjustment" as a consequence but does not specify the exact correction formula used, nor report a measured Brier-score delta for adjusted vs unadjusted outputs. This condensate fixes the formula (Bayes label-shift per Elkan 2001) but the actual cel-risk implementation of the adjustment should be traced to code — a verification task for the V0 gate. Flag: "rule specified, implementation trace pending."

## Related

- [[equations/case-control-ratio-downsampling]] — the prevalence-shift math this adjustment undoes
- [[condensates/downsample-preserves-discrimination-cuts-compute]] — the reason downsampling is used at all; this condensate is the required follow-up
- [[condensates/three-way-split-prevents-threshold-leakage]] — calibration must be fit on VAL, under corrected probabilities
<!-- TODO: verify slugs exist after batch 2/3 merge -->
- ADR-003 (cel-risk, 2026-01-20) — source decision record
- Elkan (2001); Saerens, Latinne, Decaestecker (2002) — theoretical basis
