---
type: equation
symbol: "n_{k,s}"
depends_on: []
computational_cost: "O(n)"
assumptions:
  - "stratification variable is a single categorical outcome (e.g., incident_CeD)"
  - "sample size is large enough that rounding error on per-stratum counts is negligible relative to the split fraction"
  - "the target proportions (f_TRAIN, f_VAL, f_TEST) sum to 1"
failure_modes:
  - "exact proportion drift when any stratum count is small (rounding integer allocation)"
  - "class imbalance leakage if stratification is skipped and a rare positive stratum lands disproportionately in one split"
  - "incorrect prevalence in the held-out splits when unlabeled or missing-outcome rows are not excluded before stratification"
---

# Stratified three-way split preserves per-stratum proportions across TRAIN, VAL, TEST

## Statement

For strata $s \in S$ (e.g., $s \in \{\text{case}, \text{control}\}$) and splits $k \in \{\text{TRAIN}, \text{VAL}, \text{TEST}\}$ with target fractions $(f_\text{TRAIN}, f_\text{VAL}, f_\text{TEST}) = (0.50, 0.25, 0.25)$:

$$n_{k,s} = \lfloor f_k \cdot n_s \rfloor \quad \text{with residual rows allocated deterministically to match } \sum_k n_{k,s} = n_s$$

Per-split prevalence:

$$\pi_k = \frac{n_{k,\text{case}}}{\sum_{s} n_{k,s}} \approx \pi \quad \text{for all } k$$

where $\pi$ is the overall cohort prevalence. The equality is exact in the limit of large $n_s$; integer-rounding slack is bounded by $|S|$ rows per split.

## Derivation

Stratified sampling partitions the population by stratum $s$ and independently applies the split fractions $f_k$ within each partition. Because each stratum is split with the same fractions, the within-stratum marginals $n_{k,s} / n_s$ converge to $f_k$, and the per-split prevalence

$$\pi_k = \frac{\sum_s n_{k,s} \cdot \mathbb{1}[s = \text{case}]}{\sum_s n_{k,s}}$$

reduces to the global prevalence $\pi$ up to integer-rounding. This is the standard finite-population result for proportional allocation (Cochran 1977, *Sampling Techniques*).

With a 50/25/25 split, TRAIN has twice the sample count of VAL or TEST, but the *composition* of each split matches the population. This is the property that permits VAL to be used for threshold calibration and TEST for unbiased evaluation without adjusting for shifted prevalence.

## Boundary conditions

- Applies when $n_s \gg |\text{splits}|$ for every stratum $s$; otherwise rounding causes visible drift.
- For the celiac cohort ($n_\text{case} = 148$, $n_\text{control} = 43{,}662$), a 50/25/25 split on cases yields $\{74, 37, 37\}$ — exact. On controls it yields $\{21{,}831, 10{,}916, 10{,}915\}$ — drift of one row, negligible.
- Does NOT guarantee balanced per-feature distributions (e.g., age, sex). Covariate-balanced stratification requires multi-way stratification on the join of outcome × covariates.
- Does NOT apply when prevalent vs incident cases must be kept separate in downstream splits — that case requires a two-stage allocation (see [[condensates/prevalent-restricted-to-train]]).

## Worked reference

Cohort: $n = 43{,}810$, $n_\text{case} = 148$, $n_\text{control} = 43{,}662$, $\pi = 0.00338$.

50/25/25 split stratified by `incident_CeD`:

| Stratum | n | TRAIN (50%) | VAL (25%) | TEST (25%) |
|---|---|---|---|---|
| case | 148 | 74 | 37 | 37 |
| control | 43,662 | 21,831 | 10,916 | 10,915 |
| **split total** | 43,810 | 21,905 | 10,953 | 10,952 |
| **split prevalence** $\pi_k$ | 0.00338 | 0.00338 | 0.00338 | 0.00338 |

Prevalence is preserved across all three splits to within $10^{-5}$. TRAIN carries 2× the rows of VAL or TEST but the same class composition.

## Sources

- Cochran (1977). *Sampling Techniques*, 3rd ed., Wiley.
- Steyerberg (2019). *Clinical Prediction Models*, 2nd ed., Springer — Chapter on model validation and split design.
- ADR-001 (cel-risk), 2026-01-20.

## Used by

- [[condensates/three-way-split-prevents-threshold-leakage]]
- [[condensates/prevalent-restricted-to-train]]
<!-- TODO: verify slug exists after batch 2/3 merge -->
- [[protocols/v0-strategy]] — referenced when locking `splits_strategy`
