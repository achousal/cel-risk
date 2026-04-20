---
type: condensate
depends_on: ["[[equations/perm-test-pvalue]]"]
applies_to:
  - "any model with data-dependent feature selection"
  - "high-dimensional settings (p > n)"
  - "pipelines with hyperparameter tuning"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "models passing partial-permutation p<0.05 failed full-pipeline p<0.05 in 3/4 cases"
    date: "2026-02-04"
    source: "ADR-011, operations/cellml/DESIGN.md"
falsifier: |
  If partial-permutation (shuffle after feature selection) and full-pipeline
  permutation produce p-values within ±0.01 across at least 3 datasets with
  p > n, this condensate is weakened. At least 5 such datasets -> retire.
---

# Permutation validity requires full-pipeline re-execution

## Claim

Permutation p-values are valid tests of model signal only when the FULL inner
pipeline — screening, feature selection, hyperparameter tuning — re-runs under
each permuted label set. Partial permutation (shuffling after feature
selection) leaks overfit signal into the null distribution and produces
anti-conservative p-values.

## Mechanism

Feature selection is data-dependent. With $p \gg n$ and shuffled labels, a
sufficiently flexible selector still finds features that correlate with the
shuffled labels by chance. If the null is built AFTER fixing features, the null
already contains selector-induced signal, so the observed AUROC no longer
stands out.

See [[equations/perm-test-pvalue]]: the $\text{null}_b$ distribution must
reflect the same overfitting pressure as $\text{observed}$ for the
exchangeability argument to hold.

## Actionable rule

- Gates using permutation testing MUST re-run the full pipeline per permutation.
- CLI must expose `--permute-at=labels` (correct). Reject `--permute-at=features`.
- $B$ tuned per phase: at least 50 for CI smoke, 200 for V0 gate, 1000 for
  confirmation.

## Boundary conditions

- Applies when feature count > sample count OR when tuning adapts to labels.
- Does NOT apply to pre-registered models with fixed features and no tuning;
  partial and full permutation converge in that regime.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | RF partial-perm p=0.03 vs full-perm p=0.11 pre-calibration | v0-strategy 2026-02-04 |

## Related

- [[equations/perm-test-pvalue]] — math
- [[protocols/v0-strategy]] — where this rule is applied
- ADR-004 (Stage 1 model gate) — to be migrated into this condensate plus
  sibling condensates for Stage 2 and Stage 3 of the selection workflow
