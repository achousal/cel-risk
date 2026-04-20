---
type: equation
symbol: "p_perm"
depends_on: []
computational_cost: "O(B * C_inner)"
assumptions:
  - "null-hypothesis samples are exchangeable with permuted labels"
  - "full inner pipeline re-runs under each permutation to absorb overfitting"
failure_modes:
  - "data leakage if permutation happens after any data-dependent preprocessing"
  - "under-powered p-values when B < 200 for claims near alpha"
---

# Permutation test p-value with +1 correction

## Statement

$$p = \frac{1 + \#\{\text{null}_b \ge \text{observed}\}}{1 + B}$$

- $B$: permutation count
- $\text{null}_b$: AUROC from the $b$-th permutation, trained on shuffled labels and evaluated on the unpermuted held-out fold
- $\text{observed}$: AUROC from the unpermuted pipeline

## Derivation

Exchangeability under H0 implies each permutation is equally likely under the
null. The fraction of null values meeting or exceeding the observed statistic
is a consistent estimator of the p-value.

The +1 correction (Phipson & Smyth 2010) prevents p=0 when the observed
statistic exceeds all $B$ permutations. Without correction, type-I error is
understated as $B \to \infty$.

## Boundary conditions

- Applies only when the FULL inner pipeline (screening, feature selection,
  tuning) re-runs per permutation. Permuting only at fit-time leaks overfit
  signal through fixed-feature selection. See
  [[condensates/perm-validity-full-pipeline]].
- $B$ must be large enough that the smallest reportable $p = 1/(1+B)$ is below
  the intended alpha. For alpha=0.05: $B \ge 200$. For publication: $B \ge 1000$.
- Requires outer-fold structure to be preserved.

## Worked reference

Observed AUROC = 0.72, $B$ = 200, $\#\{\text{null} \ge 0.72\}$ = 4.

$$p = \frac{1 + 4}{1 + 200} = \frac{5}{201} = 0.0249$$

Report $p = 0.025$; model signal is significant at alpha=0.05.

## Sources

- Ojala & Garriga (2010). JMLR 11:1833-1863.
- Phipson & Smyth (2010). Stat Appl Genet Mol Biol 9(1).

## Used by

- [[condensates/perm-validity-full-pipeline]]
- [[protocols/v0-strategy]] — model-gate filter
