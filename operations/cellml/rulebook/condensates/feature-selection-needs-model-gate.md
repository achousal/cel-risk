---
type: condensate
depends_on:
  - "[[equations/perm-test-pvalue]]"
  - "[[equations/consensus-rank-aggregation]]"
applies_to:
  - "cross-model feature-consensus workflows"
  - "any pipeline aggregating per-model importance rankings"
  - "high-dimensional settings (p > n) with mixed model families"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v0-strategy
    delta: "aggregating across all 4 models (unfiltered) shifts the consensus top-4 vs. aggregating only across Stage-1-passing models"
    date: "2026-02-09"
    source: "ADR-004 (three-stage feature selection, Stage 1 model gate)"
falsifier: |
  Direction claim: consensus panel derived from gated models (Stage 1 p < 0.05)
  differs from consensus panel derived from ungated models. Criterion:
  Jaccard overlap of top-N panels < 0.80 AND 95% bootstrap CI on Jaccard
  excludes 0.80. If |ΔJaccard| < 0.01 with 95% CI inside [−0.02, 0.02] (Equivalence)
  across >=3 datasets with p > n and mixed model families, this condensate is
  weakened. Retire at 5 such datasets.
---

# Cross-model feature consensus is only valid over models that passed the permutation test model gate

## Claim

Geometric-mean rank aggregation across model families (ADR-004 Stage 3) produces a deployment panel that reflects signal only when restricted to models whose label-permutation p-value is below alpha (ADR-004 Stage 1, default alpha = 0.05). Including models that failed the permutation test mixes noise rankings into the consensus; because the geometric-mean aggregator weights every included model equally, one noise model per two signal models is enough to shift the top panel. The model gate is not a courtesy check — it is a prerequisite for the consensus equation to be meaningful.

## Mechanism

See [[equations/perm-test-pvalue]] for the gate test and [[equations/consensus-rank-aggregation]] for the aggregator. A model that fails the Stage 1 gate has not demonstrated signal: its ranked importance list is consistent with what a shuffled-label pipeline would produce. Under the geometric-mean formula

$$s_i = \left( \prod_{m} N_m / r_{i,m} \right)^{1/|\mathcal{M}|}$$

each included model contributes multiplicatively. A noise model assigns effectively random $r_{i,m}$ for every protein — injecting a uniform factor into the product that decorrelates $s_i$ from the ranks that the signal-bearing models agree on. This is why ADR-004 separates the model gate (Stage 1) from the aggregator (Stage 3): the aggregator has no internal defense against garbage-in.

## Actionable rule

- V0 (strategy) gate must emit a per-model permutation-test p-value artifact before any consensus construction is attempted.
- V1 (recipe) panel-building protocol MUST consume only models from the Stage 1 pass-list.
- If zero models pass Stage 1 at alpha = 0.05, the aggregator does not run. The correct fallback is to revisit feature engineering or cohort definition, not to relax alpha or lower the gate threshold to "get a panel".
- Pass-list is locked at V0 exit. If a new model is added later, rerun the permutation test; do not assume historical inclusion transfers.

## Boundary conditions

- **Applies when model families differ.** If all "models" are the same architecture with trivial hyperparameter perturbations, their rankings are near-duplicates and the signal vs. noise diagnostic collapses. ADR-004 assumes the 4 models have meaningfully different inductive biases (LR_EN, LinSVM_cal, RF, XGBoost).
- **Does not apply to within-model ensembling.** Averaging bagged members of one RF is a different operation — they were not independently gate-tested.
- **Alpha choice is a protocol lock.** ADR-004 defaults alpha = 0.05; a protocol may declare alpha = 0.01 for stricter panels. Flipping alpha post-hoc to include more models is a tension-logging event, not a silent recovery.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | ADR-004 Stage 1 introduced precisely to filter noise models from Stage 3 consensus; per ADR-004 Rationale "consensus aggregation across noise models reduces signal" | Migrated from ADR-004 2026-02-09 |

## Related

- [[equations/perm-test-pvalue]] — Stage 1 mechanism
- [[equations/consensus-rank-aggregation]] — Stage 3 aggregator
- [[condensates/perm-validity-full-pipeline]] — the permutation test used for the gate must re-run the full inner pipeline, else Stage 1 itself is invalid
- [[condensates/stability-as-hard-filter]] — the second pre-aggregation filter (hard, not p-value)
- ADR-004 (three-stage feature selection) — canonical source
