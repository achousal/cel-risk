---
type: condensate
depends_on:
  - "[[equations/nested-cv]]"
  - "[[equations/consensus-rank-aggregation]]"
applies_to:
  - "multi-ordering recipe factorials where panel size and ordering are independent axes"
  - "pipelines that derive size rules from a sweep varying BOTH size and ordering"
  - "settings where ordering variance at fixed size is smaller than size-to-size variance"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v1-recipe
    delta: "Pooling sweep across {importance, pathway, rra} orderings before the 3-criterion rule produces a single size decision per model (or a single model-agnostic plateau) that is then crossed against ordering as a separate V1 axis; per-ordering size decisions were not computed because the pooled decision is the protocol default"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §Size rules (pooling paragraph); operations/cellml/MASTER_PLAN.md V1 Decision Architecture"
falsifier: |
  Direction claim: per-ordering size decisions (run the 3-criterion rule
  separately on sweep data filtered to ordering=o, for each o in
  {importance, pathway, rra}) disagree with the pooled-sweep decision by
  more than 2 proteins on >=3 datasets with p > n. That is, for at least
  one ordering, |p_o - p_pooled| > 2 in each of 3 datasets. Under that
  pattern, pooling is hiding genuine size-ordering interaction rather than
  averaging over irrelevant variance, and size selection is ordering-
  dependent. If |p_o - p_pooled| <= 2 for all orderings in 3 datasets
  (Equivalence-like agreement), pooling is confirmed. Retire to
  `established` at 5 confirmations; retire as failed at 5 datasets showing
  the >2-protein disagreement.
---

# Pooling sweep results across orderings before size decision makes size selection ordering-agnostic

## Claim

When a discovery sweep varies panel size `p` across multiple orderings
{o_1, ..., o_k}, the size-selection rule (e.g. 3-criterion, significance_count,
stability) MUST consume sweep data pooled across orderings, not per-ordering.
Pooling averages out ordering-induced variance at each size and yields a
single size decision that the factorial can then cross against ordering as an
independent axis. Per-ordering size selection smuggles ordering variance into
the size decision — you end up double-optimizing and cannot cleanly compare
orderings at a shared `p` in V1.

## Mechanism

At each p in the sweep, AUROC(p) varies both because of size (the true signal
the size rule wants to measure) and because of ordering (which protein order
dominates the trained model's effective feature set). If size selection runs
per-ordering, the selected p_o minimizes size-stage error conditional on
ordering o. Three consequences:

1. **Double optimization.** The size rule picks `p` to maximize the AUROC-vs-p
   curve FOR THAT ORDERING, and the V1 gate then picks the ordering to
   maximize AUROC AT THAT size. The joint optimum is not the factorial
   optimum — it is a shifted estimate that inherits optimism from both stages.
2. **Apples-to-oranges V1 comparison.** If orderings have different selected
   sizes (p_consensus = 5, p_stream = 8, p_rfe = 4), V1 cannot compare
   orderings cleanly: any ΔAUROC confounds "better ordering" with "better
   size for that ordering."
3. **Variance inflation for the size decision itself.** Per-ordering sweeps
   produce size estimates each based on k-times-less data than the pooled
   sweep, so SE(best) is larger and criteria C1/C2 of the 3-criterion rule
   (see [[condensates/three-criterion-size-rule]]) become noisier. Pooling
   preserves the effective sample size at each p.

Pooling is the statistically principled choice when the size-of-interest is
ordering-agnostic. If DESIGN.md treats size as an independent axis from
ordering (which it does: V1 tests size with ordering matched, and ordering
with size matched), then the size rule must also treat size as ordering-
agnostic at the derivation stage.

## Actionable rule

- V1 MUST pool sweep data across all orderings {importance, pathway, rra}
  before running any size rule (`three_criterion`, `three_criterion_unanimous`,
  `significance_count`, `stability`).
- Per-model size derivation pools across orderings within the model filter;
  it does NOT also filter to one ordering (see DESIGN.md §Per-model size
  derivation).
- The pooled sweep is the single input to size selection; the factorial
  then crosses the selected p against each ordering as V1 axes.
- Per-ordering sweep views ARE useful for diagnostics (plot AUROC-vs-p
  colored by ordering to eyeball interaction) but MUST NOT feed the size
  rule.
- If a new dataset shows |p_o - p_pooled| > 2 for any ordering in the
  ledger's predicted set, V1 emits a tension pointing back to this
  condensate and the size axis is flagged as Inconclusive.

## Boundary conditions

- **Breaks when ordering choice produces very different AUROC curves.** If
  orderings disagree on the SHAPE of the AUROC-vs-p curve (e.g. one ordering
  saturates at p=4, another is still rising at p=10), pooling averages two
  qualitatively different behaviors and hides the interaction rather than
  resolving it. The symptom is bimodal AUROC distribution at intermediate p;
  V1 must detect this via Levene's test on per-ordering AUROC variance at
  fixed p. If variance is significantly heterogeneous, pooling is
  inappropriate and V1 must defer to per-ordering size selection with the
  tension explicitly flagged.
- **Requires comparable sweep conditions across orderings.** All orderings
  in the pool must use the same outer-CV folds, same hyperparameter search
  budget, and same seed range. Mixing 30-seed and 10-seed sweeps across
  orderings inflates pooled SE spuriously.
- **Does not apply when the ordering determines which proteins enter the
  candidate set.** If orderings are not permutations of the same candidate
  list but rather independent selections from different trunks (e.g.
  consensus-desc picks from T1, |coef|-desc picks from T2), pooling mixes
  populations. The rule applies only when all pooled orderings operate on
  the same candidate pool.
- **Sensitive to the count of orderings pooled.** With k=2 orderings pooling
  halves ordering variance; with k=3 pooling reduces it by ~1/3 (law of
  large numbers with finite k). At k=1 pooling is a no-op. The gain scales
  poorly above k=5 and the complexity cost grows, so the default k in {2,3}
  is well-matched.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | Pooled 3-criterion rule selects 5p (>=2) / 8p (unanimous) using sweep data aggregated across {importance, pathway, rra}; V1 then crosses the locked size against each ordering as a separate axis (R1 consensus-desc, R2 stream-balanced, MS OOF/RFE). The pooled-size + independent-ordering structure is the DESIGN.md §Size rules default. | v1-recipe 2026-04-20; source DESIGN.md §Size rules; MASTER_PLAN.md V1 Decision Architecture |

## Related

- [[protocols/v1-recipe]] — applies this pooling at §2 before the size rule
  runs; T-V1-03 in the v1-recipe Known tensions section flags this choice as
  falsifiable and points here for the condensate
- [[condensates/three-criterion-size-rule]] — the rule that consumes the
  pooled sweep
- [[condensates/nested-expansion-vs-refit]] — after per-model pooled sizes
  are selected, nested expansion maps the step-down ladder; the two
  condensates together govern the size-axis derivation
- [[equations/nested-cv]] — defines the sweep-stage OOF AUROC estimator
  that is pooled
- [[equations/consensus-rank-aggregation]] — one of the orderings pooled in the
  celiac case
- DESIGN.md §Size rules — canonical specification
- DECISION_TREE_AUDIT.md §3.2 — T2-trunk-ordering confound (related but
  distinct: that confound is within T2 ordering, not across-ordering
  pooling)
