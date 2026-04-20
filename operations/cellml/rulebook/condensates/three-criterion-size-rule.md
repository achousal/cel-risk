---
type: condensate
depends_on:
  - "[[equations/nested-cv]]"
  - "[[equations/brier-decomp]]"
applies_to:
  - "pipelines with a discovery-sweep over panel sizes"
  - "AUROC-vs-p curves with an identifiable plateau"
  - "settings where the sweep-stage estimator has a standard error and is comparable across sizes"
status: provisional
confirmations: 1
evidence:
  - dataset: celiac
    gate: v1-recipe
    delta: "3-criterion rule identifies 5p (>=2 pass) and 8p (unanimous) on pooled sweep; significance count gives 4p; the unanimous plateau (8p) aligns with RF per-model plateau and the 1-SE heuristic"
    date: "2026-04-20"
    source: "operations/cellml/DESIGN.md §Size rules; operations/cellml/MASTER_PLAN.md V1 gate (Decision Architecture, panel composition + ordering)"
falsifier: |
  Direction claim: the 3-criterion rule's selected p (smallest p where >=2 of
  {C1 non-inferiority z-test, C2 within-1-SE heuristic, C3 marginal-gain
  z-test} pass) differs from the unanimous variant `three_criterion_unanimous`
  (smallest p where all 3 pass) by more than 3 proteins on >=3 datasets with
  p > n, and the unanimous p loses more than 0.02 AUROC with 95% bootstrap CI
  excluding 0 relative to the >=2 p. This means the >=2 rule is picking
  over-parsimonious panels and must be weakened (raise the consensus count,
  or reweight criteria). If |Δp| stays <=3 proteins AND |ΔAUROC| < 0.01 with
  95% CI inside [-0.02, 0.02] (Equivalence) across >=3 datasets, the rule is
  confirmed. Retire to `established` at 5 such datasets; retire as failed at
  5 datasets showing the Direction failure.
---

# Panel size is the smallest p where at least 2 of non-inferiority, 1-SE heuristic, and marginal-gain-stops-helping criteria pass

## Claim

The 3-criterion rule selects panel size as the smallest `p` in the discovery
sweep where a majority (>=2 of 3) independent statistical criteria agree that
`p` is on the performance plateau. The three criteria test non-overlapping
aspects of plateau behavior: **C1** (non-inferiority z-test against the best
p) tests that `p` is not meaningfully worse than the best; **C2** (Breiman
1-SE heuristic) tests that `p`'s mean is inside the best mean's measurement
band; **C3** (Holm-corrected two-sided marginal-gain z-test on Δ(p -> p+1))
tests that adding the next protein has stopped helping. The majority rule is
more robust than any single criterion because each criterion has distinct
failure modes (C1 is sensitive to SE, C2 is a heuristic with no
distributional basis, C3 is sensitive to noise in Δ-estimates) and the
pairwise intersections catch the cases any single criterion would miss.

## Mechanism

Each criterion alone has a known failure mode:

- **C1 alone** fails when AUROC variance is small and the plateau is shallow:
  non-inferiority rejects even tiny gaps, picking the smallest p indifferent
  to whether it is truly on the plateau.
- **C2 alone** fails when the sweep SE is large (few seeds, small effective
  sample): the 1-SE band becomes wide and many p-values pass trivially.
- **C3 alone** fails when the AUROC-vs-p curve is noisy: adjacent Δ(p -> p+1)
  estimates bounce across the null boundary and the "first insignificant
  gain" p is unstable.

Requiring >=2 criteria to agree at the same p turns these independent failure
modes into an approximate AND-gate: any p flagged by two criteria is on the
plateau under two non-overlapping notions of "plateau." This is a standard
majority-vote ensemble over weak tests. The unanimous variant
`three_criterion_unanimous` is the conservative complement — it picks p only
when all three agree, which empirically corresponds to the performance
plateau's unambiguous start. The majority and unanimous variants bracket the
plateau: majority catches where parsimony arguments start holding, unanimous
catches where they hold beyond doubt.

The composite is robust in a second sense: the sweep estimator (30-seed
averaged AUROC from [[equations/nested-cv]]) has seed-level variance that any
single criterion consumes differently. Pooling criteria reduces dependence on
a single variance assumption.

## Actionable rule

- V1 locks panel size using `three_criterion` (smallest p with >=2 passes) by
  default. The unanimous variant runs in parallel as a robustness check.
- If `three_criterion` and `three_criterion_unanimous` disagree by more than
  3 proteins, V1's decision.md must report BOTH candidate sizes in the
  recipe pair and forward both to V2 as Inconclusive on the size axis.
- Criterion thresholds are protocol locks, not tunables: C1 uses δ = 0.02
  (matches the fixed falsifier rubric Direction threshold in SCHEMA.md);
  C2 uses 1 * SE(best); C3 uses α = 0.05 under Holm correction.
- The rule runs on pooled sweep data (see
  [[condensates/size-ordering-pooling]]) and optionally on model-filtered
  sweep data for per-model plateaus (see DESIGN.md §Per-model size derivation).

## Boundary conditions

- **Requires a sweep that varies p over a range wide enough to identify a
  plateau.** If the sweep covers only small p (e.g. 4-6), C3 has nowhere to
  pass and the rule degenerates to C1+C2 voting. Minimum sweep range
  recommended: p_max - p_min >= 10.
- **Breaks on very flat AUROC-vs-p curves.** If AUROC is essentially constant
  across all sweep sizes (|max - min| < 1 * SE(best)), both C1 and C2 pass
  at the smallest p regardless of whether a true plateau exists. The rule
  will pick p_min; this is a correct "nothing saturates" output but does not
  mean p_min is the optimal size. The V1 protocol must flag |max - min| <
  1 * SE(best) in observation.md and treat the resulting p as a low-confidence
  lock.
- **Sensitive to the sweep estimator's SE definition.** C1 and C2 both
  consume SE(best); if SE is under-estimated (e.g. using cell-averaged SE
  per v1-recipe.md §3.3 caveat), both criteria become anti-conservative.
  V1 uses seed-level bootstrap SE only.
- **Criterion independence is an assumption, not a theorem.** C1 and C2
  share variance inputs; when SE(best) is large, C1 and C2 co-move. At that
  point the >=2 rule can degenerate to "C1 passes AND C2 passes" without
  C3's independent signal. Monitor in evidence; if C3's marginal contribution
  stays below ~5% across datasets, the rule simplifies to C1-or-C2.
- **Pre-registration principle (DESIGN.md §Pre-registration).** The 3-criterion
  rule consumes discovery sweep data (50 Optuna trials) to pick candidate
  sizes; the factorial re-runs under production conditions (200 trials). V1
  is the final arbiter, not the size rule itself. The rule is a
  pre-registration tool, and its accuracy is bounded by how close the
  discovery and factorial conditions are.

## Evidence

| Dataset | n | p | Phenomenon | Source gate |
|---|---|---|---|---|
| Celiac (UKBB) | 43,810 | 2,920 | Pooled sweep: `three_criterion` -> 5p; `three_criterion_unanimous` -> 8p; `significance_count` -> 4p; per-model unanimous: LR_EN 4p, LinSVM_cal 4p, RF 8p, XGBoost 9p | v1-recipe 2026-04-20; source DESIGN.md §Size rules; MASTER_PLAN.md Decision Architecture |

## Related

- [[protocols/v1-recipe]] — consumes this rule at §2 to populate the size axis
- [[condensates/size-ordering-pooling]] — specifies that sweep data must be
  pooled across orderings before this rule runs
- [[condensates/nested-expansion-vs-refit]] — addresses the tree-model
  expansion from per-model plateau down to the significance core, which
  depends on this rule's per-model p
- [[equations/nested-cv]] — provides the sweep-stage OOF AUROC estimator
  that C1/C2/C3 consume
- [[equations/brier-decomp]] — V1 reports REL alongside AUROC; size rule
  runs on AUROC only, but the sibling calibration axis is kept separate
- DESIGN.md §Size rules — canonical specification
- MASTER_PLAN.md Decision Architecture — V1 gate usage
