# Decision Tree Architecture Audit

**Date:** 2026-04-08
**Scope:** V0–V5 validation tree cohesion, chronology, and completeness

---

## 1. Cohesion: Rule Consistency Across the Tree

### 1.1 Parsimony Principle — Applied Unevenly

| Level | Parsimony rule | Codified? |
|---|---|---|
| V1 | CI includes 0 → prefer fewer proteins | Yes |
| V2 | "Ties broken by parsimony" | **Undefined** — no model complexity ordering (LR > SVM > RF > XGB?) |
| V3 | Lowest reliability preferred | **No tiebreaker** if two methods have overlapping CIs |
| V4 | Overlapping CIs → prefer none | Yes |
| V5 | Hurts calibration > helps AUPRC → prefer 1.0 | Yes |

**Gap:** V2 and V3 need explicit parsimony fallbacks. V2 says "parsimony" but never defines a model complexity ordering. V3 picks by rank without testing whether rank-1 and rank-2 are distinguishable.

### 1.2 Statistical Rigor — Inconsistent Uncertainty Handling

| Level | Accounts for uncertainty? | Method |
|---|---|---|
| V1 | Yes | z-test on SE of cell means |
| V2 | **No** | Point-estimate Pareto dominance |
| V3 | **No** | Ranking by mean reliability |
| V4 | Yes | Wald CI overlap |
| V5 | **No** | Point-estimate net benefit |

V2, V3, and V5 make decisions on point estimates without testing whether differences exceed noise. Two models 0.001 apart in Brier will show one dominating the other, even if that difference is within seed-to-seed variance.

### 1.3 V1 SE Measures the Wrong Thing (CRITICAL)

In `validate_tree.R:72-80`, V1 computes SE as:

```r
auroc_se = sd(summary_auroc_mean) / sqrt(n_cells)
```

Each `summary_auroc_mean` is already a 30-seed average for one cell. V1 then treats cells as observations. But cells within a shared recipe differ by model × calibration × weighting × downsampling — so this SE captures **downstream factor variability**, not measurement uncertainty.

**Consequence:** SE is inflated for shared recipes (which span 4 models with different performance) relative to model-specific recipes (1 model, less variance). This systematically biases toward "CI includes 0 → prefer fewer proteins" because SE is inflated.

### 1.4 V1 Cell-Count Asymmetry (CRITICAL)

| Recipe type | Cells | SE denominator | Variance includes |
|---|---|---|---|
| Shared | 108 | sqrt(108) | Model + cal + weight + downsample |
| Model-specific base | 27 | sqrt(27) | Cal + weight + downsample only |
| Nested expansion | 27 | sqrt(27) | Cal + weight + downsample only |

Comparing a shared recipe (marginal over 4 models) against a model-specific recipe (conditional on 1 model) is an apples-to-oranges comparison. The z-test assumes comparable units but the variance components differ.

**Fix options:**
- (a) Compare shared vs shared and MS vs MS separately in V1
- (b) Use a mixed-effects model on seed-level data that nests cells within recipes
- (c) Standardize the comparison unit (e.g., compute recipe performance at the seed level across 30 paired seeds)

### 1.5 V2 Pareto Axes Are Collinear

AUROC and Brier are correlated: Brier = reliability + refinement, and refinement tracks discrimination (AUROC). The Pareto front is nearly one-dimensional. A model with higher AUROC almost always has lower Brier unless its calibration is poor.

**Fix:** Switch to AUROC vs reliability (orthogonal) — isolates discrimination from calibration quality.

### 1.6 V5 Net Benefit Scale Mismatch

```r
net_benefit = auprc_gain_vs_none - reliability_cost
```

AUPRC changes are typically ~0.01–0.03. Reliability changes are typically ~0.001–0.005. Net benefit is dominated by the AUPRC term regardless of calibration damage.

**Fix:** Normalize both terms to their respective ranges, or use an explicit utility weighting.

---

## 2. Decision Chronology: Is the Ordering Optimal?

### 2.1 Current Order

```
V0: Training Strategy (gate)
V1: Recipe (panel + ordering + trunk)
V2: Model
V3: Calibration
V4: Weighting
V5: Downsampling
```

### 2.2 Assessment

| Transition | Verdict | Rationale |
|---|---|---|
| V0 → V1 | **Correct** | Training strategy defines the data distribution. Must be locked first. |
| V1 first | **Correct** | Feature space is the highest-variance decision. Everything downstream is conditioned on which proteins are in the model. |
| V2 before V3–V5 | **Correct** | Model architecture mediates how calibration, weighting, and downsampling behave. |
| V3 before V4–V5 | **Questionable** | Calibration is post-hoc. Weighting and downsampling are training-time. Causally, training-time decisions should precede post-hoc corrections. Calibration can compensate for imbalance-handling choices. |
| V4 before V5 | **Problematic** | Both address class imbalance. Sequential resolution can miss joint optima (see below). |

### 2.3 V4–V5 Interaction (CRITICAL)

Both V4 (weighting) and V5 (downsampling) address class imbalance through different mechanisms. They are substitutes/complements:

- V4 averages over all V5 levels to pick weighting. If "none" wins (because at downsampling=1.0 it's fine), it gets locked.
- V5 then resolves downsampling conditioned on "none". Might pick 2.0 because some imbalance handling is needed.
- But the global optimum might be (sqrt, 1.0) — sqrt handles imbalance so no downsampling needed.

Sequential resolution finds a **different optimum** than joint optimization.

**Fix:** Resolve V4+V5 jointly as a 3×3 grid (9 combinations per recipe×model×calibration group). Or run the tree in both orderings as a sensitivity check.

### 2.4 Recommended Ordering

```
V0: Training Strategy (gate)           — unchanged
V1: Recipe (panel + ordering + trunk)   — unchanged
V2: Model                              — unchanged
V4+V5: Imbalance handling (joint)      — 3 weighting × 3 downsampling = 9 combos
V3: Calibration                         — post-hoc, applied to imbalance-optimized model
```

At minimum, run the tree in both orderings and check if the winner is stable.

---

## 3. Completeness: Are All Routes Covered?

### 3.1 Coverage Summary

| Dimension | Coverage |
|---|---|
| Size | 4p, 5p, 6p, 7p, 8p, 9p, 19p (via nested expansion) |
| Ordering | consensus, stream-balanced, \|coef\|, OOF, RFE |
| Trunk | T1 (consensus), T2 (incident) |
| Model | 4 classifiers |
| Tuning | calibration(3) × weighting(3) × downsampling(3) |

### 3.2 Trunk × Ordering Confound

Every trunk is paired with a "natural" ordering only:

| Trunk | Orderings tested | Orderings NOT tested |
|---|---|---|
| T1 | consensus, stream-balanced, OOF, RFE | \|coef\| |
| T2 | \|coef\| only | consensus, stream-balanced, OOF, RFE |

If T2 proteins are strong but \|coef\| ordering is suboptimal for them, the factorial can't detect this. T2 is tested by exactly one recipe (R3). A single failure mode in R3 eliminates the entire T2 trunk.

**Recommendation:** Add ≥1 T2 × alternative-ordering recipe (e.g., T2 × OOF importance for a representative model) to deconfound trunk from ordering.

### 3.3 No Shared-Ordering × Model-Specific Size

R1_plateau happens to be 8p (RF's plateau), but there's no consensus × 9p (XGBoost's plateau). Model-specific recipes only use OOF and RFE orderings.

The question "does consensus ordering at XGBoost's optimal size beat OOF ordering at XGBoost's optimal size?" is unanswerable — R1_plateau tests consensus at 8p, MS_oof_XGBoost tests OOF at 9p. Size and ordering are confounded.

**Partial mitigation:** Nested expansion creates MS_oof_XGBoost_p8, comparable to R1_plateau at 8p. But the reverse (consensus at 9p) doesn't exist.

### 3.4 Ensemble Excluded

ADR-007 disables stacking. If the best clinical model is an ensemble of models, the factorial can't discover this.

**Status:** Defensible if the goal is a single interpretable model. Should be explicitly documented as a design constraint.

### 3.5 V0 Prevalent Fraction Coverage

Only 0.5 and 1.0 tested. If the optimal fraction is 0.2 or 0.3, V0 misses it.

**Recommendation:** Test {0.25, 0.5, 1.0} to bracket the space.

### 3.6 HPO Seed Variance

Each cell runs 200 Optuna trials. If the sampler seed is random, there's uncontrolled HPO variance not captured by the 30 split-seeds. If fixed, results are deterministic per cell — good for reproducibility but doesn't quantify HPO instability.

### 3.7 No Tree Cross-Validation

The V1→V5 sequential tree uses all 30 seeds for both selection and evaluation. There's no held-out seed set to confirm the selected configuration generalizes.

The holdout confirmation (Phase 5) addresses this at the data level, but not at the decision-tree level.

**Recommendation:** Split the 30 seeds into 20 selection + 10 confirmation. Run the tree on the 20, verify the winner on the 10.

---

## 4. Action Items

| Priority | Issue | Section | Fix |
|---|---|---|---|
| **P0** | V1 SE measures factor variance, not measurement noise; cell-count asymmetry | 1.3, 1.4 | Restructure V1: compare within recipe-type, or use mixed-effects model on seed-level data |
| **P0** | V4–V5 interaction: sequential resolution can miss joint optimum | 2.3 | Resolve jointly (9-cell grid) or run tree in both orderings as sensitivity check |
| **P1** | V2, V3, V5 lack uncertainty quantification | 1.2 | Add bootstrap CIs or permutation tests at all levels |
| **P1** | V2 Pareto on collinear axes | 1.5 | Switch to AUROC vs reliability (orthogonal) |
| **P1** | T2 trunk tested by single recipe | 3.2 | Add ≥1 T2 × alternative-ordering recipe |
| **P2** | V3→V4 ordering (post-hoc before training-time) | 2.2 | Move V3 after V4+V5, or validate ordering doesn't change winner |
| **P2** | V5 net_benefit scale mismatch | 1.6 | Normalize or weight AUPRC gain vs reliability cost |
| **P2** | V2, V3 missing parsimony definitions | 1.1 | Define model complexity ordering; add V3 overlap test |
| **P3** | No tree cross-validation (20/10 seed split) | 3.7 | Reserve 10 seeds for decision confirmation |
| **P3** | V0 prevalent_frac coverage | 3.5 | Test {0.25, 0.5, 1.0} |
| **P3** | Ensemble excluded by design | 3.4 | Document as explicit constraint |
| **P3** | HPO seed variance unquantified | 3.6 | Fix Optuna sampler seed or add HPO-seed replicates |

---

## 5. Resolution Log

Track decisions made against each action item.

| Item | Decision | Date | Notes |
|---|---|---|---|
| P0: V1 SE + cell-count asymmetry | **Implemented.** Stratified comparison (shared vs MS) with seed-level SE. Cross-type bridge via model-matched comparison. | 2026-04-08 | validate_tree.R rewritten. SE = mean(auroc_std)/sqrt(n_seeds). |
| P0: V4-V5 sequential interaction | **Implemented.** Joint V3 (weighting × downsampling) as 3×3 grid with normalized utility (0.5 AUPRC + 0.5 reliability). Bootstrap CIs. Parsimony toward (none, 1.0). | 2026-04-08 | Old V4+V5 merged into new V3. |
| P1: V2/V3/V5 uncertainty | **Implemented.** Bootstrap CIs at all levels: V2 (Pareto, 1000 iters), V3 (utility), V4 (reliability). | 2026-04-08 | set.seed(42) for reproducibility. |
| P1: V2 Pareto collinear axes | **Implemented.** Switched from AUROC vs Brier to AUROC vs reliability (orthogonal). | 2026-04-08 | Brier = reliability + refinement; refinement tracks AUROC. |
| P1: T2 single recipe | **Implemented.** Added R3_consensus recipe (T2 trunk, stability_freq ordering) to manifest.yaml. | 2026-04-08 | Deconfounds trunk from ordering. +108 cells → 1,566 total. |
| P2: V3→V4 ordering | **Implemented.** Calibration moved to V4 (end). Training-time decisions (V3 imbalance) now precede post-hoc (V4 calibration). | 2026-04-08 | New order: V1→V2→V3(imbalance)→V4(calibration)→V5(confirmation). |
| P2: V5 net_benefit scale | **Implemented.** Replaced raw subtraction with normalized utility (both metrics scaled to [0,1] within group). | 2026-04-08 | Equal 50/50 weight. |
| P2: V2/V3 parsimony | **Implemented.** MODEL_COMPLEXITY: LR_EN(1) < LinSVM_cal(2) < RF(3) < XGBoost(4). CALIBRATION_COMPLEXITY: logistic_intercept(1) < beta(2) < isotonic(3). | 2026-04-08 | Added to DESIGN.md, validate_tree.R, analysis program. |
| P3: Tree cross-validation | **Implemented.** V5 seed-split: 20 selection (100–119), 10 confirmation (120–129). Fallback to variance-ratio check when per-seed data unavailable. | 2026-04-08 | Flags drop > 1 SE as unstable. |
| P3: V0 prevalent_frac | **Implemented.** Expanded to {0.25, 0.5, 1.0}. V0 cell count: 40 (from 32). | 2026-04-08 | MASTER_PLAN.md updated. |
| P3: Ensemble exclusion | **Documented.** Added "Design Constraint: Single-Model Scope" section to DESIGN.md. ADR-007 updated. | 2026-04-08 | Post-factorial follow-up if no single model dominates. |
| P3: HPO seed variance | **Documented as acceptable.** Sampler seed is DERIVED from split seed (fallback cascade in optuna_utils.py). Fair within-seed comparison. Coupled variance is acceptable for factorial's relative comparisons. | 2026-04-08 | No code change. Risk: moderate. All cells share same sampler seed per split. |
| P3: Ensemble excluded | **Implemented as V6.** Post-tree informational comparison. Stacks non-dominated models from V2, compares vs locked single-model winner. Higher bar (δ=0.02, full CI > 0). Human decides. | 2026-04-08 | validate_tree.R, DESIGN.md, MASTER_PLAN.md, analysis program updated. |
