# Mathematical & Statistical Validity Audit Report

**Project**: CeliacRisks -- ML Pipeline for Incident Celiac Disease Risk Prediction
**Date**: 2026-02-07
**Auditors**: 6 automated code-review agents (5 Sonnet, 1 Opus)
**Scope**: All mathematical/statistical operations from data loading through significance testing
**Files audited**: 30+ source files across 6 domains

---

## 1. Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **MUST-FIX** | 11 | Blocking publication / clinical validity |
| **FLAG** | 10 | Requires documentation, justification, or sensitivity analysis |
| **NICE-TO-HAVE** | 5 | Enhances robustness but not blocking |
| **PASS** | 25 | Validated correct |

**Overall assessment**: The pipeline demonstrates strong mathematical rigor in its core components -- nested CV isolation, OOF prediction construction, bootstrap CIs, permutation testing (Phipson-Smyth), DCA net benefit, and prevalence adjustment. However, 11 MUST-FIX issues span feature importance aggregation, consensus ranking, calibration, and permutation resolution. Two issues are **critical blockers**: unknown residualization provenance (potential full-dataset leakage) and Platt scaling double-sigmoid.

### MUST-FIX Issues at a Glance

| # | Issue | Domain | Severity | Location |
|---|-------|--------|----------|----------|
| M1 | Unknown `_resid` provenance (leakage?) | D1 | Critical | `schema.py` |
| M2 | Median imputation before F-test inflates stats | D2 | High | `screening.py:186-190` |
| M3 | Inconsistent missing-value handling (MW vs F) | D2 | High | `screening.py` |
| M4 | Zero-fill for absent features biases mean | D3 | High | `importance.py:772-781` |
| M5 | Simplified RRA (not Kolde et al. 2012) | D3 | High | `consensus.py:348-351` |
| M6 | RRA weights 0.6/0.3/0.1 unjustified | D3 | High | `consensus.py:63-66` |
| M7 | Clustering inconsistency (graph vs hierarchical) | D3 | Medium | `corr_prune.py` / `drop_column.py` |
| M8 | Zero-positive fold handling (spw=1.0 fallback) | D4 | High | `nested_cv.py:226-236` |
| M9 | Platt scaling double-sigmoid (logit of probs) | D5 | High | `calibration.py:318-322` |
| M10 | Double calibration in stacking | D5 | High | `stacking.py:240-246` |
| M11 | Default n_perms=200 (min p~0.005) | D6 | High | `permutation_test.py:209` |

---

## 2. Per-Domain Findings

### Domain 1: Data Preparation & Sampling Validity

**Files**: `data/splits.py`, `data/filters.py`, `data/columns.py`, `data/io.py`, `data/schema.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 1.1 | Stratification correctness | PASS | -- |
| 1.2 | Prevalent case isolation (TRAIN only) | PASS | -- |
| 1.3 | Temporal missing fill (`min_date - 1`) | FLAG | Medium |
| 1.4 | Seed derivation (`XOR 0x5678`) | PASS | -- |
| 1.5 | `_resid` suffix provenance | **FLAG (Critical)** | Critical |
| 1.6 | Listwise deletion MCAR assumption | FLAG | Medium |
| 1.7 | Column resolution (proteins vs metadata) | PASS | -- |

#### 1.1 Stratification -- PASS

Two-stage stratified split (`splits.py:425-530`) correctly preserves class ratios. `build_working_strata()` implements fallback hierarchy (outcome+sex+age3 -> outcome+sex -> outcome) ensuring min_count=2 per stratum. sklearn's `train_test_split(stratify=...)` handles proportional allocation.

#### 1.2 Prevalent Case Isolation -- PASS

Multiple defensive layers prevent leakage:
1. Prevalent excluded before split (`save_splits.py:378-387`)
2. Split on controls + incident only (`save_splits.py:410-418`)
3. Prevalent added to TRAIN after split (`save_splits.py:438-440`)
4. Explicit validation via `check_prevalent_in_eval()` with `strictness="error"` (`save_splits.py:481-496`)
5. Test coverage confirms no prevalent in VAL/TEST (`test_cli_save_splits.py:195-206`)

#### 1.3 Temporal Missing Fill -- FLAG (Medium)

**Location**: `splits.py:333-359`

Missing dates filled with `min_date - 1`, placing missing samples at timeline start. Warning issued if missingness > 5%. If missingness is informative (e.g., correlated with case status), this creates temporal cohort effects. **Recommendation**: Report CeD_date missingness rate; if > 5%, conduct sensitivity analysis with alternative imputation strategies.

#### 1.4 Seed Derivation -- PASS

`second_seed = random_state ^ 0x5678` (`splits.py:482`). XOR is bijective (zero collision risk within a run). MT19937 streams seeded with XOR-derived values are effectively statistically independent.

#### 1.5 `_resid` Suffix Provenance -- FLAG (Critical)

**Location**: `schema.py:72-74`

All 2,920 protein columns end with `_resid`, implying upstream covariate adjustment. **No code in the repository creates or documents the residualization process.** If residualization used the full dataset (not split-aware), this is test set leakage that invalidates all downstream results.

**Action required**:
1. Document residualization provenance (covariates, reference population, regression method)
2. If full-dataset: re-run with split-aware residualization or conduct sensitivity analysis
3. If external reference: document and justify

#### 1.6 Listwise Deletion -- FLAG (Medium)

**Location**: `filters.py:78-84`

Missing age/BMI rows are dropped (listwise deletion). MCAR assumption not validated. **Recommendation**: Report missingness rate; if > 5%, conduct Little's MCAR test and sensitivity analysis.

#### 1.7 Column Resolution -- PASS

`columns.py` correctly separates proteins (suffix `_resid`) from metadata (age, BMI, sex, ethnicity). Feature selection operates on `protein_cols` only; metadata included as fixed covariates.

---

### Domain 2: Feature Screening & Statistical Tests

**Files**: `features/screening.py`, `features/kbest.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 2.1 | Mann-Whitney U test | PASS | -- |
| 2.2 | F-test (ANOVA) | FLAG | Low |
| 2.3 | Multiple testing correction (2,920 tests) | FLAG | Medium |
| 2.4 | Median imputation before F-test | **MUST-FIX** | High |
| 2.5 | SelectKBest vs screening.py equivalence | FLAG | Medium |
| 2.6 | Missing value handling consistency | **MUST-FIX** | High |
| 2.7 | Effect size reporting | NICE-TO-HAVE | Low |

#### 2.1 Mann-Whitney U -- PASS

Two-sided test with `method="asymptotic"`, `min_n_per_group=10` (`screening.py:97`). Pairwise deletion for missing values. scipy handles ties correctly via tie correction.

#### 2.2 F-test -- FLAG (Low)

`f_classif` from sklearn (`screening.py:194`). Normality/homoscedasticity assumptions not documented. With N=43,960, CLT justification is sound. **Recommendation**: Add CLT justification in docstring.

#### 2.3 Multiple Testing -- FLAG (Medium, mitigated)

2,920 simultaneous comparisons, no FDR/Bonferroni. Expected ~146 false positives at alpha=0.05. **This is intentional screening** (Stage 0 pre-filter, 2920 -> 1000) with downstream protection via nested CV, permutation testing, and FDR-corrected consensus (ADR-004). **Recommendation**: Add module docstring documenting the three-stage design.

#### 2.4 Median Imputation Before F-test -- MUST-FIX (High)

**Location**: `screening.py:186-190`

```python
med = Xp.median(axis=0, skipna=True)
Ximp = Xp.fillna(med)
F, pvals = f_classif(Ximp.to_numpy(dtype=float), y)
```

Median imputation reduces variance and inflates F-statistics (anti-conservative). Direction of bias: **more false positives**. Note: CLAUDE.md states "Missing proteins: Zero", so current data may not be affected, but the code path exists and could affect future datasets.

**Recommendation**: Use pairwise deletion (consistent with Mann-Whitney) or document the bias for future datasets.

#### 2.5 kbest.py vs screening.py -- FLAG (Medium)

`kbest.py:compute_f_classif_scores()` does NOT call `_impute_proteins` before `f_classif`, while `screening.py` does. Not equivalent when input has missing values. `kbest.py` appears to be legacy. **Recommendation**: Add deprecation warning to `kbest.py` or enforce consistency.

#### 2.6 Missing Value Handling Consistency -- MUST-FIX (High)

**Location**: `screening.py:78-89` (Mann-Whitney: pairwise deletion) vs `screening.py:186-190` (F-test: median imputation)

Different missing data strategies produce non-comparable p-values. Switching `screen_method` in config changes protein rankings due to missing data handling, not just test choice. **Recommendation**: Standardize on pairwise deletion for both methods.

#### 2.7 Effect Size -- NICE-TO-HAVE (Low)

Mann-Whitney reports absolute mean difference; F-test reports F-score. Neither reports standardized effect sizes (Cohen's d, rank-biserial). Sufficient for screening; standardized measures would aid cross-study comparison.

---

### Domain 3: Feature Importance, Selection & Consensus

**Files**: `features/importance.py`, `features/grouped_importance.py`, `features/drop_column.py`, `features/rfe.py`, `features/rfe_engine.py`, `features/nested_rfe.py`, `features/stability.py`, `features/corr_prune.py`, `features/consensus.py`, `features/panels.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 3.1 | OOF grouped PFI (cluster permutation) | PASS | -- |
| 3.2 | L1-normalized coef (scale dependence) | FLAG | Medium |
| 3.3 | Zero-fill for absent features | **MUST-FIX** | High |
| 3.4 | RRA implementation (simplified) | **MUST-FIX** | High |
| 3.5 | RRA weights (0.6/0.3/0.1) | **MUST-FIX** | High |
| 3.6 | Clustering inconsistency | **MUST-FIX** | Medium |
| 3.7 | RFE knee detection formula | PASS | -- |
| 3.8 | Stability threshold cascade | FLAG | Medium |
| 3.9 | Panel refill greedy algorithm | FLAG | Low |
| 3.10 | Drop-column additive assumption | FLAG | Low |

#### 3.1 OOF Grouped PFI -- PASS

Permutation respects cluster structure: all features in a group are permuted together (`importance.py:442-450`, `grouped_importance.py:398-406`). Within-cluster correlation preserved.

#### 3.2 L1-Normalized Coefficients -- FLAG (Medium)

`importance.py:190-197` computes `abs_coefs / coef_sum` (L1 normalization). Scale-dependent if features not standardized. Pipeline includes `StandardScaler` as `pre` step, mitigating this. `rfe_engine.py:201-207` uses raw `abs(coef)` without normalization (appropriate for RFE ordering). **Recommendation**: Document preprocessing dependency in docstring.

#### 3.3 Zero-Fill for Absent Features -- MUST-FIX (High)

**Location**: `importance.py:772-781`

```python
else:
    val = 0.0  # Feature absent from this fold
# ...
"mean_importance": float(np.mean(values_array)),  # Includes zeros
```

Feature selected in 3/5 folds with importance [0.05, 0.04, 0.06]: zero-fill mean = 0.030 vs nanmean = 0.050 (40% deflation). Systematically penalizes features that are strong but variably selected across folds.

**Recommendation**: Replace with `np.nan` and `np.nanmean`, or use fold-count weighted mean. The existing `n_folds_nonzero` column enables this.

#### 3.4 RRA Implementation -- MUST-FIX (High)

**Location**: `consensus.py:348-351`

```python
reciprocal_ranks = [1.0 / r for r in ranks]
row["consensus_score"] = float(gmean(reciprocal_ranks))
```

Computes geometric mean of reciprocal ranks -- **not** the formal RRA method from Kolde et al. (2012), which uses beta-model p-values on normalized ranks. The simplification:
- Has no formal null hypothesis testing
- Produces no p-values for consensus ranking
- Is sensitive to list lengths (ranks not normalized by list size)
- The function name `robust_rank_aggregate` is misleading

The line 15 comment acknowledges this: "No external R dependencies (vs Stuart's p-value method)".

**Recommendation**:
1. Rename to `geometric_mean_rank_aggregate`
2. Normalize ranks before reciprocal: `1.0 / (r / n)` where `n` is list length
3. Document deviation from Kolde et al. (2012)
4. Consider optional formal RRA via `rpy2` + R `RobustRankAggreg` package

#### 3.5 RRA Weights -- MUST-FIX (High)

**Location**: `consensus.py:63-66, 220-225`

OOF importance: 0.6, essentiality: 0.3, stability: 0.1. No cross-validation, sensitivity analysis, or citation. When OOF importance is missing, effective weights silently shift to 75%/25% essentiality/stability.

**Recommendation**:
1. Run sensitivity analysis: compute top-20 under 3-5 weight configs; report Jaccard overlap
2. Document rationale in ADR or docstring
3. Log warning when any input signal is missing (effective weight change)

#### 3.6 Clustering Inconsistency -- MUST-FIX (Medium)

**Location**: `corr_prune.py:148-156` (graph DFS components) vs `drop_column.py:574-595` (scipy hierarchical, average linkage)

Connected components are transitive: A~B and B~C implies {A,B,C} one cluster even if corr(A,C) < threshold. Average linkage is not transitive: same features could be split into different clusters by drop_column.

**Recommendation**: Standardize on one method. Refactor `_cluster_panel_features` in `drop_column.py` to use `corr_prune.build_correlation_graph` + `find_connected_components`.

#### 3.7 RFE Knee Detection -- PASS

`rfe_engine.py:319-351`. Perpendicular distance formula `|ax + by + c| / sqrt(a^2 + b^2)` is mathematically correct. Normalization to [0,1] ensures comparable scales. Edge cases handled: flat curve, 2 points, all collinear.

#### 3.8 Stability Thresholds -- FLAG (Medium)

0.75 for repeat stability (`stability.py:72`), 0.90 for consensus (`consensus.py:481`). Not formally justified. Silent fallback to top-N if no proteins meet threshold (`stability.py:164-169`). **Recommendation**: Document rationale; log warning (not just info) on fallback.

#### 3.9 Panel Refill Greedy -- FLAG (Low)

`corr_prune.py:515-534`. Order-dependent greedy algorithm. Deterministic via `sorted(key=(-selection_freq, name))`. Standard approach; acceptable. **Recommendation**: Document order-dependence.

#### 3.10 Drop-Column Additive Assumption -- FLAG (Low)

`drop_column.py:190-203`. LOCO-style one-cluster-at-a-time importance assumes additive effects. Higher-order interactions missed. Standard limitation. **Recommendation**: Document limitation and suggest targeted multi-cluster ablation for suspected biological interactions.

---

### Domain 4: Model Training & Hyperparameter Optimization

**Files**: `models/optuna_search.py`, `models/hyperparams.py`, `models/nested_cv.py`, `models/registry.py`, `models/training.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 4.1 | Nested CV data leakage | PASS | -- |
| 4.2 | OOF prediction matrix coverage | PASS | -- |
| 4.3 | scale_pos_weight per fold | **MUST-FIX** | High |
| 4.4 | XGBoost spw=1.0 fallback (zero-positive fold) | **MUST-FIX** | High |
| 4.5 | Optuna TPE trial count | PASS | -- |
| 4.6 | Multi-objective Pareto selection | PASS | -- |
| 4.7 | Pipeline cloning independence | PASS | -- |
| 4.8 | Feature scaling within CV | PASS | -- |
| 4.9 | Random seed propagation | NICE-TO-HAVE | Low |

#### 4.1 Nested CV -- PASS

Inner CV folds are strictly subsets of outer training folds. `search.fit(X.iloc[train_idx], y[train_idx])` ensures no outer validation data leaks into inner optimization.

#### 4.2 OOF Matrix -- PASS

Initialized with NaN (`nested_cv.py:169`). Assigned via `preds[repeat_num, test_idx]` from `RepeatedStratifiedKFold`. Explicit validation: `RuntimeError` if any NaN remains after CV loop (`nested_cv.py:484-486`).

#### 4.3-4.4 Zero-Positive Fold Handling -- MUST-FIX (High)

**Location**: `nested_cv.py:226-236`, `registry.py:289-312`

When a fold has zero positive samples (statistically likely with 0.34% prevalence in 5-fold CV), `compute_scale_pos_weight_from_y` returns `spw=1.0` fallback. The model trains with incorrect class weight, produces under-confident predictions (~0.5), and contaminates the OOF matrix. Downstream metrics (AUROC, Brier, stacking) biased downward.

**Recommendation**: Detect zero-positive folds early and skip model training. Assign global prevalence as neutral prediction. Record skipped fold in metadata.

#### 4.5 Optuna TPE -- PASS

Default 400 trials (far exceeds 40 minimum). `check_tpe_hyperband_trials()` warns if < 40 with TPE+Hyperband.

#### 4.6 Pareto Selection -- PASS

All strategies (knee, extreme_auroc, balanced) are deterministic. `np.argmin/argmax` returns first occurrence for ties.

#### 4.7 Pipeline Cloning -- PASS

sklearn `clone()` creates truly independent estimators. No shared mutable state across folds.

#### 4.8 Feature Scaling -- PASS

`StandardScaler` is inside the Pipeline. sklearn's CV infrastructure fits it only on training fold data.

#### 4.9 Per-Fold Seed Diversity -- NICE-TO-HAVE (Low)

All folds use `random_state=42` for estimators. Deriving fold-specific seeds (`random_state + split_idx`) would improve ensemble diversity. Expected impact: < 1% AUROC.

---

### Domain 5: Calibration, Prevalence & Ensemble

**Files**: `models/calibration.py`, `models/prevalence.py`, `models/stacking.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 5.1 | OOF-posthoc calibration data flow | PASS | -- |
| 5.2 | Platt scaling (sigmoid method) | **MUST-FIX** | High |
| 5.3 | Isotonic regression (PAV) | PASS | -- |
| 5.4 | ECE binning (uniform, 10 bins) | NICE-TO-HAVE | Low |
| 5.5 | Calibration intercept/slope (logit clipping) | PASS | -- |
| 5.6 | Saerens prevalence adjustment | FLAG | Medium |
| 5.7 | Stacking meta-learner (L2, class_weight) | PASS | -- |
| 5.8 | Double calibration in stacking | **MUST-FIX** | High |

#### 5.1 OOF Data Flow -- PASS

Each sample's prediction comes from a model that never saw it (`nested_cv.py:198-424`). Calibrator fitted on genuinely held-out predictions. Explicit NaN check enforces completeness.

#### 5.2 Platt Scaling -- MUST-FIX (High)

**Location**: `calibration.py:318-322`

```python
log_odds = logit(oof_clean)
self.calibrator_ = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
self.calibrator_.fit(log_odds.reshape(-1, 1), y_clean)
```

Applies `logit()` to probabilities **before** fitting logistic regression. This creates a double-sigmoid transformation with no statistical justification. Platt scaling should fit directly on raw scores/probabilities.

**Recommendation**: Fit on `oof_clean` directly (not logit-transformed):
```python
self.calibrator_.fit(oof_clean.reshape(-1, 1), y_clean)
```

**Reference**: Platt (1999); Niculescu-Mizil & Caruana (2005).

#### 5.3 Isotonic Regression -- PASS

sklearn `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")`. PAV monotonicity guaranteed. Boundary handling correct.

#### 5.4 ECE Binning -- NICE-TO-HAVE (Low)

10 uniform bins. With 0.34% prevalence, most predictions cluster in low bins. Adaptive (quantile-based) binning would improve resolution. Not a blocker.

#### 5.5 Calibration Metrics -- PASS

`logit()` uses `eps=1e-7` clipping, preventing `log(0)`. No regularization (`C=np.inf`) correct for calibration intercept/slope measurement.

#### 5.6 Saerens Prevalence -- FLAG (Medium)

**Location**: `prevalence.py:54-114`

Uses closed-form intercept shift (`logit(p) + logit(prev_new) - logit(prev_old)`), not the full Saerens (2002) EM algorithm. Valid as first-order approximation. Edge cases handled. **Recommendation**: Validate that `mean(adjusted_probs) ~ target_prev` on validation data. Implement full EM if poor match.

#### 5.7 Stacking Meta-Learner -- PASS

L2-regularized logistic regression (`C=1.0`) with `class_weight="balanced"`. With 2-3 base models and ~43k samples, overfitting risk is negligible (1:14,000+ parameter-to-sample ratio).

#### 5.8 Double Calibration -- MUST-FIX (High)

**Location**: `stacking.py:240-246`

Base models produce calibrated predictions (via `oof_posthoc`). Meta-learner trained on these calibrated predictions. Then meta-learner output is re-calibrated via `CalibratedClassifierCV(method="isotonic")`. This double calibration can over-smooth predictions and reduce discrimination.

**Recommendation**: Default `calibrate_meta=False` when base models are already calibrated. Add diagnostic logging comparing Brier/AUROC with vs without meta-calibration.

---

### Domain 6: Evaluation Metrics & Significance Testing

**Files**: `metrics/discrimination.py`, `metrics/thresholds.py`, `metrics/dca.py`, `metrics/bootstrap.py`, `significance/permutation_test.py`

| # | Item | Finding | Severity |
|---|------|---------|----------|
| 6.1 | AUROC computation | PASS | -- |
| 6.2 | Threshold strategies (Youden, fixed-spec, F1) | PASS | -- |
| 6.3 | DCA net benefit formula | PASS | -- |
| 6.4 | DCA integration (trapezoidal) | PASS | -- |
| 6.5 | Bootstrap (stratified, percentile) | PASS | -- |
| 6.6 | Phipson-Smyth correction | PASS | -- |
| 6.7 | Permutation target (y_train only) | PASS | -- |
| 6.8 | Cross-fold meta-analysis (pooled null) | FLAG | Medium |
| 6.9 | Default n_perms=200 | **MUST-FIX** | High |
| 6.10 | Bootstrap BCa vs percentile | NICE-TO-HAVE | Low |

#### 6.1 AUROC -- PASS

Delegates to sklearn `roc_auc_score`. Mann-Whitney interpretation valid for extreme imbalance (tests rank ordering). Validates both classes present.

#### 6.2 Thresholds -- PASS

Youden's J (`tpr - fpr`), fixed specificity at 0.95, max F1 -- all mathematically correct. Edge cases: single class, empty thresholds, NaN/inf all handled with safe fallbacks.

#### 6.3 DCA Net Benefit -- PASS

`NB = TP/n - FP/n * pt/(1-pt)` matches Vickers & Elkin (2006). Treat-all uses prevalence-based formula. Treat-none = 0.0.

#### 6.4 DCA Integration -- PASS

numpy `trapezoid` (trapezoidal rule). Zero-crossing uses linear interpolation. Boundary conditions handled.

#### 6.5 Bootstrap -- PASS

Stratified sampling preserves class ratio. Percentile method uses 2.5%/97.5% quantiles (correct alpha/2). Default n_boot=1000. Seeded RNG for reproducibility.

#### 6.6 Phipson-Smyth -- PASS

`p = (1 + #{null >= obs}) / (1 + B)` matches Phipson & Smyth (2010). Prevents p=0.

#### 6.7 Permutation Target -- PASS

Only `y_train` permuted. X features and `y_test` labels remain intact. Full pipeline (screening, feature selection, HPO) re-run per permutation.

#### 6.8 Cross-Fold Meta-Analysis -- FLAG (Medium)

**Location**: `significance/aggregation.py:212-337`

Uses pooled-null distribution (all null AUROCs across folds/seeds) rather than Fisher's or Stouffer's method. Valid when null distributions are directly comparable (same model, same scenario). More powerful than Fisher's for this use case. **Recommendation**: Document the choice and cite Westfall & Young (1993).

#### 6.9 Permutation Count -- MUST-FIX (High)

**Location**: `permutation_test.py:209`

Default `n_perms=200` yields min p = 1/201 ~ 0.005. For p < 0.01 claims, 1000+ recommended. Docstring already notes "B >= 1000 for final validation" but default is 200.

**Recommendation**: Increase default to 1000. Add warning if n_perms < 1000 in publication context.

#### 6.10 BCa Bootstrap -- NICE-TO-HAVE (Low)

Percentile method used. BCa would be superior for skewed distributions (common with extreme imbalance). Acceptable for current use with n_boot >= 1000.

---

## 3. Cross-Cutting Concerns

### 3.1 DRY Violations

- **Mann-Whitney and F-stat**: Both implemented in `screening.py` and partially in `kbest.py`. Consolidate or deprecate `kbest.py`.
- **Correlation clustering**: Two algorithms (`corr_prune.py` graph DFS, `drop_column.py` hierarchical). Should share one implementation.

### 3.2 Missing Confidence Intervals

- Feature importance estimates lack CIs (no bootstrap on importance).
- RRA consensus scores lack uncertainty quantification.
- Threshold estimates lack CIs.

### 3.3 Inconsistencies

- Missing data: pairwise deletion (Mann-Whitney) vs median imputation (F-test) within same module.
- Clustering: connected components vs average-linkage hierarchical for same conceptual task.
- Coefficient importance: L1-normalized in OOF path, raw absolute in RFE path.

### 3.4 Documentation Gaps

- `_resid` provenance not documented anywhere in the codebase.
- RRA weights (0.6/0.3/0.1) not justified.
- Stability thresholds (0.75, 0.90) not justified.
- Simplified RRA vs formal Kolde et al. (2012) not documented.

---

## 4. Validation Checklist

### Formulas Verified Against Literature

| Formula | Reference | Status |
|---------|-----------|--------|
| DCA net benefit `NB = TP/n - FP/n * pt/(1-pt)` | Vickers & Elkin (2006) | PASS |
| Phipson-Smyth `p = (1+#{null>=obs})/(1+B)` | Phipson & Smyth (2010) | PASS |
| Saerens prevalence shift `logit(p) + delta` | Saerens et al. (2002) (simplified) | FLAG -- first-order approximation |
| OOF stacking (held-out meta-features) | Wolpert (1992) | PASS |
| Youden's J `TPR - FPR` | Youden (1950) | PASS |
| Perpendicular distance elbow detection | Standard computational geometry | PASS |
| Stratified bootstrap (class-preserving) | Efron & Tibshirani (1993) | PASS |

### Assumptions Documented

| Assumption | Documented? | Validated? |
|------------|------------|------------|
| MCAR for listwise deletion | No | No |
| Normality for F-test | No | Justified by CLT (N=43,960) |
| Feature additivity (drop-column) | No | Standard LOCO limitation |
| P(X\|Y) invariance (Saerens) | Yes (docstring) | Not empirically validated |
| Residualization provenance | No | **Unknown** |

### Edge Cases Handled

| Edge Case | Status |
|-----------|--------|
| Single class in fold (no positives) | Handled but with degraded fallback (M8) |
| All predictions identical | Handled (threshold defaults to 0.5) |
| Empty null distribution | Raises ValueError |
| Zero coefficients | Handled (returns uniform importance) |
| Predictions exactly 0 or 1 | Clipped via epsilon (1e-7) |

---

## 5. Prioritized Action Plan

### P0: Critical Blockers (before publication)

1. **M1**: Resolve `_resid` provenance -- document or re-run with split-aware residualization
2. **M9**: Fix Platt scaling -- fit on raw probabilities, not logit(probabilities)

### P1: High-Impact Fixes (before final results)

3. **M4**: Replace zero-fill with NaN-aware aggregation in `importance.py`
4. **M5**: Rename/normalize RRA; document deviation from Kolde et al.
5. **M6**: Run weight sensitivity analysis (Jaccard overlap of top-20)
6. **M8**: Skip zero-positive folds instead of training with spw=1.0
7. **M10**: Default `calibrate_meta=False` when base models are calibrated
8. **M11**: Increase default n_perms from 200 to 1000

### P2: Should-Fix (before peer review)

9. **M2/M3**: Standardize missing data handling in screening (pairwise deletion for both)
10. **M7**: Standardize clustering method across corr_prune and drop_column
11. Document RRA weights rationale, stability thresholds, and _resid provenance

### P3: Nice-to-Have (future improvement)

12. Add standardized effect sizes (Cohen's d, rank-biserial)
13. Implement adaptive ECE binning
14. Add BCa bootstrap option
15. Per-fold seed diversity for ensemble
16. Add CIs to importance estimates

---

## 6. References

- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Kolde, R. et al. (2012). Robust rank aggregation for gene list integration and meta-analysis. Bioinformatics, 28(4), 573-580.
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. ICML.
- Ojala, M. & Garriga, G.C. (2010). Permutation tests for studying classifier performance. JMLR, 11, 1833-1863.
- Phipson, B. & Smyth, G.K. (2010). Permutation P-values should never be zero. Statistical Applications in Genetics and Molecular Biology, 9(1).
- Platt, J. (1999). Probabilistic outputs for support vector machines. Advances in Large Margin Classifiers.
- Saerens, M. et al. (2002). Adjusting the outputs of a classifier to new a priori probabilities. Neural Computation, 14(1), 21-41.
- Van Calster, B. et al. (2016). A calibration hierarchy for risk models. Statistics in Medicine, 35(20), 3524-3545.
- Vickers, A.J. & Elkin, E.B. (2006). Decision curve analysis. Medical Decision Making, 26(6), 565-574.
- Westfall, P.H. & Young, S.S. (1993). Resampling-Based Multiple Testing. Wiley.
- Wolpert, D.H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.
- Youden, W.J. (1950). Index for rating diagnostic tests. Cancer, 3(1), 32-35.

---

## Appendix: Positive Findings (Validated Correct)

The following 25 items were validated as mathematically/statistically correct:

1. Stratified split preserves class ratios (D1.1)
2. Prevalent cases isolated to TRAIN with multi-layer validation (D1.2)
3. Seed derivation via XOR is bijective and independent (D1.4)
4. Column resolution correctly separates proteins from metadata (D1.7)
5. Mann-Whitney U test: two-sided, asymptotic, ties handled (D2.1)
6. OOF grouped PFI preserves cluster structure (D3.1)
7. RFE knee detection formula mathematically correct (D3.7)
8. Nested CV: no data leakage between outer/inner folds (D4.1)
9. OOF prediction matrix covers all samples exactly once per repeat (D4.2)
10. Optuna TPE: 400 trials >> 40 minimum (D4.5)
11. Pareto selection deterministic and reproducible (D4.6)
12. Pipeline cloning creates independent estimators (D4.7)
13. Feature scaling within CV: no leakage (D4.8)
14. OOF-posthoc calibration uses genuinely held-out predictions (D5.1)
15. Isotonic regression: PAV monotonicity, boundary handling correct (D5.3)
16. Calibration logit clipping prevents log(0) (D5.5)
17. Stacking meta-learner: appropriate regularization and class weighting (D5.7)
18. AUROC delegates to sklearn correctly (D6.1)
19. Youden's J, fixed-spec, max-F1 all correct (D6.2)
20. DCA net benefit matches Vickers & Elkin (2006) (D6.3)
21. DCA integration (trapezoidal rule) correct (D6.4)
22. Bootstrap: stratified, percentile, alpha/2, n=1000 (D6.5)
23. Phipson-Smyth correction prevents p=0 (D6.6)
24. Permutation: only y_train permuted, full pipeline re-run (D6.7)
25. Numerical stability: epsilon clipping applied consistently (D5.5, D6.6)
