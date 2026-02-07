# Mathematical & Statistical Validity Audit Plan

## Context

Full-pipeline audit of the CeliacRisks ML codebase (incident celiac disease risk prediction from proteomics). Goal: catalog and validate every mathematical/statistical operation from data loading through significance testing. Three exploration agents already mapped 50+ mathematical operations across 30+ source files. This plan launches targeted code-reviewer agents to produce a formal audit report.

---

## Exploration Summary: What We Found

### MUST-FIX Issues (9 total, blocking publication/clinical validity)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1 | **Unknown residualization process** -- `_resid` suffix implies upstream covariate adjustment. No code found. Leakage status unknown. | `schema.py`, external? | If residualization used full dataset (not split-aware), test set leakage invalidates all results |
| 2 | **TRAIN/VAL-TEST distribution mismatch** -- TRAIN has prevalent (post-diagnosis) + incident (pre-diagnosis) biomarkers. VAL/TEST incident-only | `splits.py:326-366` | Model trains on mixed signal, evaluated on different distribution |
| 3 | **Temporal missing value placement** -- Missing dates set to `min_date - 1` (earliest). If missingness is informative, creates temporal leakage | `splits.py:348-359` | Train/test contamination in temporal split mode |
| 4 | **No multiple testing correction at screening** -- 2,920 Mann-Whitney/F-tests with no FDR/Bonferroni. Expected ~146 false positives at alpha=0.05 | `screening.py` | False discovery inflation (mitigated downstream but not formally corrected) |
| 5 | **Zero-filling for absent features** -- Features missing from a CV fold get importance=0, biasing aggregated means downward | `importance.py:778` | Underestimates importance of features not selected in all folds |
| 6 | **RRA weights arbitrary (0.6/0.3/0.1)** -- No cross-validation, sensitivity analysis, or statistical justification | `consensus.py:109-113` | Consensus panel composition sensitive to weight choice |
| 7 | **Simplified RRA** -- Uses geometric mean of reciprocal ranks, not Stuart's permutation-based p-value method | `consensus.py:249-388` | No p-values or significance thresholds for consensus ranking |
| 8 | **No cross-fold permutation meta-analysis** -- Per-fold p-values computed independently, no Fisher's/Stouffer's combination | `permutation_test.py:377-423` | Cannot make overall significance statement across folds |
| 9 | **200 permutations** yields min p=0.005 -- Insufficient for p<0.01 claims; 1000+ recommended for publication | `permutation_test.py` | Resolution limit on significance |

### NICE-TO-HAVE Issues (12 total, enhances robustness)

- Age bin clinical justification (40/60 cutpoints arbitrary)
- Unstandardized effect sizes (mean diff, not Cohen's d)
- Median imputation before F-tests inflates statistics
- Normality assumption unvalidated for ANOVA F-test
- Inconsistent clustering: graph (corr_prune) vs hierarchical (drop_column)
- Percentile bootstrap (not BCa) for skewed distributions
- No uncertainty quantification (CIs) on importance estimates
- Rank normalization loses magnitude info in consensus
- Greedy panel algorithms (suboptimal but tractable)
- ECE uses uniform binning (adaptive better for imbalanced data)
- Multi-size panels not nested (larger panel may not contain smaller)
- DRY violations: Mann-Whitney and F-stat implemented twice

### Positive Findings (validated correct)

- OOF-posthoc calibration uses genuinely held-out predictions (no leakage)
- Phipson-Smyth p-value correction (prevents p=0)
- Stratified bootstrap preserves class ratio
- Numerical stability: epsilon clipping on logit/log consistently applied
- Edge cases (single class, zero positives) handled with guards
- All hyperparameters configurable via YAML
- Prevalence adjustment correctly implements Saerens (2002)
- DCA net benefit formula matches Vickers & Elkin (2006)

---

## Audit Execution Plan: 6 Code-Review Agents

### Domain 1: Data Preparation & Sampling Validity (Sonnet)

**Files**: `data/splits.py`, `data/filters.py`, `data/columns.py`, `data/io.py`, `data/schema.py`

**Verify**:
- Stratification preserves class ratio across 18 strata (3 age x 3 sex x 2)
- Downsampling formula: `round(n_cases_stratum x ratio)` sums correctly
- Prevalent cases ONLY in TRAIN (no leakage to VAL/TEST indices)
- Temporal missing fill logic (`min_date - 1`) -- flag as risk
- Seed derivation `seed XOR 0x5678` -- collision/independence analysis
- `_resid` suffix documentation -- flag unknown provenance
- Listwise deletion assumes MCAR -- flag assumption

### Domain 2: Feature Screening & Statistical Tests (Sonnet)

**Files**: `features/screening.py`, `features/kbest.py`

**Verify**:
- Mann-Whitney U: two-sided correct, asymptotic valid for n>=10
- F-test: normality/homoscedasticity assumptions stated
- Multiple testing: 2920 comparisons, no correction -- is this intentional screening?
- Median imputation BEFORE testing: quantify bias direction
- SelectKBest vs screening.py: numerical equivalence of F-statistics
- Missing value handling: pairwise deletion (MWU) vs median impute (F-test) consistency

### Domain 3: Feature Importance, Selection & Consensus (Opus)

**Files**: `features/importance.py`, `features/grouped_importance.py`, `features/drop_column.py`, `features/rfe.py`, `features/rfe_engine.py`, `features/nested_rfe.py`, `features/stability.py`, `features/corr_prune.py`, `features/consensus.py`, `features/panels.py`

**Verify**:
- OOF grouped PFI: permutation respects cluster structure
- L1-normalized |coef| for linear models: scale-dependent?
- Zero-fill vs NaN for absent features: bias quantification
- RRA geometric mean vs Stuart's method: document deviation
- Weight sensitivity: 0.6/0.3/0.1 -- would 0.5/0.3/0.2 change top-20?
- Clustering consistency: graph DFS vs scipy hierarchical -- same clusters?
- RFE knee detection: perpendicular distance formula correct
- Stability threshold cascade: 0.75 (repeat) -> 0.90 (consensus)
- Panel refill greedy: order-dependence analysis
- Drop-column: additive assumption (no interaction effects documented)

### Domain 4: Model Training & Hyperparameter Optimization (Sonnet)

**Files**: `models/optuna_search.py`, `models/hyperparams.py`, `models/nested_cv.py`, `models/registry.py`, `models/training.py`

**Verify**:
- Nested CV: no data leakage between outer/inner folds
- OOF prediction matrix: all samples covered exactly once per repeat
- Class imbalance: `scale_pos_weight = n_neg/n_pos` per fold (not global)
- XGBoost fallback: `spw=1.0` when no positives -- downstream effect
- Optuna TPE: min 40 trials for effective Bayesian modeling
- Multi-objective Pareto: selection deterministic and reproducible
- Pipeline cloning: verify `clone()` creates independent estimators

### Domain 5: Calibration, Prevalence & Ensemble (Sonnet)

**Files**: `models/calibration.py`, `models/prevalence.py`, `models/stacking.py`

**Verify**:
- OOF-posthoc: pooled predictions genuinely held-out (trace data flow)
- Platt scaling: `C=inf` (no regularization) correct for calibration
- Isotonic: PAV monotonicity, boundary handling (0/1 predictions)
- ECE: 10 bins uniform -- verify deterministic edges
- Calibration intercept/slope: logit clipping prevents log(0)
- Saerens: `P(X|Y)` invariance assumption documented and justified
- Stacking: L2 meta-learner on OOF -- overfitting risk with few base models
- Stacking calibration: isotonic on meta-predictions (double calibration?)

### Domain 6: Evaluation Metrics & Significance Testing (Sonnet)

**Files**: `metrics/discrimination.py`, `metrics/thresholds.py`, `metrics/dca.py`, `metrics/bootstrap.py`, `significance/permutation_test.py`

**Verify**:
- AUROC: Mann-Whitney interpretation correct for imbalanced data
- Threshold strategies: Youden, fixed-spec, max-F1 all mathematically correct
- DCA net benefit: `NB = TP/n - FP/n x pt/(1-pt)` matches Vickers 2006
- DCA integration: trapezoidal rule correct, zero-crossing interpolation valid
- Bootstrap: stratified, percentile method, alpha/2 quantiles (not alpha)
- Permutation: Phipson-Smyth `(1+#{null>=obs})/(1+B)` correct
- Permutation: only y_train permuted (not X, not y_test)
- Cross-fold: no meta-analysis -- flag as gap
- 200 permutations: resolution limit p>=0.005 -- flag for publication

---

## Deliverables

All outputs written to `analysis/docs/development/`:

1. **`MATHEMATICAL_VALIDITY_AUDIT_PLAN.md`** -- This plan (copied from plan file on execution start)
2. **`MATHEMATICAL_VALIDITY_AUDIT.md`** -- Final audit report

Audit report structure:
1. Executive Summary (issue counts by severity)
2. Per-domain findings (MUST-FIX + NICE-TO-HAVE with file:line, formula, impact, recommendation)
3. Cross-cutting concerns (duplicates, inconsistencies, missing CIs)
4. Validation checklist (formulas match literature, assumptions documented, edge cases handled)
5. References (cited statistical methods)

---

## Verification

After audit report is written:
- Spot-check 3 formulas against cited papers (Saerens 2002, Vickers 2006, Phipson-Smyth 2010)
- Verify code paths traced by reviewers match actual function calls
- Run `pytest tests/ -v` to confirm existing tests pass (no regressions from audit observations)

---

## Agent Model Selection

| Domain | Model | Rationale |
|--------|-------|-----------|
| 1. Data Prep | Sonnet | Complex sampling, critical for downstream validity |
| 2. Screening | Sonnet | Multiple testing, distributional assumptions |
| 3. Feature Selection | Opus | Most complex domain (10 files, consensus algorithms, 9 MUST-FIX items) |
| 4. Training/Optuna | Sonnet | Standard ML, moderate complexity |
| 5. Calibration | Sonnet | Calibration theory, Saerens assumptions |
| 6. Eval/Significance | Sonnet | Bootstrap theory, permutation validity |

All 6 agents run with `subagent_type=code-reviewer`. Domains 1-2 and 4-6 in parallel (Sonnet). Domain 3 (Opus) may run concurrently or sequentially based on context budget.

Unresolved questions: none.
