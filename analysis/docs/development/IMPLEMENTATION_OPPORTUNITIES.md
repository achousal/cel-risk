# Implementation Opportunities

**Date**: 2026-01-31
**Version**: 1.0.0
**Status**: Active roadmap

---

## Summary

This document catalogs high-value implementation opportunities for the CeD-ML pipeline, prioritized by impact and effort. All items are production-ready and align with existing architecture decisions.

**Dataset context**: 43,960 samples, 2,920 protein features, 0.34% prevalence (148 incident cases)

**Current state**: Production-ready ML pipeline with stacking ensemble, OOF-posthoc calibration, temporal validation, panel optimization, and cross-model consensus. Test coverage: 82%+.

---

## Priority Matrix

| Priority | Effort | Items |
|----------|--------|-------|
| P0 (Critical) | < 4 hours | Test suite for new utils, integrate style.py, commit refactoring |
| P1 (High Impact) | 1-3 days | Clinical deployment module, SHAP explainability, panel size curves |
| P2 (Performance) | 2-5 days | RFE parallelization, pipeline profiling |
| P3 (Scientific) | 3-7 days | Cross-cohort validation, temporal validation, uncertainty quantification |
| P4 (Infrastructure) | 2-5 days | Pre-commit hooks, CI/CD pipeline |

---


## P1: HIGH-IMPACT FEATURES (1-3 days)

### 4. Clinical Deployment Module

**Status**: Complete 617-line specification ([DEPLOYMENT.md](DEPLOYMENT.md)) marked "speculative". Needs implementation.

**Motivation**: Move pipeline from research-grade to deployment-grade for clinical screening applications.

**Deliverable**: New module + CLI command

**Module structure**:
```
src/ced_ml/deployment/
├── __init__.py
├── prevalence.py       # adjust_probabilities_for_prevalence()
├── wrapper.py          # DeploymentModel class
├── validation.py       # detect_drift(), validate_calibration()
├── reporting.py        # stratify_patients(), generate_model_card()
└── batch.py           # batch_predict()
```

**New CLI command**:
```bash
ced deploy \
    --model-path results/LR_EN/split_seed0/core/final_model.pkl \
    --cohort-path new_patients.parquet \
    --deployment-prevalence 0.0034 \
    --thresholds-json thresholds.json \
    --output-dir results/deployment/

# Outputs:
# - risk_scores.csv (patient_id, risk_score, risk_category, deployment_date)
# - model_card.yaml (version, performance, limitations)
# - calibration_plot.png (before/after prevalence adjustment)
# - drift_report.html (feature distribution shifts)
```

**Key features**:
1. **Prevalence adjustment**: Logit-scale adjustment from training prevalence (16.7%) to deployment prevalence (user-specified)
2. **Risk stratification**: Assign patients to low/medium/high risk tiers
3. **Drift detection**: Statistical tests (KS, Wasserstein) for feature distribution shifts
4. **Calibration validation**: Compare predicted vs observed probabilities on deployment cohort
5. **Model card generation**: Standardized metadata (version, performance, limitations)

**Testing requirements**:
- Unit tests for prevalence adjustment (edge cases: very low/high prevalence)
- Integration test: end-to-end deployment workflow on toy data
- Validate AUROC unchanged after prevalence adjustment (discrimination preserved)

**Acceptance criteria**:
- Matches [DEPLOYMENT.md](DEPLOYMENT.md) specification
- >= 80% test coverage for deployment module
- Passes on synthetic cohort with known prevalence shift

**Effort**: 2-3 days

**References**: [ADR-010: Prevalence Adjustment](../adr/ADR-010-prevalence-adjustment.md), Steyerberg (2019) Ch. 13

---

### 5. Model Explainability Suite (SHAP)

**Status**: Clinical ML requires interpretability. Current pipeline has metrics but no local/global explanations.

**Motivation**:
- Clinicians need to understand **why** a patient is high-risk (which proteins drive prediction)
- Regulatory requirements for "explainable AI" in healthcare
- Scientific discovery: identify novel biomarker interactions

**Deliverable**: New module + CLI command + plots

**Module structure**:
```
src/ced_ml/explain/
├── __init__.py
├── shap_values.py      # TreeExplainer, KernelExplainer wrappers
├── waterfall.py        # Per-patient feature attributions
├── summary.py          # Cohort-level feature importance
└── interactions.py     # Protein-protein interaction detection
```

**New CLI command**:
```bash
ced explain \
    --model-path results/XGBoost/split_seed0/core/final_model.pkl \
    --data-path data/test_split_seed0.parquet \
    --sample-ids 12345,67890,... \
    --output-dir plots/shap/

# Or explain entire cohort
ced explain \
    --model-path results/ensemble/final_model.pkl \
    --data-path data/test_split_seed0.parquet \
    --n-samples 100 \
    --output-dir plots/shap/
```

**Outputs**:
1. **Waterfall plots** (`waterfall_patient_{id}.png`): Per-patient feature contributions
   - Show top 20 proteins pushing risk up/down
   - Annotate base rate, feature effects, final prediction
2. **Beeswarm summary** (`summary_cohort.png`): Cohort-level feature importance
   - Rank proteins by |SHAP value|
   - Show distribution of effects (positive/negative)
3. **Dependence plots** (`dependence_{protein}.png`): Marginal effect of each top-10 protein
   - X-axis: protein level, Y-axis: SHAP value
   - Color by interaction feature (auto-detected)
4. **Interaction matrix** (`interactions_heatmap.png`): Protein-protein interactions (top 20 × 20)

**Implementation notes**:
- Use `shap.TreeExplainer` for tree models (XGBoost, RF) - fast, exact
- Use `shap.KernelExplainer` for linear models (LR_EN) - slower, sampling-based
- For ensemble: compute SHAP for each base model, aggregate weighted by stacking coefficients
- Cache SHAP values to avoid recomputation

**Testing requirements**:
- Unit tests: SHAP values sum to (prediction - base_rate)
- Integration test: Generate all plot types on toy data
- Validate: Top features from SHAP match feature importance from model

**Acceptance criteria**:
- Works for all model types (LR_EN, RF, XGBoost, ensemble)
- Generates publication-ready plots
- Execution time: < 5 min for 100 samples on XGBoost

**Effort**: 2-3 days

**Dependencies**: `shap>=0.43.0`, `matplotlib>=3.7.0`

**References**: Lundberg & Lee (2017), SHAP documentation

---

### 6. Panel Size Optimization Curves

**Status**: Panel optimization exists ([ADR-013: Feature Selection](../adr/ADR-013-four-strategy-feature-selection.md)) but no visualization of performance vs. panel size trade-off.

**Motivation**:
- Clinical labs need to balance performance vs. cost
- Justify panel size choice: "127 proteins achieve AUROC=0.89; 50 proteins achieve AUROC=0.87 at 60% cost reduction"
- Identify elbow point in performance curve

**Deliverable**: Extension to [panel_curve.py](../../src/ced_ml/plotting/panel_curve.py)

**New function**:
```python
def plot_panel_size_tradeoff(
    panel_sizes: List[int],         # [10, 20, 50, 100, 200, ...]
    metrics_by_size: Dict[str, List[float]],  # {"AUROC": [...], "AUPRC": [...]}
    cost_per_protein: float = 100.0,  # $ per protein assay
    output_path: Path = None
) -> Figure:
    """
    Plot performance vs. panel size with cost overlay.

    Features:
    - Dual-axis: AUROC/AUPRC (left), total cost (right)
    - Annotate elbow point (max performance/cost ratio)
    - Shade confidence intervals from bootstrap
    """
```

**CLI integration**:
```bash
# After running optimize-panel
ced plot-panel-tradeoff \
    --run-id 20260127_115115 \
    --model LR_EN \
    --cost-per-protein 100 \
    --output plots/panel_tradeoff.png
```

**Output plot**:
```
Left Y-axis: AUROC (0.7 - 0.9)
Right Y-axis: Total cost ($1,000 - $30,000)
X-axis: Panel size (10 - 300 proteins)

Features:
- Two lines: AUROC (blue), AUPRC (orange)
- Cost curve (red, dashed, right Y-axis)
- Vertical line at elbow point (e.g., 75 proteins)
- Annotation: "Elbow: 75 proteins, AUROC=0.86, cost=$7,500"
- CI ribbons (bootstrap 95% CI)
```

**Testing requirements**:
- Unit test: Elbow detection algorithm (second derivative max)
- Visual regression test: Compare output plot to reference

**Acceptance criteria**:
- Generates publication-ready figure
- Elbow point detection agrees with manual inspection
- Cost calculation matches (n_proteins × cost_per_protein)

**Effort**: 1 day


## P2: PERFORMANCE OPTIMIZATIONS (2-5 days)

### 8. RFE Parallelization

**Status**: Nested RFECV takes **22 hours** ([ADR-013](../adr/ADR-013-four-strategy-feature-selection.md)). Current implementation is serial: 2,920 features × nested CV × bootstrap.

**Motivation**:
- Reduce nested RFE runtime from 22 hours to ~3-5 hours (4-8x speedup on HPC)
- Enable routine use of nested RFE (currently only for scientific discovery due to runtime)

**Deliverable**: Parallel RFE implementation in [features/nested_rfe.py](../../src/ced_ml/features/nested_rfe.py)

**Implementation**:
```python
from joblib import Parallel, delayed
from ced_ml.config.training import FeatureSelectionConfig

def _rfe_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator,
    n_features_to_select: int,
    step: float,
    cv,
    scoring: str
) -> Dict:
    """Single fold RFE - parallelizable unit of work."""
    from sklearn.feature_selection import RFECV

    rfe = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        n_jobs=1  # Inner parallelism disabled (outer loop handles it)
    )
    rfe.fit(X_train, y_train)

    return {
        "n_features": rfe.n_features_,
        "ranking": rfe.ranking_,
        "support": rfe.support_,
        "cv_scores": rfe.cv_results_["mean_test_score"]
    }

def nested_rfe_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    outer_cv,
    config: FeatureSelectionConfig,
    n_jobs: int = -1  # Use all available cores
) -> Dict:
    """Nested RFE with outer-loop parallelization."""

    # Parallelize across outer CV folds
    fold_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_rfe_single_fold)(
            X[train_idx], y[train_idx],
            estimator, config.n_features_to_select,
            config.rfe_step, config.inner_cv, config.scoring
        )
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y))
    )

    # Aggregate results across folds
    return aggregate_rfe_results(fold_results)
```

**Configuration**:
```yaml
# configs/training_config.yaml
feature_selection:
  strategy: nested_rfe
  rfe_step: 0.1
  n_jobs: 8  # Parallelize across 8 cores (or -1 for all cores)
```

**Expected speedup**:
- Local (8 cores): 22 hours → 3-4 hours (6-7x)
- HPC (32 cores): 22 hours → 1-2 hours (11-22x)

**Trade-offs**:
- Memory usage increases (each worker holds copy of data)
- Requires careful random seed management (each worker gets deterministic seed)

**Testing requirements**:
- Unit test: Results identical to serial version (same random seed)
- Performance test: Measure speedup on toy data (expect linear scaling up to memory limit)
- Smoke test: Run on full dataset with n_jobs=2

**Acceptance criteria**:
- Speedup >= 4x on 8-core machine
- Results numerically identical to serial version (deterministic)
- Memory usage < 2x serial version

**Effort**: 2-3 days

---

## P3: SCIENTIFIC ENHANCEMENTS (3-7 days)

### 10. Cross-Cohort Validation Framework

**Status**: Trained on UK Biobank. No external validation on independent cohorts.

**Motivation**:
- Assess generalization to other populations (age, sex, ethnicity, geography)
- Detect platform-specific effects (proteomics assay, batch effects)
- Regulatory requirement: demonstrate robustness before clinical deployment

**Deliverable**: External validation module + CLI command

**Module structure**:
```
src/ced_ml/validation/
├── __init__.py
├── schema_check.py     # Validate feature names, types, ranges
├── harmonization.py    # Map protein names across platforms (e.g., Olink → SomaScan)
├── missing_handler.py  # Handle missing features (imputation, feature subset)
└── calibration_check.py # Recalibration for new cohort
```

**New CLI command**:
```bash
ced validate-external \
    --model-bundle results/LR_EN/split_seed0/core/final_model.pkl \
    --cohort-path external_cohort.parquet \
    --cohort-name "Netherlands_Lifelines" \
    --output-dir results/external_validation/

# Outputs:
# - feature_mismatch_report.txt (missing/extra features)
# - performance_metrics.json (AUROC, AUPRC, Brier on external cohort)
# - calibration_before_after.png (calibration curve before/after recalibration)
# - recommendation.txt ("Deploy as-is" / "Recalibrate" / "Retrain required")
```

**Workflow**:
1. **Schema validation**: Check for missing/extra features
   - Missing features: Options (fail, impute, drop from panel)
   - Extra features: Ignore (model only uses features it was trained on)
2. **Feature harmonization**: Map protein names across platforms
   - Example: UK Biobank uses Olink IDs, external cohort uses UniProt IDs
   - Require user-provided mapping CSV
3. **Distribution checks**: Compare feature distributions (KS test, Wasserstein distance)
   - Flag features with significant shifts (p < 0.01)
4. **Performance evaluation**: Compute metrics on external cohort
   - AUROC (expect similar to test set if no data shift)
   - Calibration (expect degradation if prevalence differs)
5. **Recalibration** (if needed): Refit isotonic regression on external cohort
6. **Recommendation**: Decision tree based on performance
   - AUROC drop < 5%: Deploy as-is
   - AUROC drop 5-10%: Recalibrate
   - AUROC drop > 10%: Retrain required

**Example output**:
```
External Validation Report
Cohort: Netherlands_Lifelines (n=5,230, prevalence=0.8%)

Feature Validation:
  ✓ 2,920 features matched
  ✗ 15 features missing (imputed with median)
  ⚠ 42 features show distribution shift (p < 0.01)

Performance (before recalibration):
  AUROC: 0.84 (vs 0.89 on UK Biobank test) ← 5.6% drop
  Calibration slope: 0.72 (underconfident)

Performance (after recalibration):
  AUROC: 0.84 (unchanged, as expected)
  Calibration slope: 0.96 (well-calibrated)

Recommendation: Recalibrate before deployment
```

**Testing requirements**:
- Unit tests: Schema validation, harmonization logic
- Integration test: Run on synthetic external cohort with known shifts

**Acceptance criteria**:
- Handles missing features gracefully (imputation or panel subset)
- Detects distribution shifts (flags proteins with KS p < 0.01)
- Recalibration improves calibration without changing AUROC

**Effort**: 3-5 days

---

### 11. Temporal Validation Analysis

**Status**: UK Biobank data is cross-sectional (random train/val/test split). No time-based validation.

**Motivation**:
- Assess model stability over time (biomarker distributions may drift)
- Detect secular trends (e.g., changing CeD incidence, changing assay platforms)
- Regulatory requirement: demonstrate prospective validity

**Deliverable**: Time-aware splitting + drift detection

**Extension to [data/splits.py](../../src/ced_ml/data/splits.py)**:
```python
def create_temporal_splits(
    data: pd.DataFrame,
    time_col: str = "baseline_date",
    train_years: Tuple[int, int] = (2006, 2011),  # 5 years
    val_years: Tuple[int, int] = (2011, 2013),    # 2 years
    test_years: Tuple[int, int] = (2013, 2015),   # 2 years
    config: SplitsConfig = None
) -> Dict[str, pd.DataFrame]:
    """
    Create time-based splits (no overlap in time).

    Returns:
        {"train": df_train, "val": df_val, "test": df_test}
    """
    data["year"] = pd.to_datetime(data[time_col]).dt.year

    train_mask = data["year"].between(*train_years)
    val_mask = data["year"].between(*val_years)
    test_mask = data["year"].between(*test_years)

    return {
        "train": data[train_mask],
        "val": data[val_mask],
        "test": data[test_mask]
    }
```

**New CLI command**:
```bash
ced create-temporal-splits \
    --infile data/input.parquet \
    --time-col baseline_date \
    --train-years 2006-2011 \
    --val-years 2011-2013 \
    --test-years 2013-2015 \
    --output-dir data/temporal_splits/

# Train on temporal split
ced train \
    --config configs/training_config.yaml \
    --data-dir data/temporal_splits/ \
    --model LR_EN
```

**Drift detection**:
```bash
ced detect-drift \
    --baseline-run 20260127_115115 \  # Trained on 2006-2011
    --current-run 20260131_143022 \   # Trained on 2011-2015
    --output plots/drift_report.html

# Output: HTML report showing:
# - Feature distribution shifts (KS test p-values)
# - Performance degradation (AUROC over time)
# - Calibration drift (calibration slope over time)
```

**Testing requirements**:
- Unit test: Temporal splits have no time overlap
- Integration test: Train on early years, test on later years

**Acceptance criteria**:
- Splits are temporally non-overlapping
- Drift detection flags proteins with KS p < 0.01
- Report quantifies performance degradation over time

**Effort**: 2-3 days

---

### 12. Uncertainty Quantification Enhancement

**Status**: Bootstrap confidence intervals exist ([metrics/bootstrap.py](../../src/ced_ml/metrics/bootstrap.py)), but no conformal prediction or calibrated uncertainty.

**Motivation**:
- Clinical decisions require calibrated uncertainty (e.g., "Patient risk: 4.2% ± 1.5%")
- Identify patients near decision boundaries (high uncertainty → defer to expert)
- Regulatory requirement: quantify prediction uncertainty

**Deliverable**: Prediction intervals for individual patients

**Module structure**:
```
src/ced_ml/uncertainty/
├── __init__.py
├── conformal.py        # Conformal prediction intervals
├── calibrated_ci.py    # Temperature scaling for uncertainty
└── ensemble_variance.py # Ensemble disagreement as uncertainty proxy
```

**Implementation: Conformal Prediction**
```python
# conformal.py
from typing import Tuple
import numpy as np

class ConformalPredictor:
    """
    Compute prediction intervals using conformal prediction.

    References:
        Shafer & Vovk (2008), "A Tutorial on Conformal Prediction"
    """

    def __init__(self, model, alpha: float = 0.05):
        self.model = model
        self.alpha = alpha  # Miscoverage rate (e.g., 0.05 for 95% CI)
        self.calibration_scores = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Compute nonconformity scores on calibration set."""
        preds = self.model.predict_proba(X_cal)[:, 1]
        self.calibration_scores = np.abs(preds - y_cal)  # Absolute error

    def predict_interval(
        self,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals.

        Returns:
            point_estimate: Predicted probabilities
            lower_bound: Lower bound of (1-alpha) prediction interval
            upper_bound: Upper bound of (1-alpha) prediction interval
        """
        point_estimate = self.model.predict_proba(X_test)[:, 1]

        # Compute quantile of calibration scores
        q = np.quantile(self.calibration_scores, 1 - self.alpha)

        # Prediction interval: [p - q, p + q]
        lower_bound = np.clip(point_estimate - q, 0, 1)
        upper_bound = np.clip(point_estimate + q, 0, 1)

        return point_estimate, lower_bound, upper_bound
```

**CLI integration**:
```bash
ced predict-with-uncertainty \
    --model-path results/LR_EN/final_model.pkl \
    --data-path data/test.parquet \
    --output predictions_with_ci.csv

# Output CSV:
# patient_id, risk_score, ci_lower_95, ci_upper_95, interval_width
# 12345, 0.023, 0.015, 0.034, 0.019  ← Tight interval (high confidence)
# 67890, 0.041, 0.018, 0.089, 0.071  ← Wide interval (uncertain)
```

**Use case: Uncertainty-aware triage**
```python
# Flag high-uncertainty patients for expert review
df["interval_width"] = df["ci_upper_95"] - df["ci_lower_95"]
high_uncertainty = df[df["interval_width"] > 0.05]  # Wide CI

print(f"{len(high_uncertainty)} patients flagged for expert review")
# These patients are near decision boundary → uncertain classification
```

**Testing requirements**:
- Unit test: Conformal intervals achieve nominal coverage (95% of true values in [lower, upper])
- Integration test: Run on test set, validate coverage

**Acceptance criteria**:
- Coverage: 95% ± 2% (on test set)
- Interval width correlates with ensemble variance (for ensemble models)
- Execution time: < 1 second per 1000 patients

**Effort**: 3-4 days

**References**: Shafer & Vovk (2008), Angelopoulos & Bates (2021)

---

## P4: ANALYSIS & REPORTING (1-3 days)

### 13. Model Comparison Dashboard

**Status**: Pipeline trains 4 models (LR_EN, RF, XGBoost, SVM) + stacking ensemble. Comparison is manual (inspect individual plots).

**Motivation**:
- Simplify model selection (visualize all models on one page)
- Identify complementary models for ensemble (low correlation between predictions)
- Publication-ready comparison figure

**Deliverable**: HTML interactive dashboard

**New CLI command**:
```bash
ced compare-models \
    --run-id 20260127_115115 \
    --output model_comparison_dashboard.html

# Or compare specific models
ced compare-models \
    --run-id 20260127_115115 \
    --models LR_EN,RF,XGBoost \
    --output comparison_subset.html
```

**Dashboard sections**:
1. **ROC curves overlaid**: All models on one plot
   - Color-coded by model
   - Annotate AUROC in legend
2. **PR curves overlaid**: Precision-recall for all models
3. **Calibration curves overlaid**: Predicted vs observed probabilities
4. **DCA overlaid**: Net benefit across threshold range
5. **Performance table**: AUROC, AUPRC, Brier, calibration slope (sortable)
6. **Feature importance comparison**: Venn diagram of top-20 proteins
   - Identify proteins selected by all models (consensus)
   - Identify model-specific proteins
7. **Execution time / cost comparison**: Bar chart
8. **Ensemble performance**: Stacking ensemble vs best base model

**Interactive features**:
- Hover: Show exact metric values
- Click legend: Toggle model visibility
- Download: Export plots as PNG/SVG

**Testing requirements**:
- Visual regression test: Compare HTML output to reference
- Unit test: Validate metric calculations

**Acceptance criteria**:
- Dashboard loads in < 2 seconds
- All plots render correctly in Chrome, Firefox, Safari
- Metrics match individual model reports

**Effort**: 2-3 days

**Dependencies**: `plotly>=5.0.0` (for interactive plots)

---

## Exclusions (Out of Scope)

These are intentionally NOT included:

1. **GUI/Web interface**: Pipeline is CLI-first, HPC-oriented
2. **Real-time prediction API**: Deployment is batch-oriented (clinical screening cohorts)
3. **AutoML integration**: Current Optuna-based hyperparameter optimization is sufficient
4. **Multi-task learning**: Pipeline is single-outcome (incident CeD), not multi-disease
5. **Deep learning models**: Current linear/tree models are interpretable and sufficient (AUROC ~0.89)

---

## Success Metrics

How to measure impact:

1. **Code quality**: Coverage >= 82%, all linters pass
2. **Performance**: RFE runtime < 5 hours (from 22 hours)
3. **Clinical readiness**: Deployment module validates on external cohort

---

**Last Updated**: 2026-01-31
**Maintainer**: Andres Chousal
**Status**: Active roadmap (update quarterly)
