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


## P1: HIGH-IMPACT FEATURES

### 5. Model Explainability Suite (SHAP)

**Status**: Clinical ML requires interpretability. Current pipeline has metrics but no local/global explanations.

**Motivation**:
- Clinicians need to understand **why** a patient is high-risk (which proteins drive prediction)
- Regulatory requirements for "explainable AI" in healthcare
- Scientific discovery: identify novel biomarker interactions

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

### 11. Temporal Validation Analysis

**Status**: UK Biobank data is cross-sectional (random train/val/test split). No time-based validation.

**Motivation**:
- Assess model stability over time (biomarker distributions may drift)
- Detect secular trends (e.g., changing CeD incidence, changing assay platforms)
- Regulatory requirement: demonstrate prospective validity

**Deliverable**: Time-aware splitting + drift detection

**Testing requirements**:
- Unit test: Temporal splits have no time overlap
- Integration test: Train on early years, test on later years

**Acceptance criteria**:
- Splits are temporally non-overlapping
- Drift detection flags proteins with KS p < 0.01
- Report quantifies performance degradation over time

**Effort**: 2-3 days

---

## P4: ANALYSIS & REPORTING (1-3 days)

### 13. Model Comparison Dashboard

**Status**: Pipeline trains 4 models (LR_EN, RF, XGBoost, SVM) + stacking ensemble. Comparison is manual (inspect individual plots).

**Motivation**:
- Simplify model selection (visualize all models on one page)
- Identify complementary models for ensemble (low correlation between predictions)
- Publication-ready comparison figure

**Deliverable**: HTML interactive dashboard

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

## Exclusions (Out of Scope)

These are intentionally NOT included:

1. **GUI/Web interface**: Pipeline is CLI-first, HPC-oriented
4. **Multi-task learning**: Pipeline is single-outcome (incident CeD), not multi-disease
5. **Deep learning models**: Current linear/tree models are interpretable and sufficient (AUROC ~0.89)

---

## Success Metrics

---

**Last Updated**: 2026-01-31
**Maintainer**: Andres Chousal
**Status**: Active roadmap (update quarterly)
