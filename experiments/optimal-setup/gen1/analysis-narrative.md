# Optimal Setup: Analysis Narrative

## Overview

This document traces the decision chain that converges on the locked analysis configuration for prediagnostic celiac disease risk prediction from plasma proteomics.

---

## 1. Model x Panel Size Landscape (Fig 1)

**Question:** Does feature ordering strategy matter? Which model-size combinations produce the best discrimination?

**Method:** 264 configurations (4 models x 22 panel sizes x 3 feature orders) evaluated via 5-fold CV with 10 random seeds. Pooled test AUROC reported.

**Figure:** Three side-by-side heatmaps (RRA, Importance, Pathway ordering), each showing model (y) x panel size (x) with AUROC fill. Red dashed line marks the 4-protein BH-significant core.

**Results:**
- Performance patterns are consistent across all three ordering strategies. Order explains only 5% of AUROC variance (ANOVA).
- Top AUROCs cluster in the 8-15 protein range across all models.
- The 4-protein BH-significant panel (TGM2, CPA2, ITGB7, GIP) is marked as the statistically grounded floor. These 4 proteins survive BH correction over the full 2920-protein universe (all p_adj < 0.033).

**Decision:** Feature ordering is not load-bearing. Pathway order adopted as default for downstream analyses (marginal advantage, biologically interpretable).

---

## 2. Pareto Frontier: Discrimination vs Calibration (Fig 2)

**Question:** Which model achieves the best trade-off between AUROC (discrimination) and Brier score (calibration)?

**Method:** All 264 base-model configurations plotted as AUROC vs Brier. Pareto-optimal points identified (maximize AUROC, minimize Brier simultaneously).

**Figure:** Scatter plot with color = model, size = panel size, Pareto front as dashed step line. Pareto-optimal points labeled.

**Results:**
- Linear SVM (calibrated) dominates the Pareto frontier, appearing in the majority of optimal trade-off points.
- SVM achieves competitive AUROC while maintaining lower Brier scores than tree-based models (RF, XGBoost) and comparable calibration to Logistic Regression (EN).
- XGBoost shows poor calibration (high Brier) despite competitive AUROC; RF shows high variance at small panel sizes.

**Decision:** Linear SVM (calibrated) selected as the primary model. All downstream analyses use LinSVM_cal.

---

## 3. Training Strategy: Incident Validation (Fig 3)

**Question:** Which training population and class-weighting scheme optimizes prediction of incident celiac disease?

**Method:** 12 combinations (3 training strategies x 4 weight schemes) compared via 5-fold CV on a 134-protein panel selected by bootstrap stability. Primary metric: AUPRC (appropriate for class imbalance). Locked 20% test set held out.

**Figure:** Two-panel figure.
- Panel A: Dot plot of mean AUPRC by strategy-weight combination. Incident-only + log weights highlighted.
- Panel B: Top 20 feature coefficients from the winning model, colored by direction (risk/protective), annotated with bootstrap stability frequency.

**Results:**
- **Best strategy:** Incident-only training with log class weights (mean AUPRC = 0.215, mean AUROC = 0.867).
- Incident-only strategies consistently outperform incident+prevalent and prevalent-only on AUPRC.
- Balanced weighting collapses performance for incident-only (AUPRC = 0.125), suggesting moderate upweighting is needed but extreme rebalancing overshoots.
- **Locked test set:** AUROC = 0.908 [0.827, 0.978], AUPRC = 0.188 [0.090, 0.363].

**Top features (|coefficient|):**
- TGM2 (protective, -0.637) -- strongest signal, 100% bootstrap stability
- CPA2 (risk, 0.229) -- 100% stability
- MUC2 (risk, 0.205) -- 100% stability
- CLEC4G (protective, -0.144) -- 99% stability
- CXCL9 (risk, 0.142) -- 100% stability

**Decision:** Training on incident cases only with log class weights. Final elastic-net model retains 28/134 features with non-zero coefficients.

---

## 4. Optimal Setup Description

The locked configuration for all downstream analyses:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Core panel** | TGM2, CPA2, ITGB7, GIP (4 proteins) | BH-corrected significance over 2920-protein universe |
| **Operating panel** | 10 proteins (core + 6 extensions) | Saturation plateau in sweep (Fig 1) |
| **Feature order** | Pathway | Marginal advantage, biologically interpretable (Fig 1) |
| **Model** | Linear SVM (calibrated) | Pareto-optimal discrimination-calibration (Fig 2) |
| **Training population** | Incident cases + controls | Incident-only outperforms prevalent inclusion (Fig 3) |
| **Class weighting** | Log weights | Best AUPRC among weighting schemes (Fig 3) |
| **Validation** | Pooled test AUROC = 0.861 (sweep), 0.908 (incident validation) | Consistent across experiments |

### Key numbers

- Sweep operating point: AUROC 0.861, Brier 0.085 (pathway, LinSVM_cal, 10p)
- Incident validation test: AUROC 0.908 [0.827, 0.978]
- BH-significant proteins: 4/2920 (all p_adj < 0.033)
- Training strategy ANOVA: model choice explains 32% of variance, panel size 22%, order 5%

---

## Figure Index

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 | `narrative/fig01_order_heatmaps.R` | 3 heatmaps: model x panel size, one per feature order |
| Fig 2 | `narrative/fig02_pareto_all_models.R` | AUROC vs Brier Pareto frontier, all models |
| Fig 3 | `narrative/fig03_training_strategy.R` | Strategy comparison (AUPRC) + feature coefficients |

## Artifact Pointers

- Compiled sweep results: `results/compiled_results_aggregated.csv`
- RRA significance (universe-corrected): `results/experiments/rra_universe_sensitivity/rra_significance_corrected.csv`
- Incident validation summary: `results/incident_validation/summary_report.md`
- Feature coefficients: `results/incident_validation/feature_coefficients.csv`
- Strategy comparison: `results/incident_validation/strategy_comparison.csv`
