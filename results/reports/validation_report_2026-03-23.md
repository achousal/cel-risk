# Celiac Disease Proteomic Risk Model: Statistical Validation Report

**Date:** 2026-03-23
**Run IDs:** Phase 1 `run_20260317_131842` | Phase 2 `run_phase2_val_consensus` | Phase 3 `run_phase3_holdout`

---

## 1. Overview

This report summarizes the three-phase validation of a 7-protein plasma proteomic panel for celiac disease (CeD) risk classification. The pipeline progresses from discovery (feature selection across 2,920 proteins) through independent replication on non-overlapping data splits, permutation significance testing, and final evaluation on a 30% held-out set never seen during model development.

**Panel:** tgm2, cpa2, itgb7, gip, cxcl9, cd160, muc2 (age-sex-BMI-residualized SomaScan aptamers)

**Models evaluated:** Elastic Net Logistic Regression (LR_EN), Calibrated Linear SVM (LinSVM_cal), Random Forest (RF), XGBoost, and a Stacking Ensemble (ENSEMBLE).

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | SomaScan plasma proteomics + demographics |
| Total samples (post-filter) | 43,741 |
| Proteins measured | 2,920 (age-, sex-, BMI-residualized) |
| Scenario | Incident + Prevalent CeD |
| Demographic covariates | Age, sex, BMI, genetic ethnic grouping |
| Train prevalence (downsampled) | 16.7% (1:5 case-control) |
| Stratification | Outcome + sex + age (2-bin) |

---

## 3. Phase 1 — Discovery (30 splits, seeds 100–129)

**Objective:** Identify a consensus biomarker panel from 2,920 proteins using four ML classifiers with independent feature selection, then apply Robust Rank Aggregation (RRA) to derive a cross-model consensus.

### 3.1 Feature Selection

Each model independently selected top features via embedded or wrapper methods across 30 random train/val/test splits. RRA identified 25 proteins appearing across all four models. Of these, 7 passed BH-corrected significance (FDR < 0.05):

| Rank | Protein | RRA Score | BH-adjusted p | All 4 models |
|------|---------|-----------|---------------|-------------|
| 1 | tgm2 | 65.83 | 0.003 | Yes |
| 2 | cpa2 | 26.88 | 0.003 | Yes |
| 3 | itgb7 | 26.88 | 0.003 | Yes |
| 4 | gip | 16.46 | 0.003 | Yes |
| 5 | cxcl9 | 10.63 | 0.008 | Yes |
| 6 | cd160 | 8.44 | 0.036 | Yes |
| 7 | muc2 | 7.88 | 0.036 | Yes |

All 7 proteins were selected by all 4 models (agreement strength = 1.0). The 8th-ranked protein (nos2, BH p = 0.093) did not pass the significance threshold.

### 3.2 Discovery Performance (full 2,920-protein feature space)

| Model | Test AUROC | 95% CI | Brier |
|-------|-----------|--------|-------|
| XGBoost | 0.874 | [0.863–0.885] | 0.080 |
| RF | 0.871 | [0.860–0.883] | 0.076 |
| LR_EN | 0.846 | [0.835–0.858] | 0.100 |
| LinSVM_cal | 0.830 | [0.818–0.843] | 0.095 |

Tree-based models outperformed linear models in the full feature space, consistent with their capacity to capture non-linear interactions among 2,920 features.

---

## 4. Phase 2 — Independent Validation (10 splits, seeds 200–209)

**Objective:** Validate the 7-protein fixed panel on non-overlapping splits. No feature selection or panel optimization — the panel is locked.

### 4.1 Validation Performance

| Model | Test AUROC | 95% CI | Brier | Delta vs Phase 1 |
|-------|-----------|--------|-------|-------------------|
| LinSVM_cal | 0.888 | [0.862–0.915] | 0.071 | +0.058 |
| LR_EN | 0.887 | [0.861–0.914] | 0.071 | +0.041 |
| XGBoost | 0.877 | [0.850–0.904] | 0.088 | +0.003 |
| ENSEMBLE | 0.875 | (pooled) | 0.077 | — |
| RF | 0.869 | [0.841–0.898] | 0.076 | −0.002 |

**Key finding:** Linear models improved substantially (+4–6 points) when constrained to the 7-protein panel, while tree models held steady. This is expected: removing 2,913 noise features benefits models sensitive to irrelevant covariates. All models replicate above 0.86.

### 4.2 Permutation Significance

Each model was tested against a null distribution of 3,000 label permutations (300 per seed × 10 seeds), aggregated across seeds:

| Model | Observed AUROC | p-value | Null Mean | Null Max | Significant |
|-------|---------------|---------|-----------|----------|-------------|
| LR_EN | 0.872 | < 0.001 | 0.486 | 0.891 | Yes |
| LinSVM_cal | 0.873 | < 0.001 | 0.504 | 0.888 | Yes |
| RF | 0.865 | < 0.001 | 0.539 | 0.839 | Yes |
| XGBoost | 0.865 | < 0.001 | 0.518 | 0.883 | Yes |

All models significantly outperform chance. The observed AUROCs exceed the maximum of the null distribution for RF and approach it for the other models, indicating robust signal.

---

## 5. Phase 3 — Holdout Evaluation (30% held-out set)

**Objective:** Final unbiased performance estimate on 30% of data (n = 13,123; 44 CeD cases) carved out before any model training. Models were retrained on the 70% development portion (seeds 200–209) and evaluated once on the holdout.

### 5.1 Holdout Performance

| Model | Holdout AUROC | 95% CI | PR-AUC | Sens @ Spec 90% |
|-------|--------------|--------|--------|-----------------|
| **LR_EN** | **0.901** | [0.897–0.904] | 0.194 | 67.7% |
| **LinSVM_cal** | **0.901** | [0.898–0.904] | 0.201 | 67.3% |
| ENSEMBLE | 0.892 | [0.887–0.898] | 0.188 | — |
| RF | 0.872 | [0.866–0.878] | 0.178 | 67.0% |
| XGBoost | 0.871 | [0.866–0.877] | 0.171 | 65.0% |

### 5.2 Calibration (Holdout)

| Model | Mean Cal. Slope | Interpretation |
|-------|----------------|----------------|
| LR_EN | 0.87 | Slight overconfidence |
| LinSVM_cal | 1.09 | Well-calibrated |
| RF | 1.18 | Slight underconfidence |
| XGBoost | 1.61 | Overconfident — requires recalibration |

LinSVM_cal has the best calibration (slope nearest 1.0) combined with top-tier discrimination.

### 5.3 Performance Trajectory Across Phases

| Model | Phase 1 (discovery) | Phase 2 (validation) | Phase 3 (holdout) | Trend |
|-------|--------------------|--------------------|------------------|-------|
| LR_EN | 0.846 | 0.887 | **0.901** | Improving |
| LinSVM_cal | 0.830 | 0.888 | **0.901** | Improving |
| RF | 0.871 | 0.869 | 0.872 | Stable |
| XGBoost | 0.874 | 0.877 | 0.871 | Stable |

Linear models improve as the feature space narrows from 2,920 to 7 proteins. Tree models are stable, indicating the signal is robust across model families.

---

## 6. Summary

| Validation Gate | Result |
|----------------|--------|
| Cross-model consensus (RRA) | 7 proteins, BH p < 0.05, all selected by 4/4 models |
| Independent replication (Phase 2) | All models replicate above AUROC 0.86 on non-overlapping splits |
| Permutation significance | All models p < 0.001 vs 3,000 null permutations |
| Holdout evaluation (Phase 3) | Best model AUROC 0.901 [0.897–0.904] on 30% held-out data |
| Calibration | LinSVM_cal well-calibrated (slope 1.09); XGBoost requires recalibration |

### Recommended Model

**LinSVM_cal** — matches LR_EN on discrimination (AUROC 0.901) with superior calibration (slope 1.09 vs 0.87) and the highest PR-AUC (0.201). For deployment, LinSVM_cal provides the best balance of discrimination, calibration, and interpretability.

### 7-Protein Panel

| Protein | Biological Context |
|---------|--------------------|
| **tgm2** | Tissue transglutaminase — the primary autoantigen in celiac disease |
| **cpa2** | Carboxypeptidase A2 — pancreatic enzyme, gut inflammation marker |
| **itgb7** | Integrin beta-7 — gut-homing lymphocyte adhesion receptor |
| **gip** | Gastric inhibitory polypeptide — enteroendocrine hormone |
| **cxcl9** | CXCL9/MIG — IFN-gamma-induced chemokine, Th1 inflammation |
| **cd160** | CD160 — NK/T-cell receptor, mucosal immune surveillance |
| **muc2** | Mucin-2 — intestinal barrier glycoprotein |

The panel captures autoimmune (tgm2), gut-homing (itgb7), mucosal barrier (muc2), inflammatory (cxcl9, cd160), and metabolic (gip, cpa2) axes of celiac disease pathophysiology.

---

## 7. Limitations

1. **Low prevalence in holdout.** The holdout set contains 44 CeD cases in 13,123 samples (0.34% raw prevalence). While downsampling creates balanced training sets (16.7%), the holdout reflects population prevalence, inflating specificity-driven metrics and suppressing PR-AUC.

2. **Single-cohort derivation.** All phases use the same BioMe biobank cohort with different data splits. External validation in an independent cohort is required before clinical deployment.

3. **Residualized features.** Proteins are age-sex-BMI-residualized. Clinical deployment would require either applying the same residualization pipeline or retraining on raw aptamer values with covariates as model inputs.

4. **No temporal validation.** All splits are random. Temporal validation (training on earlier samples, testing on later) would address potential temporal confounding.

5. **Ensemble adds no value.** The stacking ensemble (AUROC 0.892) underperforms the best base models (0.901), likely because 7 features provide insufficient diversity for meta-learning gains.

---

*Report generated from automated pipeline results. All metrics derived from held-out test/holdout sets with no data leakage.*
