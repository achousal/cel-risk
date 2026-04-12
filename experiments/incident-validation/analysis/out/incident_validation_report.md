# Incident Validation: Three-Model Comparison

## Pipeline
```
UK Biobank proteomics (N=44,174; 2,920 Olink proteins)
  |
  v
Locked 80/20 dev/test split (seed=42, stratified by sex)
  |-- Dev: 119 incident + 150 prevalent + 35,100 controls
  |-- Test: 29 incident + 8,776 controls (LOCKED, touched once)
  |
  v
Bootstrap stability feature selection (Wald, 100 resamples)
  |-- LR_EN:  top 200/resample, threshold >= 70% --> 134 proteins
  |-- SVM L1: top 150/resample, threshold >= 50% --> 130 proteins
  |-- SVM L2: top 150/resample, threshold >= 50% --> 130 proteins
  |
  v
Correlation pruning (|r| > 0.85, keep more stable)
  |
  v
3 x 4 factorial: strategy x weight_scheme
  |-- Strategies: incident_only, incident_prevalent, prevalent_only
  |-- Weights:    none, log, sqrt, balanced
  |-- 5-fold outer CV, 3-fold inner Optuna (AUPRC objective)
  |
  v
Select best (strategy, weight) by mean CV AUPRC
  |
  v
Refit on full dev set with winning config
  |
  v
Evaluate on locked test set (bootstrap CIs)
```

## Model Configurations

| Parameter          | LR_EN                 | SVM L1                    | SVM L2                    |
|--------------------|-----------------------|---------------------------|---------------------------|
| Model              | ElasticNet LogReg     | LinearSVC (L1) + Sigmoid  | LinearSVC (L2) + Sigmoid  |
| Tuned HPs          | C, l1_ratio           | C                         | C                         |
| Optuna trials      | 50                    | 50                        | 50                        |
| Bootstrap resamples| 100                   | 100                       | 100                       |
| Stability top_k    | 200                   | 150                       | 150                       |
| Stability threshold| 70%                   | 50%                       | 50%                       |
| Panel size         | 134                   | 130                       | 130                       |

## Strategy Comparison

All three models select **incident_only** as the best training strategy.

| Model       | Best strategy     | Best weight | CV AUPRC        | Test AUPRC  | Test AUROC  |
|-------------|-------------------|-------------|-----------------|-------------|-------------|
| LR_EN       | incident_only     | log         | 0.215 +/- 0.061 | 0.188       | 0.908       |
| SVM L1      | incident_only     | log         | 0.235 +/- 0.067 | 0.203       | 0.913       |
| SVM L2      | incident_only     | none        | 0.223 +/- 0.053 | 0.217       | 0.918       |

**Key findings:**

1. **incident_only dominates** across all three models. Prevalent cases add noise
   rather than signal -- likely because prevalent CeD reflects post-diagnosis
   biology (dietary changes, treatment effects) that diverges from pre-diagnostic
   proteomic signatures.

2. **Weight scheme matters less than strategy.** Within incident_only, the top
   weight schemes (log, sqrt, none) are within 1 SD of each other. The SVMs
   prefer lighter or no weighting; LR prefers log. Balanced weighting is
   consistently worst (over-corrects the imbalance).

3. **Test set performance is tightly grouped.** AUPRC ranges 0.188-0.210, AUROC
   0.908-0.918. CIs overlap substantially. No model clearly dominates on the
   locked test set -- the 29 incident test cases limit statistical power.

## Feature Stability

### Model sparsity

| Model  | Panel | Non-zero | Sparsity |
|--------|-------|----------|----------|
| LR_EN  | 134   | 28       | 79%      |
| SVM L1 | 134   | 96       | 28%      |
| SVM L2 | 134   | 134      | 0%      |

### Cross-model core features

**28 proteins** have non-zero coefficients in all three models.
Of these, **26/28 (93%)** have consistent sign (direction of effect) across all models.

Top core features by mean importance rank:

| Protein | Sign consistent | Min stability | LR coef | SVM L1 coef | SVM L2 coef |
|---------|-----------------|---------------|---------|-------------|-------------|
| TGM2         | yes             | 1.00          | -0.6368  | -0.1099      | -0.0191      |
| CKMT1A_CKMT1B | yes             | 1.00          | +0.1419  | +0.0356      | +0.0082      |
| MUC2         | yes             | 1.00          | +0.2052  | +0.0339      | +0.0049      |
| CLEC4G       | yes             | 0.99          | -0.1437  | -0.0320      | -0.0052      |
| APOA1        | yes             | 0.80          | -0.1107  | -0.0291      | -0.0073      |
| NOS2         | yes             | 1.00          | +0.0699  | +0.0308      | +0.0095      |
| CPA2         | yes             | 1.00          | +0.2286  | +0.0263      | +0.0036      |
| CD160        | yes             | 1.00          | +0.0620  | +0.0221      | +0.0072      |
| CXCL11       | yes             | 1.00          | +0.0760  | +0.0170      | +0.0043      |
| TNFRSF8      | yes             | 1.00          | +0.0422  | +0.0146      | +0.0063      |
| CXCL9        | yes             | 1.00          | +0.1418  | +0.0109      | +0.0019      |
| CCL11        | yes             | 1.00          | +0.0542  | +0.0183      | +0.0031      |
| MRPS16       | yes             | 0.78          | +0.0886  | +0.0130      | +0.0019      |
| PPP1R14D     | yes             | 1.00          | +0.0071  | +0.0151      | +0.0066      |
| JUN          | yes             | 0.71          | +0.0631  | +0.0112      | +0.0020      |
| SLC9A3R2     | yes             | 0.81          | +0.0888  | +0.0103      | +0.0009      |
| KLRD1        | yes             | 1.00          | +0.0611  | +0.0101      | +0.0018      |
| ITGB7        | yes             | 1.00          | +0.0254  | +0.0079      | +0.0035      |
| TIGIT        | yes             | 1.00          | +0.0232  | +0.0091      | +0.0029      |
| RNASET2      | yes             | 0.88          | +0.0727  | +0.0062      | +0.0016      |

### Interpretation

The core feature set is dominated by:

- **TGM2** (transglutaminase 2): strongest signal in all models, negative
  coefficient. TGM2 is the autoantigen in celiac disease -- lower circulating
  levels pre-diagnosis may reflect tissue sequestration or immune complex
  formation.

- **Gut-epithelial markers** (MUC2, RBP2, FABP1, CPA2): intestinal integrity
  and absorptive function proteins. Elevated pre-diagnosis suggests subclinical
  mucosal changes.

- **Immune/inflammatory** (CXCL9, CXCL11, CCL11, NOS2, CD160): IFN-gamma
  responsive chemokines and NK/T cell markers. Consistent with the Th1-driven
  immune response in CeD pathogenesis.

- **CLEC4G, APOA1** (negative): hepatic/metabolic markers whose decrease may
  reflect systemic inflammation or liver-gut axis perturbation.

## Recommendation

For the factorial (V0 gate), the incident validation confirms:

1. **Lock incident_only** as the training strategy across all models.
2. **Weight scheme is a secondary factor** -- test log/none but not balanced.
3. **LR_EN is most parsimonious** (28 features, test AUPRC 0.188) but see
   calibration section -- LR_EN is miscalibrated without post-hoc adjustment.
4. **SVM L2 has the best point estimate** (test AUPRC 0.217) and best
   calibration -- recommended as primary model.
5. **SVM L1 is a middle ground** (96 features, test AUPRC 0.203) if moderate
   sparsity is desired.

See companion analyses for calibration metrics (calibration_metrics.csv),
decision curve analysis (fig7_dca), SHAP-based feature importance (fig9-11),
and saturation curve (fig8_saturation).

## Calibration Assessment

| Metric                  | LR_EN   | SVM L1  | SVM L2  | Ideal |
|-------------------------|---------|---------|---------|-------|
| ECE                     | 0.0115  | 0.0012  | 0.0009  | 0     |
| ICI (LOESS)             | 0.0112  | 0.0016  | 0.0010  | 0     |
| Brier score             | 0.0040  | 0.0030  | 0.0030  | 0     |
| Brier reliability       | 0.0011  | 0.0004  | 0.0003  | 0     |
| Brier resolution        | 0.0004  | 0.0006  | 0.0006  | high  |
| Calibration intercept   | -1.892  | 0.263   | 0.317   | 0     |
| Calibration slope       | 0.984   | 1.077   | 1.086   | 1     |
| Spiegelhalter z         | -7.840  | 0.124   | 0.018   | ~0    |
| Spiegelhalter p         | 4.52e-15 | 0.901   | 0.986   | >.05  |

**Interpretation:** LR_EN shows systematic overestimation (large negative
intercept, Spiegelhalter p < 0.001). Both SVMs are well-calibrated
(Spiegelhalter p > 0.9, near-zero ICI). The CalibratedClassifierCV sigmoid
wrapper on the SVMs is doing meaningful work.

## Feature Overlap (UpSet decomposition)

| Intersection          | Count |
|-----------------------|-------|
| All 3 models (core)   | 28    |
| SVM L1 + SVM L2 only  | 68    |
| SVM L2 only           | 38    |

The regularization hierarchy is clean: LR_EN active features are a strict
subset of SVM L1's, which are a strict subset of SVM L2's. Elastic net
selects the most stable core.

## Saturation Curve (LR_EN, incident_only + log)

| Panel size | CV AUPRC        | Test AUPRC            |
|-----------:|-----------------|-----------------------|
|          5 | 0.125 +/- 0.046 | 0.135 [0.039, 0.294] |
|          8 | 0.139 +/- 0.060 | 0.131 [0.040, 0.274] |
|         10 | 0.174 +/- 0.055 | 0.153 [0.054, 0.304] |
|         15 | 0.170 +/- 0.058 | 0.144 [0.059, 0.305] |
|         20 | 0.180 +/- 0.059 | 0.151 [0.065, 0.317] |
|         25 | 0.210 +/- 0.061 | 0.170 [0.079, 0.342] |
|         28 | 0.201 +/- 0.058 | 0.173 [0.082, 0.342] |
|         40 | 0.200 +/- 0.062 | 0.178 [0.089, 0.346] |
|         60 | 0.207 +/- 0.055 | 0.178 [0.090, 0.345] |
|         80 | 0.208 +/- 0.058 | 0.183 [0.090, 0.357] |
|        100 | 0.209 +/- 0.058 | 0.181 [0.089, 0.351] |
|        134 | 0.210 +/- 0.061 | 0.176 [0.086, 0.344] |

**Knee at N~25-28.** CV AUPRC plateaus near the LR_EN non-zero count.
Test AUPRC is flat from N=25 onwards -- the 134-feature panel offers no
test-set advantage over ~25 features. Elastic net regularization arrived
at a near-optimal panel size without needing this curve as a guide.

## Figure Inventory

| Figure                        | Description                                           |
|-------------------------------|-------------------------------------------------------|
| fig1_strategy_heatmap         | 3x4 AUPRC heatmap: strategy x weight, faceted         |
| fig2_fold_auprc               | CV AUPRC per fold, best config per model              |
| fig3_test_roc_pr              | ROC and PR curves on locked test set                  |
| fig4_feature_rank_comparison  | Top 30 features by cross-model importance rank       |
| fig5_core_features            | Core proteins: coefficient direction and magnitude   |
| fig6_calibration              | Reliability diagrams with LOESS smooth                |
| fig6_bootstrap_forest         | Bootstrap 95% CI forest plot (AUPRC + AUROC)         |
| fig7_dca                      | Decision curve analysis (net benefit vs threshold)   |
| fig7_feature_upset            | UpSet plot of feature overlap across models           |
| fig8_saturation               | Performance vs panel size (saturation curve)          |
| fig9_shap_beeswarm            | SHAP beeswarm plots (top 20 features, 3 models)      |
| fig10_shap_bar                | Mean |SHAP| bar chart (top 15 features, 3 models)    |
| fig11_shap_dependence         | SHAP dependence for top 5 core features (LR_EN)      |
