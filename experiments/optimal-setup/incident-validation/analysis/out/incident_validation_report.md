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

| Model       | Best strategy     | Best weight | CV AUPRC        | Test AUPRC       | Test AUROC       |
|-------------|-------------------|-------------|-----------------|------------------|------------------|
| LR_EN       | incident_only     | log         | 0.215 +/- 0.061 | 0.188 [.09,.36]  | 0.908 [.83,.98]  |
| SVM L1      | incident_only     | sqrt        | 0.227 +/- 0.071 | 0.199 [.10,.38]  | 0.913 [.84,.97]  |
| SVM L2      | incident_only     | none        | 0.226 +/- 0.056 | 0.210 [.10,.40]  | 0.918 [.83,.98]  |

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
| SVM L1 | 130   | 67       | 48%      |
| SVM L2 | 130   | 130      | 0%       |

### Cross-model core features

**27 proteins** have non-zero coefficients in all three models.
Of these, **26/27 (96%)** have consistent sign (direction of effect) across all models.

Top core features by mean importance rank:

| Protein | Sign consistent | Min stability | LR coef | SVM L1 coef | SVM L2 coef |
|---------|-----------------|---------------|---------|-------------|-------------|
| TGM2         | yes             | 1.00          | -0.6368  | -0.1460      | -0.0221      |
| CKMT1A_CKMT1B | yes             | 1.00          | +0.1419  | +0.0407      | +0.0093      |
| CPA2         | yes             | 1.00          | +0.2286  | +0.0424      | +0.0041      |
| MUC2         | yes             | 1.00          | +0.2052  | +0.0404      | +0.0058      |
| CLEC4G       | yes             | 0.95          | -0.1437  | -0.0398      | -0.0060      |
| APOA1        | yes             | 0.66          | -0.1107  | -0.0264      | -0.0071      |
| NOS2         | yes             | 1.00          | +0.0699  | +0.0359      | +0.0104      |
| CD160        | yes             | 1.00          | +0.0620  | +0.0241      | +0.0082      |
| CXCL11       | yes             | 1.00          | +0.0760  | +0.0218      | +0.0045      |
| CXCL9        | yes             | 1.00          | +0.1418  | +0.0198      | +0.0015      |
| CCL11        | yes             | 1.00          | +0.0542  | +0.0210      | +0.0037      |
| MRPS16       | yes             | 0.60          | +0.0886  | +0.0164      | +0.0022      |
| SLC9A3R2     | yes             | 0.68          | +0.0888  | +0.0168      | +0.0013      |
| TNFRSF8      | yes             | 1.00          | +0.0422  | +0.0086      | +0.0066      |
| RBP2         | NO              | 1.00          | +0.1128  | +0.0149      | -0.0002      |
| JUN          | yes             | 0.53          | +0.0631  | +0.0129      | +0.0024      |
| PPP1R14D     | yes             | 1.00          | +0.0071  | +0.0138      | +0.0069      |
| ITGB7        | yes             | 1.00          | +0.0254  | +0.0128      | +0.0039      |
| KLRD1        | yes             | 0.97          | +0.0611  | +0.0103      | +0.0027      |
| RNASET2      | yes             | 0.76          | +0.0727  | +0.0095      | +0.0019      |

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
2. **Weight scheme is a secondary factor** -- test log/sqrt/none but not balanced.
3. **LR_EN provides the most parsimonious model** (28 features) with only modest
   performance cost. For interpretability-first applications, LR_EN is preferred.
4. **SVM L2 has the best point estimate** (AUPRC 0.210) but uses all 130 features
   with no sparsity. Useful as an upper-bound benchmark.
5. **SVM L1 is a middle ground** (67 features, AUPRC 0.199) if moderate sparsity
   is desired.
