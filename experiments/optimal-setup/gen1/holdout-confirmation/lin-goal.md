# Linear Interpretability Goal

## Objective

Find the smallest protein panel where every feature has a defensible biological role, using a linear model, that is statistically non-inferior to the best configuration.

This is a constrained optimization, not a maximization:

```
min |panel|
  s.t. AUROC_linear(panel) >= AUROC_best - delta
       every protein in panel has a nameable biological mechanism
       model is linear (coefficients = interpretable effect directions)
```

The deliverable is a sparse, explainable score: each protein maps to a biological axis, coefficients give direction and magnitude, and the panel clears pre-specified performance floors.

---

## Candidate Panel: 8 Stable Proteins

From incident validation (incident_only + log weighting, 5-fold CV), 8 proteins were non-zero with consistent sign direction across all folds:

| Protein | Direction | Mean coef | Biological axis |
|---------|-----------|-----------|----------------|
| TGM2 | - | -0.617 | Mucosal integrity (celiac autoantigen) |
| CPA2 | + | 0.231 | Pancreatic / metabolic |
| MUC2 | + | 0.172 | Mucosal barrier |
| CXCL9 | + | 0.137 | IFN-gamma driven immune activation |
| CKMT1A/B | + | 0.132 | Mitochondrial energy metabolism |
| NOS2 | + | 0.079 | Inflammatory nitric oxide |
| TNFRSF8 | + | 0.051 | Immune activation (CD30) |
| ITGB7 | + | 0.029 | Gut-homing integrin |

Four biological axes covered: mucosal integrity, immune surveillance, inflammation, metabolism. No statistical bystanders -- each protein has published celiac biology.

This set is the natural intersection of:
- Statistical stability (100% bootstrap frequency, all-fold consistency)
- The sweep's 8-10p optimum (AUROC ~0.854)
- The BH-locked 4p core (tgm2, cpa2, itgb7, gip are a subset)

---

## Acceptance Gate: 3-Criterion Conjunction

All three must pass on the held-out set:

### 1. Discrimination

```
AUROC >= 0.85
```

Rationale: best model (XGBoost, 10p) achieves 0.878 on holdout. Non-inferiority margin delta = 0.03 (standard for biomarker companion diagnostics, FDA/EMA guidance). Floor = 0.878 - 0.03 ~ 0.85.

### 2. Calibration

```
Brier <= 0.10
```

Rationale: probability estimates must be reliable for risk stratification. LinSVM_cal achieves 0.077 on the current holdout config -- well within range. Brier > 0.10 indicates probability estimates are not trustworthy for clinical thresholds.

### 3. Parsimony

```
|panel| <= 10
AND every protein is biologically annotated
```

Rationale: each protein must map to a named pathway. A reviewer must be able to read the panel and understand *why* each protein is there. No features retained purely for statistical lift.

### Optional 4th: Clinical Utility

```
Net benefit > treat-all on DCA at threshold range [0.5%, 2.0%]
```

Rationale: at population screening-relevant thresholds, the model must add clinical value beyond default strategies (treat-all, treat-none).

---

## Reference: Current Holdout Performance

Pooled across 10 seeds, current locked config (10p panel):

| Model | AUROC | PR-AUC | Brier | Sens@95Spec |
|-------|-------|--------|-------|-------------|
| XGBoost | 0.878 | 0.717 | 0.098 | 0.632 |
| RF | 0.877 | 0.715 | 0.079 | 0.636 |
| ENSEMBLE | 0.874 | 0.695 | 0.077 | 0.644 |
| LinSVM_cal | 0.873 | 0.721 | 0.077 | 0.664 |
| LR_EN | 0.871 | 0.720 | 0.076 | 0.652 |

LinSVM_cal is already within delta=0.005 of the best AUROC, with the best Brier and best Sens@95Spec. The linear model is not conceding meaningful discrimination.

True held-out ensemble (n=13,123, 44 incident cases): mean AUROC 0.894, range [0.874, 0.904].

---

## Decision Logic

```
IF 8p LinSVM_cal clears all 3 gates on holdout:
    ACCEPT 8p as the final interpretable panel

ELSE IF 8p fails discrimination (AUROC < 0.85):
    Expand to 10p (add next 2 most stable from sweep)
    Re-evaluate against same gates

ELSE IF 8p fails calibration (Brier > 0.10):
    Re-calibrate (Platt/isotonic) and re-evaluate
    If still fails: flag calibration as a separate problem
```

Fallback to 10p remains interpretable -- the sweep showed proteins 9-10 are still biologically annotatable (from the RRA/pathway ranking).

---

## Ensemble Exclusion Rationale

Ensemble models (stacking LinSVM_cal + XGBoost + RF + LR_EN) were evaluated and excluded from the interpretability goal.

### Empirical: ensemble does not earn its complexity

Holdout performance (pooled, 10 seeds, current 10p config):

| | AUROC | PR-AUC | Brier |
|---|---|---|---|
| LinSVM_cal | 0.873 | 0.721 | 0.077 |
| ENSEMBLE | 0.874 | 0.695 | 0.077 |
| Gap | +0.001 | -0.026 | 0.000 |

The ensemble gains 0.001 AUROC but loses 0.026 PR-AUC -- strictly worse on the metric that matters most for rare-event prediction (44 incident cases in 13,123 samples). The 50-trial Optuna budget produces weak out-of-fold predictions for the meta-learner, limiting ensemble upside (sweep finding: ensemble underperforms base models at 50 trials).

### Structural: interpretability cost is not justified

Linear models provide **intrinsic** interpretability: coefficient sign, magnitude, and protein name tell the full story. A reviewer reads "TGM2 coef = -0.62" and understands it directly.

Ensemble models require **post-hoc** interpretability (SHAP): attributions are approximate, vary per sample, and are harder to defend in a paper or regulatory submission.

### Quantitative inclusion threshold

For ensemble to re-enter consideration:

```
AUROC_ensemble - AUROC_linear > delta_complexity = 0.02
```

This is the minimum improvement that justifies losing intrinsic interpretability -- a meaningful, publishable gain, not noise. At the current gap of +0.001, the ensemble is ~20x below threshold.

### When to revisit

- Production trial budget (300+ Optuna trials) where the meta-learner gets better OOF inputs
- Evidence of genuine interaction effects that linear models miss (testable via SHAP interaction values at the 8p panel)
- Use case shift from "interpretable risk score" to "deployed screening tool" where black-box is acceptable

Until one of these conditions is met, ensemble adds complexity without compensating gain.

---

## Next Action

Run the 8-stable-protein panel through the holdout pipeline:
- Model: LinSVM_cal
- Training: incident_only + log weighting
- Panel: tgm2, cpa2, muc2, cxcl9, ckmt1a_ckmt1b, nos2, tnfrsf8, itgb7
- Evaluate against the 3-criterion gate
