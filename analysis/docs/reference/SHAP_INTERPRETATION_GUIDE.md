# SHAP Interpretation Guide

## What are SHAP Values?

SHAP (SHapley Additive exPlanations) values provide a principled way to understand how each feature (protein biomarker) contributes to a model's prediction for an individual sample.

**Key concept**: SHAP values represent the *deviation from the model's average prediction* caused by each feature. A positive SHAP value means that feature is pushing the prediction higher (toward disease risk); a negative value means it's pushing the prediction lower.

### Before You Start

SHAP values explain the *base model's native output scale*, NOT the final calibrated probability used for clinical decisions. This is intentional and important - it helps you understand the model's raw signal independent of calibration adjustments.

---

## Reading SHAP Plots

### 1. Bar Plot (Feature Importance)

Shows the average magnitude (|SHAP|) of each feature's contribution across all test samples, ranked by importance.

**What to look for**:
- Features at the top have the strongest overall influence on predictions
- Red bars indicate positive contributions (risk factors)
- Blue bars indicate negative contributions (protective factors)
- The bar length (SHAP magnitude) indicates consistency and strength

**Clinical interpretation**:
- Proteins with large bars are candidates for focused experimental validation
- The ranking is *patient-agnostic* (global view)

### 2. Beeswarm Plot (Feature Distribution)

Shows the distribution of SHAP values for each feature across all test samples. Each dot represents one sample.

**What to look for**:
- Horizontal spread: How much does this feature's contribution vary between samples?
- Color gradient (blue → red): From negative contribution → positive contribution
- Clustering: Do samples cluster into distinct groups?

**Clinical interpretation**:
- High horizontal spread = high variability between patients
- Left-clustered dots = most samples benefit from this feature (protective)
- Right-clustered dots = most samples have disease-promoting values for this feature

### 3. Waterfall Plot (Individual Sample)

Explains a single patient's prediction step-by-step.

**What to look for**:
- Base value (gray bar on left): Model's average prediction
- Feature bars: How each feature moves the prediction up or down
- Final prediction (gray bar on right): The model's final score

**Reading the path**:
1. Start from the base value (average prediction)
2. Features pushed in order of descending impact
3. Final bar shows where you end up

**Categories**:
- **TP (True Positive)**: High-risk prediction, patient later diagnosed with CeD
- **FP (False Positive)**: High-risk prediction, patient remained disease-free
- **FN (False Negative)**: Low-risk prediction, patient later diagnosed with CeD
- **TN (True Negative)**: Low-risk prediction, patient remained disease-free

**Clinical interpretation**:
- Study FP and FN waterfall plots to understand misclassifications
- Use to identify feature combinations that drive risk in high-risk individuals

### 4. Dependence Plot (Feature-Risk Relationship)

Shows how an individual feature's value (x-axis) relates to its SHAP contribution (y-axis).

**What to look for**:
- Slope: Is the relationship monotonic (always up/down) or non-linear?
- Scatter: Do samples with the same feature value have different SHAP values? (Indicates interactions)
- Color gradient: Interaction with another feature

**Clinical interpretation**:
- Steep positive slope = stronger risk factor (more weight in predictions)
- Flat trend = weak association (feature is noise)
- Non-linear relationship = complex dose-response curve
- Color variation = feature works differently in different contexts (pathway interactions)

---

## SHAP Output Scales

SHAP values are computed in the base model's native output scale. **This is NOT the final operating probability used clinically.**

| Model | Output Scale | SHAP Units | Interpretation |
|-------|--------------|-----------|---|
| Logistic Regression | log-odds | $\ln(\text{odds})$ | Log probability ratio |
| XGBoost | log-odds | $\ln(\text{odds})$ | Log probability ratio |
| Random Forest | raw (log-odds equivalent) | Model-specific | Feature influence magnitude |
| SVM | margin | Distance from decision boundary | Geometric units |

**Important**: SHAP log-odds values can be transformed back to probability, but the clinical decision uses `y_prob_adjusted` (prevalence-adjusted on validation set). The SHAP explanation is *independent* of this adjustment - it explains the raw signal.

### Example Interpretation (Log-Odds Scale)
- SHAP value = +0.5 (log-odds) → multiplies odds by ~1.65x
- SHAP value = -0.5 (log-odds) → multiplies odds by ~0.61x

---

## Key Caveats

### 1. SHAP ≠ Causality

SHAP values show *association* and *feature importance*, not cause-and-effect. A high protein level may be a marker of disease, not a cause.

**Example**: Elevated anti-tTG antibodies have high SHAP values because they're markers of CeD, but SHAP doesn't explain *why* they're elevated.

### 2. Pre-Calibration Values

SHAP values explain the model *before* calibration. They don't account for the prevalence adjustment applied for clinical use.

**Why**: Pre-calibration SHAP values are more stable and interpretable as pure feature importance.

### 3. Correlated Features in Random Forest

Random Forest uses "interventional" SHAP (perturbing features independently), which assumes feature independence. In high-correlation proteomics data (e.g., pathway proteins), this causes:
- **Misattribution**: Effects spread across correlated features unpredictably
- **Noisier rankings**: Similar proteins get different importance scores
- **Interaction effects**: Non-obvious feature combinations

**How to identify**: Look for similar proteins with very different SHAP values despite similar distributions. This suggests correlation-induced misattribution.

**Recommendation**: Use SHAP for *hypothesis generation*, not causal inference. Validate findings with experimental biology.

### 4. Background Strategy

SHAP compares samples to a "background" (reference distribution). Our strategy:

- **Default (`controls_only`)**: Background is healthy controls. SHAP values show deviation from healthy baseline.
- **Interpretation**: "How does this patient differ from a typical healthy person?"
- **Not comparable to**: Different background strategies (e.g., all patients) - results will differ significantly.

### 5. Waterfall Classification

Waterfall categories use the operating threshold (0.95 specificity on validation set). If no validation set exists, the test set threshold is used.

**Important**: Test-set thresholds can be optimistic. Review waterfall plots with awareness of potential overfitting on the test set.

---

## Workflow: Using SHAP for Clinical Decision Support

### Step 1: Global Understanding
- Review **bar plot** to identify top 5-10 influential proteins
- Read **beeswarm plot** to understand distribution and variability
- Note proteins with unexpected directionality

### Step 2: High-Risk Individual Review
- Select a high-risk prediction (FP or borderline)
- View **waterfall plot** for that patient
- Identify 3-5 dominant features
- Compare to typical waterfall patterns

### Step 3: Feature Validation
- Review the dependence plots for top proteins
- Check for non-linearity or unexpected interactions
- Cross-reference with biological pathways

### Step 4: Misclassification Analysis
- Find FN (missed diagnoses) and FP (false alarms) waterfall plots
- Identify common feature patterns
- Hypothesize: Are these known markers? Are there data quality issues?

### Step 5: Threshold Decisions
- Review ROC/PR curves alongside waterfall examples
- Use TP/FP examples to calibrate your confidence in the threshold
- Consider FN waterfall patterns when setting thresholds for screening

---

## Common Pitfalls

1. **Comparing SHAP values across different models**: Different output scales make direct comparison invalid. Use feature *rankings*, not absolute values.

2. **Over-interpreting individual SHAP values**: One patient's SHAP values don't prove anything - patterns across many patients matter.

3. **Treating SHAP importance as pathway enrichment**: SHAP ranks features by prediction influence, not biological significance. Similar-importance proteins may be in different pathways.

4. **Assuming zero SHAP = no effect**: Zero just means "average contribution." The feature may still be important for specific subgroups.

5. **Forgetting the calibration step**: Clinical decisions use calibrated probabilities, not raw SHAP values. Always check the operating curve (ROC) alongside SHAP explanations.

---

## Questions to Ask When Reviewing SHAP

- [ ] Are the top features biologically plausible for CeD?
- [ ] Do waterfall patterns make sense for TP vs. FN?
- [ ] Are there unexpected correlations between high-SHAP features?
- [ ] Does the FP waterfall suggest data quality issues (e.g., missing values, outliers)?
- [ ] Are any features independent risk factors in literature? (Validate SHAP findings)
- [ ] How robust are rankings to different random seeds / background samples?

---

## References

- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Aas, K., Jøsund, O. H., & Lundberg, S. M. (2021). Explaining individual predictions when features are dependent. arXiv:2104.14560.
- Merz, B., Blöschl, G., & Parajka, J. (2020). Flood frequency regionalization. In *Handbook of Flood Risk Assessment and Management* (pp. 109-138). Routledge.
- Tonekaboni, S., Joshi, S., McCradden, M. D., & Goldenberg, A. (2019). What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use. In *Machine Learning for Healthcare Conference* (pp. 359-380). PMLR.

---

## Troubleshooting

**Q: Why do two similar proteins have very different SHAP values?**
A: (1) Correlation with other features, (2) different distributions in training data, (3) Random Forest feature interactions, (4) data quality differences.

**Q: Why is my high-risk patient FP not obviously sick?**
A: The model detected a biomarker signature consistent with CeD, but the patient may be (1) truly disease-free despite markers, (2) in pre-clinical phase, (3) have assay error.

**Q: Can I trust SHAP for this patient?**
A: Check (1) Is the patient similar to training data? (outliers are less reliable), (2) Are predictions consistent across folds? (model confidence), (3) Are top features known CeD markers?

---

**For further technical details**, see [ADR-007 (OOF Stacking)](../adr/ADR-007-oof-stacking-ensemble.md) and [ADR-008 (Calibration)](../adr/ADR-008-oof-posthoc-calibration.md).
