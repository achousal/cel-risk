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
- Bar length shows mean |SHAP| (absolute magnitude) across all samples
- This is an unsigned ranking -- it does not distinguish risk factors from protective factors (use the beeswarm for directionality)

**Clinical interpretation**:
- Proteins with large bars are candidates for focused experimental validation
- The ranking is *patient-agnostic* (global view)

### 2. Beeswarm Plot (Feature Distribution)

Shows the distribution of SHAP values for each feature across all test samples. Each dot represents one sample.

**What to look for**:
- Horizontal spread: How much does this feature's contribution vary between samples?
- Color gradient (blue to red): Low feature value to high feature value (not SHAP sign)
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

**Sample selection** (automated, one per category):
- **TP (highest risk)**: True positive with the highest predicted probability
- **FP (highest risk)**: False positive with the highest predicted probability
- **FN (highest risk missed)**: False negative with the highest predicted probability -- clinically critical missed case
- **TN (near threshold)**: True negative closest to the decision boundary -- shows what almost triggered a positive

**Clinical interpretation**:
- Study FP and FN waterfall plots to understand misclassifications
- Use to identify feature combinations that drive risk in high-risk individuals

### 4. Scatter Plot (Feature-Risk Relationship)

Shows how an individual feature's value (x-axis) relates to its SHAP contribution (y-axis). Uses the modern `shap.plots.scatter` API (replaces the legacy `shap.dependence_plot`).

**What to look for**:
- Slope: Is the relationship monotonic (always up/down) or non-linear?
- Scatter: Do samples with the same feature value have different SHAP values? (Indicates interactions)
- Color gradient: Auto-detected interaction feature (strongest interacting feature is selected automatically)

**Clinical interpretation**:
- Steep positive slope = stronger risk factor (more weight in predictions)
- Flat trend = weak association (feature is noise)
- Non-linear relationship = complex dose-response curve
- Color variation = feature works differently in different contexts (pathway interactions)

### 5. Heatmap Plot (Per-Sample Feature Attribution Matrix)

Shows SHAP values as a color-encoded matrix with samples on the x-axis and features on the y-axis. Samples are hierarchically clustered to group similar attribution patterns together.

**What to look for**:
- Clusters of samples with similar SHAP patterns (potential disease subtypes)
- Features with consistent sign across all samples vs. variable sign
- Subgroups where different features dominate

**Clinical interpretation**:
- Distinct sample clusters may correspond to biological subtypes or risk profiles
- Features with universally positive SHAP values are consistent risk markers
- Heterogeneous patterns suggest context-dependent feature importance

---

## SHAP Output Scales

SHAP values are computed in the base model's native output scale. **This is NOT the final operating probability used clinically.**

| Model | Output Scale | SHAP Units | Interpretation |
|-------|--------------|-----------|---|
| LR_EN / LR_L1 | log-odds | ln(odds) | Log probability ratio |
| XGBoost | log-odds (raw) | ln(odds) | Log probability ratio (tree_path_dependent default) |
| Random Forest | raw or probability | Model-specific | `raw` = internal tree scale (NOT log-odds); `probability` when using interventional perturbation |
| LinSVM_cal | margin | Distance from hyperplane | Linear decision function units |

**Note**: RF output scale depends on config (`tree_model_output`). With the default `auto`, RF resolves to `probability` (interventional). XGBoost with `auto` resolves to `raw` (log-odds with tree_path_dependent). See `shap_schema.py` for details.

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

SHAP compares samples to a "background" (reference distribution). Available strategies (configured via `background_strategy` in `shap_schema.py`):

- **`random_train` (default)**: Random sample from the full training set (cases + controls). Neutral baseline.
- **`controls_only`**: Background is healthy controls only. SHAP values show deviation from healthy baseline. Interpretation: "How does this patient differ from a typical healthy person?"
- **`stratified`**: Class-proportional sampling from training set.

**Important**: Results are NOT comparable across different background strategies. If you switch strategies, all SHAP values and rankings change.

### 5. Waterfall Classification

Waterfall TP/FP/FN/TN classification uses the operating threshold from the validation set (`val_threshold`). If no validation set exists, the test set threshold is used as fallback. The threshold value depends on the configured operating point (e.g., 0.95 specificity) and is computed during the evaluation stage.

**Important**: Test-set thresholds can be optimistic. Review waterfall plots with awareness of potential overfitting when the fallback is used.

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
- Review the scatter plots for top proteins
- Check for non-linearity or unexpected interactions
- Use the heatmap to identify patient subgroups with distinct feature patterns
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

1. **Comparing SHAP values across different models without normalization**: Different output scales make raw comparison invalid. If you need cross-model aggregation, use explicit per-model normalization (enabled in `ced consensus-panel` with `--ranking-signal oof_shap --shap-explicit-normalization`) before averaging.

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
