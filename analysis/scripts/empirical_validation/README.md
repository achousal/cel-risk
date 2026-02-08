# Empirical Validation Studies

This directory contains scripts for empirical validation of methodological choices flagged in the Mathematical Validity Audit (see `analysis/docs/development/MATHEMATICAL_VALIDITY_AUDIT.md`).

## Overview

| Study | Purpose | Addresses | Status |
|-------|---------|-----------|--------|
| **T1** | Consensus weight sensitivity | FLAG M6 (RRA weights 0.6/0.3/0.1) | ✅ Ready |
| **T2** | Stacking calibration benchmark | FLAG M10 (double calibration) | ✅ Ready |
| **T4** | Prevalence adjustment validation | FLAG D5.6 (Saerens implementation) | ✅ Ready |

## Quick Start

### Using Synthetic Data (Testing/Development)

```bash
# T1: Consensus weight sensitivity
python t1_consensus_weight_sensitivity.py --synthetic

# T2: Stacking calibration benchmark
python t2_stacking_calibration_benchmark.py --synthetic

# T4: Prevalence adjustment validation
python t4_prevalence_adjustment_validation.py --synthetic
```

### Using Real Pipeline Results

```bash
# First, ensure you have a completed pipeline run
cd /path/to/CeliacRisks
ced run-pipeline --models LR_EN,RF,XGBoost --split-seeds 0,1,2

# Get the run ID (e.g., 20260208_120000)
RUN_ID=$(ls -t results/ | head -1)

# Run validation studies
cd analysis/scripts/empirical_validation

# T1: Test consensus weight configurations
python t1_consensus_weight_sensitivity.py --run-id $RUN_ID

# T2: Compare stacking calibration strategies
python t2_stacking_calibration_benchmark.py --run-id $RUN_ID

# T4: Validate prevalence adjustment
python t4_prevalence_adjustment_validation.py --run-id $RUN_ID
```

## Study Details

### T1: Consensus Weight Sensitivity Analysis

**Purpose**: Validate the consensus weight configuration (0.6/0.3/0.1 for OOF/essentiality/stability) by comparing top-k protein selection under multiple weight configurations.

**Metrics**:
- Jaccard overlap for top-k (k=10, 20, 50, 100)
- Spearman/Kendall rank correlation for full rankings
- Selection frequency stability across configurations

**Usage**:
```bash
# Default: 7 predefined weight configurations
python t1_consensus_weight_sensitivity.py --run-id <RUN_ID>

# Custom weight configurations
python t1_consensus_weight_sensitivity.py --run-id <RUN_ID> \
    --weight-configs "0.6,0.3,0.1" "0.5,0.4,0.1" "0.7,0.2,0.1"

# Synthetic data with custom parameters
python t1_consensus_weight_sensitivity.py --synthetic \
    --n-proteins 500 --n-models 3
```

**Outputs**:
- `weight_sensitivity_results.csv`: Comparison metrics for each configuration
- `consensus_ranking_*.csv`: Full rankings for each configuration
- `jaccard_overlap.png`: Jaccard similarity heatmaps
- `rank_correlation.png`: Spearman/Kendall correlations
- `sensitivity_heatmap.png`: Summary heatmap
- `summary_report.txt`: Text summary with interpretation

**Interpretation**:
- **High correlation (ρ > 0.9, τ > 0.8)**: Weight choice is robust
- **High Jaccard (> 0.8) for small k**: Top proteins are stable across configurations
- **Lower overlap for larger k**: Expected (more variation in middle ranks)

---

### T2: Stacking Calibration Benchmark

**Purpose**: Evaluate whether meta-learner calibration (`calibrate_meta=True/False`) improves performance when base models are already OOF-posthoc calibrated.

**Metrics**:
- AUROC (discrimination)
- PR-AUC (precision-recall trade-off)
- Brier score (overall calibration quality)
- ECE (expected calibration error)

**Usage**:
```bash
# Default: test all calibration methods (sigmoid, isotonic, none)
python t2_stacking_calibration_benchmark.py --run-id <RUN_ID>

# Specific calibration methods only
python t2_stacking_calibration_benchmark.py --run-id <RUN_ID> \
    --calibration-methods sigmoid none

# Custom CV splits
python t2_stacking_calibration_benchmark.py --run-id <RUN_ID> --n-splits 10

# Synthetic data
python t2_stacking_calibration_benchmark.py --synthetic --n-samples 10000
```

**Outputs**:
- `stacking_calibration_results.csv`: Detailed results for each config/fold
- `metrics_comparison.png`: Bar plots comparing metrics
- `auroc_vs_brier.png`: Discrimination vs calibration trade-off
- `calibration_curves.png`: Calibration curves for each configuration
- `summary_report.txt`: Aggregated results with interpretation

**Interpretation**:
- If `calibrate_meta=True` does **not** improve Brier/ECE significantly, use `calibrate_meta=False` to avoid over-smoothing
- AUROC should remain stable (discrimination independent of calibration)
- Lower Brier and ECE indicate better calibration

---

### T4: Prevalence Adjustment Validation

**Purpose**: Validate that Saerens prevalence adjustment correctly shifts predicted probabilities to match target prevalence: `mean(adjusted_probs) ≈ target_prevalence`

**Tests**:
1. **Agreement**: |mean(adjusted) - target| < tolerance (default: 0.001)
2. **Calibration preservation**: Calibration slope/intercept before vs after
3. **Discrimination preservation**: AUROC unchanged
4. **Multiple target prevalences**: Test range of realistic scenarios
5. **Sensitivity to training prevalence**: Validate assumption P(X|Y) invariance

**Usage**:
```bash
# Default: test 7 target prevalences (0.001 to 0.10)
python t4_prevalence_adjustment_validation.py --run-id <RUN_ID>

# Custom target prevalences
python t4_prevalence_adjustment_validation.py --run-id <RUN_ID> \
    --target-prevalences 0.001 0.005 0.01 0.05

# Specific model
python t4_prevalence_adjustment_validation.py --run-id <RUN_ID> \
    --model-name LR_EN

# Synthetic data with custom sample prevalence
python t4_prevalence_adjustment_validation.py --synthetic \
    --sample-prevalence 0.167 --n-samples 10000
```

**Outputs**:
- `prevalence_validation_results.csv`: Results for each target prevalence
- `prevalence_agreement.png`: Target vs adjusted mean (scatter + error)
- `auroc_preservation.png`: AUROC before/after adjustment
- `calibration_metrics.png`: Calibration slope/intercept changes
- `summary_table.png`: Summary table with pass/fail indicators
- `summary_report.txt`: Detailed results with interpretation

**Interpretation**:
- **Absolute error < 0.001**: Excellent agreement (PASS)
- **AUROC change < 0.001**: Discrimination preserved (expected)
- **Calibration intercept changes**: Expected (adjustment shifts intercept)
- **Calibration slope stable**: Good (adjustment doesn't distort calibration)

**Assumptions** (Saerens method is valid when):
1. P(X|Y) distributions are similar between training and deployment
2. Only the prior P(Y) changes (prevalence shift)
3. Feature distributions remain stable

---

## Output Structure

All studies save results to:
```
results/empirical_validation/
├── t1_<run_id>_<timestamp>/
│   ├── weight_sensitivity_results.csv
│   ├── consensus_ranking_*.csv
│   ├── jaccard_overlap.png
│   ├── rank_correlation.png
│   ├── sensitivity_heatmap.png
│   └── summary_report.txt
├── t2_<run_id>_<timestamp>/
│   ├── stacking_calibration_results.csv
│   ├── metrics_comparison.png
│   ├── auroc_vs_brier.png
│   ├── calibration_curves.png
│   └── summary_report.txt
└── t4_<run_id>_<timestamp>/
    ├── prevalence_validation_results.csv
    ├── prevalence_agreement.png
    ├── auroc_preservation.png
    ├── calibration_metrics.png
    ├── summary_table.png
    └── summary_report.txt
```

## Dependencies

All scripts require the CeliacRisks Python environment:

```bash
cd analysis
source .venv/bin/activate  # or conda activate ced-ml
```

Key dependencies:
- `numpy`, `pandas`, `scipy`: Core computation
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Metrics and models
- `ced_ml`: Project modules (features, models, evaluation)

## Integration with Audit Document

After running these studies, update `analysis/docs/development/MATHEMATICAL_VALIDITY_AUDIT.md`:

1. **Section 5C**: Add empirical results to "Consolidated Tuning Backlog (Empirical)"
2. **FLAG items**: Update with empirical evidence
   - M6: Add Jaccard/correlation results from T1
   - M10: Add Brier/ECE comparison from T2
   - D5.6: Add agreement test results from T4

Example update:
```markdown
### FLAG M6: RRA Weights (0.6/0.3/0.1) -- EMPIRICALLY VALIDATED

**Empirical study (T1, 2026-02-08)**:
- Tested 7 weight configurations
- Mean Spearman ρ: 0.95, min: 0.89
- Jaccard@10: mean 0.92, min 0.85
- **Conclusion**: Current weights are robust; top-20 stable across configurations
```

## Troubleshooting

### "No module named 'ced_ml'"
```bash
cd analysis
source .venv/bin/activate
pip install -e .
```

### "Run directory not found"
Ensure the run ID exists:
```bash
ls -la results/
```

### "OOF predictions file not found"
The pipeline must complete aggregation:
```bash
ced aggregate-splits --run-id <RUN_ID>
```

### Synthetic data runs but results look unrealistic
Adjust parameters:
```bash
python t1_consensus_weight_sensitivity.py --synthetic \
    --n-proteins 1000 --n-models 5
```

## Next Steps

After running all three studies:

1. **Review results**: Check summary reports and plots
2. **Document findings**: Update audit document (section 5C)
3. **Make recommendations**:
   - If weights are robust (T1): Document and keep current configuration
   - If calibration doesn't help (T2): Set `calibrate_meta=False` as default
   - If prevalence adjustment passes (T4): No code changes needed
4. **Consider additional studies** (optional):
   - T1 extension: Stability across different CV seeds
   - T2 extension: Low-prevalence stress testing
   - T4 extension: Feature distribution drift sensitivity

## References

- **Audit Document**: `analysis/docs/development/MATHEMATICAL_VALIDITY_AUDIT.md`
- **ADR-004**: Feature selection strategy (consensus weights)
- **ADR-007**: Stacking ensemble design
- **ADR-008**: OOF-posthoc calibration
- **Saerens et al. (2002)**: Prevalence adjustment method
- **Kolde et al. (2012)**: Formal RRA (for comparison)
