# Factorial Experiment Analysis

This directory contains tools for analyzing factorial experiments to investigate sources of variability in model performance, feature selection, and risk scores.

## Scripts

### `investigate_factorial.py` (Original)
**Design**: 2×2 factorial
- Factor 1: prevalent_train_frac (0.5 vs 1.0)
- Factor 2: train_control_per_case (1:1 vs 1:5)
- Total configs: 4

**Use case**: Original prevalence/ratio experiment

### `investigate_factorial_extended.py` (New)
**Design**: 3-factor factorial
- Factor 1: fold_size (k in k-fold CV)
- Factor 2: train_size (absolute N)
- Factor 3: calibration_method (isotonic, platt, oof_posthoc)

**Feature selection**: Fixed at **hybrid** for all experiments

**Use case**: Investigating calibration robustness, fold size impact, and training set size effects

## Experimental Setup

### Factor Levels

| Factor | Recommended Levels | Rationale |
|--------|-------------------|-----------|
| **fold_size** | 3, 5, 10 | Small k (3) → large folds, less variance<br>Large k (10) → small folds, more variance |
| **train_size** | 298, 596, 894 | Corresponds to 1:1, 1:3, 1:5 control:case ratios<br>Tests learning curve effects |
| **calib_method** | isotonic, platt, oof_posthoc | Different calibration strategies (see ADR-014) |

### Example: 3×3×3 = 27 Configurations

```
k3_N298_isotonic    k3_N298_platt    k3_N298_oof_posthoc
k3_N596_isotonic    k3_N596_platt    k3_N596_oof_posthoc
k3_N894_isotonic    k3_N894_platt    k3_N894_oof_posthoc

k5_N298_isotonic    k5_N298_platt    k5_N298_oof_posthoc
k5_N596_isotonic    k5_N596_platt    k5_N596_oof_posthoc
k5_N894_isotonic    k5_N894_platt    k5_N894_oof_posthoc

k10_N298_isotonic   k10_N298_platt   k10_N298_oof_posthoc
k10_N596_isotonic   k10_N596_platt   k10_N596_oof_posthoc
k10_N894_isotonic   k10_N894_platt   k10_N894_oof_posthoc
```

Each config should be run with multiple random seeds (5-10 recommended) and 2 models (e.g., LR_EN, XGBoost).

**Total runs**: 27 configs × 5 seeds × 2 models = **270 runs**

## Running Experiments

### Step 1: Configure Factorial Design

Create config files for each factor combination. Example for `k5_N596_isotonic`:

```yaml
# configs/factorial/k5_N596_isotonic.yaml
cv_config:
  n_splits: 5
  strategy: stratified

training:
  n_train: 596
  train_control_per_case: 3  # (596 - 149) / 149 ≈ 3

feature_selection:
  method: hybrid  # FIXED for all experiments
  stability:
    threshold: 0.75
  kbest:
    k: 100

calibration:
  method: isotonic

models:
  - LR_EN
  - XGBoost

split_seeds:
  - 0
  - 1
  - 2
  - 3
  - 4
```

### Step 2: Run Pipeline for Each Config

```bash
# Run all configs in batch
for config in configs/factorial/*.yaml; do
    ced run-pipeline --config "$config" --output-suffix "$(basename $config .yaml)"
done

# Or run individually
ced run-pipeline --config configs/factorial/k5_N596_isotonic.yaml
```

### Step 3: Analyze Results

```bash
# Analyze latest run
python investigate_factorial_extended.py

# Or specify run ID
python investigate_factorial_extended.py --run-id run_20260202_120000

# Custom output location
python investigate_factorial_extended.py \
    --results-dir results/ \
    --output-dir analysis/factorial_results/
```

## Outputs

The analysis script generates:

| File | Description |
|------|-------------|
| `metrics_all.csv` | Raw metrics for all runs (one row per split_seed) |
| `comparison_table.csv` | Aggregated stats by config (mean ± std, 95% CI) |
| `main_effects.csv` | Main effect tests for each factor |
| `interactions.csv` | Two-way interaction tests (fold × train, fold × calib, train × calib) |
| `power_analysis.csv` | Statistical power for each comparison |
| `summary.md` | Human-readable report with interpretation |

## Statistical Tests

### Main Effects
For each factor (fold_size, train_size, calib_method):
- Pairwise comparisons between all levels
- Paired t-tests (matched by split_seed)
- Bonferroni correction for multiple comparisons
- Effect sizes (Cohen's d)

**Example**: Does k=5 produce significantly different AUROC than k=10?

### Interactions
Two-way ANOVA to detect interaction effects:
- **fold_size × train_size**: Does fold size impact change with training set size?
- **fold_size × calib_method**: Does calibration quality depend on fold size?
- **train_size × calib_method**: Does calibration method effectiveness vary with N?

**Interpretation**: Significant interaction → the effect of one factor depends on the level of another

### Multiple Testing Correction
- **Main effects**: Bonferroni correction across all pairwise comparisons
- **Interactions**: Uncorrected p < 0.05 (exploratory)

## Research Questions

### Fold Size (k in CV)
1. Does smaller k (larger folds) improve calibration quality?
2. Are features more stable with larger k (more iterations)?
3. What's the compute/performance trade-off?

### Train Set Size (N)
4. Does performance plateau after a certain N?
5. How does N affect feature selection consistency?
6. Can we achieve adequate performance with smaller N (faster iteration)?

### Calibration Method
7. Which method produces the best-calibrated risk scores?
8. Does optimal method depend on N or k?
9. How robust is each method to distribution shift?

## Interpretation Guide

### Cohen's d Effect Sizes
- **|d| < 0.2**: Negligible difference (likely not practically important)
- **0.2 ≤ |d| < 0.5**: Small effect (may be important at scale)
- **0.5 ≤ |d| < 0.8**: Medium effect (likely important)
- **|d| ≥ 0.8**: Large effect (definitely important)

### Statistical Power
- **Power < 0.5**: Underpowered (may miss true effects)
- **0.5 ≤ Power < 0.8**: Moderate (acceptable for exploratory)
- **Power ≥ 0.8**: Well-powered (standard threshold)

### Eta-Squared (η²) for Interactions
- **η² < 0.01**: Small interaction effect
- **0.01 ≤ η² < 0.06**: Medium interaction
- **η² ≥ 0.06**: Large interaction (investigate further)

## Example Workflows

### Quick Test (Subset Design)
Test a 2×2×2 subset to validate pipeline:
```
Fold sizes: 5, 10
Train sizes: 298, 894
Calib methods: isotonic, platt
Total: 8 configs × 3 seeds × 2 models = 48 runs
```

### Full Factorial (Comprehensive)
3×3×3 design with 5 seeds:
```
Total: 27 configs × 5 seeds × 2 models = 270 runs
Estimated time: ~45 hours on HPC (parallel), ~10 days sequential
```

### Sensitivity Analysis (One-Factor-at-a-Time)
Fix baseline (k=5, N=596, isotonic), vary one factor:
```
Fold size: k3, k5, k10 (3 configs)
Train size: N298, N596, N894 (3 configs)
Calib method: isotonic, platt, oof_posthoc (3 configs)
Total: 9 configs × 5 seeds × 2 models = 90 runs
```

## References

- [ADR-006](../adr/ADR-006-nested-cv.md): Nested CV structure
- [ADR-013](../adr/ADR-013-four-strategy-feature-selection.md): Feature selection framework
- [ADR-014](../adr/ADR-014-oof-posthoc-calibration.md): OOF-posthoc calibration
- [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d): Effect size interpretation
- [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction): Multiple testing adjustment
