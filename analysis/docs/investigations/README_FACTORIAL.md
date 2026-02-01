# 2x2x2 Factorial Experiment: Training Set Composition

Standalone experiment investigating how training set composition affects model performance.

## Design

**Factors** (8 cells total):

| Factor | Levels | Description |
|--------|--------|-------------|
| `n_cases` | 50, 149 | Number of incident cases in training |
| `ratio` | 1, 5 | Controls per case |
| `prevalent_frac` | 0.5, 1.0 | Fraction of prevalent cases included |

**Derived**: `train_N = n_cases + round(prevalent_frac * n_cases) + ratio * n_cases`

**Models**: LR_EN, XGBoost
**Panel**: Fixed 25-protein panel (`data/fixed_panel.csv`)
**Seeds**: 10 (configurable)
**Total runs**: 8 cells x 10 seeds x 2 models = 160

## Three Guardrails

### 1. Fixed TEST set
One global stratified test split (seed=42). All cells evaluate on the same TEST set.
Prevalent cases are excluded from TEST entirely.

### 2. Paired seeds (nesting)
For each seed, the same shuffled pools are used. Cells take **prefixes** from the shuffled pools:
- `n_cases=50` is a strict subset of `n_cases=149` within the same seed
- This reduces variance and enables paired statistical tests

### 3. Frozen features + hyperparams
- **Panel**: Fixed 25-protein panel (no feature selection during experiment)
- **Hyperparams**: Tuned once on baseline cell (n_cases=149, ratio=5, prev_frac=0.5), then frozen

## Usage

### Option 1: Complete workflow (RECOMMENDED)

Run all three steps (tune, experiment, analyze) with a single command:

```bash
python run_factorial_complete.py \
    --data-path ../../../data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path ../../../data/fixed_panel.csv \
    --output-dir ../../../results/factorial_2x2x2 \
    --n-seeds 10
```

This will:
1. Tune baseline hyperparameters via Optuna (if `frozen_hyperparams.yaml` doesn't exist)
2. Run the full factorial experiment (160 runs with default settings)
3. Analyze results and generate all visualizations

Optional flags:
- `--models LR_EN XGBoost` -- models to run (default: both)
- `--skip-tuning` -- skip tuning if hyperparams already exist (requires `--hyperparams-path`)
- `--hyperparams-path PATH` -- use existing hyperparameters
- `--top-k K` -- number of top features for Jaccard overlap analysis (default: 15)

### Option 2: Individual steps (for debugging or customization)

#### Step 1: Tune baseline hyperparams

```bash
python run_factorial_2x2x2.py \
    --data-path ../../../data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path ../../../data/fixed_panel.csv \
    --output-dir ../../../results/factorial_2x2x2 \
    --tune-baseline
```

Saves `frozen_hyperparams.yaml` via Optuna (50 trials per model).

#### Step 2: Run experiment

```bash
python run_factorial_2x2x2.py \
    --data-path ../../../data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path ../../../data/fixed_panel.csv \
    --output-dir ../../../results/factorial_2x2x2 \
    --hyperparams-path ../../../results/factorial_2x2x2/frozen_hyperparams.yaml \
    --n-seeds 10
```

Outputs `factorial_results.csv` (one row per seed x cell x model) and `feature_importances.csv` (one row per seed x cell x model x feature).

#### Step 3: Analyze and visualize

```bash
python analyze_factorial_2x2x2.py \
    --results ../../../results/factorial_2x2x2/factorial_results.csv \
    --top-k 15
```

Generates:
- CSV summaries: `main_effects.csv`, `interactions.csv`, `feature_jaccard.csv`
- Visualizations: `main_effects_{model}.png`, `interactions_{model}.png`, `score_distributions_{model}.png`, `cell_means_{model}.png`, `jaccard_heatmap_{model}.png`
- Markdown reports: `summary_{model}.md`, `feature_overlap.md`

Optional flags:
- `--feature-importances PATH` -- override path to `feature_importances.csv` (default: sibling of results file)
- `--top-k K` -- number of top features for Jaccard overlap analysis (default: 15)
- `--output-dir DIR` -- override output directory (default: `analysis/` subdirectory)

## Sampling Procedure

For each seed s:
1. Shuffle `I_pool` (incident), `P_pool` (prevalent), `C_pool` (controls) with seed s
2. For each cell `(n_cases, ratio, prevalent_frac)`:
   - `I = first n_cases from I_pool`
   - `P = first round(prevalent_frac * n_cases) from P_pool`
   - `C = first (ratio * n_cases) from C_pool`
   - Train on `I + P + C`

Prefixes guarantee nesting: smaller cells are subsets of larger cells within the same seed.

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| AUROC | Discrimination | Area under ROC curve |
| PRAUC | Discrimination | Area under precision-recall curve |
| Brier | Calibration | Brier score (lower = better) |
| cal_slope | Calibration | Logistic calibration slope (ideal = 1) |
| cal_intercept | Calibration | Logistic calibration intercept (ideal = 0) |
| sens_at_spec95 | Clinical | Sensitivity at 95% specificity |
| mean_prob_incident | Score distribution | Mean predicted probability for incident cases (test set) |
| mean_prob_prevalent | Score distribution | Mean predicted probability for prevalent cases (train set) |
| mean_prob_control | Score distribution | Mean predicted probability for controls (test set) |
| score_gap | Score distribution | `mean_prob_incident - mean_prob_prevalent` (positive = incident scored higher) |

## Statistical Analysis

### Main effects (paired contrasts)
For each factor, compute `high - low` averaged over other factors, paired by seed:
- Mean delta, 95% CI, Cohen's d (paired), paired t-test

### Interactions
- **2-way**: Does the effect of factor A depend on the level of factor B?
- **3-way**: Does the n_cases x ratio interaction depend on prevalent_frac?

### Feature importance overlap (Jaccard)
For each (model, seed, n_cases, ratio) combination, the top-K features by importance
are compared between `prevalent_frac=0.5` and `prevalent_frac=1.0`.
Jaccard similarity measures how much the two feature sets overlap:
- **Jaccard = 1.0**: identical top-K features regardless of prevalent fraction
- **Jaccard << 1.0**: including more prevalent cases shifts which biomarkers the model relies on

This is computed from `feature_importances.csv` (absolute LR coefficients or XGB `feature_importances_`).

### Interpretation (Cohen's d)
- `|d| < 0.2`: negligible
- `0.2 <= |d| < 0.5`: small
- `0.5 <= |d| < 0.8`: medium
- `|d| >= 0.8`: large

## Output Schema

`factorial_results.csv`:
```
seed, n_cases, ratio, prevalent_frac, train_N, model,
AUROC, PRAUC, Brier, cal_slope, cal_intercept, sens_at_spec95,
mean_prob_incident, mean_prob_prevalent, mean_prob_control, score_gap,
runtime_s
```

`feature_importances.csv`:
```
seed, n_cases, ratio, prevalent_frac, model, feature, importance
```

`analysis/` directory:

**CSV outputs:**
- `main_effects.csv` -- paired contrasts for all metrics (including score distributions)
- `interactions.csv` -- 2-way and 3-way interaction tests
- `feature_jaccard.csv` -- per-cell Jaccard similarity of top-K features across prevalent_frac levels

**Markdown reports:**
- `feature_overlap.md` -- markdown summary of feature overlap analysis
- `summary_{model}.md` -- per-model markdown report (includes Jaccard section)

**Visualization plots (PNG):**
- `main_effects_{model}.png` -- forest plots of main effects with 95% CIs (* = p<0.05)
- `interactions_{model}.png` -- 2-way interaction plots for AUROC, Brier, sens_at_spec95
- `score_distributions_{model}.png` -- box plots of mean predicted probabilities by case type
- `cell_means_{model}.png` -- bar plots of cell means for AUROC, PRAUC, Brier, sens_at_spec95
- `jaccard_heatmap_{model}.png` -- heatmap of feature overlap across design factors

## Seed Count Guidance

| Purpose | Seeds | Total runs |
|---------|-------|------------|
| Quick validation | 2 | 32 |
| Real inference | 10 | 160 |
| High precision | 20 | 320 |
