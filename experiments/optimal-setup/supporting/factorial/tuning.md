# Factorial Experiment: Cell-Specific Hyperparameter Tuning

## Overview

The factorial experiment now uses **cell-specific hyperparameter tuning** to ensure fair comparisons across all factorial conditions. Each of the 8 cells (n_cases × ratio × prevalent_frac) gets hyperparameters optimized specifically for its data distribution.

## Why Cell-Specific Tuning?

**Problem with shared hyperparameters:**
- Different sample sizes (N=50 vs N=745) require different regularization strengths
- Different class ratios (1:1 vs 1:5) affect optimal model configurations
- Using hyperparams tuned on one cell gives that cell an unfair advantage

**Solution:**
- Tune hyperparameters separately for each factorial cell
- Use train/val split within each cell (prevents test set leakage)
- Store cell-specific hyperparams in `cell_hyperparams.yaml`

## Workflow

### Step 1: Cell-Specific Hyperparameter Tuning

Tune hyperparameters for all 8 factorial cells:

```bash
python experiments/optimal-setup/supporting/factorial/run_factorial_2x2x2.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path data/fixed_panel.csv \
    --output-dir results/factorial_2x2x2 \
    --tune-cells \
    --n-trials 50 \
    --models LR_EN XGBoost
```

**Runtime estimate:** ~2-3 hours (8 cells × 2 models × 50 trials = 800 Optuna runs)

**Outputs:**
- `results/factorial_2x2x2/cell_hyperparams.yaml` - Cell-specific hyperparameters
- Console logs showing val/test AUROC for each cell

**Example output structure:**
```yaml
n50_r1_p0.5:
  LR_EN:
    C: 0.0123
    l1_ratio: 0.45
    penalty: elasticnet
    solver: saga
    max_iter: 2000
  XGBoost:
    n_estimators: 250
    max_depth: 6
    learning_rate: 0.08
    ...

n50_r1_p1.0:
  LR_EN:
    C: 0.0089  # Different from n50_r1_p0.5!
    l1_ratio: 0.52
    ...
```

### Step 2: Run Full Factorial Experiment

Run all seeds using the cell-specific hyperparameters:

**Local (sequential):**
```bash
python experiments/optimal-setup/supporting/factorial/run_factorial_2x2x2.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path data/fixed_panel.csv \
    --output-dir results/factorial_2x2x2 \
    --hyperparams-path results/factorial_2x2x2/cell_hyperparams.yaml \
    --n-seeds 10 \
    --models LR_EN XGBoost
```

**Local (parallel with all CPUs):**
```bash
python experiments/optimal-setup/supporting/factorial/run_factorial_2x2x2.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path data/fixed_panel.csv \
    --output-dir results/factorial_2x2x2 \
    --hyperparams-path results/factorial_2x2x2/cell_hyperparams.yaml \
    --n-seeds 10 \
    --models LR_EN XGBoost \
    --parallel
```

**HPC (SLURM with 32 CPUs):**
```bash
sbatch experiments/optimal-setup/supporting/factorial/submit_factorial_hpc.sh
```

**Runtime estimates:**
- Sequential: ~30-60 minutes (10 seeds × 8 cells × 2 models = 160 runs)
- Parallel (32 CPUs): ~5-10 minutes (160 runs / 32 workers)

**Outputs:**
- `results/factorial_2x2x2/factorial_results.csv` - Main results table
- `results/factorial_2x2x2/feature_importances.csv` - Feature importance per run
- `results/factorial_2x2x2/test_indices.csv` - Fixed test set indices

## Technical Details

### Guardrails Against Test Set Leakage

1. **Fixed test set (seed=42):** Created once, never changes across all tuning and evaluation
2. **Train/val split within cells:** Each cell uses 80/20 train/val split for hyperparameter optimization
3. **Test set never used for tuning:** Test AUROC is logged but never optimized
4. **Paired seeds:** Same shuffled pools across cells (nesting guarantee)

### Cell Key Format

Cells are identified by keys: `n{n_cases}_r{ratio}_p{prev_frac}`

Examples:
- `n50_r1_p0.5` → 50 cases, ratio=1, prevalent_frac=0.5
- `n149_r5_p1.0` → 149 cases, ratio=5, prevalent_frac=1.0

### Tuning Process Per Cell

For each cell:
1. Sample cell using `TUNING_SEED=0` (reproducible)
2. Split cell into train (80%) and val (20%)
3. Run Optuna (50 trials default) optimizing val AUROC
4. Save best hyperparams to `cell_hyperparams.yaml`
5. Log both val AUROC (tuned) and test AUROC (held out)

## Expected Hyperparameter Patterns

Based on theory, expect:

- **Larger N → weaker regularization:** `C` should increase with sample size
- **Balanced classes → different tree depth:** ratio=1 may favor shallower trees
- **More prevalent cases → more regularization:** prevalent_frac affects effective sample size

## Validation

Check that tuning worked correctly:

```python
import yaml
import pandas as pd

# Load cell hyperparams
with open("results/factorial_2x2x2/cell_hyperparams.yaml") as f:
    hp = yaml.safe_load(f)

# Extract C values for LR_EN across cells
c_values = {cell: params["LR_EN"]["C"] for cell, params in hp.items()}
df_c = pd.DataFrame(list(c_values.items()), columns=["cell", "C"])
print(df_c.sort_values("C"))

# Expect: larger cells (n149) should have higher C than smaller cells (n50)
```

## Analysis Script

See `experiments/optimal-setup/supporting/factorial/analyze_factorial_2x2x2.py` for:
- Main effects plots (n_cases, ratio, prevalent_frac)
- Interaction plots
- Hyperparameter comparison across cells
- Statistical tests (ANOVA, Tukey HSD)

## Comparison to Baseline Approach

**Old approach (baseline tuning):**
- Tune on baseline cell (n=149, r=5, p=0.5) using test set
- Apply same hyperparams to all cells
- **Problems:** Test leakage, baseline advantage, suboptimal for other cells

**New approach (cell-specific tuning):**
- Tune each cell using train/val split (no test leakage)
- Each cell gets optimal hyperparams for its conditions
- **Benefits:** Fair comparisons, valid statistical inference, honest uncertainty

## Parallelization

### Local Parallelization

The script supports multiprocessing via `--n-jobs` or `--parallel`:

```bash
# Use all available CPUs
python run_factorial_2x2x2.py ... --parallel

# Use specific number of CPUs
python run_factorial_2x2x2.py ... --n-jobs 16

# Sequential (default)
python run_factorial_2x2x2.py ... --n-jobs 1
```

**Speed comparison (10 seeds, 8 cells, 2 models = 160 jobs):**
- Sequential (1 CPU): ~30-60 minutes
- Parallel (8 CPUs): ~5-10 minutes
- Parallel (32 CPUs): ~2-5 minutes

### HPC Parallelization

For SLURM clusters, use the provided submission script:

```bash
# Edit paths in submit_factorial_hpc.sh first
sbatch experiments/optimal-setup/supporting/factorial/submit_factorial_hpc.sh
```

The script:
1. Allocates 32 CPUs and 64GB RAM
2. Tunes cell-specific hyperparams (if not already done)
3. Runs full experiment in parallel
4. Saves logs to `logs/factorial_<jobid>.{out,err}`

**Customization:**
- Adjust `--cpus-per-task` in SLURM header
- Adjust `--mem` based on dataset size
- Modify paths in script variables section

## Notes

- Tuning is expensive but only needs to run once
- Hyperparameters can be reused across multiple analysis runs
- If time is limited, reduce `--n-trials` (minimum 20-30 recommended)
- For quick testing, can tune single cell and copy hyperparams (but document this limitation)
- **Parallelization speeds up experiment runs but NOT hyperparameter tuning** (Optuna trials are sequential within each cell)
