# Factorial Experiment: 2×2×2 Design

## Quick Start

### Local (Parallel, Recommended)

```bash
# Step 1: Tune cell-specific hyperparameters (~2-3 hours)
python analysis/docs/investigations/run_factorial_2x2x2.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path data/fixed_panel.csv \
    --output-dir results/factorial_2x2x2 \
    --tune-cells \
    --n-trials 50

# Step 2: Run experiment with parallelization (~5-10 min with 8 CPUs)
python analysis/docs/investigations/run_factorial_2x2x2.py \
    --data-path data/Celiac_dataset_proteomics_w_demo.parquet \
    --panel-path data/fixed_panel.csv \
    --output-dir results/factorial_2x2x2 \
    --hyperparams-path results/factorial_2x2x2/cell_hyperparams.yaml \
    --n-seeds 10 \
    --parallel
```

### HPC (SLURM or LSF)

**SLURM:**
```bash
# Submit job (handles both tuning and experiment)
sbatch analysis/docs/investigations/submit_factorial_hpc.sh

# Monitor progress
tail -f logs/factorial_<jobid>.out
```

**LSF:**
```bash
# Submit job
bsub < analysis/docs/investigations/submit_factorial_hpc_lsf.sh

# Monitor progress
tail -f logs/factorial_<jobid>.out
```

Both scripts:
- Auto-detect repository root (no path editing needed)
- Match settings from `configs/pipeline_hpc.yaml`
- Allocate 8 CPUs, 8GB per CPU, 24-hour walltime
- Use `acc_Chipuk_Laboratory` account and `premium` queue

---

## Experimental Design

### Factorial Structure

**3 Factors × 2 Levels Each = 8 Cells:**

| Factor | Levels | Rationale |
|--------|--------|-----------|
| `n_cases` | 50, 149 | Sample size effect (all vs half of incident cases) |
| `ratio` | 1, 5 | Class imbalance effect (balanced vs 5:1 controls:cases) |
| `prevalent_frac` | 0.5, 1.0 | Prevalent case inclusion (half vs all) |

**Per cell:** 10 seeds × 2 models (LR_EN, XGBoost) = 20 runs
**Total:** 8 cells × 20 runs = 160 model fits per experiment

### Scientific Questions

1. **Sample size effect:** Does using all 149 incident cases vs 50 improve performance?
2. **Class imbalance effect:** Does balanced (1:1) vs imbalanced (1:5) ratio affect discrimination?
3. **Prevalent case inclusion:** Do prevalent cases improve or harm incident CeD prediction?
4. **Interactions:** Are there synergistic effects between factors?

---

## Guardrails (Statistical Rigor)

### 1. Fixed Test Set
- Created once with seed=42
- Stratified 25% of incident/control cases
- **Never changes** across all tuning and evaluation
- Ensures all cells compared on identical held-out data

### 2. Cell-Specific Hyperparameter Tuning
- Each cell gets hyperparameters optimized for its data distribution
- Prevents unfair advantages from shared hyperparams tuned on one cell
- Uses train/val split (80/20) **within each cell** for tuning
- Test set never used for optimization (only final evaluation logging)

### 3. Paired Seeds (Nesting)
- Same shuffled pools across cells within each seed
- Cells take prefixes of shuffled pools (nesting guarantee)
- Ensures seed=0 for n_cases=50 is a subset of seed=0 for n_cases=149

### 4. Frozen Features
- 25-protein panel fixed across all cells
- Eliminates feature selection as confounding variable

---

## Parallelization

### How It Works

The experiment parallelizes at the **job level** (seed × cell × model):
- 10 seeds × 8 cells × 2 models = 160 independent jobs
- Each job trains one model on one cell with one seed
- Jobs run in parallel via Python `multiprocessing.Pool`

### Performance

| CPUs | Runtime (160 jobs) | Speedup |
|------|-------------------|---------|
| 1 (sequential) | ~30-60 min | 1.0× |
| 8 | ~5-10 min | 5-7× |
| 16 | ~3-5 min | 10-14× |
| 32 | ~2-3 min | 18-25× |

**Note:** Hyperparameter tuning (Step 1) is NOT parallelized - Optuna trials run sequentially within each cell.

### Parallel Options

```bash
# Use all available CPUs
--parallel

# Use specific number of CPUs
--n-jobs 16

# Sequential (default)
--n-jobs 1
```

---

## Outputs

```
results/factorial_2x2x2/
├── cell_hyperparams.yaml         # Cell-specific frozen hyperparameters
├── test_indices.csv              # Fixed test set indices
├── factorial_results.csv         # Main results (160 rows)
└── feature_importances.csv       # Feature importance per run
```

### Results Schema

**factorial_results.csv:**
- `seed`: Random seed (0-9)
- `n_cases`, `ratio`, `prevalent_frac`: Cell configuration
- `train_N`: Training set size
- `model`: LR_EN or XGBoost
- `AUROC`, `PRAUC`, `Brier`: Discrimination metrics
- `cal_slope`, `cal_intercept`: Calibration metrics
- `sens_at_spec95`: Sensitivity at 95% specificity
- `mean_prob_incident`, `mean_prob_prevalent`, `mean_prob_control`: Score distributions
- `score_gap`: Incident - Prevalent mean score
- `runtime_s`: Training time

---

## Analysis

See [analyze_factorial_2x2x2.py](analysis/docs/investigations/analyze_factorial_2x2x2.py) for:

- **Main effects plots:** Marginal effect of each factor
- **Interaction plots:** Synergistic effects between factors
- **Statistical tests:** ANOVA, Tukey HSD post-hoc comparisons
- **Hyperparameter comparison:** How hyperparams vary across cells

Example analysis:
```bash
python analysis/docs/investigations/analyze_factorial_2x2x2.py \
    --results results/factorial_2x2x2/factorial_results.csv \
    --output results/factorial_2x2x2/analysis
```

---

## Testing

Verify parallel implementation matches sequential:

```bash
python analysis/docs/investigations/test_parallel.py
```

Expected output:
```
Sequential time: 45.3s
Parallel time: 13.2s
Speedup: 3.43x
Max AUROC difference: 0.00e+00
✓ Parallel results match sequential (numerically identical)
```

---

## Files

| File | Purpose |
|------|---------|
| [run_factorial_2x2x2.py](run_factorial_2x2x2.py) | Main experiment script |
| [FACTORIAL_CELL_TUNING.md](FACTORIAL_CELL_TUNING.md) | Detailed tuning documentation |
| [submit_factorial_hpc.sh](submit_factorial_hpc.sh) | SLURM submission script (8 CPUs, 24h) |
| [submit_factorial_hpc_lsf.sh](submit_factorial_hpc_lsf.sh) | LSF submission script (8 CPUs, 24h) |
| [test_parallel.py](test_parallel.py) | Parallelization test |
| [analyze_factorial_2x2x2.py](analyze_factorial_2x2x2.py) | Analysis and visualization |

---

## Troubleshooting

### Parallelization Issues

**Problem:** `OSError: [Errno 24] Too many open files`
**Solution:** Reduce `--n-jobs` or increase file descriptor limit:
```bash
ulimit -n 4096
```

**Problem:** Out of memory errors
**Solution:** Reduce `--n-jobs` (each worker loads full dataset into memory)

### Hyperparameter Tuning Issues

**Problem:** Tuning is too slow
**Solution:** Reduce `--n-trials` (minimum 20-30 recommended)

**Problem:** Want to skip tuning for quick testing
**Solution:** Use default hyperparameters (omit `--hyperparams-path`), but document this limitation

---

## Citation

If using this factorial design in publications, cite the methodology:

> We evaluated the effects of sample size, class imbalance, and prevalent case inclusion on incident Celiac Disease prediction using a 2×2×2 factorial design with cell-specific hyperparameter optimization and a fixed held-out test set to ensure valid statistical comparisons across all experimental conditions.

---

## Changelog

**2026-02-01:**
- Added cell-specific hyperparameter tuning (replaces baseline-only tuning)
- Fixed test set leakage (now uses train/val split within cells)
- Added parallelization support (multiprocessing.Pool)
- Added HPC SLURM submission script
- Updated documentation with parallelization benchmarks
