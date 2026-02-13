# CLI Reference

Complete reference for the `ced` command-line interface.

---

## Global Options

All commands support:

| Option | Description |
|--------|-------------|
| `--log-level` | Logging level: `debug`, `info`, `warning`, `error` (default: `info`) |
| `--version` | Show version and exit |
| `--help` | Show command help |

---

## Pipeline Commands

### `ced run-pipeline`

**Full end-to-end workflow orchestration (RECOMMENDED)**

Runs the complete ML pipeline: splits, training, aggregation, and optional evaluation.

```bash
ced run-pipeline [OPTIONS]
```

**HPC Mode Logging:**
When running in HPC mode (`--hpc`), a detailed submission log is automatically created at:
```
logs/run_{RUN_ID}/submission.log
```
This log captures orchestrator submission details, training job IDs, and barrier progress metadata.

HPC runs now use a sentinel-based orchestrator job (instead of long LSF `-w done(...)` chains):
- Completion log: `logs/run_{RUN_ID}/sentinels/completed.log`
- Wrapper + orchestrator scripts: `logs/run_{RUN_ID}/scripts/`
- Job manifest: `logs/run_{RUN_ID}/scripts/jobs_manifest.json`
- Orchestrator runtime log: `logs/run_{RUN_ID}/orchestrator.log`

Per-stage waits are tuned under `hpc.orchestrator` in `configs/pipeline_hpc.yaml`:
- `poll_interval`
- `training_timeout`, `post_timeout`, `perm_timeout`, `panel_timeout`, `consensus_timeout`
- `max_concurrent_submissions`

Troubleshooting:
```bash
cat logs/run_<RUN_ID>/sentinels/completed.log
cat logs/run_<RUN_ID>/orchestrator.log
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | PATH | - | Training config YAML |
| `--pipeline-config` | PATH | - | Pipeline execution config YAML |
| `--infile` | PATH | Auto-detect | Input proteomics file |
| `--split-dir` | PATH | Auto | Splits directory |
| `--outdir` | PATH | `results/` | Output directory |
| `--models` | TEXT | Config | Comma-separated model names |
| `--split-seeds` | TEXT | Config | Comma-separated split seeds |
| `--dry-run` | FLAG | False | Show what would run without executing |
| `--no-aggregate` | FLAG | False | Skip aggregation step |
| `--no-optimize` | FLAG | False | Skip panel optimization |

**Examples:**
```bash
# Auto-detect data, run full pipeline
ced run-pipeline

# Specify models and splits
ced run-pipeline --models LR_EN,RF,XGBoost --split-seeds 0,1,2

# Dry run to see planned execution
ced run-pipeline --dry-run
```

---

## Data Preparation

### `ced save-splits`

Generate train/val/test splits with stratification.

```bash
ced save-splits [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | PATH | - | Splits config YAML |
| `--infile` | PATH | **Required** | Input proteomics file |
| `--outdir` | PATH | `splits/` | Output directory |
| `--mode` | CHOICE | `development` | `development` or `holdout` |
| `--scenarios` | TEXT | `IncidentOnly` | Scenarios (repeatable) |
| `--n-splits` | INT | 3 | Number of random splits |
| `--val-size` | FLOAT | 0.25 | Validation proportion |
| `--test-size` | FLOAT | 0.25 | Test proportion |
| `--seed-start` | INT | 0 | Starting random seed |

**Examples:**
```bash
# Basic splits
ced save-splits --infile data/proteomics.parquet --outdir splits/

# Multiple splits with holdout
ced save-splits --infile data/proteomics.parquet \
    --mode holdout \
    --n-splits 5 \
    --seed-start 42
```

### `ced convert-to-parquet`

Convert CSV files to Parquet format.

```bash
ced convert-to-parquet --infile INPUT.csv --outfile OUTPUT.parquet
```

---

## Training Commands

### `ced train`

Train a single ML model on a specific split.

```bash
ced train [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | PATH | - | Training config YAML |
| `--infile` | PATH | **Required** | Input proteomics file |
| `--split-dir` | PATH | **Required** | Splits directory |
| `--outdir` | PATH | `results/` | Output directory |
| `--model` | CHOICE | **Required** | Model: `LR_EN`, `RF`, `XGBoost`, `SVM` |
| `--split-seed` | INT | **Required** | Split seed to train on |
| `--run-id` | TEXT | Auto | Run ID (timestamp if not specified) |

**Examples:**
```bash
# Train logistic regression
ced train --infile data/proteomics.parquet \
    --split-dir splits/ \
    --model LR_EN \
    --split-seed 0

# Train with specific config
ced train --config configs/training_config.yaml \
    --infile data/proteomics.parquet \
    --split-dir splits/ \
    --model RF \
    --split-seed 42
```

### `ced train-ensemble`

Train stacking ensemble meta-learner.

```bash
ced train-ensemble [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID with trained base models |
| `--config`, `-c` | PATH | - | Training config YAML |
| `--base-models` | TEXT | Auto | Comma-separated base model names |
| `--split-seed` | INT | 0 | Split seed |

**Examples:**
```bash
# Train ensemble on existing run
ced train-ensemble --run-id 20260127_115115

# Specify base models
ced train-ensemble --run-id 20260127_115115 \
    --base-models LR_EN,RF,XGBoost
```

---

## Results Aggregation

### `ced aggregate-splits`

Aggregate results across multiple split seeds.

```bash
ced aggregate-splits [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID to aggregate |
| `--model` | TEXT | All | Specific model (default: all) |
| `--metrics-only` | FLAG | False | Only aggregate metrics |

**Examples:**
```bash
# Aggregate all models
ced aggregate-splits --run-id 20260127_115115

# Aggregate specific model
ced aggregate-splits --run-id 20260127_115115 --model LR_EN
```

---

## Panel Optimization

### `ced optimize-panel`

Optimize feature panel size via aggregated RFE.

```bash
ced optimize-panel [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID with aggregated results |
| `--model` | TEXT | Auto | Model to optimize |
| `--min-features` | INT | 5 | Minimum panel size |
| `--max-features` | INT | 50 | Maximum panel size |
| `--step` | INT | 1 | RFE step size |

**Examples:**
```bash
# Optimize panel for a model
ced optimize-panel --run-id 20260127_115115 --model LR_EN

# Custom panel range
ced optimize-panel --run-id 20260127_115115 \
    --model RF \
    --min-features 10 \
    --max-features 30
```

### `ced consensus-panel`

Build cross-model consensus panel using Robust Rank Aggregation.

```bash
ced consensus-panel [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID with optimized models |
| `--models` | TEXT | All | Comma-separated models |
| `--top-k` | INT | 20 | Consensus panel size |

**Examples:**
```bash
# Build consensus from all models
ced consensus-panel --run-id 20260127_115115

# Specify models and panel size
ced consensus-panel --run-id 20260127_115115 \
    --models LR_EN,RF,XGBoost \
    --top-k 15
```

---

## Significance Testing

### `ced permutation-test`

Test model significance via label permutation testing.

Tests the null hypothesis that model performance is no better than chance
by comparing observed AUROC against a null distribution from B label permutations.

```bash
ced permutation-test [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID to test |
| `--model` | TEXT | All | Specific model (default: all base models) |
| `--split-seed-start` | INT | 0 | First split seed to test |
| `--n-split-seeds` | INT | 1 | Number of consecutive seeds to test |
| `--n-perms` | INT | 200 | Number of permutations per seed |
| `--metric` | TEXT | `auroc` | Metric (only `auroc` supported) |
| `--n-jobs` | INT | 1 | Parallel jobs |
| `--outdir` | PATH | Auto | Output directory |
| `--random-state` | INT | 42 | Random seed |
| `--aggregate-only` | FLAG | False | Only aggregate existing per-seed results |

**Usage Modes:**

1. **Local parallel mode**: Runs all permutations locally
   ```bash
   ced permutation-test --run-id 20260127_115115 --model LR_EN --n-jobs 4
   ```

2. **HPC mode**: Orchestrator submits one full-command job per seed
   ```bash
   ced run-pipeline --hpc  # includes permutation testing if enabled
   ```

**Output:**
```
results/run_{RUN_ID}/{MODEL}/significance/
    permutation_test_results_seed{N}.csv  # Summary per seed
    null_distribution_seed{N}.csv         # Full null per seed
    aggregated_significance.csv           # Pooled result (multi-seed)
```

**Interpretation:**
- p < 0.05: Strong evidence of generalization above chance
- p < 0.10: Marginal evidence
- p >= 0.10: No evidence above chance

**Examples:**
```bash
# Test all models with 4 parallel jobs
ced permutation-test --run-id 20260127_115115 --n-jobs 4

# Test specific model with 200 permutations
ced permutation-test --run-id 20260127_115115 \
    --model LR_EN \
    --n-perms 200

# Aggregate existing per-seed results only
ced permutation-test --run-id 20260127_115115 \
    --model LR_EN \
    --aggregate-only
```

---

## Evaluation

### `ced eval-holdout`

Evaluate trained models on holdout set.

```bash
ced eval-holdout [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-id` | TEXT | **Required** | Run ID to evaluate |
| `--model` | TEXT | All | Specific model |
| `--holdout-file` | PATH | Auto | Holdout data file |

**Examples:**
```bash
# Evaluate all models on holdout
ced eval-holdout --run-id 20260127_115115

# Evaluate specific model
ced eval-holdout --run-id 20260127_115115 --model ENSEMBLE
```

---

## Configuration

### `ced config validate`

Validate configuration file.

```bash
ced config validate CONFIG_FILE [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--command` | CHOICE | - | Command context: `train`, `splits`, `pipeline` |
| `--strict` | FLAG | False | Treat warnings as errors |

**Examples:**
```bash
# Validate training config
ced config validate configs/training_config.yaml --command train

# Strict validation
ced config validate configs/training_config.yaml --command train --strict
```

### `ced config diff`

Compare two configuration files.

```bash
ced config diff CONFIG1 CONFIG2 [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output`, `-o` | PATH | stdout | Output file for diff |

**Examples:**
```bash
# Compare configs
ced config diff configs/local.yaml configs/hpc.yaml

# Save diff to file
ced config diff configs/v1.yaml configs/v2.yaml --output diff.txt
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CED_RESULTS_DIR` | Override default results directory |
| `CED_DATA_DIR` | Override default data directory |
| `SEED_GLOBAL` | Force global random seed (debugging) |
| `RUN_MODELS` | Override model list |
| `DRY_RUN` | Enable dry-run mode |

---

## Output Directory Structure

```
results/run_{YYYYMMDD_HHMMSS}/
    run_metadata.json              # Run configuration and metadata
    {MODEL}/
        splits/
            split_seed{N}/
                core/
                    val_metrics.csv
                    test_metrics.csv
                    {MODEL}__final_model.joblib
                preds/
                    train_oof__{MODEL}.csv
                    test_preds__{MODEL}.csv
                cv/
                    oof_importance__{MODEL}.csv  # If importance enabled
        aggregated/
            metrics/
                aggregated_metrics.csv
            panels/
                optimized_panel.csv
                drop_column_validation__{MODEL}.csv
            importance/
                oof_importance__{MODEL}.csv
        significance/
            permutation_test_results.csv
            null_distribution.csv
```

---

## See Also

- [ARTIFACTS.md](ARTIFACTS.md) - Complete output file reference
- [FEATURE_SELECTION.md](FEATURE_SELECTION.md) - Feature selection methods
- [ADR-011](../adr/ADR-011-permutation-testing.md) - Permutation testing methodology
