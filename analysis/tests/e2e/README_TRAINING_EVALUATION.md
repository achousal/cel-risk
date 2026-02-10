# Training and Evaluation E2E Tests

**Purpose**: End-to-end tests for the complete training and evaluation workflow.

**File**: `test_training_evaluation_workflow.py`

**Coverage**: train -> aggregate-splits -> optimize-panel -> eval-holdout

---

## Test Classes

### 1. TestTrainingWorkflow

Tests the training stage of the pipeline.

**Tests**:
- `test_train_single_model`: Train one model (LR_EN) with one split
- `test_train_multiple_models`: Train two models (LR_EN, RF) with same split

**Validates**:
- Training outputs (model artifacts, predictions, metrics)
- Run metadata creation and consistency
- Required file structure

**Speed**: Slow (requires model training)

### 2. TestAggregationWorkflow

Tests results aggregation across multiple splits.

**Tests**:
- `test_aggregate_single_model`: Train on 2 splits, then aggregate

**Validates**:
- Aggregated metrics and summary statistics
- Pooled predictions
- Aggregated directory structure

**Speed**: Slow (requires training + aggregation)

### 3. TestPanelOptimizationWorkflow

Tests panel optimization after aggregation.

**Tests**:
- `test_optimize_panel_after_aggregation`: Full train -> aggregate -> optimize-panel workflow

**Validates**:
- Panel optimization outputs
- Panel curves and optimal panel selection
- Handles missing RFE data gracefully

**Speed**: Slow (requires training + aggregation + optimization)

**Note**: Panel optimization may skip if no RFE data available (depends on config).

### 4. TestHoldoutEvaluationWorkflow

Tests holdout evaluation after training.

**Tests**:
- `test_eval_holdout_after_training`: Train model and evaluate on holdout set

**Validates**:
- Holdout metrics
- Holdout predictions
- DCA outputs (if enabled)

**Speed**: Slow (requires training + evaluation)

### 5. TestCompleteWorkflow

Tests the complete end-to-end workflow.

**Tests**:
- `test_complete_workflow_single_model`: All stages with one model
- `test_complete_workflow_multi_model`: All stages with multiple models

**Validates**:
- Full pipeline execution
- Cross-model coordination
- Shared run_id behavior
- Output consistency

**Speed**: Slow (longest tests, full workflow)

**Coverage**: This is the key E2E test covering the entire pipeline.

### 6. TestDeterministicBehavior

Tests deterministic and reproducible behavior.

**Tests**:
- `test_same_seed_produces_same_results`: Verify same seed produces identical results

**Validates**:
- Deterministic training with fixed seeds
- Reproducibility across runs
- Consistent metrics

**Speed**: Slow (requires multiple training runs)

---

## Running Tests

### Run all tests
```bash
pytest tests/e2e/test_training_evaluation_workflow.py -v
```

### Run fast tests only (none in this file, all marked slow)
```bash
pytest tests/e2e/test_training_evaluation_workflow.py -v -m "not slow"
```

### Run specific test class
```bash
pytest tests/e2e/test_training_evaluation_workflow.py::TestCompleteWorkflow -v
```

### Run specific test
```bash
pytest tests/e2e/test_training_evaluation_workflow.py::TestCompleteWorkflow::test_complete_workflow_single_model -v
```

### Run with detailed output
```bash
pytest tests/e2e/test_training_evaluation_workflow.py -vv -s
```

---

## Test Data

All tests use fixtures from `conftest.py`:

- **small_proteomics_data**: 180 samples (120 controls, 48 incident, 12 prevalent), 10 proteins
- **fast_training_config**: 2-fold CV, minimal Optuna trials, small feature grids
- **tmp_path**: Isolated temporary directory for each test

---

## Expected Runtime

| Test Class | Tests | Runtime (approx) |
|------------|-------|------------------|
| TestTrainingWorkflow | 2 | 20-40s |
| TestAggregationWorkflow | 1 | 30-50s |
| TestPanelOptimizationWorkflow | 1 | 40-60s |
| TestHoldoutEvaluationWorkflow | 1 | 30-50s |
| TestCompleteWorkflow | 2 | 60-120s |
| TestDeterministicBehavior | 1 | 40-60s |
| **Total** | **8** | **220-380s (4-6 min)** |

**Note**: Times vary based on hardware and parallelization.

---

## Test Fixtures

### Required Fixtures

From `conftest.py`:
- `small_proteomics_data`: Minimal proteomics dataset (180 samples, 10 proteins)
- `fast_training_config`: Fast training config (2-fold CV, minimal iterations)
- `tmp_path`: Pytest built-in temporary directory fixture

### Fixture Characteristics

**small_proteomics_data**:
- 180 samples total
- 120 controls, 48 incident, 12 prevalent
- 10 protein features (PROT_000_resid ... PROT_009_resid)
- Demographics: age, BMI, sex, Genetic_ethnic_grouping
- Signal in first 3 proteins (separates incident from controls)
- Balanced for stratification (2 sex x 3 age bins)

**fast_training_config**:
- scenario: IncidentOnly
- CV: 2 folds, 1 repeat, 2 inner folds
- Optuna: disabled for speed
- Features: mannwhitney screening, top 8 features, k_grid [3, 5]
- Calibration: enabled, isotonic, oof_posthoc
- Thresholds: youden, fixed_spec 0.95

---

## Output Validation

Each test validates specific outputs:

### Training Outputs
- `core/val_metrics.csv`: Validation metrics
- `core/test_metrics.csv`: Test metrics
- `preds/train_oof__MODEL.csv`: Out-of-fold predictions
- `preds/test_preds__MODEL.csv`: Test predictions
- `run_metadata.json`: Run metadata

### Aggregation Outputs
- `aggregated/*metrics*.csv`: Aggregated metrics
- `aggregated/*pooled*.csv`: Pooled predictions across splits
- `aggregated/plots/`: Aggregated plots (if enabled)

### Panel Optimization Outputs (optional)
- `panel_optimization/*panel*.csv`: Panel selection results
- `panel_optimization/*panel*.png`: Panel curves

### Holdout Evaluation Outputs
- `*metrics*.csv` or `*metrics*.json`: Holdout metrics
- `*preds*.csv`: Holdout predictions
- `*dca*.csv`: Decision curve analysis (if enabled)

---

## Troubleshooting

### Common Issues

**1. "Training failed"**
- Check that fixtures are loaded correctly
- Verify small_proteomics_data has required columns
- Check fast_training_config is valid YAML

**2. "Aggregation failed: No split directories found"**
- Ensure training completed successfully
- Check that split_seed directories exist in model directory
- Verify run_id is correct

**3. "Panel optimization skipped"**
- Expected behavior if no RFE data available
- Depends on feature_selection_strategy in config
- Not an error, just a skip

**4. "Holdout evaluation failed"**
- Check that model artifact (.pkl) exists
- Verify holdout indices file exists
- Ensure infile path is correct

**5. "Same seed produces different results"**
- Check for unseeded randomness in code
- Verify config has random_state set
- May be due to parallel execution (set n_jobs=1)

### Debug Commands

```bash
# Run with full output
pytest tests/e2e/test_training_evaluation_workflow.py::TestCompleteWorkflow::test_complete_workflow_single_model -vv -s

# Run with pdb on failure
pytest tests/e2e/test_training_evaluation_workflow.py::TestTrainingWorkflow::test_train_single_model --pdb

# Run single test with verbose output
pytest tests/e2e/test_training_evaluation_workflow.py::TestTrainingWorkflow::test_train_single_model -vv -s
```

---

## Coverage Gaps Addressed

This test file addresses the following gaps identified in E2E_TEST_INVENTORY.md:

1. **Complete training-to-evaluation workflow**: Full pipeline test covering train -> aggregate -> optimize-panel -> eval-holdout
2. **Holdout evaluation**: Direct tests for `ced eval-holdout` command (previously untested)
3. **Multi-model coordination**: Tests for shared run_id across multiple models
4. **Deterministic behavior**: Explicit test for reproducibility with same seed

---

## Integration with Existing Tests

This test file complements existing e2e tests:

- `test_full_pipeline_all_stages.py`: Tests `ced run-pipeline` (orchestrator)
- `test_training_evaluation_workflow.py`: Tests individual commands in sequence (NEW)
- `test_run_id_*.py`: Tests run_id auto-detection and metadata
- `test_pipeline_holdout.py`: Tests holdout split generation
- `test_holdout_workflow_complete.py`: Tests holdout workflow (if exists)

**Distinction**: This file tests the manual workflow where users run individual commands (`ced train`, `ced aggregate-splits`, etc.) rather than the orchestrated `ced run-pipeline` command.

---

## Future Enhancements

Potential additions:

1. **Error recovery tests**: Interrupted training, corrupted outputs, partial results
2. **Large-scale tests**: Use larger fixtures (1000+ samples, 100+ proteins) to test performance
3. **Parallel execution tests**: Multi-model training with parallel execution
4. **Config validation tests**: Test various config combinations
5. **Cross-validation tests**: Test different CV strategies (stratified, grouped, temporal)

---

**Last Updated**: 2026-02-09
**Maintainer**: Andres Chousal (Chowell Lab)
