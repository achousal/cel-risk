# End-to-End Test Implementation Plan for cel-risk ML Pipeline

## Summary

Add end-to-end tests to fill critical gaps:
- **P1**: Full pipeline smoke test with all stages (ensemble, panel optimization, consensus, permutation testing)
- **P2**: Config system e2e (YAML -> CLI override precedence through actual commands)
- **P3**: Complete holdout workflow (splits -> train -> eval-holdout)

## Analysis of Existing Infrastructure

### Current Test Coverage
- **Existing**: 126 E2E tests in 24 files, ~4865 lines of E2E test code
- **Gap P1**: No test runs full `ced run-pipeline` with all stages enabled
- **Gap P2**: Config system has unit tests but no e2e validation through CLI
- **Gap P3**: Holdout has 3 tests but no complete workflow test

### Existing Fixtures (tests/e2e/conftest.py)
```python
small_proteomics_data(tmp_path)  # 180 samples, 10 proteins, balanced demographics
minimal_proteomics_data(tmp_path)  # 200 samples, 15 proteins
fast_training_config(tmp_path)  # 2-fold CV, no Optuna, minimal grids
minimal_splits_config(tmp_path)  # 2 splits, seed_start=42
SHARED_RUN_ID = "20260128_E2ETEST"
```

### Pipeline Orchestration (run_pipeline.py)
8 stages:
1. Generate splits
2. Train base models (per model x seed)
3. Aggregate base models (per model)
4. Train ensemble (per seed)
5. Aggregate ensemble
6. Optimize panel (per model)
7. Generate consensus panel
8. Permutation testing (per model x seed)

### Pipeline Config Structure (pipeline_local.yaml)
```yaml
environment: local
paths:
  infile: ../data/...
  splits_dir: ../splits
  results_dir: ../results
configs:
  splits: configs/splits_config.yaml
  training: configs/training_config.yaml
pipeline:
  models: [LR_EN, RF]
  ensemble: true
  consensus: true
  optimize_panel: true
  permutation_test: false
```

### Config Loading Chain
1. Defaults (config/defaults.py)
2. YAML file (_base support + deep merge)
3. CLI overrides (dot-notation: cv.folds=10)
4. Pydantic validation

---

## Priority 1: Full Pipeline Smoke Test

**Objective**: Test complete end-to-end pipeline with all stages enabled (ensemble, panel optimization, consensus)

**File to create**: `tests/e2e/test_full_pipeline_all_stages.py` (~400 lines)

### Test Classes and Methods

#### Class: TestFullPipelineAllStages
**Purpose**: Validate complete multi-stage pipeline execution

**Test 1**: `test_full_pipeline_with_ensemble_and_consensus()`
- **Runtime target**: 5-8 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - All 8 stages complete without errors
  - Ensemble model trained and aggregated
  - Panel optimization runs for all models
  - Consensus panel generated with RRA rankings
  - Cross-stage artifacts accessible (OOF preds, aggregated importance)
- **Data**: small_proteomics_data (180 samples, fast)
- **Config**: fast_training_config (2-fold CV, no Optuna)
- **Models**: LR_EN, RF (2 models for consensus)
- **Split seeds**: 0, 1 (2 seeds for aggregation)
- **CLI invocation**:
  ```python
  result = runner.invoke(cli, [
      "run-pipeline",
      "--infile", str(data),
      "--split-dir", str(splits_dir),
      "--outdir", str(results_dir),
      "--config", str(config),
      "--models", "LR_EN,RF",
      "--split-seeds", "0,1",
      # All stages enabled (no --no-* flags)
  ], catch_exceptions=False)
  ```
- **Assertions**:
  - Exit code 0
  - Base model dirs exist: `run_*/LR_EN/splits/split_seed{0,1}/`
  - Aggregated dirs exist: `run_*/LR_EN/aggregated/`
  - Ensemble dir exists: `run_*/ENSEMBLE/splits/split_seed{0,1}/`
  - Ensemble aggregated exists: `run_*/ENSEMBLE/aggregated/`
  - Panel optimization outputs: `run_*/LR_EN/panel_optimization/`
  - Consensus panel: `run_*/consensus_panel/`
  - Key files verified:
    - `oof_predictions_pooled.csv` (aggregated)
    - `feature_importance_aggregated.csv` (aggregated)
    - `rfe_curve.csv` (panel optimization)
    - `consensus_ranking.csv` (consensus)

**Test 2**: `test_full_pipeline_with_permutation_testing()`
- **Runtime target**: 10-15 minutes (permutation testing is expensive)
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Permutation testing stage completes
  - p-values computed for all model/seed combinations
  - Permutation results saved correctly
- **Data**: small_proteomics_data
- **Config**: fast_training_config + override permutation params
- **Models**: LR_EN (single model to reduce runtime)
- **Split seeds**: 0 (single seed)
- **Permutation params**: n_perms=50, n_jobs=1 (reduced for speed)
- **CLI invocation**:
  ```python
  result = runner.invoke(cli, [
      "run-pipeline",
      "--infile", str(data),
      "--split-dir", str(splits_dir),
      "--outdir", str(results_dir),
      "--config", str(config),
      "--models", "LR_EN",
      "--split-seeds", "0",
      "--no-ensemble",  # Disable to save time
      "--no-consensus",
      "--no-optimize-panel",
      # Permutation test enabled (default is disabled)
      "--permutation-n-perms", "50",
      "--permutation-n-jobs", "1",
  ], catch_exceptions=False)
  ```
- **Assertions**:
  - Exit code 0
  - Permutation results exist: `run_*/LR_EN/splits/split_seed0/permutation_test/`
  - Files verified:
    - `permutation_results.csv` (p-value, observed metric, null distribution)
    - `permutation_histogram.png` (visualization)

**Test 3**: `test_full_pipeline_validates_cross_stage_compatibility()`
- **Runtime target**: 6-10 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - OOF predictions from training accessible to aggregation
  - Aggregated importance accessible to panel optimization
  - Panel optimization results accessible to consensus
- **Strategy**: Run full pipeline, then verify file existence and content integrity
- **Data**: small_proteomics_data
- **Models**: LR_EN, RF
- **Split seeds**: 0, 1
- **Assertions**:
  - Load and validate OOF predictions schema
  - Load and validate aggregated importance schema
  - Load and validate panel optimization recommendations
  - Load and validate consensus rankings
  - Verify cross-model consistency (same proteins in consensus panel appear in both model's importance rankings)

#### Class: TestFullPipelineErrorHandling
**Purpose**: Validate error handling and graceful failures

**Test 1**: `test_pipeline_handles_fixed_panel_strategy()`
- **Runtime target**: 3-5 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Pipeline auto-disables panel optimization and consensus when using fixed_panel strategy
  - Warning logged about auto-disabled stages
- **Config**: Create fixed_panel config with predefined panel
- **CLI invocation**: Full pipeline with fixed_panel config
- **Assertions**:
  - Panel optimization dir does NOT exist
  - Consensus panel dir does NOT exist
  - Log contains "disabling panel optimization and consensus"

**Test 2**: `test_pipeline_fails_gracefully_on_insufficient_samples()`
- **Runtime target**: <1 minute (fast failure)
- **Marker**: None (fast test)
- **Validates**:
  - Pipeline fails early with clear error when samples insufficient for splits
- **Data**: Create tiny dataset (30 samples total)
- **Assertions**:
  - Exit code != 0
  - Error message contains "insufficient samples" or "split size"

### New Fixtures Needed

**Fixture 1**: `ultra_fast_training_config(tmp_path)`
```python
@pytest.fixture
def ultra_fast_training_config(tmp_path):
    """Minimal config for full pipeline testing with all stages.

    Optimized for speed while maintaining all pipeline stages:
    - 2-fold CV, 1 repeat
    - No Optuna
    - Minimal feature selection (screen_top_n=8, k_grid=[3])
    - Ensemble enabled with fast meta-learner
    """
    config = {
        "scenario": "IncidentOnly",
        "cv": {
            "folds": 2,
            "repeats": 1,
            "inner_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
        },
        "optuna": {"enabled": False},
        "features": {
            "feature_selection_strategy": "multi_stage",
            "kbest_scope": "protein",
            "screen_method": "mannwhitney",
            "screen_top_n": 6,  # Very small for speed
            "k_grid": [3],  # Single k value
            "stability_thresh": 0.6,
            "corr_thresh": 0.85,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {
            "objective": "youden",
            "fixed_spec": 0.95,
        },
        "evaluation": {
            "n_boot": 10,  # Minimal bootstraps
        },
        "lr": {
            "C_min": 0.1,
            "C_max": 10.0,
            "C_points": 2,
            "l1_ratio": [0.5],
            "solver": "saga",
            "max_iter": 500,
        },
        "rf": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
        "ensemble": {
            "method": "stacking",
            "base_models": ["LR_EN", "RF"],
            "meta_model": {
                "type": "logistic_regression",
                "penalty": "l2",
                "C": 1.0,
            },
        },
    }

    config_path = tmp_path / "ultra_fast_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
```

**Fixture 2**: `fixed_panel_config(tmp_path)`
```python
@pytest.fixture
def fixed_panel_config(tmp_path):
    """Config with fixed_panel strategy for testing auto-disable logic."""
    config = {
        "scenario": "IncidentOnly",
        "cv": {"folds": 2, "repeats": 1, "inner_folds": 2},
        "optuna": {"enabled": False},
        "features": {
            "feature_selection_strategy": "fixed_panel",
            "fixed_panel_features": [
                "PROT_000_resid",
                "PROT_001_resid",
                "PROT_002_resid",
            ],
        },
        "lr": {
            "C_min": 0.1,
            "C_max": 10.0,
            "C_points": 2,
            "l1_ratio": [0.5],
        },
    }

    config_path = tmp_path / "fixed_panel_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
```

---

## Priority 2: Config System E2E Tests

**Objective**: Validate config loading chain through actual CLI commands (not just unit tests)

**File to create**: `tests/e2e/test_config_system_e2e.py` (~350 lines)

### Test Classes and Methods

#### Class: TestConfigLoadingChain
**Purpose**: Validate YAML -> override -> CLI precedence

**Test 1**: `test_yaml_config_loads_correctly()`
- **Runtime target**: <30 seconds
- **Marker**: None (fast)
- **Validates**:
  - YAML config values used when no CLI overrides
- **Strategy**:
  - Create custom YAML config with non-default values
  - Run `ced train` with config file
  - Verify run_metadata.json contains YAML values
- **Assertions**:
  - `cv.folds` matches YAML value (not default)
  - `features.screen_top_n` matches YAML value
  - `scenario` matches YAML value

**Test 2**: `test_cli_overrides_yaml_config()`
- **Runtime target**: <30 seconds
- **Marker**: None (fast)
- **Validates**:
  - CLI overrides take precedence over YAML
- **Strategy**:
  - Create YAML config with cv.folds=5
  - Run `ced train` with `--override cv.folds=3`
  - Verify run_metadata.json shows cv.folds=3
- **Assertions**:
  - Overridden value appears in metadata
  - Non-overridden values from YAML preserved

**Test 3**: `test_nested_override_precedence()`
- **Runtime target**: <30 seconds
- **Marker**: None (fast)
- **Validates**:
  - Dot-notation overrides work correctly (features.screen_top_n=100)
- **Strategy**:
  - YAML: features.screen_top_n=500
  - CLI: --override features.screen_top_n=100
  - Verify metadata shows 100
- **Assertions**:
  - Nested override applied
  - Sibling nested values from YAML preserved

**Test 4**: `test_boolean_override_parsing()`
- **Runtime target**: <10 seconds
- **Marker**: None (fast)
- **Validates**:
  - Boolean CLI overrides parsed correctly (true/false/yes/no/1/0)
- **Strategy**:
  - Test all boolean variants
  - Verify metadata contains correct boolean type
- **Assertions**:
  - "true", "True", "yes", "1" -> True
  - "false", "False", "no", "0" -> False

**Test 5**: `test_list_override_parsing()`
- **Runtime target**: <10 seconds
- **Marker**: None (fast)
- **Validates**:
  - List CLI overrides parsed correctly (comma-separated)
- **Strategy**:
  - Override k_grid=[5,10,15]
  - Verify metadata contains list of ints
- **Assertions**:
  - List parsed correctly
  - Element types preserved (int vs str)

#### Class: TestConfigPathResolution
**Purpose**: Validate relative path resolution

**Test 1**: `test_relative_paths_resolved_from_config_dir()`
- **Runtime target**: <30 seconds
- **Marker**: None (fast)
- **Validates**:
  - Relative paths in config resolved relative to config file location
- **Strategy**:
  - Create config in tmp_path/configs/ with infile="../data/test.parquet"
  - Create data in tmp_path/data/test.parquet
  - Run command and verify path resolution worked
- **Assertions**:
  - Command locates data file correctly
  - No FileNotFoundError

**Test 2**: `test_absolute_paths_unchanged()`
- **Runtime target**: <10 seconds
- **Marker**: None (fast)
- **Validates**:
  - Absolute paths in config not modified
- **Strategy**:
  - Config with absolute path
  - Verify metadata shows same absolute path
- **Assertions**:
  - Path unchanged
  - No resolution attempted

#### Class: TestConfigValidation
**Purpose**: Validate Pydantic validation errors surface correctly

**Test 1**: `test_invalid_config_fails_with_clear_error()`
- **Runtime target**: <5 seconds
- **Marker**: None (fast)
- **Validates**:
  - Invalid config values rejected with clear error messages
- **Strategy**:
  - Create config with cv.folds=1 (invalid: < 2)
  - Run `ced train`
  - Verify error message clear and actionable
- **Assertions**:
  - Exit code != 0
  - Error message contains "cv.folds" or "folds"
  - Error message contains "validation" or "invalid"

**Test 2**: `test_missing_required_field_fails_gracefully()`
- **Runtime target**: <5 seconds
- **Marker**: None (fast)
- **Validates**:
  - Missing required config fields produce clear errors
- **Strategy**:
  - Create config missing required field (e.g., cv section)
  - Run command
  - Verify error message identifies missing field
- **Assertions**:
  - Exit code != 0
  - Error message mentions missing field name

#### Class: TestPipelineConfigIntegration
**Purpose**: Validate pipeline config (pipeline_local.yaml) loading

**Test 1**: `test_pipeline_config_loads_all_sections()`
- **Runtime target**: <5 seconds
- **Marker**: None (fast)
- **Validates**:
  - Pipeline config sections (paths, configs, pipeline) loaded correctly
- **Strategy**:
  - Create pipeline config with all sections
  - Mock run-pipeline (dry-run or early exit)
  - Verify config sections accessible
- **Assertions**:
  - paths.infile resolved
  - configs.training resolved
  - pipeline.models parsed as list

**Test 2**: `test_pipeline_config_overrides_propagate_to_training()`
- **Runtime target**: <30 seconds
- **Marker**: None (fast)
- **Validates**:
  - Pipeline-level overrides reach training config
- **Strategy**:
  - Pipeline config specifies training config
  - Training config has cv.folds=5
  - Run with pipeline config
  - Verify training uses cv.folds=5
- **Assertions**:
  - Training metadata shows pipeline config values

### New Fixtures Needed

**Fixture 1**: `custom_config_hierarchy(tmp_path)`
```python
@pytest.fixture
def custom_config_hierarchy(tmp_path):
    """Create a config directory hierarchy for path resolution tests.

    Structure:
        tmp_path/
            configs/
                test_config.yaml (with ../data/test.parquet)
                base_config.yaml (base config for _base testing)
            data/
                test.parquet
    """
    configs_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    configs_dir.mkdir()
    data_dir.mkdir()

    # Create test data
    data = pd.DataFrame({
        "SAMPLE_ID": [f"S{i:04d}" for i in range(100)],
        "Incident_Celiac": [0] * 80 + [1] * 20,
        "age": np.random.randint(30, 70, 100),
        "PROT_000_resid": np.random.randn(100),
        "PROT_001_resid": np.random.randn(100),
        "PROT_002_resid": np.random.randn(100),
    })
    data_path = data_dir / "test.parquet"
    data.to_parquet(data_path, index=False)

    # Create base config
    base_config = {
        "cv": {"folds": 3, "repeats": 1},
        "optuna": {"enabled": False},
    }
    base_config_path = configs_dir / "base_config.yaml"
    with open(base_config_path, "w") as f:
        yaml.dump(base_config, f)

    # Create test config (inherits from base)
    test_config = {
        "_base": "base_config.yaml",
        "infile": "../data/test.parquet",
        "scenario": "IncidentOnly",
        "features": {"screen_top_n": 10},
    }
    test_config_path = configs_dir / "test_config.yaml"
    with open(test_config_path, "w") as f:
        yaml.dump(test_config, f)

    return {
        "config_path": test_config_path,
        "data_path": data_path,
        "base_config_path": base_config_path,
    }
```

---

## Priority 3: Complete Holdout Workflow

**Objective**: Test full holdout workflow end-to-end (splits -> train -> eval-holdout)

**File to create**: `tests/e2e/test_holdout_workflow_complete.py` (~250 lines)

### Test Classes and Methods

#### Class: TestCompleteHoldoutWorkflow
**Purpose**: Validate end-to-end holdout evaluation workflow

**Test 1**: `test_full_holdout_workflow_development_mode()`
- **Runtime target**: 2-3 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Complete workflow: splits (development mode) -> train -> eval-holdout (using test set as holdout)
- **Strategy**:
  - Generate splits in development mode (no true holdout)
  - Train model
  - Use test set indices as holdout for eval-holdout
  - Verify holdout evaluation completes
- **Data**: small_proteomics_data
- **Assertions**:
  - Splits generated (train/val/test)
  - Model trained successfully
  - Holdout evaluation completes
  - Holdout predictions saved
  - Holdout metrics computed (AUROC, PR-AUC, etc.)

**Test 2**: `test_holdout_workflow_with_aggregated_model()`
- **Runtime target**: 4-6 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Holdout evaluation on aggregated model (trained on multiple splits)
- **Strategy**:
  - Train model on 2 splits
  - Aggregate results
  - Evaluate aggregated model on holdout
- **Data**: small_proteomics_data
- **Assertions**:
  - Aggregated model artifact exists
  - Holdout evaluation uses aggregated predictions
  - Metrics reflect aggregated performance

**Test 3**: `test_holdout_predictions_schema_validation()`
- **Runtime target**: 2-3 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Holdout predictions have correct schema and content
- **Strategy**:
  - Run full workflow
  - Load holdout predictions CSV
  - Verify schema matches expected
- **Assertions**:
  - Required columns present: SAMPLE_ID, y_true, y_pred_proba, y_pred
  - No missing values
  - Probability range [0, 1]
  - Sample IDs match holdout indices

#### Class: TestHoldoutWorkflowErrorHandling
**Purpose**: Validate error handling in holdout workflow

**Test 1**: `test_holdout_workflow_missing_indices_fails()`
- **Runtime target**: <1 minute
- **Marker**: None (fast)
- **Validates**:
  - Workflow fails gracefully when holdout indices missing
- **Assertions**:
  - Exit code != 0
  - Error message clear and actionable

**Test 2**: `test_holdout_workflow_index_mismatch_fails()`
- **Runtime target**: 2-3 minutes
- **Marker**: `@pytest.mark.slow`
- **Validates**:
  - Workflow fails when holdout indices don't match data
- **Strategy**:
  - Train on dataset with 180 samples
  - Provide holdout indices [200, 201, 202] (out of range)
  - Verify error caught and reported
- **Assertions**:
  - Exit code != 0
  - Error message mentions index mismatch

### New Fixtures Needed

**Fixture 1**: `holdout_mode_splits_config(tmp_path)`
```python
@pytest.fixture
def holdout_mode_splits_config(tmp_path):
    """Splits config for holdout validation mode (larger holdout set).

    Uses mode='holdout_validation' with separate holdout set.
    """
    config = {
        "mode": "holdout_validation",
        "scenarios": ["IncidentOnly"],
        "n_splits": 1,
        "val_size": 0.15,
        "test_size": 0.15,
        "holdout_size": 0.30,  # Larger holdout set
        "seed_start": 42,
        "train_control_per_case": 5.0,
    }

    config_path = tmp_path / "holdout_splits_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
```

---

## Implementation Order

1. **Start with P2** (config system e2e) - Fastest to implement, foundational for other tests
2. **Then P1** (full pipeline) - Highest value, exercises most code paths
3. **Finally P3** (holdout workflow) - Straightforward once P1 patterns established

---

## Runtime Targets Summary

| Priority | File | Test Classes | Tests | Fast | Slow | Total Runtime |
|----------|------|--------------|-------|------|------|---------------|
| P1 | test_full_pipeline_all_stages.py | 2 | 5 | 1 | 4 | 35-50 min |
| P2 | test_config_system_e2e.py | 4 | 13 | 12 | 1 | 5-10 min |
| P3 | test_holdout_workflow_complete.py | 2 | 5 | 2 | 3 | 12-18 min |
| **Total** | **3 files** | **8 classes** | **23 tests** | **15** | **8** | **52-78 min** |

---

## Expected Artifacts Created

### Test Files (3 new files)
1. `tests/e2e/test_full_pipeline_all_stages.py` (~400 lines)
2. `tests/e2e/test_config_system_e2e.py` (~350 lines)
3. `tests/e2e/test_holdout_workflow_complete.py` (~250 lines)

### Fixtures (5 new fixtures in conftest.py)
1. `ultra_fast_training_config(tmp_path)` - Minimal config for full pipeline
2. `fixed_panel_config(tmp_path)` - Fixed panel strategy config
3. `custom_config_hierarchy(tmp_path)` - Config directory hierarchy for path tests
4. `holdout_mode_splits_config(tmp_path)` - Holdout validation mode config
5. (Optional) Helper function: `verify_pipeline_stage_outputs(run_dir, stage_name, expected_files)`

---

## Validation Strategy

Each test validates:
1. **Exit code** - 0 for success, non-zero for expected failures
2. **Output structure** - Expected directories and files exist
3. **File content** - Key files contain expected schema/values
4. **Cross-stage compatibility** - Outputs from stage N consumed by stage N+1
5. **Error messages** - Failures produce clear, actionable messages

---

## Unresolved questions

None. All information needed for implementation is available from existing test patterns.
