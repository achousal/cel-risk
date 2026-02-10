# E2E Test Runner Guide

**Purpose**: Complete guide for running, interpreting, and maintaining the CeliacRisks E2E test suite.

**Last Updated**: 2026-02-09

---

## Quick Start

```bash
# Navigate to analysis directory
cd /Users/andreschousal/Projects/Elahi_Lab/CeliacRisks/analysis

# Run fast tests only (recommended for development)
python -m pytest tests/e2e/ -v -m "not slow" --tb=short

# Run all tests including slow integration tests
python -m pytest tests/e2e/ -v --tb=short

# Run specific test file
python -m pytest tests/e2e/test_full_pipeline_all_stages.py -v

# Run with coverage report
python -m pytest tests/e2e/ -v --cov=ced_ml --cov-report=term-missing

# Run tests matching a pattern
python -m pytest tests/e2e/ -v -k "calibration"

# Run single test with detailed output
python -m pytest tests/e2e/test_full_pipeline_all_stages.py::TestFullPipelineAllStages::test_full_pipeline_with_ensemble -vv -s
```

---

## Test Suite Structure

### Current E2E Test Coverage (162 tests total)

| Test File | Tests | Speed | Purpose |
|-----------|-------|-------|---------|
| `test_basic_workflow.py` | 12 | Fast | Basic pipeline, splits, data conversion |
| `test_cli_smoke.py` | 17 | Fast | CLI smoke tests and error handling |
| `test_config_system_e2e.py` | 11 | Fast | Config loading, validation, inheritance |
| `test_e2e_calibration_workflows.py` | 10 | Slow | Calibration strategies (OOF, per-fold) |
| `test_e2e_fixed_panel_workflows.py` | 9 | Fast | Fixed panel extraction and validation |
| `test_e2e_multi_model_workflows.py` | 10 | Mixed | Cross-model coordination and consensus |
| `test_e2e_output_structure.py` | 20+ | Fast | Output directory and file structure validation |
| `test_e2e_temporal_workflows.py` | 10+ | Slow | Temporal validation workflows |
| `test_full_pipeline_all_stages.py` | 4 | Slow | Complete pipeline with ensemble |
| `test_holdout_workflow_complete.py` | 10+ | Slow | Holdout evaluation workflows |
| `test_pipeline_*.py` | 40+ | Mixed | Various pipeline stages and integration |
| `test_run_id_*.py` | 10+ | Fast | Run-ID auto-detection and coordination |

### Test Markers

```python
@pytest.mark.slow  # Tests requiring model training (10-60s each)
# No marker = Fast tests (<1s each)
```

**Usage**:
```bash
# Skip slow tests during development
pytest tests/e2e/ -m "not slow"

# Run only slow tests (full integration)
pytest tests/e2e/ -m slow
```

---

## Test Fixtures

### Data Fixtures

| Fixture | Samples | Proteins | Use Case |
|---------|---------|----------|----------|
| `tiny_proteomics_data` | 20 | 5 | Error handling tests |
| `small_proteomics_data` | 180 | 10 | Fast integration tests |
| `minimal_proteomics_data` | 200 | 15 | Standard e2e tests |
| `temporal_proteomics_data` | 200 | 15 | Temporal validation tests |

**Sample Distribution** (minimal_proteomics_data):
- Controls: 150
- Incident CeD: 30
- Prevalent CeD: 20
- Demographics: Balanced age/sex/ethnicity for stratification

### Config Fixtures

| Fixture | CV | Optuna | Features | Use Case |
|---------|-----|--------|----------|----------|
| `ultra_fast_training_config` | 2-fold | No | screen_top_n=6, k=[3] | Full pipeline tests |
| `fast_training_config` | 2-fold | No | screen_top_n=8, k=[3,5] | Integration tests |
| `minimal_training_config` | 2-fold | No | screen_top_n=10, k=[3,5] | Standard tests |
| `fixed_panel_training_config` | 2-fold | No | Fixed panel (3 proteins) | Fixed panel tests |

### Helper Fixtures

```python
SHARED_RUN_ID = "20260128_E2ETEST"  # Consistent run-id for coordination tests

def extract_run_id_from_dir(results_dir: Path) -> str:
    """Extract run_id from results directory."""

def verify_run_metadata(run_dir: Path, expected_model: str, expected_split_seed: int):
    """Verify run_metadata.json structure and content."""
```

---

## Test Categories and Examples

### 1. Full Pipeline Tests

**File**: `test_full_pipeline_all_stages.py`

**What they test**:
- Complete 8-stage pipeline: splits → train → aggregate → ensemble → panel → consensus
- Cross-stage artifact compatibility
- Stage auto-disable logic (e.g., fixed_panel disables panel optimization)
- Error handling for insufficient data

**Example**:
```python
def test_full_pipeline_with_ensemble(
    small_proteomics_data,
    ultra_fast_training_config,
    tmp_path
):
    """
    Run complete pipeline with 2 models, 2 seeds, ensemble training.
    Validates all stages complete successfully and produce expected outputs.
    """
    # Pre-generate splits
    # Run full pipeline with ensemble
    # Verify base models trained
    # Verify ensemble trained and aggregated
```

**Runtime**: 5-8 minutes per test (marked `@pytest.mark.slow`)

### 2. Config System Tests

**File**: `test_config_system_e2e.py`

**What they test**:
- YAML config loading and validation
- CLI override precedence
- Config inheritance (`_base` directive)
- Path resolution (relative to config file)
- Boolean and list parsing from CLI

**Example**:
```python
def test_cli_override_takes_precedence_over_yaml(tmp_path):
    """
    YAML: cv.folds=5
    CLI: --override cv.folds=3
    Expected: cv.folds=3 in run_metadata.json
    """
```

**Runtime**: <30 seconds per test

### 3. Calibration Workflow Tests

**File**: `test_e2e_calibration_workflows.py`

**What they test**:
- OOF-posthoc calibration strategy
- Per-fold calibration strategy
- Calibration plot generation
- Metrics recording (before/after calibration)
- Strategy comparison

**Example**:
```python
def test_oof_posthoc_produces_calibrated_predictions(
    minimal_proteomics_data,
    tmp_path
):
    """
    Train with OOF-posthoc calibration.
    Verify calibrated predictions in [0,1] range.
    Check calibration curve plot exists.
    """
```

**Runtime**: 2-5 minutes per test (slow)

### 4. Fixed Panel Tests

**File**: `test_e2e_fixed_panel_workflows.py`

**What they test**:
- Panel extraction from aggregated results
- Training with pre-defined panel
- Fixed panel metadata recording
- Unbiased validation workflow (discovery → validation split)
- Panel file format validation

**Example**:
```python
def test_discovery_then_validation_workflow(tmp_path):
    """
    1. Train on discovery split with feature selection
    2. Extract stable panel
    3. Train on validation split with fixed panel
    4. Compare performance
    """
```

**Runtime**: <1 minute per test (fast)

### 5. Multi-Model Coordination Tests

**File**: `test_e2e_multi_model_workflows.py`

**What they test**:
- Multiple models sharing run_id
- Cross-model aggregation
- Consensus panel generation (RRA)
- Ensemble auto-detection
- Metadata consistency

**Example**:
```python
def test_consensus_panel_integrates_all_models(tmp_path):
    """
    Train LR_EN and RF.
    Generate consensus panel via RRA.
    Verify panel contains features from both models.
    Check geometric mean ranking.
    """
```

**Runtime**: 1-3 minutes per test (mixed)

### 6. Output Structure Tests

**File**: `test_e2e_output_structure.py`

**What they test**:
- Directory structure matches spec
- Required files exist (predictions, metrics, plots)
- File schemas (column names, data types)
- Metadata completeness
- Artifact references (e.g., OOF predictions used by aggregation)

**Example**:
```python
def test_training_predictions_have_correct_schema(tmp_path):
    """
    Train model.
    Load predictions CSV.
    Verify columns: SAMPLE_ID, y_true, y_pred_proba, y_pred
    Check no missing values, probabilities in [0,1].
    """
```

**Runtime**: <1 minute per test (fast)

### 7. Holdout Workflow Tests

**File**: `test_holdout_workflow_complete.py`

**What they test**:
- Holdout split generation
- Holdout evaluation on trained models
- Holdout predictions schema
- Metrics computation (AUROC, PR-AUC, calibration)
- Error handling (missing indices, index mismatch)

**Example**:
```python
def test_full_holdout_workflow_development_mode(tmp_path):
    """
    1. Generate splits in development mode
    2. Train model
    3. Use test set as pseudo-holdout
    4. Run eval-holdout
    5. Verify predictions and metrics
    """
```

**Runtime**: 2-4 minutes per test (slow)

---

## Common Test Patterns

### Pattern 1: CLI Testing with CliRunner

```python
from click.testing import CliRunner
from ced_ml.cli.main import cli

def test_example(minimal_proteomics_data, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "train",
            "--model", "LR_EN",
            "--infile", str(minimal_proteomics_data),
            "--split-seed", "0",
            "--results-dir", str(tmp_path / "results"),
        ],
        catch_exceptions=False,  # Raise exceptions for debugging
    )

    # Validate exit code
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Validate outputs
    assert (tmp_path / "results" / "LR_EN").exists()
```

### Pattern 2: Pre-Generate Splits

```python
def _generate_splits(runner, data_path, splits_dir, scenario="IncidentOnly", seeds=(0, 1)):
    """Pre-generate splits to avoid mismatch with training config."""
    for seed in seeds:
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile", str(data_path),
                "--outdir", str(splits_dir),
                "--mode", "development",
                "--scenarios", scenario,
                "--n-splits", "1",
                "--seed-start", str(seed),
            ],
        )
        if result.exit_code != 0:
            return result
    return result
```

### Pattern 3: Find Run Directory

```python
def _find_run_dir(results_dir):
    """Return the single run_* directory inside results_dir."""
    run_dirs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    assert len(run_dirs) == 1, f"Expected 1 run dir, found {len(run_dirs)}"
    return run_dirs[0]
```

### Pattern 4: Validate File Schema

```python
import pandas as pd

def test_predictions_schema(tmp_path):
    # Run training
    # ...

    # Load predictions
    preds_path = run_dir / "LR_EN" / "split_seed0" / "preds" / "test" / "predictions.csv"
    preds = pd.read_csv(preds_path)

    # Validate schema
    assert "SAMPLE_ID" in preds.columns
    assert "y_true" in preds.columns
    assert "y_pred_proba" in preds.columns
    assert "y_pred" in preds.columns

    # Validate content
    assert len(preds) > 0
    assert preds["y_pred_proba"].between(0, 1).all()
    assert preds["y_true"].isin([0, 1]).all()
```

---

## Troubleshooting

### Common Issues

#### 1. Tests fail with "FileNotFoundError"

**Symptom**: Cannot find splits, data, or results files.

**Solutions**:
```bash
# Run with verbose output to see paths
pytest tests/e2e/test_full_pipeline_all_stages.py::test_full_pipeline_with_ensemble -vv -s

# Check that tmp_path is used correctly
# Verify splits pre-generation succeeded
# Ensure results_dir passed to all commands
```

#### 2. Tests timeout or run too slowly

**Symptom**: Tests exceed 60s timeout.

**Solutions**:
```bash
# Skip slow tests during development
pytest tests/e2e/ -m "not slow"

# Use smallest fixture (small_proteomics_data)
# Use ultra_fast_training_config (2-fold CV, minimal grids)
# Reduce number of models/seeds tested
```

#### 3. Non-deterministic test failures

**Symptom**: Tests pass/fail randomly.

**Solutions**:
- Check for unseeded randomness: `grep -r "np.random" tests/e2e/`
- Verify fixtures set `random_state=42` or use `rng = np.random.default_rng(42)`
- Ensure split generation uses fixed seeds

#### 4. CLI tests fail with SystemExit

**Symptom**: `result.exit_code != 0`

**Solutions**:
```python
# Print CLI output for debugging
if result.exit_code != 0:
    print("CLI OUTPUT:", result.output)
    if result.exception:
        import traceback
        traceback.print_exception(
            type(result.exception),
            result.exception,
            result.exception.__traceback__,
        )

# Check for:
# - Missing required arguments
# - Invalid config values
# - Missing input files
# - Insufficient permissions (tmp_path)
```

#### 5. Split generation fails

**Symptom**: `save-splits` exits with error.

**Common causes**:
- Dataset too small for stratified splitting (use `minimal_proteomics_data` not `tiny_proteomics_data`)
- Scenario mismatch between splits and training config
- Insufficient samples per stratification group

**Solution**:
```python
# Use pre-generated splits matching training config scenario
_generate_splits(runner, data_path, splits_dir, scenario="IncidentOnly", seeds=(0,1))

# Verify data has enough samples
assert len(pd.read_parquet(data_path)) >= 100
```

---

## Test Maintenance

### Adding New Tests

**When to add**:
1. New CLI command implemented → Add smoke test + integration test
2. New output file format → Add schema validation test
3. Bug fix → Add regression test
4. New feature → Add feature-specific tests

**Where to add**:
- Run-ID workflows → `test_run_id_*.py`
- Output validation → `test_e2e_output_structure.py`
- Config system → `test_config_system_e2e.py`
- Full pipeline → `test_full_pipeline_all_stages.py`
- Specialized workflows → Create new `test_e2e_<feature>_workflows.py`

**Template**:
```python
@pytest.mark.slow  # If requires model training
def test_new_feature(minimal_proteomics_data, fast_training_config, tmp_path):
    """
    Test: Brief description of what's being tested.

    Steps:
    1. Setup (data, splits, config)
    2. Execute command/workflow
    3. Validate outputs
    4. Verify behavior
    """
    # Setup
    runner = CliRunner()
    splits_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"

    # Execute
    result = runner.invoke(cli, [...])
    assert result.exit_code == 0

    # Validate
    run_dir = _find_run_dir(results_dir)
    assert (run_dir / "expected_output.csv").exists()

    # Verify
    df = pd.read_csv(run_dir / "expected_output.csv")
    assert len(df) > 0
```

### Updating Tests After Code Changes

**Checklist**:
1. Update tests validating changed behavior
2. Add new tests for new functionality
3. Remove tests for deprecated features
4. Update expected outputs if schema changes
5. Run full test suite: `pytest tests/e2e/ -v`
6. Check coverage: `pytest tests/e2e/ --cov=ced_ml --cov-report=term-missing`

### Before Commit

```bash
# 1. Run fast tests (< 2 min)
pytest tests/e2e/ -m "not slow" -v

# 2. Run slow tests (< 10 min)
pytest tests/e2e/ -m slow -v

# 3. Check coverage (optional)
pytest tests/e2e/ --cov=ced_ml --cov-report=html
open htmlcov/index.html

# 4. Verify no skipped tests (check for fixture issues)
pytest tests/e2e/ -v | grep SKIPPED
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run fast E2E tests
        run: |
          cd analysis
          pytest tests/e2e/ -v -m "not slow" --tb=short

  slow-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: fast-tests
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run slow E2E tests
        run: |
          cd analysis
          pytest tests/e2e/ -v -m slow --tb=short

  coverage:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: [fast-tests, slow-tests]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests with coverage
        run: |
          cd analysis
          pytest tests/e2e/ --cov=ced_ml --cov-report=xml --cov-fail-under=70
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./analysis/coverage.xml
```

---

## Coverage Targets

| Component | Target | Current (2026-02-09) | Priority |
|-----------|--------|----------------------|----------|
| Data I/O | >= 90% | ~82% | High |
| Feature selection | >= 85% | ~71% | High |
| Model training | >= 80% | ~68% | High |
| Calibration | >= 85% | ~79% | Medium |
| CLI commands | >= 75% | ~63% | Medium |
| Plotting | >= 60% | ~45% | Low |

**Overall Target**: >= 70% (gating for CI/CD)

**Gaps**:
- `cli/train_ensemble.py`: 12% coverage (blocked by stacking bugs)
- `cli/optimize_panel.py`: 7% coverage (blocked by RFE bugs)
- `cli/permutation_test.py`: 5% coverage (needs integration tests)
- `features/rfe.py`: 11% coverage (blocked by implementation gaps)

---

## Performance Benchmarks

### Test Execution Times (local macOS, M1)

| Test Category | Tests | Time (fast only) | Time (all) |
|--------------|-------|------------------|------------|
| Basic workflows | 12 | 15s | 45s |
| CLI smoke tests | 17 | 10s | 30s |
| Config system | 11 | 8s | 20s |
| Calibration workflows | 10 | - | 5m |
| Fixed panel | 9 | 12s | 2m |
| Multi-model | 10 | 20s | 3m |
| Output structure | 20 | 15s | 1m |
| Full pipeline | 4 | - | 8m |
| Holdout workflows | 10 | - | 6m |
| **Total** | **162** | **~2 min** | **~30 min** |

**CI/CD Times** (GitHub Actions, ubuntu-latest):
- Fast tests: ~3 minutes
- Slow tests: ~15 minutes
- Full suite with coverage: ~20 minutes

---

## Related Documentation

- [E2E_TEST_INVENTORY.md](E2E_TEST_INVENTORY.md) - Complete test file listing
- [E2E_TESTING_GUIDE.md](E2E_TESTING_GUIDE.md) - Development patterns and best practices
- [CLI_REFERENCE.md](../docs/reference/CLI_REFERENCE.md) - CLI command reference
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - System architecture
- [CLAUDE.md](../../CLAUDE.md) - Project overview

---

**Maintainer**: Andres Chousal (Chowell Lab)
**Last Updated**: 2026-02-09
