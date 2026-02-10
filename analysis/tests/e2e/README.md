# End-to-End Tests for CeliacRisks ML Pipeline

**Purpose**: Comprehensive E2E test suite for the complete `ced run-pipeline` workflow and all CLI commands.

**Status**: Production-ready with 194 tests across 30 files (2026-02-09).

---

## Quick Start

```bash
# Navigate to analysis directory
cd analysis/

# Fast tests only (recommended for development) - ~2 minutes
pytest tests/e2e/ -v -m "not slow" --tb=short

# All tests - ~5-10 minutes
pytest tests/e2e/ -v --tb=short

# Specific test file
pytest tests/e2e/test_full_pipeline_all_stages.py -v

# With coverage
pytest tests/e2e/ -v --cov=ced_ml --cov-report=term-missing

# Run tests matching pattern
pytest tests/e2e/ -v -k "calibration"

# Single test with detailed output
pytest tests/e2e/test_cli_smoke.py::TestCliSmoke::test_train_smoke -vv -s
```

---

## Test Suite Overview

### Quick Stats

- **Total Tests**: 194 E2E tests
- **Test Files**: 30
- **Coverage**: Full pipeline, CLI commands, config system, calibration, feature selection
- **Runtime**: ~2 min (fast), ~5-10 min (full)
- **Framework**: pytest with Click CliRunner
- **Fixtures**: Deterministic small datasets (10-200 samples, 5-15 proteins)

### Test Categories

| Category | Files | Tests | Speed | Purpose |
|----------|-------|-------|-------|---------|
| **Full Pipeline** | test_full_pipeline_all_stages.py | 4 | Slow | Complete 8-stage workflow |
| **CLI Smoke** | test_cli_smoke.py | 13 | Fast | All CLI commands |
| **Config System** | test_config_system_e2e.py | 11 | Fast | Config loading, validation |
| **Training+Eval** | test_training_evaluation_workflow.py | 8 | Mixed | Train → aggregate → panel → eval |
| **Feature Selection** | test_feature_selection_workflow.py | 11 | Mixed | Model gate → evidence → consensus |
| **Calibration** | test_e2e_calibration_workflows.py | 10 | Slow | OOF-posthoc, per-fold strategies |
| **Fixed Panel** | test_e2e_fixed_panel_workflows.py | 9 | Fast | Panel extraction, validation |
| **Multi-Model** | test_e2e_multi_model_workflows.py | 10 | Mixed | Cross-model coordination |
| **Output Structure** | test_e2e_output_structure.py | 20+ | Fast | Directory/file validation |
| **Temporal** | test_e2e_temporal_workflows.py | 10+ | Slow | Temporal validation |
| **Holdout** | test_holdout_workflow_*.py | 10+ | Mixed | Holdout evaluation |
| **Run-ID** | test_run_id_*.py | 10+ | Fast | Run-ID auto-detection |
| **Pipeline Stages** | test_pipeline_*.py | 40+ | Mixed | Individual stage tests |
| **Basic** | test_basic_workflow.py | 12 | Fast | Basic workflows |

---

## Test Details

### 1. Full Pipeline Tests

**File**: `test_full_pipeline_all_stages.py`

**Purpose**: Test complete end-to-end pipeline with all 8 stages.

**What's tested**:
- Complete workflow: data → splits → train → aggregate → ensemble → panel → consensus → permutation
- Cross-stage artifact compatibility (OOF predictions, importance rankings)
- Stage auto-disable logic (fixed_panel bypasses panel optimization)
- Error handling (insufficient data, missing files)

**Example**:
```bash
pytest tests/e2e/test_full_pipeline_all_stages.py -v
```

---

### 2. CLI Smoke Tests

**File**: `test_cli_smoke.py`

**Purpose**: Fast smoke tests for all major CLI commands to catch integration failures.

**Commands tested**:
- save-splits, train, aggregate-splits
- optimize-panel, consensus-panel, permutation-test
- config validate, config diff

**Features**:
- Fast execution (~60-90s total)
- Minimal fixtures (180 samples, 10 proteins)
- Graceful failure handling
- Error path testing

**Example**:
```bash
pytest tests/e2e/test_cli_smoke.py -v
```

---

### 3. Training + Evaluation Workflow

**File**: `test_training_evaluation_workflow.py`

**Purpose**: Test training → aggregation → panel optimization → holdout evaluation workflow.

**What's tested**:
- Model training with multiple models (LR_EN, RF)
- Shared run_id coordination across models and splits
- Results aggregation across splits
- Panel optimization (with graceful skip when unavailable)
- Holdout evaluation (with graceful skip when missing)
- Deterministic behavior with fixed seeds

**Example**:
```bash
pytest tests/e2e/test_training_evaluation_workflow.py -v
```

**Test classes**:
- `TestTrainingWorkflow` - Model training (2 tests)
- `TestAggregationWorkflow` - Results aggregation (1 test)
- `TestPanelOptimizationWorkflow` - Panel optimization (1 test)
- `TestHoldoutEvaluationWorkflow` - Holdout evaluation (1 test)
- `TestCompleteWorkflow` - Full pipeline (2 tests)
- `TestDeterministicBehavior` - Reproducibility (1 test)

---

### 4. Feature Selection Workflow

**File**: `test_feature_selection_workflow.py`

**Purpose**: Test three-stage feature selection: Model Gate → Per-Model Evidence → RRA Consensus.

**What's tested**:

**Stage 1: Model Gate (Permutation Testing)**
- Permutation test p-value computation
- Multi-model significance validation

**Stage 2: Per-Model Evidence**
- OOF importance file generation
- Stability selection consistency

**Stage 3: RRA Consensus**
- Consensus panel generation
- Geometric mean rank aggregation correctness
- OOF importance integration

**Full Workflow Integration**
- All three stages together
- Cross-stage artifact flow

**Error Handling**
- Insufficient models, missing run_id, zero permutations

**Example**:
```bash
pytest tests/e2e/test_feature_selection_workflow.py -v

# Fast tests only (error handling)
pytest tests/e2e/test_feature_selection_workflow.py -v -m "not slow"

# Specific stage
pytest tests/e2e/test_feature_selection_workflow.py::TestStage1ModelGate -v
```

**Test classes**:
- `TestStage1ModelGate` - Permutation testing (2 tests)
- `TestStage2PerModelEvidence` - OOF importance, stability (2 tests)
- `TestStage3RRAConsensus` - Consensus panel (3 tests)
- `TestFullWorkflow` - All stages (1 test)
- `TestErrorHandling` - Edge cases (3 tests)

---

### 5. Config System Tests

**File**: `test_config_system_e2e.py`

**Purpose**: Test YAML config loading, inheritance, and validation.

**What's tested**:
- YAML config loading
- Config hierarchy (YAML → env vars → CLI flags)
- Validation and error messages
- Config diff functionality

---

### 6. Calibration Workflows

**File**: `test_e2e_calibration_workflows.py`

**Purpose**: Test calibration strategies (OOF-posthoc, per-fold).

**What's tested**:
- OOF-posthoc calibration on held-out predictions
- Per-fold calibration strategies
- Calibration artifact generation

---

### 7. Other Test Suites

**Fixed Panel**: Panel extraction and validation
**Multi-Model**: Cross-model coordination and consensus
**Output Structure**: Directory and file structure validation
**Temporal**: Temporal validation workflows
**Holdout**: Holdout evaluation workflows
**Run-ID**: Run-ID auto-detection and propagation

For a complete test inventory, see [E2E_TEST_INVENTORY.md](../E2E_TEST_INVENTORY.md).

---

## Fixtures and Test Data

### Key Fixtures (from conftest.py)

**Datasets**:
- `tiny_dataset`: 180 samples (90 controls, 45 incident, 45 prevalent), 10 proteins
- `small_dataset`: 500 samples, 50 proteins
- `minimal_dataset`: 50 samples, 5 proteins

**Features**:
- All datasets use deterministic seeds (42)
- First 5 proteins have structured signal for validation
- Demographics: age, BMI, sex, genetic_ethnic_grouping
- Missing ethnicity: 17% (matches real data)

**Splits**:
- 50/25/25 train/val/test ratio (matches production)
- Stratified by CeD status
- Prevalent cases in train only (matches ADR-002)

### Creating New Fixtures

For new tests, prefer reusing existing fixtures. If creating new fixtures:

1. Use small datasets (50-200 samples max)
2. Set deterministic seeds
3. Include structured signal in first N features
4. Follow existing naming conventions
5. Add to conftest.py

---

## Running Tests

### Development Workflow

```bash
# Fast feedback loop (skip slow tests)
pytest tests/e2e/ -v -m "not slow"

# Run single test for debugging
pytest tests/e2e/test_cli_smoke.py::TestCliSmoke::test_train_smoke -vv -s

# Run tests matching pattern
pytest tests/e2e/ -v -k "training"
```

### CI/CD

```bash
# Full test suite (all E2E tests)
pytest tests/e2e/ -v --tb=short

# With coverage
pytest tests/e2e/ -v --cov=ced_ml --cov-report=xml --cov-report=term
```

### Test Markers

- `@pytest.mark.slow`: Tests that train models (10-60s each). Skip with `-m "not slow"`.
- No marker: Fast tests (validation, CLI parsing, file checks).

---

## Troubleshooting

### Common Issues

**1. Tests fail with "XGBoost not installed"**
- Expected on systems without XGBoost
- These tests are skipped automatically

**2. Tests fail with "OpenMP runtime not found" (macOS)**
- Solution: `brew install libomp`
- Or skip XGBoost tests

**3. Tests fail with "Insufficient data for splits"**
- Dataset too small for current test configuration
- Use larger fixture or reduce CV folds

**4. Ensemble tests fail**
- Known issue: stacking meta-learner bug
- Skip with `-k "not ensemble"`

**5. Permutation tests timeout**
- Reduce n_perms for development (use n_perms=10)
- Full runs use n_perms=200 (HPC recommended)

### Debugging Failed Tests

```bash
# Verbose output with stack traces
pytest tests/e2e/test_name.py -vv -s --tb=long

# Drop into debugger on failure
pytest tests/e2e/test_name.py -v --pdb

# Show print statements
pytest tests/e2e/test_name.py -v -s
```

---

## Writing New Tests

### Test Structure Template

```python
import pytest
from click.testing import CliRunner
from ced_ml.cli.main import cli

class TestNewFeature:
    """Test description."""

    def test_basic_functionality(self, tmp_path, tiny_dataset):
        """Test basic success path."""
        runner = CliRunner()

        # Arrange
        infile = tmp_path / "data.parquet"
        tiny_dataset.to_parquet(infile)

        # Act
        result = runner.invoke(cli, [
            "command",
            "--infile", str(infile),
            "--outdir", str(tmp_path / "output")
        ])

        # Assert
        assert result.exit_code == 0
        assert (tmp_path / "output" / "expected_file.csv").exists()

    @pytest.mark.slow
    def test_slow_integration(self, tmp_path, small_dataset):
        """Test that requires training models."""
        # ...
```

### Best Practices

1. **Use appropriate fixtures**: tiny_dataset for fast tests, small_dataset for slow tests
2. **Mark slow tests**: Use `@pytest.mark.slow` for tests that train models
3. **Use tmp_path**: All file I/O should use pytest's tmp_path fixture
4. **Test one thing**: Each test should have a clear, focused purpose
5. **Deterministic**: Use fixed seeds, stable fixtures
6. **Fast by default**: Prefer fast tests; only use slow when necessary
7. **Graceful skips**: Use `pytest.skip()` for optional paths instead of failures
8. **Clear names**: Test names should describe what they test

---

## Maintenance

### Adding New Tests

1. Identify test category (CLI, workflow, validation)
2. Choose appropriate test file or create new one
3. Use existing fixtures when possible
4. Follow naming conventions
5. Add to test inventory (E2E_TEST_INVENTORY.md)
6. Update this README if new category

### Test Review Checklist

- [ ] Uses appropriate fixture (tiny/small/minimal)
- [ ] Marked as `@pytest.mark.slow` if trains models
- [ ] Uses tmp_path for all file I/O
- [ ] Deterministic (fixed seeds)
- [ ] Clear, focused purpose
- [ ] Fast (<5s) or justified as slow
- [ ] Good test name (describes what it tests)
- [ ] Added to inventory if new file

---

## Test Inventory

For a complete detailed inventory of all E2E tests with line counts, test classes, and purposes, see:

[E2E_TEST_INVENTORY.md](../E2E_TEST_INVENTORY.md)

---

## Additional Resources

- **Project docs**: `CeliacRisks/CLAUDE.md` - Project overview and workflows
- **CLI reference**: `analysis/docs/reference/CLI_REFERENCE.md` - Complete CLI documentation
- **Architecture**: `analysis/docs/ARCHITECTURE.md` - System architecture
- **ADRs**: `analysis/docs/adr/` - Architecture decision records
