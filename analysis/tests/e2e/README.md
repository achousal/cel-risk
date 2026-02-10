# End-to-End Tests for CeliacRisks ML Pipeline

**Purpose**: Comprehensive E2E test suite for the complete `ced run-pipeline` workflow and all CLI commands.

**Status**: Production-ready with 162 tests, 87.5% pass rate on representative subset (2026-02-09).

---

## Quick Start

```bash
# Navigate to analysis directory
cd /Users/andreschousal/Projects/Elahi_Lab/CeliacRisks/analysis

# Fast tests only (recommended for development) - ~2 minutes
python -m pytest tests/e2e/ -v -m "not slow" --tb=short

# Run all tests - ~30 minutes
python -m pytest tests/e2e/ -v --tb=short

# Run specific test file
python -m pytest tests/e2e/test_full_pipeline_all_stages.py -v

# Run with coverage
python -m pytest tests/e2e/ -v --cov=ced_ml --cov-report=term-missing

# Run tests matching pattern
python -m pytest tests/e2e/ -v -k "calibration"
```

---

## Test Organization

### Test Files (27 total)

| Category | Files | Tests | Purpose |
|----------|-------|-------|---------|
| **Full Pipeline** | `test_full_pipeline_all_stages.py` | 4 | Complete 8-stage workflow |
| **Config System** | `test_config_system_e2e.py` | 11 | Config loading, validation, inheritance |
| **CLI Smoke** | `test_cli_smoke.py` | 17 | CLI commands and error handling |
| **Calibration** | `test_e2e_calibration_workflows.py` | 10 | Calibration strategies |
| **Fixed Panel** | `test_e2e_fixed_panel_workflows.py` | 9 | Panel extraction and validation |
| **Multi-Model** | `test_e2e_multi_model_workflows.py` | 10 | Cross-model coordination |
| **Output Structure** | `test_e2e_output_structure.py` | 20+ | Directory and file validation |
| **Temporal** | `test_e2e_temporal_workflows.py` | 10+ | Temporal validation |
| **Holdout** | `test_holdout_workflow_complete.py` | 10+ | Holdout evaluation |
| **Run-ID** | `test_run_id_*.py` | 10+ | Run-ID auto-detection |
| **Pipeline Stages** | `test_pipeline_*.py` | 40+ | Individual stage tests |
| **Basic** | `test_basic_workflow.py` | 12 | Basic workflows |

**Total**: 162 tests across 27 files

---

## Test Categories

### 1. Full Pipeline Integration (test_full_pipeline_all_stages.py)

**What's tested**:
- Complete 8-stage pipeline: splits → train → aggregate → ensemble → panel → consensus → permutation
- Cross-stage artifact compatibility
- Stage auto-disable logic (e.g., fixed_panel bypasses panel optimization)
- Error handling (insufficient data)

**Example**:
```bash
# Run full pipeline with ensemble
ced run-pipeline \
  --infile data.parquet \
  --models LR_EN,RF \
  --split-seeds 0,1
```

**Tests verify**:
- Base models trained and aggregated
- Ensemble meta-learner trained
- Panel optimization completed
- Consensus panel generated
- All outputs have correct structure

**Status**: 4/4 PASSED

---

### 2. Config System (test_config_system_e2e.py)

**What's tested**:
- YAML config loading
- CLI override precedence
- Config inheritance (`_base`)
- Path resolution (relative to config file)
- Validation error messages

**Example**:
```yaml
# config.yaml
_base: base_config.yaml
cv:
  folds: 5
```

```bash
# CLI override
ced train --config config.yaml --override cv.folds=3

# Expected: cv.folds=3 in run_metadata.json
```

**Status**: 11/11 PASSED

---

### 3. CLI Smoke Tests (test_cli_smoke.py)

**What's tested**:
- All CLI commands execute without error
- Error handling for common failures
- Full workflows (single model, multi-model)

**Commands tested**:
- `ced save-splits`
- `ced train`
- `ced train-ensemble` (fails - known bug)
- `ced aggregate-splits`
- `ced optimize-panel`
- `ced consensus-panel`
- `ced permutation-test` (fails - needs integration)
- `ced eval-holdout` (fails - needs implementation)
- `ced config validate`
- `ced config diff`

**Status**: 13/17 PASSED (4 known failures)

---

### 4. Calibration Workflows (test_e2e_calibration_workflows.py)

**What's tested**:
- OOF-posthoc calibration (primary strategy, ADR-008)
- Per-fold calibration (alternative)
- Calibration plots (reliability diagrams)
- Metrics (ECE, Brier score)

**Strategies**:
```yaml
# OOF-posthoc (recommended)
calibration:
  enabled: true
  method: isotonic
  strategy: oof_posthoc

# Per-fold (alternative)
calibration:
  enabled: true
  method: isotonic
  strategy: per_fold
```

**Status**: 10/10 PASSED

---

### 5. Fixed Panel Workflows (test_e2e_fixed_panel_workflows.py)

**What's tested**:
- Panel extraction from importance rankings
- Training with pre-defined panel
- Discovery-validation workflow

**Workflow**:
```bash
# 1. Discovery: Train with feature selection
ced train --model LR_EN --split-seed 0

# 2. Extract panel
ced optimize-panel --run-id <RUN_ID>

# 3. Validation: Train with fixed panel
ced train --model LR_EN --split-seed 1 --config fixed_panel_config.yaml
```

**Status**: 9/9 PASSED

---

### 6. Multi-Model Coordination (test_e2e_multi_model_workflows.py)

**What's tested**:
- Multiple models sharing run_id
- Independent model aggregation
- Consensus panel (RRA)
- Ensemble auto-detection

**Workflow**:
```bash
# Train multiple models
ced train --model LR_EN,RF,XGBoost --split-seed 0,1

# Generate consensus panel
ced consensus-panel --run-id <RUN_ID>
```

**Status**: 10/10 PASSED

---

## Test Fixtures

All fixtures in `conftest.py`.

### Data Fixtures

| Fixture | Samples | Proteins | Use Case |
|---------|---------|----------|----------|
| `tiny_proteomics_data` | 20 | 5 | Error handling |
| `small_proteomics_data` | 180 | 10 | Fast integration |
| `minimal_proteomics_data` | 200 | 15 | Standard e2e |
| `temporal_proteomics_data` | 200 | 15 | Temporal validation |

**Key features**:
- Deterministic (seed=42)
- Balanced demographics
- Signal in first 3-5 proteins
- Realistic incident/prevalent distinction

### Config Fixtures

| Fixture | CV | Optuna | Features | Use Case |
|---------|-----|--------|----------|----------|
| `ultra_fast_training_config` | 2-fold | No | screen_top_n=6 | Full pipeline |
| `fast_training_config` | 2-fold | No | screen_top_n=8 | Integration |
| `minimal_training_config` | 2-fold | No | screen_top_n=10 | Standard |
| `fixed_panel_training_config` | 2-fold | No | Fixed panel | Fixed panel |

**Speed optimization**:
- 2-fold CV (minimum for stratification)
- No Optuna (manual grids)
- Small feature grids
- Small estimators (30 trees)

---

## Running Tests

### Development Workflow

```bash
# 1. Fast iteration (< 2 min)
pytest tests/e2e/ -m "not slow" -v

# 2. Specific test
pytest tests/e2e/test_full_pipeline_all_stages.py::TestFullPipelineAllStages::test_full_pipeline_with_ensemble -vv -s

# 3. Tests matching pattern
pytest tests/e2e/ -k "calibration" -v

# 4. Full suite with coverage (< 30 min)
pytest tests/e2e/ -v --cov=ced_ml --cov-report=html
open htmlcov/index.html
```

### Pre-Commit Checklist

```bash
# Fast tests
pytest tests/e2e/ -m "not slow" -v

# Full suite (if touching core modules)
pytest tests/e2e/ -v

# Check for skipped tests
pytest tests/e2e/ -v | grep SKIPPED
```

---

## Test Patterns

### CLI Testing

```python
from click.testing import CliRunner
from ced_ml.cli.main import cli

def test_cli_command(minimal_proteomics_data, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["train", "--model", "LR_EN", "--infile", str(minimal_proteomics_data)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
```

### Output Validation

```python
def test_output_structure(tmp_path):
    # Run training
    # ...

    # Find run directory
    run_dirs = [d for d in results_dir.iterdir() if d.name.startswith("run_")]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Validate structure
    assert (run_dir / "LR_EN" / "splits" / "split_seed0" / "model.pkl").exists()
```

### Schema Validation

```python
def test_predictions_schema(tmp_path):
    # Run training, load predictions
    preds = pd.read_csv(preds_path)

    # Validate
    assert set(preds.columns) >= {"SAMPLE_ID", "y_true", "y_pred_proba", "y_pred"}
    assert preds["y_pred_proba"].between(0, 1).all()
```

---

## Known Issues

### Failing Tests (4/32 in representative subset)

1. **test_train_ensemble_smoke** (FAILED)
   - Issue: Stacking meta-learner OOF aggregation bug
   - Workaround: Use `--no-ensemble`
   - Priority: High

2. **test_permutation_test_smoke** (FAILED)
   - Issue: Needs integration validation
   - Workaround: Skip permutation testing
   - Priority: Low

3. **test_eval_holdout_smoke** (FAILED)
   - Issue: Implementation gap
   - Workaround: Use test set evaluation
   - Priority: Medium

4. **test_full_workflow_two_models_ensemble** (FAILED)
   - Issue: Same as test_train_ensemble_smoke
   - Workaround: Test multi-model without ensemble
   - Priority: High

---

## Coverage Metrics

**Overall**: 50% (up from 11% baseline)

**High Coverage** (>= 80%):
- `plotting/calibration.py`: 92%
- `plotting/calibration_reliability.py`: 93%
- `utils/metadata.py`: 93%
- `plotting/roc_pr.py`: 89%
- `plotting/style.py`: 96%

**Needs Improvement** (< 40%):
- `cli/optimize_panel.py`: 4%
- `cli/consensus_panel.py`: 5%
- `cli/permutation_test.py`: 5%
- `cli/eval_holdout.py`: 0%

---

## Documentation

- **E2E_TEST_SUMMARY.md** - Executive summary of test coverage
- **E2E_RUNNER_GUIDE.md** - Detailed execution guide and troubleshooting
- **E2E_TESTING_GUIDE.md** - Development patterns and best practices
- **E2E_TEST_INVENTORY.md** - Complete test file listing
- **E2E_TEST_STATUS.md** - Latest test run results and analysis
- **README.md** (this file) - Quick reference

---

## Maintenance

### Adding Tests

**When to add**:
- New CLI command → Add smoke test + integration test
- New output format → Add schema validation
- Bug fix → Add regression test

**Template**:
```python
@pytest.mark.slow  # If requires model training
def test_new_feature(minimal_proteomics_data, fast_training_config, tmp_path):
    """Test new feature."""
    runner = CliRunner()
    result = runner.invoke(cli, [...])
    assert result.exit_code == 0
```

### Before Commit

```bash
pytest tests/e2e/ -m "not slow" -v  # Fast tests
pytest tests/e2e/ -v                 # Full suite
```

---

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Fast E2E tests
  run: pytest tests/e2e/ -m "not slow" --tb=short
  timeout-minutes: 10

- name: Slow E2E tests
  run: pytest tests/e2e/ -m slow --tb=short
  timeout-minutes: 30
```

---

**Maintainer**: Andres Chousal (Chowell Lab)
**Last Updated**: 2026-02-09
