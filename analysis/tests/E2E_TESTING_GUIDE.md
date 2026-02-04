# E2E Testing Guide

**Purpose**: Practical guide for running, maintaining, and extending end-to-end tests for the CeliacRisks ML pipeline.

**Status**: Production-ready test suite with 1,271 tests covering full pipeline workflows (2026-01-28).

---

## Quick Start

```bash
# Run all E2E tests (fast only)
cd analysis/
pytest tests/test_e2e_*.py -v -m "not slow"

# Run specific workflow tests
pytest tests/test_e2e_run_id_workflows.py -v
pytest tests/test_e2e_calibration_workflows.py -v

# Run slow integration tests (includes full training)
pytest tests/test_e2e_runner.py -v -m slow

# Run with coverage
pytest tests/test_e2e_*.py --cov=ced_ml --cov-report=term-missing

# Debug mode (verbose output)
pytest tests/test_e2e_runner.py::TestE2ERunner::test_full_pipeline -vv -s
```

---

## Test Organization

### Test File Structure

| File | Purpose | Classes | Speed |
|------|---------|---------|-------|
| `test_e2e_runner.py` | Core pipeline workflows | 11 classes | Mixed (fast + slow) |
| `test_e2e_pipeline.py` | Basic integration tests | 2 classes | Mixed |
| `test_e2e_fixed_panel_workflows.py` | Panel extraction and validation | 4 classes | Fast |
| `test_e2e_calibration_workflows.py` | Calibration strategies | 5 classes | Slow |
| `test_e2e_temporal_workflows.py` | Temporal validation | 5 classes | Slow |
| `test_e2e_run_id_workflows.py` | Run-ID auto-detection | 7 classes | Fast |
| `test_e2e_multi_model_workflows.py` | Cross-model workflows | 6 classes | Fast |
| `test_e2e_output_structure.py` | Output validation | 5 classes | Fast |

### Test Markers

- `@pytest.mark.slow`: Tests that train models (10-60s each). Skip with `-m "not slow"` for fast development.
- No marker: Fast tests (validation, CLI parsing, file existence checks).

### Test Isolation

Each test:
- Uses isolated `tmp_path` fixture for outputs
- Creates independent train/val/test splits
- Cleans up artifacts automatically
- Runs deterministically with fixed seeds

---

## Running Tests Locally

### Development Workflow

```bash
# 1. Install dev dependencies
pip install -e ".[dev]"

# 2. Fast iteration (skip slow tests)
pytest tests/test_e2e_*.py -v -m "not slow"

# 3. Run slow tests before commit
pytest tests/test_e2e_runner.py -v -m slow

# 4. Check coverage
pytest tests/test_e2e_*.py --cov=ced_ml --cov-report=html
open htmlcov/index.html
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run fast E2E tests
  run: pytest tests/test_e2e_*.py -v -m "not slow"

- name: Run slow E2E tests
  run: pytest tests/test_e2e_runner.py -v -m slow
  timeout-minutes: 30
```

### HPC Testing

```bash
# Test HPC pipeline execution (dry run mode)
cd analysis/
ced run-pipeline --hpc --dry-run

# Validate generated submission scripts
ls -la hpc_jobs/
cat hpc_jobs/CeD_*.lsf
```

---

## Test Design Patterns

### 1. Minimal Fixtures

```python
@pytest.fixture
def minimal_proteomics_data(tmp_path):
    """Create smallest viable proteomics dataset (10 proteins, 100 samples)."""
    np.random.seed(42)
    data = {
        "SampleID": [f"S{i:03d}" for i in range(100)],
        "Age": np.random.uniform(20, 80, 100),
        "BMI": np.random.uniform(18, 35, 100),
        "Sex": np.random.choice(["M", "F"], 100),
        "Genetic ethnic grouping": np.random.choice(
            ["European", "African", "Asian", "Missing"], 100
        ),
        "CaseControl_final": [1] * 20 + [0] * 80,
    }
    for i in range(10):
        data[f"Protein_{i}_resid"] = np.random.randn(100)
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.parquet"
    df.to_parquet(file_path, index=False)
    return file_path
```

**Why**: Fast tests (< 1s), deterministic, isolated state.

### 2. CLI Testing Pattern

```python
from click.testing import CliRunner
from ced_ml.cli.main import cli

def test_cli_command(minimal_proteomics_data, tmp_path):
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
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert (tmp_path / "results" / "LR_EN").exists()
```

**Why**: Tests real CLI interface, captures output, validates exit codes.

### 3. Output Validation Pattern

```python
def test_training_outputs(minimal_proteomics_data, tmp_path):
    # Run training
    result = runner.invoke(cli, ["train", ...])
    assert result.exit_code == 0

    # Validate output structure
    results_dir = tmp_path / "results" / "LR_EN" / "split_seed0"
    assert (results_dir / "model.pkl").exists()
    assert (results_dir / "preds" / "test" / "predictions.csv").exists()
    assert (results_dir / "plots" / "roc_curve.png").exists()

    # Validate predictions format
    preds = pd.read_csv(results_dir / "preds" / "test" / "predictions.csv")
    assert "y_true" in preds.columns
    assert "y_pred_proba" in preds.columns
    assert len(preds) > 0
```

**Why**: Catches regressions in output structure, file naming, data formats.

### 4. Slow Integration Tests

```python
@pytest.mark.slow
def test_full_pipeline(minimal_proteomics_data, tmp_path):
    """Full training workflow with real model (slow: ~20s)."""
    runner = CliRunner()

    # Train model
    result = runner.invoke(cli, ["train", "--model", "LR_EN", ...])
    assert result.exit_code == 0

    # Aggregate results
    result = runner.invoke(cli, ["aggregate-splits", ...])
    assert result.exit_code == 0

    # Validate aggregated outputs
    agg_dir = tmp_path / "results" / "LR_EN" / "aggregated"
    assert (agg_dir / "metrics_summary.json").exists()
    assert (agg_dir / "calibration_plot.png").exists()
```

**Why**: Tests real model training, multi-stage workflows, integration points.

---

## Fixture Guide

### Core Fixtures

| Fixture | Purpose | Speed | Scope |
|---------|---------|-------|-------|
| `tmp_path` | Isolated temp directory (pytest built-in) | Fast | function |
| `minimal_proteomics_data` | 10 proteins, 100 samples | Fast | function |
| `training_config` | Minimal training config (3-fold CV, 5 Optuna trials) | Fast | function |
| `splits_config` | Minimal splits config (2 splits, 0.3 test size) | Fast | function |

### Custom Fixtures (Examples)

```python
@pytest.fixture
def training_config(tmp_path):
    """Minimal training config for fast tests."""
    return {
        "cv": {"n_outer": 3, "n_repeats": 1, "n_inner": 2},
        "optuna": {"enabled": True, "n_trials": 5, "sampler": "tpe"},
        "features": {
            "feature_selection_strategy": "hybrid_stability",
            "screen_top_n": 10,
            "k_grid": [5],
            "stability_thresh": 0.5,
        },
        "calibration": {"enabled": True, "strategy": "per_fold", "method": "isotonic"},
        "thresholds": {"objective": "fixed_spec", "fixed_spec": 0.95},
    }

@pytest.fixture
def run_id():
    """Generate unique run ID for test isolation."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
```

### Fixture Reuse

```python
# conftest.py (shared fixtures)
@pytest.fixture
def minimal_proteomics_data(tmp_path):
    """Shared across all test files."""
    # ... (see above)

# test_e2e_run_id_workflows.py
def test_aggregate_with_run_id(minimal_proteomics_data, tmp_path):
    # Fixture automatically injected
    assert minimal_proteomics_data.exists()
```

---

## Troubleshooting

### Common Issues

**1. Tests fail with "FileNotFoundError"**
```bash
# Symptom: Cannot find data file or results directory
# Solution: Check tmp_path isolation and file path construction
pytest tests/test_e2e_runner.py::test_full_pipeline -vv -s
# Inspect printed paths for errors
```

**2. Tests timeout or run too slowly**
```bash
# Symptom: Tests exceed 60s timeout
# Solution: Skip slow tests during development
pytest tests/test_e2e_*.py -v -m "not slow"
# Or reduce fixture size (fewer proteins, fewer CV folds)
```

**3. Non-deterministic test failures**
```bash
# Symptom: Tests pass/fail randomly
# Solution: Check for unseeded randomness
grep -r "np.random" tests/  # Find unseeded calls
# Add explicit seeds in fixtures and test setup
```

**4. CLI tests fail with "SystemExit"**
```bash
# Symptom: result.exit_code != 0
# Solution: Print CLI output for debugging
print(result.output)
print(result.exception)
# Check for missing required arguments or invalid config
```

**5. Aggregation tests fail with "No split_seed* directories found"**
```bash
# Symptom: aggregate-splits cannot find training outputs
# Solution: Ensure training completed successfully first
assert (tmp_path / "results" / "LR_EN" / "split_seed0" / "model.pkl").exists()
# Check for training errors before aggregation
```

### Debug Commands

```bash
# Print detailed test output
pytest tests/test_e2e_runner.py::test_full_pipeline -vv -s

# Run single test with pdb on failure
pytest tests/test_e2e_runner.py::test_full_pipeline --pdb

# List all tests without running
pytest tests/test_e2e_*.py --collect-only

# Run tests matching pattern
pytest tests/test_e2e_*.py -k "calibration"
```

---

## Best Practices

### Do's

1. **Isolate state**: Use `tmp_path` for all file I/O (no shared state between tests).
2. **Deterministic**: Fix seeds (`np.random.seed(42)`, `random_state=42`).
3. **Fast by default**: Mark slow tests with `@pytest.mark.slow`.
4. **Validate outputs**: Check file existence, format, and content.
5. **Test CLI interface**: Use `CliRunner` to test real user workflows.
6. **Minimal fixtures**: Use smallest viable data (10 proteins, 3-fold CV).
7. **Clear assertions**: Include failure messages (`assert x, f"Expected {x}"`)
8. **Test error paths**: Validate failure modes (missing files, invalid inputs).

### Don'ts

1. **No shared state**: Avoid writing to project directories during tests.
2. **No network calls**: Mock external APIs or skip tests requiring network.
3. **No hardcoded paths**: Use `tmp_path` or fixture-generated paths.
4. **No long-running tests**: Keep fast tests < 1s, slow tests < 60s.
5. **No test interdependencies**: Each test must run independently.
6. **No debug prints**: Remove `print()` statements before commit (use logging if needed).
7. **No large datasets**: Keep fixtures minimal (< 100 samples, < 20 proteins).
8. **No unstable selectors**: Avoid timing-dependent assertions.

### Performance Optimization

```python
# Bad: Trains full model for each test (20s x 10 tests = 200s)
def test_predictions_format():
    result = train_full_model()  # Slow!
    assert result["predictions"].shape[1] == 2

# Good: Test prediction logic directly (< 1s)
def test_predictions_format():
    mock_preds = np.array([[0.1, 0.9], [0.8, 0.2]])
    df = create_predictions_df(mock_preds)
    assert df.shape[1] == 2
```

### Test Maintenance

1. **Run tests before commit**: `pytest tests/test_e2e_*.py -v`
2. **Update tests with code changes**: Keep tests synced with CLI/API changes.
3. **Add regression tests**: New bugs require new tests to prevent recurrence.
4. **Monitor coverage**: Aim for >= 80% coverage on core logic.
5. **Review test output**: Check for warnings, deprecations, resource leaks.
6. **Prune obsolete tests**: Remove tests for deprecated features.

---

## Test Execution Strategy

### Local Development

```bash
# Fast iteration loop (< 10s)
pytest tests/test_e2e_run_id_workflows.py -v -m "not slow"

# Pre-commit validation (< 60s)
pytest tests/test_e2e_*.py -v -m "not slow"

# Full validation before PR (< 10 min)
pytest tests/ -v --cov=ced_ml --cov-report=term-missing
```

### CI/CD Pipeline

```bash
# Stage 1: Fast tests (< 2 min, fail fast)
pytest tests/test_e2e_*.py -v -m "not slow"

# Stage 2: Slow tests (< 10 min, parallel if possible)
pytest tests/test_e2e_runner.py -v -m slow

# Stage 3: Coverage report (gating for >= 80%)
pytest tests/ --cov=ced_ml --cov-report=xml --cov-fail-under=80
```

### HPC Validation

```bash
# Test HPC submission workflow locally (dry run)
cd analysis/
ced run-pipeline --hpc --dry-run --models LR_EN --split-seeds 0

# Submit jobs to HPC scheduler
ced run-pipeline --hpc --models LR_EN --split-seeds 0,1,2

# Monitor job status
bjobs -w | grep CeD_

# Check job output logs
ls -la logs/hpc/
```

---

## Coverage Targets

| Component | Target | Current | Priority |
|-----------|--------|---------|----------|
| Data I/O | >= 90% | 82% | High |
| Feature selection | >= 85% | 71% | High |
| Model training | >= 80% | 68% | High |
| Calibration | >= 85% | 79% | Medium |
| CLI commands | >= 75% | 63% | Medium |
| Plotting | >= 60% | 45% | Low |

**Gap Analysis**:
- Missing: Holdout evaluation E2E tests
- Missing: Panel optimization validation tests (edge cases)
- Missing: Error recovery tests (interrupted training, corrupted outputs)
- Missing: Multi-user HPC coordination tests

---

## Related Documentation

- [E2E_TEST_INVENTORY.md](E2E_TEST_INVENTORY.md) - Test file listing and coverage status
- [analysis/docs/reference/CLI_REFERENCE.md](../docs/reference/CLI_REFERENCE.md) - CLI command reference
- [analysis/docs/reference/FEATURE_SELECTION.md](../docs/reference/FEATURE_SELECTION.md) - Feature selection workflows
- [CLAUDE.md](../../CLAUDE.md) - Project overview and workflows

---

**Last Updated**: 2026-01-28
**Maintainer**: Andres Chousal (Chowell Lab)
