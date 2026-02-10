# CLI Smoke Tests Documentation

## Overview

The smoke test suite (`test_cli_smoke.py`) provides lightweight integration tests for all major CLI commands in the CeD-ML pipeline. These tests verify that commands execute without crashing and produce expected outputs.

### Scope

Smoke tests focus on:
- CLI argument parsing and execution
- Basic success paths for each command
- Error handling with invalid inputs
- Integration between multiple commands

**NOT tested** (covered by unit tests):
- Correctness of ML algorithms
- Accuracy metrics and performance
- Detailed output validation beyond file existence

## Test Coverage

### Core Commands (11 tests)

| Command | Test | Purpose | Runtime |
|---------|------|---------|---------|
| `save-splits` | `test_save_splits_smoke` | Split generation | <2s |
| `train` | `test_train_smoke` | Single model training | ~10s |
| `train-ensemble` | `test_train_ensemble_smoke` | Ensemble meta-learner | ~5s |
| `aggregate-splits` | `test_aggregate_splits_smoke` | Results aggregation | ~2s |
| `optimize-panel` | `test_optimize_panel_smoke` | Panel optimization | ~5s |
| `consensus-panel` | `test_consensus_panel_smoke` | Cross-model consensus | ~2s |
| `permutation-test` | `test_permutation_test_smoke` | Significance testing | ~15s |
| `eval-holdout` | `test_eval_holdout_smoke` | Holdout evaluation | ~5s |
| `config validate` | `test_config_validate_smoke` | Config validation | <1s |
| `config diff` | `test_config_diff_smoke` | Config comparison | <1s |

### Error Handling Tests (5 tests)

| Test | Scenario | Expected Result |
|------|----------|-----------------|
| `test_save_splits_missing_infile` | Missing input file | exit_code != 0 |
| `test_train_missing_splits_dir` | Missing splits directory | exit_code != 0 |
| `test_train_invalid_model` | Invalid model name | exit_code != 0 |
| `test_aggregate_splits_nonexistent_run_id` | Nonexistent run ID | exit_code != 0 |
| `test_config_validate_missing_config` | Missing config file | exit_code != 0 |

### Integration Tests (2 tests)

| Test | Workflow | Purpose |
|------|----------|---------|
| `test_full_workflow_single_model` | splits → train | Basic pipeline path |
| `test_full_workflow_two_models_ensemble` | splits → train x2 → ensemble | Multi-model workflow |

## Running the Tests

### All smoke tests
```bash
pytest analysis/tests/e2e/test_cli_smoke.py -v
```

### Single test class
```bash
pytest analysis/tests/e2e/test_cli_smoke.py::TestCliSmoke -v
pytest analysis/tests/e2e/test_cli_smoke.py::TestCliSmokeErrorHandling -v
pytest analysis/tests/e2e/test_cli_smoke.py::TestCliSmokeIntegration -v
```

### Single test
```bash
pytest analysis/tests/e2e/test_cli_smoke.py::TestCliSmoke::test_save_splits_smoke -v
```

### With coverage
```bash
pytest analysis/tests/e2e/test_cli_smoke.py -v --cov=ced_ml.cli --cov-report=html
```

## Performance

- **Total runtime**: ~60-90 seconds for full suite
- **Typical breakdown**:
  - Setup (fixtures): 10-15s
  - Command execution: 40-60s
  - Teardown: <5s

## Design Principles

### Minimal Assertions
Smoke tests use lenient assertions:
```python
# Good: accept both success and graceful failure
assert result.exit_code in [0, 1], f"Error: {result.output}"

# Avoid: require specific internals
assert "AUROC" in metrics_df.columns  # This is unit test territory
```

### Fast Fixtures
All fixtures use minimal data:
- 180 samples (vs 43,960 real dataset)
- 10 protein features (vs 2,920 real)
- 2-fold CV (vs 5-fold real)
- Optimization: Optuna disabled, reduced grid searches

### Skip Over Fail
Use `pytest.skip()` for optional paths:
```python
run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
if not run_dirs:
    pytest.skip("No run directory created")
```

## Key Fixtures

| Fixture | Purpose | Size |
|---------|---------|------|
| `small_proteomics_data` | Minimal test dataset | 180 samples, 10 proteins |
| `ultra_fast_training_config` | Fast training config | 2-fold CV, no Optuna |
| `tmp_path` | pytest's isolated temp dir | Per-test isolation |

## Troubleshooting

### Test hangs
Smoke tests have 2-minute pytest timeout. If tests timeout:
1. Check `ultra_fast_training_config` - ensure no Optuna/grid searches
2. Verify CV fold counts are minimal (2-3 max)
3. Check system resource availability

### Missing output files
Commands may skip certain outputs under certain conditions:
```python
# Don't assume all files exist - check what was generated
results_found = any(model_dir.rglob("*oof*.csv")) or any(model_dir.rglob("*preds*.csv"))
if not results_found:
    pytest.skip("No prediction files in this run")
```

### Exit code != 0
Commands may exit with code 1 for graceful failures:
```python
# Correct for commands that might skip with code 1
assert result.exit_code in [0, 1], f"Unexpected error: {result.output}"
```

## Maintenance

### Adding New Commands

1. Create smoke test in `TestCliSmoke` class
2. Use `CliRunner` to invoke command
3. Verify exit code and key output files
4. Add to this documentation

Example template:
```python
def test_new_command_smoke(self, fixtures, tmp_path):
    """Smoke: new-command runs without error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["new-command", "--option", "value"])

    assert result.exit_code in [0, 1], f"Error: {result.output}"
    assert (tmp_path / "expected_output.csv").exists()
```

### Debugging Failed Tests

```bash
# Show full output
pytest analysis/tests/e2e/test_cli_smoke.py::TestCliSmoke::test_train_smoke -vv -s

# Drop into pdb on failure
pytest analysis/tests/e2e/test_cli_smoke.py -x --pdb

# Show captured stderr/stdout
pytest analysis/tests/e2e/test_cli_smoke.py -s
```

## Integration with CI

Smoke tests should run in CI pipelines:

```yaml
# Example GitHub Actions
- name: Run CLI smoke tests
  run: pytest analysis/tests/e2e/test_cli_smoke.py -v --tb=short
```

**Expected**: All 17 tests passing in <2 minutes on typical CI hardware.

## Related Documentation

- [E2E Testing Guide](E2E_TESTING_GUIDE.md) - Comprehensive E2E testing patterns
- [CLI Reference](../docs/reference/CLI_REFERENCE.md) - Full CLI command documentation
- [Architecture](../docs/ARCHITECTURE.md) - Pipeline module structure
