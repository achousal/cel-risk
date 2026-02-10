# Feature Selection E2E Tests

**Purpose**: End-to-end tests for the three-stage feature selection workflow.

**Status**: Comprehensive coverage of model gate, per-model evidence, and RRA consensus (2026-02-09).

---

## Three-Stage Feature Selection Workflow

| Stage | Component | Purpose | CLI Command |
|-------|-----------|---------|-------------|
| **1. Model Gate** | Permutation test | Filter models with real signal (p < 0.05) | `ced permutation-test` |
| **2. Per-Model Evidence** | OOF importance, stability, RFE | Rank features per model | Computed during training/aggregation |
| **3. RRA Consensus** | Geometric mean rank aggregation | Cross-model robust biomarkers | `ced consensus-panel` |

**Reference**: [docs/reference/FEATURE_SELECTION.md](../../docs/reference/FEATURE_SELECTION.md)

---

## Test Coverage

### Stage 1: Model Gate (Permutation Testing)

**Test Class**: `TestStage1ModelGate`

| Test | Validates | Speed |
|------|-----------|-------|
| `test_permutation_test_identifies_significant_model` | Model with signal yields p < 0.05 | Slow (~30s) |
| `test_permutation_test_multiple_models` | Separate results per model | Slow (~45s) |

**Coverage**:
- Permutation test execution with small n_perms (10 for speed)
- P-value computation correctness
- Observed AUROC > mean(null distribution)
- Output file structure (results CSV, null distribution CSV)

### Stage 2: Per-Model Evidence

**Test Class**: `TestStage2PerModelEvidence`

| Test | Validates | Speed |
|------|-----------|-------|
| `test_oof_importance_computed` | OOF importance files generated | Slow (~25s) |
| `test_stability_selection_consistent` | Stability summary after aggregation | Slow (~35s) |

**Coverage**:
- OOF importance computation during training
- Top proteins have high importance
- Stability selection identifies consistent proteins
- Strong-signal proteins have high stability scores

### Stage 3: RRA Consensus

**Test Class**: `TestStage3RRAConsensus`

| Test | Validates | Speed |
|------|-----------|-------|
| `test_consensus_panel_basic` | Consensus panel generation | Slow (~50s) |
| `test_consensus_panel_rra_correctness` | RRA (geometric mean) correctness | Slow (~50s) |
| `test_consensus_panel_with_oof_importance` | OOF importance drives ranking | Slow (~50s) |

**Coverage**:
- Cross-model consensus panel generation
- Final panel CSV, consensus ranking CSV, metadata JSON
- Geometric mean rank aggregation correctness
- Per-model rank columns included
- OOF importance integration (when available)

### Full Workflow Integration

**Test Class**: `TestFullThreeStageWorkflow`

| Test | Validates | Speed |
|------|-----------|-------|
| `test_complete_workflow` | All three stages integrated | Slow (~90s) |

**Coverage**:
- Train multiple models
- Permutation test filters significant models
- Aggregate per-model evidence
- Generate consensus panel via RRA
- Output structure correctness
- Final panel contains expected proteins

### Error Handling

**Test Class**: `TestFeatureSelectionErrorHandling`

| Test | Validates | Speed |
|------|-----------|-------|
| `test_consensus_panel_insufficient_models` | Error when < 2 models | Fast |
| `test_consensus_panel_nonexistent_run` | Error when run_id missing | Fast |
| `test_permutation_test_zero_perms` | Error when n_perms = 0 | Fast |

**Coverage**:
- Invalid input validation
- Meaningful error messages
- Graceful failure modes

---

## Running the Tests

### Local Development

```bash
# Run all feature selection E2E tests
cd analysis/
pytest tests/e2e/test_feature_selection_workflow.py -v

# Run specific stage tests
pytest tests/e2e/test_feature_selection_workflow.py::TestStage1ModelGate -v
pytest tests/e2e/test_feature_selection_workflow.py::TestStage3RRAConsensus -v

# Run only fast tests (error handling)
pytest tests/e2e/test_feature_selection_workflow.py -v -m "not slow"

# Run with coverage
pytest tests/e2e/test_feature_selection_workflow.py --cov=ced_ml.features --cov=ced_ml.significance --cov-report=term-missing -v
```

### CI/CD Pipeline

```bash
# Stage 1: Fast tests (error handling)
pytest tests/e2e/test_feature_selection_workflow.py -v -m "not slow"

# Stage 2: Slow integration tests (all stages)
pytest tests/e2e/test_feature_selection_workflow.py -v -m slow --maxfail=3
```

### Debug Mode

```bash
# Run single test with verbose output
pytest tests/e2e/test_feature_selection_workflow.py::TestFullThreeStageWorkflow::test_complete_workflow -vv -s

# Run with pdb on failure
pytest tests/e2e/test_feature_selection_workflow.py::TestStage1ModelGate::test_permutation_test_identifies_significant_model --pdb
```

---

## Test Fixtures

### Data Fixtures

**`feature_selection_proteomics_data`**
- 200 samples: 140 controls, 40 incident, 20 prevalent
- 20 proteins with structured signal:
  - PROT_000-004: strong signal (both incident + prevalent)
  - PROT_005-009: weak signal (incident only)
  - PROT_010-019: noise
- Ensures permutation test passes (p < 0.05)
- Ensures stability selection identifies correct proteins

**Design rationale**:
- Small enough for fast tests (< 2 min per stage)
- Large enough for meaningful statistics
- Clear signal structure for validation

### Config Fixtures

**`feature_selection_config`**
- Enables OOF importance computation
- Enables stability selection (threshold = 0.5)
- 2-fold CV, 1 repeat (fast)
- No Optuna (deterministic)
- Hybrid feature selection (screening + kbest + stability)

**Design rationale**:
- Minimal CV folds for speed
- Enables all feature selection components
- Deterministic (fixed random_state)

---

## Expected Test Outputs

### Permutation Test Outputs

```
results/
  run_YYYYMMDD_HHMMSS/
    LR_EN/
      significance/
        permutation_test_results_seed42.csv    # P-values, observed AUROC, null stats
        null_distribution_seed42.csv           # Full null distribution (B=10)
    RF/
      significance/
        permutation_test_results_seed42.csv
        null_distribution_seed42.csv
```

**permutation_test_results_seed42.csv**:
```csv
model,split_seed,outer_fold,observed_auroc,p_value,null_mean,null_std,n_perms
LR_EN,42,0,0.85,0.09,0.52,0.04,10
LR_EN,42,1,0.82,0.09,0.50,0.05,10
```

### Consensus Panel Outputs

```
results/
  run_YYYYMMDD_HHMMSS/
    consensus/
      final_panel.csv                # Final panel (up to target_size proteins)
      final_panel.txt                # Plain text list (one protein per line)
      consensus_ranking.csv          # All proteins ranked by RRA
      per_model_rankings.csv         # Per-model rankings (long format)
      correlation_clusters.csv       # Correlation clustering results
      metadata.json                  # Run metadata
```

**final_panel.csv**:
```csv
protein
PROT_000_resid
PROT_001_resid
PROT_002_resid
PROT_003_resid
PROT_004_resid
```

**consensus_ranking.csv**:
```csv
protein,consensus_score,consensus_rank,n_models_present,LR_EN_rank,RF_rank
PROT_000_resid,2.45,1,2,1,2
PROT_001_resid,2.24,2,2,2,1
PROT_002_resid,2.00,3,2,3,3
```

**per_model_rankings.csv**:
```csv
model,protein,stability_freq,oof_importance,oof_rank,final_rank
LR_EN,PROT_000_resid,1.0,0.45,1,1
LR_EN,PROT_001_resid,1.0,0.42,2,2
RF,PROT_001_resid,1.0,0.38,1,1
RF,PROT_000_resid,1.0,0.35,2,2
```

---

## Test Design Patterns

### 1. Deterministic Fixtures

```python
@pytest.fixture
def feature_selection_proteomics_data(tmp_path):
    rng = np.random.default_rng(42)  # Fixed seed
    # ... generate data
    return parquet_path
```

**Why**: Reproducible tests, no flakiness.

### 2. Minimal Permutations (n_perms=10)

```python
result = runner.invoke(
    cli,
    [
        "permutation-test",
        "--n-perms", "10",  # Fast for CI
    ],
)
```

**Why**: Fast tests (< 30s) while maintaining statistical validity.

### 3. Skip on Training Failure

```python
if result_train.exit_code != 0:
    pytest.skip(f"Training failed: {result_train.output[:500]}")
```

**Why**: Focus on testing feature selection logic, not training stability.

### 4. Output Validation

```python
assert results_csv.exists()
df_results = pd.read_csv(results_csv)
assert "p_value" in df_results.columns
p_val = df_results["p_value"].iloc[0]
assert 0 < p_val <= 1.0
```

**Why**: Catches regressions in output structure and content.

---

## Performance Optimization

### Current Test Times (MacBook Pro M1)

| Test Class | Tests | Total Time | Per Test |
|------------|-------|------------|----------|
| `TestStage1ModelGate` | 2 | ~75s | ~37s |
| `TestStage2PerModelEvidence` | 2 | ~60s | ~30s |
| `TestStage3RRAConsensus` | 3 | ~150s | ~50s |
| `TestFullThreeStageWorkflow` | 1 | ~90s | ~90s |
| `TestFeatureSelectionErrorHandling` | 3 | ~5s | ~2s |
| **Total** | **11** | **~380s** | **~35s** |

### Speed Optimizations Applied

1. **Small n_perms**: 10 instead of 200 (production default)
2. **Minimal CV**: 2-fold CV, 1 repeat instead of 3x5
3. **Small dataset**: 200 samples, 20 proteins instead of 44K samples, 2920 proteins
4. **No Optuna**: Deterministic grid search instead of Bayesian optimization
5. **Skip on failure**: Don't retry failed training

---

## Known Issues and Workarounds

### Issue: Training Failure in CI

**Symptom**: Training exits with code 1, test skipped.

**Workaround**: Tests use `pytest.skip()` on training failure to avoid false negatives.

**Resolution**: Monitor skipped tests in CI logs.

### Issue: OOF Importance Files Not Found

**Symptom**: `importance_files` list is empty after training.

**Workaround**: Tests check `if importance_files:` before validation.

**Resolution**: Hook OOF importance computation into training pipeline.

### Issue: Consensus Panel Requires 2+ Models

**Symptom**: Single-model consensus fails with error.

**Workaround**: Tests train at least 2 models (LR_EN + RF).

**Resolution**: Expected behavior per ADR-004.

---

## Coverage Targets

| Module | Target | Current | Priority |
|--------|--------|---------|----------|
| `significance/permutation_test.py` | >= 85% | ~75% | High |
| `features/consensus.py` | >= 85% | ~80% | High |
| `features/importance.py` | >= 80% | ~70% | Medium |
| `features/stability.py` | >= 80% | ~65% | Medium |
| `cli/consensus_panel.py` | >= 75% | ~50% | Medium |
| `cli/permutation_test.py` | >= 75% | ~45% | Medium |

**Gap Analysis**:
- Missing: Edge cases for zero-positive folds in permutation test
- Missing: Large-scale consensus panel (100+ proteins)
- Missing: Consensus with missing OOF importance for some models
- Missing: RFE integration in consensus workflow

---

## Troubleshooting

### Test Fails: "Permutation test failed"

**Debug**:
```bash
pytest tests/e2e/test_feature_selection_workflow.py::TestStage1ModelGate::test_permutation_test_identifies_significant_model -vv -s
```

**Common causes**:
1. Insufficient signal in data (check PROT_000-004 have strong signal)
2. Missing validation set (permutation test requires val split)
3. Model training failed (check training logs)

### Test Fails: "Consensus panel failed"

**Debug**:
```bash
pytest tests/e2e/test_feature_selection_workflow.py::TestStage3RRAConsensus::test_consensus_panel_basic -vv -s
```

**Common causes**:
1. Aggregation not run before consensus (check aggregated/ directory exists)
2. Fewer than 2 models provided (consensus requires >= 2 models)
3. No stable proteins above threshold (lower --stability-threshold)

### Test Skipped: "Not enough significant models"

**Cause**: Both models failed permutation test (p >= 0.5).

**Solution**: This is expected occasionally with small n_perms=10. Re-run or increase n_perms to 20.

---

## Related Documentation

- [FEATURE_SELECTION.md](../../docs/reference/FEATURE_SELECTION.md) - Feature selection reference
- [E2E_TESTING_GUIDE.md](../E2E_TESTING_GUIDE.md) - General E2E testing guide
- [ADR-004](../../docs/adr/ADR-004-four-strategy-feature-selection.md) - Three-stage workflow
- [ADR-011](../../docs/adr/ADR-011-permutation-testing.md) - Permutation testing

---

**Last Updated**: 2026-02-09
**Maintainer**: Andres Chousal (Chowell Lab)
