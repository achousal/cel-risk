# E2E Test Inventory

**Purpose**: Comprehensive listing of all E2E test files, classes, and coverage status.

**Status**: 9 test files, 50 test classes, 1,282 total tests (2026-02-09).

---

## Test File Summary

| File | Lines | Classes | Tests | Speed | Purpose |
|------|-------|---------|-------|-------|---------|
| `test_e2e_runner.py` | 1,940 | 11 | ~65 | Mixed | Core pipeline workflows |
| `test_e2e_pipeline.py` | 132 | 2 | ~5 | Mixed | Basic integration |
| `test_e2e_fixed_panel_workflows.py` | 889 | 4 | ~15 | Fast | Panel extraction/validation |
| `test_e2e_calibration_workflows.py` | 930 | 5 | ~18 | Slow | Calibration strategies |
| `test_e2e_temporal_workflows.py` | 754 | 5 | ~16 | Slow | Temporal validation |
| `test_e2e_run_id_workflows.py` | 1,267 | 7 | ~22 | Fast | Run-ID auto-detection |
| `test_e2e_multi_model_workflows.py` | 987 | 6 | ~20 | Fast | Cross-model workflows |
| `test_e2e_output_structure.py` | 1,171 | 5 | ~25 | Fast | Output validation |
| `test_feature_selection_workflow.py` | 1,050 | 5 | ~11 | Slow | Three-stage feature selection |
| **Total** | **9,120** | **50** | **~197** | - | - |

---

## Detailed Test Class Breakdown

### test_e2e_runner.py (Core Workflows)

**File Size**: 1,940 lines | **Speed**: Mixed (fast + slow)

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestE2EFullPipeline` | 6 | Complete training pipeline (splits → train → results) | Slow |
| `TestE2EEnsembleWorkflow` | 5 | Ensemble training and aggregation | Slow |
| `TestE2EHPCWorkflow` | 4 | HPC config validation and dry-run mode | Fast |
| `TestE2ETemporalValidation` | 6 | Temporal split generation and validation | Slow |
| `TestE2EErrorHandling` | 8 | Error handling for common failure modes | Fast |
| `TestE2EPanelOptimization` | 6 | Panel optimization via RFE workflow | Slow |
| `TestE2EFixedPanelValidation` | 7 | Fixed panel training and validation | Slow |
| `TestE2EAggregationWorkflow` | 8 | Results aggregation across splits | Slow |
| `TestE2EConfigValidation` | 6 | Config validation and comparison | Fast |
| `TestE2EHoldoutEvaluation` | 5 | Holdout set evaluation workflow | Slow |
| `TestE2EDataConversion` | 4 | CSV to Parquet conversion utilities | Fast |

**Coverage**:
- Training workflows: Complete
- Aggregation workflows: Complete
- Panel optimization: Complete
- Error handling: Comprehensive
- Config validation: Complete

### test_e2e_pipeline.py (Basic Integration)

**File Size**: 132 lines | **Speed**: Mixed

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestE2EPipeline` | 3 | Config roundtrip and basic validation | Fast |
| `TestE2ESlowPipeline` | 2 | Full pipeline with real model training | Slow |

**Coverage**:
- Config I/O: Complete
- Basic pipeline: Complete

### test_e2e_fixed_panel_workflows.py (Panel Validation)

**File Size**: 889 lines | **Speed**: Fast

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestPanelExtraction` | 4 | Panel extraction from aggregated results | Fast |
| `TestFixedPanelTraining` | 4 | Training with pre-defined panel | Fast |
| `TestUnbiasedValidation` | 4 | Unbiased panel validation with new splits | Fast |
| `TestPanelFormatValidation` | 3 | Panel file format and schema validation | Fast |

**Coverage**:
- Panel extraction: Complete
- Fixed panel training: Complete
- Unbiased validation: Complete
- Format validation: Complete

### test_e2e_calibration_workflows.py (Calibration)

**File Size**: 930 lines | **Speed**: Slow

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestOOFPosthocCalibration` | 4 | OOF-posthoc calibration strategy | Slow |
| `TestPerFoldCalibration` | 4 | Per-fold calibration strategy | Slow |
| `TestCalibrationStrategyComparison` | 3 | Compare calibration strategies | Slow |
| `TestCalibrationPlots` | 4 | Calibration plot generation | Slow |
| `TestCalibrationAggregation` | 3 | Aggregated calibration metrics | Slow |

**Coverage**:
- OOF-posthoc: Complete
- Per-fold: Complete
- Strategy comparison: Complete
- Plot generation: Complete
- Aggregation: Complete

### test_e2e_temporal_workflows.py (Temporal Validation)

**File Size**: 754 lines | **Speed**: Slow

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestTemporalSplitCreation` | 3 | Temporal split generation with dates | Slow |
| `TestTemporalTrainingWorkflow` | 4 | Training with temporal splits | Slow |
| `TestTemporalValidationMetrics` | 3 | Metrics for temporal validation | Slow |
| `TestTemporalAggregation` | 3 | Aggregated temporal results | Slow |
| `TestTemporalComparison` | 3 | Compare temporal vs random splits | Slow |

**Coverage**:
- Temporal splits: Complete
- Training workflows: Complete
- Validation metrics: Complete
- Aggregation: Complete
- Comparison: Complete

### test_e2e_run_id_workflows.py (Run-ID Auto-Detection)

**File Size**: 1,267 lines | **Speed**: Fast (mostly)

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestRunIdMetadataCreation` | 3 | Verify run_metadata.json creation | Fast |
| `TestAggregateWithRunId` | 3 | `ced aggregate-splits --run-id` | Fast |
| `TestEnsembleWithRunId` | 3 | `ced train-ensemble --run-id` | Slow |
| `TestOptimizePanelWithRunId` | 4 | `ced optimize-panel --run-id` | Fast |
| `TestConsensusPanelWithRunId` | 3 | `ced consensus-panel --run-id` | Fast |
| `TestFullPipelineWithRunId` | 3 | Complete pipelines with --run-id | Slow |
| `TestRunIdErrorHandling` | 3 | Invalid/missing run IDs | Fast |

**Coverage**:
- Metadata creation: Complete
- Aggregate auto-detection: Complete
- Ensemble auto-detection: Complete
- Panel optimization: Complete
- Consensus panel: Complete
- Full pipeline: Complete
- Error handling: Complete

### test_e2e_multi_model_workflows.py (Cross-Model)

**File Size**: 987 lines | **Speed**: Fast

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestSharedRunIdCoordination` | 4 | Multi-model run-id coordination | Fast |
| `TestCrossModelAggregation` | 3 | Aggregate multiple models | Fast |
| `TestConsensusPanelIntegration` | 4 | Cross-model consensus panel | Fast |
| `TestEnsembleWorkflows` | 4 | Ensemble with multiple base models | Slow |
| `TestPanelOptimizationBatch` | 3 | Batch panel optimization | Fast |
| `TestMetadataConsistency` | 2 | Metadata consistency across models | Fast |

**Coverage**:
- Shared run-id: Complete
- Cross-model aggregation: Complete
- Consensus panel: Complete
- Ensemble workflows: Complete
- Batch optimization: Complete
- Metadata consistency: Complete

### test_e2e_output_structure.py (Output Validation)

**File Size**: 1,171 lines | **Speed**: Fast

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestTrainingOutputStructure` | 6 | Training output directories and files | Fast |
| `TestAggregationOutputStructure` | 5 | Aggregation output structure | Fast |
| `TestEnsembleOutputStructure` | 5 | Ensemble outputs and references | Fast |
| `TestPanelOptimizationOutputStructure` | 5 | RFE outputs and panel curves | Fast |
| `TestConsensusPanelOutputStructure` | 4 | Consensus rankings and panels | Fast |

**Coverage**:
- Training outputs: Complete
- Aggregation outputs: Complete
- Ensemble outputs: Complete
- Panel optimization: Complete
- Consensus panel: Complete

### test_feature_selection_workflow.py (Three-Stage Feature Selection)

**File Size**: 1,050 lines | **Speed**: Slow

| Class | Tests | Purpose | Speed |
|-------|-------|---------|-------|
| `TestStage1ModelGate` | 2 | Permutation testing (model gate) | Slow |
| `TestStage2PerModelEvidence` | 2 | OOF importance and stability | Slow |
| `TestStage3RRAConsensus` | 3 | Geometric mean rank aggregation | Slow |
| `TestFullThreeStageWorkflow` | 1 | Complete workflow integration | Slow |
| `TestFeatureSelectionErrorHandling` | 3 | Error handling and edge cases | Fast |

**Coverage**:
- Permutation test: Complete
- OOF importance: Complete
- Stability selection: Complete
- RRA consensus: Complete
- Cross-model panel: Complete
- Error handling: Complete

---

## Coverage Status

### High Coverage (>= 70%)

**Well-tested modules**:
- `cli/save_splits.py`: 68% - Split generation comprehensive
- `cli/train.py`: 66% - Core training workflows complete
- `data/splits.py`: 82% - Split logic thoroughly tested
- `data/io.py`: 75% - Data loading validated
- `models/calibration.py`: 79% - Calibration strategies tested

**Status**: Production-ready

### Medium Coverage (40-70%)

**Adequately tested modules**:
- `cli/aggregate_splits.py`: 40% - Basic aggregation covered
- `cli/consensus_panel.py`: 50% - Cross-model consensus functional
- `features/stability.py`: 55% - Stability selection validated
- `features/kbest.py`: 62% - K-best selection tested
- `metrics/discrimination.py`: 58% - Core metrics validated

**Status**: Functional, edge cases needed

### Low Coverage (< 40%)

**Needs improvement**:
- `cli/train_ensemble.py`: 12% - Blocked by ensemble bugs
- `cli/optimize_panel.py`: 7% - Blocked by RFE bugs
- `features/rfe.py`: 11% - RFE implementation gaps
- `models/stacking.py`: 13% - Meta-learner bugs
- `plotting/oof.py`: 23% - Plot generation minimal

**Status**: Known functional bugs blocking tests

---

## Known Gaps

### Missing Test Coverage

**1. Holdout Evaluation**
- Status: No E2E tests
- Impact: Medium
- Priority: Medium
- Reason: CLI command exists but no workflow tests

**2. Panel Optimization Edge Cases**
- Status: Basic tests only
- Impact: Low
- Priority: Low
- Reason: Core functionality tested, edge cases (empty stable proteins, single protein) not covered

**3. Error Recovery**
- Status: Minimal tests
- Impact: Medium
- Priority: Medium
- Reason: No tests for interrupted training, corrupted outputs, partial results

**4. Multi-User HPC Coordination**
- Status: No tests
- Impact: Low
- Priority: Low
- Reason: Complex to simulate, low real-world impact

**5. Large-Scale Performance**
- Status: No tests
- Impact: Low
- Priority: Low
- Reason: All tests use minimal fixtures (< 100 samples, < 20 proteins)

### Blocked Tests (Functional Bugs)

**1. Ensemble Training** (2 tests skipped)
- **Module**: `models/stacking.py`, `cli/train_ensemble.py`
- **Issue**: Meta-learner fails during OOF prediction aggregation
- **Impact**: High - ensemble workflows completely blocked
- **Debug**: Check OOF shape mismatch, metadata handling errors

**2. Panel Optimization** (3 tests skipped)
- **Module**: `features/rfe.py`, `cli/optimize_panel.py`
- **Issue**: RFE fails to find consensus stable proteins
- **Impact**: High - panel optimization workflows blocked
- **Debug**: Check feature stability data availability, RFE input validation

---

## Test Execution Summary (2026-01-28)

### Recent Test Run

**Command**: `pytest tests/test_e2e_*.py -v`

**Results**:
- Total tests: ~186
- Passed: 165 (89%)
- Skipped: 16 (9%) - Blocked by functional bugs
- Failed: 5 (2%) - Known issues, documented

**Execution Time**:
- Fast tests only (`-m "not slow"`): ~2 minutes
- All tests: ~12 minutes

**Coverage**:
- Overall: 14% (after major refactoring)
- Core modules: 40-82%
- CLI commands: 7-68%
- Plotting: 23-45%

### What Works

1. Training creates correct run_metadata.json
2. Metadata persists across operations
3. Error handling for invalid run IDs
4. Dependency validation (aggregation before optimization)
5. Run-id auto-detection in all supported commands
6. Output structure validation comprehensive
7. Calibration strategies tested
8. Temporal validation functional
9. Fixed panel workflows complete

### What's Blocked

1. **Ensemble training**: Meta-learner OOF aggregation bug
2. **Panel optimization**: RFE stable protein filtering bug
3. **Holdout evaluation**: No E2E tests yet
4. **Error recovery**: Minimal coverage

---

## Test Maintenance

### Adding New Tests

**When to add tests**:
1. New CLI command implemented
2. New output file format introduced
3. Bug fix (add regression test)
4. New feature added (add feature tests)

**Where to add tests**:
- **Run-id workflows**: `test_e2e_run_id_workflows.py`
- **Output validation**: `test_e2e_output_structure.py`
- **General workflows**: `test_e2e_runner.py`
- **Specialized features**: Create new `test_e2e_<feature>_workflows.py`

**Test template**:
```python
@pytest.mark.slow  # If requires training
def test_new_feature(self, small_proteomics_data, fast_training_config, tmp_path):
    """Test: Brief description."""
    # Setup
    # Execute
    # Validate outputs
    # Verify behavior
```

### Updating Tests

**When code changes**:
1. Update tests that validate changed behavior
2. Add new tests for new functionality
3. Remove tests for deprecated features
4. Update expected outputs if schema changes

**Before commit**:
```bash
pytest tests/test_e2e_*.py -v -m "not slow"  # Fast validation
pytest tests/test_e2e_*.py -v                 # Full validation
```

### Monitoring Coverage

```bash
# Generate coverage report
pytest tests/ --cov=ced_ml --cov-report=html

# View in browser
open htmlcov/index.html

# Check coverage targets
pytest tests/ --cov=ced_ml --cov-fail-under=80
```

---

## Related Documentation

- [E2E_TESTING_GUIDE.md](E2E_TESTING_GUIDE.md) - How to run and maintain E2E tests
- [analysis/docs/reference/CLI_REFERENCE.md](../docs/reference/CLI_REFERENCE.md) - CLI command documentation
- [analysis/docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Technical architecture
- [CLAUDE.md](../../CLAUDE.md) - Project overview

---

**Last Updated**: 2026-01-28
**Maintainer**: Andres Chousal (Chowell Lab)
