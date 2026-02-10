# E2E Test Run Commands

Quick reference for running the training and evaluation E2E tests.

---

## Basic Commands

### Run all E2E tests for training/evaluation workflow
```bash
cd analysis/
pytest tests/e2e/test_training_evaluation_workflow.py -v
```

### Run with detailed output
```bash
pytest tests/e2e/test_training_evaluation_workflow.py -vv -s
```

---

## Run Specific Test Classes

### Training workflow tests
```bash
pytest tests/e2e/test_training_evaluation_workflow.py::TestTrainingWorkflow -v
```

### Aggregation workflow tests
```bash
pytest tests/e2e/test_training_evaluation_workflow.py::TestAggregationWorkflow -v
```

### Complete workflow tests
```bash
pytest tests/e2e/test_training_evaluation_workflow.py::TestCompleteWorkflow -v
```

---

## Expected Results

### Successful run
```
====== 6 passed, 2 skipped, XX warnings in XX.XXs ======
```

- 6 passed: Core workflow tests
- 2 skipped: Panel optimization and holdout evaluation (expected when dependencies unavailable)

### Runtime
- Full suite: ~70-90 seconds
- Individual tests: 10-30 seconds each

---

**Last Updated**: 2026-02-09
**Maintainer**: Andres Chousal (Chowell Lab)
