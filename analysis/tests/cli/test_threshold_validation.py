"""Tests for threshold-on-test validation (V-06 fix)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ced_ml.cli.orchestration.context import TrainingContext
from ced_ml.cli.orchestration.evaluation_stage import evaluate_splits
from ced_ml.config.training_schema import TrainingConfig


def test_threshold_on_test_requires_explicit_flag():
    """Test that threshold-on-test requires allow_test_thresholding=True."""
    # Setup config with no validation set (val_size=0)
    config = TrainingConfig()
    config.cv.folds = 1  # This means no validation set
    config.allow_test_thresholding = False  # Default: disallow threshold-on-test

    # Create minimal context with no validation set
    ctx = TrainingContext(config=config)

    # Create dummy training data
    ctx.X_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    ctx.y_train = np.array([0, 1, 0, 1])
    ctx.train_idx = np.array([0, 1, 2, 3])
    ctx.cat_train = np.array(["control", "case", "control", "case"])

    # No validation set (empty val_idx means has_validation_set=False)
    ctx.X_val = pd.DataFrame({"f1": []})
    ctx.y_val = np.array([])
    ctx.val_idx = np.array([])
    ctx.cat_val = np.array([])

    # Create dummy test data
    ctx.X_test = pd.DataFrame({"f1": [5, 6]})
    ctx.y_test = np.array([0, 1])
    ctx.test_idx = np.array([4, 5])
    ctx.cat_test = np.array(["control", "case"])

    # Create minimal pipeline
    ctx.final_pipeline = LogisticRegression()
    ctx.final_pipeline.fit(ctx.X_train, ctx.y_train)

    # Mock OOF predictions
    ctx.oof_preds = np.array([[0.3, 0.6, 0.4, 0.7]])

    # Set train prevalence
    ctx.train_prev = 0.5

    # Should raise ValueError because allow_test_thresholding=False
    with pytest.raises(ValueError, match="threshold-on-test not explicitly allowed"):
        evaluate_splits(ctx)


def test_threshold_on_test_allowed_with_flag():
    """Test that threshold-on-test works when allow_test_thresholding=True."""
    # Setup config with no validation set (val_size=0)
    config = TrainingConfig()
    config.cv.folds = 1  # This means no validation set
    config.allow_test_thresholding = True  # Explicitly allow threshold-on-test

    # Create minimal context with no validation set
    ctx = TrainingContext(config=config)

    # Create dummy training data
    ctx.X_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    ctx.y_train = np.array([0, 1, 0, 1])
    ctx.train_idx = np.array([0, 1, 2, 3])
    ctx.cat_train = np.array(["control", "case", "control", "case"])

    # No validation set (empty val_idx means has_validation_set=False)
    ctx.X_val = pd.DataFrame({"f1": []})
    ctx.y_val = np.array([])
    ctx.val_idx = np.array([])
    ctx.cat_val = np.array([])

    # Create dummy test data
    ctx.X_test = pd.DataFrame({"f1": [5, 6]})
    ctx.y_test = np.array([0, 1])
    ctx.test_idx = np.array([4, 5])
    ctx.cat_test = np.array(["control", "case"])

    # Create minimal pipeline
    ctx.final_pipeline = LogisticRegression()
    ctx.final_pipeline.fit(ctx.X_train, ctx.y_train)

    # Mock OOF predictions
    ctx.oof_preds = np.array([[0.3, 0.6, 0.4, 0.7]])

    # Set train prevalence
    ctx.train_prev = 0.5

    # Should NOT raise (allow_test_thresholding=True)
    updated_ctx = evaluate_splits(ctx)

    # Verify that val_threshold is None (computed on test)
    assert updated_ctx.val_threshold is None
    assert updated_ctx.test_metrics is not None


def test_normal_validation_threshold_flow():
    """Test that validation threshold works normally when validation set exists."""
    # Setup config with validation set
    config = TrainingConfig()
    config.cv.folds = 5  # This means validation set exists
    config.allow_test_thresholding = False  # Default

    # Create minimal context WITH validation set
    ctx = TrainingContext(config=config)

    # Create dummy training data
    ctx.X_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    ctx.y_train = np.array([0, 1, 0, 1])
    ctx.train_idx = np.array([0, 1, 2, 3])
    ctx.cat_train = np.array(["control", "case", "control", "case"])

    # Create dummy validation data
    ctx.X_val = pd.DataFrame({"f1": [5, 6]})
    ctx.y_val = np.array([0, 1])
    ctx.val_idx = np.array([4, 5])
    ctx.cat_val = np.array(["control", "case"])

    # Create dummy test data
    ctx.X_test = pd.DataFrame({"f1": [7, 8]})
    ctx.y_test = np.array([0, 1])
    ctx.test_idx = np.array([6, 7])
    ctx.cat_test = np.array(["control", "case"])

    # Create minimal pipeline
    ctx.final_pipeline = LogisticRegression()
    ctx.final_pipeline.fit(ctx.X_train, ctx.y_train)

    # Mock OOF predictions
    ctx.oof_preds = np.array([[0.3, 0.6, 0.4, 0.7]])

    # Set train prevalence
    ctx.train_prev = 0.5

    # Should work normally
    updated_ctx = evaluate_splits(ctx)

    # Verify that val_threshold is set (computed on validation)
    assert updated_ctx.val_threshold is not None
    assert updated_ctx.val_metrics is not None
    assert updated_ctx.test_metrics is not None
