"""Tests for StackingEnsemble data handling and edge cases.

Tests cover:
- NaN handling in OOF predictions
- File-based collection functions
- Index mismatch detection (length and values)
- Successful collection with matching indices
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ced_ml.models.stacking import (
    StackingEnsemble,
    collect_oof_predictions,
    collect_split_predictions,
)


def test_nan_handling_in_oof():
    """Test handling of NaN values in OOF predictions."""
    y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    oof_dict = {
        "LR_EN": np.array([[0.3, 0.2, np.nan, 0.8, 0.7, 0.3, 0.2, 0.9, 0.1, 0.6]]),
        "RF": np.array([[0.4, 0.3, 0.5, 0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.7]]),
    }

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )

    # Should handle NaN by dropping those samples (with warning)
    ensemble.fit_from_oof(oof_dict, y_train)
    assert ensemble.is_fitted_


def test_collect_oof_predictions_file_not_found():
    """Test error when OOF file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_split_predictions_file_not_found():
    """Test error when split predictions file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN"],
                split_seed=0,
                split_name="test",
            )


def test_collect_oof_predictions_from_files():
    """Test collecting OOF predictions from CSV files."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create mock directory structure and files
        for model in ["LR_EN", "RF"]:
            model_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            model_dir.mkdir(parents=True)

            # Create mock OOF CSV
            oof_df = pd.DataFrame(
                {
                    "idx": np.arange(100),
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                    "y_prob_repeat1": rng.uniform(0, 1, 100),
                }
            )
            oof_df.to_csv(model_dir / f"train_oof__{model}.csv", index=False)

        # Collect
        oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
            results_dir,
            base_models=["LR_EN", "RF"],
            split_seed=0,
        )

        assert "LR_EN" in oof_dict
        assert "RF" in oof_dict
        assert oof_dict["LR_EN"].shape == (2, 100)  # 2 repeats, 100 samples
        assert len(y_train) == 100
        assert len(train_idx) == 100
        assert cat_train is None  # No category column in test data


def test_collect_oof_predictions_index_length_mismatch():
    """Test error when base models have different sample counts in OOF predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with 100 samples
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100),
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(lr_dir / "train_oof__LR_EN.csv", index=False)

        # Create RF with 80 samples (mismatch)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(80),
                "y_true": rng.integers(0, 2, 80),
                "y_prob_repeat0": rng.uniform(0, 1, 80),
            }
        ).to_csv(rf_dir / "train_oof__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index length mismatch.*OOF predictions"):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_oof_predictions_index_value_mismatch():
    """Test error when base models have different index values in OOF predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-99
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100),
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(lr_dir / "train_oof__LR_EN.csv", index=False)

        # Create RF with different indices (100-199 instead of 0-99)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100, 200),  # Different indices
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(rf_dir / "train_oof__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*OOF predictions"):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_split_predictions_index_length_mismatch():
    """Test error when base models have different sample counts in test predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with 50 test samples
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50),
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(lr_dir / "test_preds__LR_EN.csv", index=False)

        # Create RF with 40 test samples (mismatch)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(40),
                "y_true": rng.integers(0, 2, 40),
                "y_prob": rng.uniform(0, 1, 40),
            }
        ).to_csv(rf_dir / "test_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index length mismatch.*test predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="test",
            )


def test_collect_split_predictions_index_value_mismatch():
    """Test error when base models have different index values in test predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-49
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50),
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(lr_dir / "test_preds__LR_EN.csv", index=False)

        # Create RF with different indices
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50, 100),  # Different indices
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(rf_dir / "test_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*test predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="test",
            )


def test_collect_split_predictions_val_index_mismatch():
    """Test error when base models have different index values in val predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-29
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(30),
                "y_true": rng.integers(0, 2, 30),
                "y_prob": rng.uniform(0, 1, 30),
            }
        ).to_csv(lr_dir / "val_preds__LR_EN.csv", index=False)

        # Create RF with different indices
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(30, 60),  # Different indices
                "y_true": rng.integers(0, 2, 30),
                "y_prob": rng.uniform(0, 1, 30),
            }
        ).to_csv(rf_dir / "val_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*val predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="val",
            )


def test_collect_oof_predictions_matching_indices_succeeds():
    """Test that collecting OOF predictions succeeds when indices match."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Use same fixed seed for reproducible indices
        shared_indices = np.arange(100)

        for model in ["LR_EN", "RF", "XGBoost"]:
            model_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            model_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "idx": shared_indices,  # Same indices for all models
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                }
            ).to_csv(model_dir / f"train_oof__{model}.csv", index=False)

        # Should succeed without error
        oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
            results_dir,
            base_models=["LR_EN", "RF", "XGBoost"],
            split_seed=0,
        )

        assert len(oof_dict) == 3
        assert len(y_train) == 100
        np.testing.assert_array_equal(train_idx, shared_indices)
        assert cat_train is None  # No category column in test data
