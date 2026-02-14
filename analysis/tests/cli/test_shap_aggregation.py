"""
Tests for SHAP importance aggregation.

This module tests that OOF SHAP importance is correctly aggregated across splits
by the aggregate_shap_importance function.

Coverage:
- Multi-split aggregation with numerical correctness
- Empty splits (no SHAP files)
- Single split (stability = 1.0, std = NaN)
- Different features across splits (union handling)
- Filename contract with persistence stage
- Sorting by mean_abs_shap descending
- Stability calculation (n_splits_present / total_splits)
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ced_ml.cli.aggregation.orchestrator import aggregate_shap_importance


@pytest.fixture
def mock_shap_splits():
    """Create temp dirs simulating multiple split outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split_dirs = []
        for i in range(3):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            cv_dir = split_dir / "cv"
            cv_dir.mkdir(parents=True)

            df = pd.DataFrame(
                {
                    "feature": ["protein_A", "protein_B", "protein_C"],
                    "mean_abs_shap": [0.5 + i * 0.1, 0.3 + i * 0.05, 0.1 + i * 0.02],
                    "std_abs_shap": [0.05, 0.03, 0.01],
                    "median_abs_shap": [0.45, 0.28, 0.09],
                    "n_folds_nonzero": [5, 5, 3],
                }
            )
            df.to_csv(cv_dir / "oof_shap_importance__LR_EN.csv", index=False)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        yield split_dirs, output_dir


@pytest.fixture
def logger():
    """Create a logger for tests."""
    return logging.getLogger("test")


def test_aggregate_shap_multi_split(mock_shap_splits, logger):
    """Test aggregation across multiple splits with same features."""
    split_dirs, output_dir = mock_shap_splits

    result = aggregate_shap_importance(split_dirs, "LR_EN", output_dir, logger)

    # Verify return value
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

    # Verify output file exists
    expected_path = output_dir / "importance" / "oof_shap_importance__LR_EN.csv"
    assert expected_path.exists()

    # Verify columns
    expected_cols = ["feature", "mean_abs_shap", "std_abs_shap", "stability", "rank"]
    assert list(result.columns) == expected_cols

    # Verify features
    assert set(result["feature"]) == {"protein_A", "protein_B", "protein_C"}

    # Verify numerical correctness
    # protein_A: mean([0.5, 0.6, 0.7]) = 0.6
    protein_a = result[result["feature"] == "protein_A"].iloc[0]
    np.testing.assert_allclose(protein_a["mean_abs_shap"], 0.6, rtol=1e-5)
    # std([0.5, 0.6, 0.7]) = 0.1
    np.testing.assert_allclose(protein_a["std_abs_shap"], 0.1, rtol=1e-5)
    assert protein_a["stability"] == 1.0

    # protein_B: mean([0.3, 0.35, 0.4]) = 0.35
    protein_b = result[result["feature"] == "protein_B"].iloc[0]
    np.testing.assert_allclose(protein_b["mean_abs_shap"], 0.35, rtol=1e-5)
    # std([0.3, 0.35, 0.4]) = 0.05
    np.testing.assert_allclose(protein_b["std_abs_shap"], 0.05, rtol=1e-5)
    assert protein_b["stability"] == 1.0

    # protein_C: mean([0.1, 0.12, 0.14]) = 0.12
    protein_c = result[result["feature"] == "protein_C"].iloc[0]
    np.testing.assert_allclose(protein_c["mean_abs_shap"], 0.12, rtol=1e-5)
    # std([0.1, 0.12, 0.14]) = 0.02
    np.testing.assert_allclose(protein_c["std_abs_shap"], 0.02, rtol=1e-5)
    assert protein_c["stability"] == 1.0

    # Verify rank column is 1-indexed and sorted by mean_abs_shap descending
    assert result["rank"].min() == 1
    assert result["rank"].max() == 3
    assert result.iloc[0]["rank"] == 1  # protein_A (highest)
    assert result.iloc[1]["rank"] == 2  # protein_B
    assert result.iloc[2]["rank"] == 3  # protein_C (lowest)


def test_aggregate_shap_no_files(logger):
    """Test aggregation when no SHAP CSV files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create empty split dirs
        split_dirs = []
        for i in range(3):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            cv_dir = split_dir / "cv"
            cv_dir.mkdir(parents=True)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance(split_dirs, "LR_EN", output_dir, logger)

        # Should return None and not crash
        assert result is None

        # Verify no output file created
        expected_path = output_dir / "importance" / "oof_shap_importance__LR_EN.csv"
        assert not expected_path.exists()


def test_aggregate_shap_single_split(logger):
    """Test aggregation with a single split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create single split dir
        split_dir = tmpdir_path / "split_seed0" / "LR_EN"
        cv_dir = split_dir / "cv"
        cv_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "feature": ["protein_X", "protein_Y"],
                "mean_abs_shap": [0.8, 0.4],
                "std_abs_shap": [0.1, 0.05],
                "median_abs_shap": [0.75, 0.38],
                "n_folds_nonzero": [5, 5],
            }
        )
        df.to_csv(cv_dir / "oof_shap_importance__LR_EN.csv", index=False)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance([split_dir], "LR_EN", output_dir, logger)

        # Verify return value
        assert result is not None
        assert len(result) == 2

        # Verify stability is 1.0 (present in 1 out of 1 split)
        assert all(result["stability"] == 1.0)

        # Verify mean_abs_shap values are preserved
        protein_x = result[result["feature"] == "protein_X"].iloc[0]
        np.testing.assert_allclose(protein_x["mean_abs_shap"], 0.8, rtol=1e-5)
        # std should be NaN for single split (pandas std with ddof=1)
        assert pd.isna(protein_x["std_abs_shap"])

        protein_y = result[result["feature"] == "protein_Y"].iloc[0]
        np.testing.assert_allclose(protein_y["mean_abs_shap"], 0.4, rtol=1e-5)

        # Verify rank
        assert protein_x["rank"] == 1
        assert protein_y["rank"] == 2


def test_aggregate_shap_different_features(logger):
    """Test aggregation when splits have different feature sets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Split 1: features [A, B, C]
        split1_dir = tmpdir_path / "split_seed0" / "LR_EN"
        cv1_dir = split1_dir / "cv"
        cv1_dir.mkdir(parents=True)
        df1 = pd.DataFrame(
            {
                "feature": ["protein_A", "protein_B", "protein_C"],
                "mean_abs_shap": [0.6, 0.4, 0.2],
                "std_abs_shap": [0.05, 0.03, 0.01],
                "median_abs_shap": [0.58, 0.38, 0.18],
                "n_folds_nonzero": [5, 5, 3],
            }
        )
        df1.to_csv(cv1_dir / "oof_shap_importance__LR_EN.csv", index=False)

        # Split 2: features [B, C, D]
        split2_dir = tmpdir_path / "split_seed1" / "LR_EN"
        cv2_dir = split2_dir / "cv"
        cv2_dir.mkdir(parents=True)
        df2 = pd.DataFrame(
            {
                "feature": ["protein_B", "protein_C", "protein_D"],
                "mean_abs_shap": [0.5, 0.3, 0.1],
                "std_abs_shap": [0.04, 0.02, 0.01],
                "median_abs_shap": [0.48, 0.28, 0.09],
                "n_folds_nonzero": [5, 4, 2],
            }
        )
        df2.to_csv(cv2_dir / "oof_shap_importance__LR_EN.csv", index=False)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance([split1_dir, split2_dir], "LR_EN", output_dir, logger)

        # Verify all features from union are present
        assert result is not None
        assert len(result) == 4
        assert set(result["feature"]) == {"protein_A", "protein_B", "protein_C", "protein_D"}

        # Verify stability for features present in different numbers of splits
        protein_a = result[result["feature"] == "protein_A"].iloc[0]
        assert protein_a["stability"] == 0.5  # 1 out of 2 splits

        protein_b = result[result["feature"] == "protein_B"].iloc[0]
        assert protein_b["stability"] == 1.0  # 2 out of 2 splits

        protein_c = result[result["feature"] == "protein_C"].iloc[0]
        assert protein_c["stability"] == 1.0  # 2 out of 2 splits

        protein_d = result[result["feature"] == "protein_D"].iloc[0]
        assert protein_d["stability"] == 0.5  # 1 out of 2 splits

        # Verify mean_abs_shap for protein_B (present in both)
        # mean([0.4, 0.5]) = 0.45
        np.testing.assert_allclose(protein_b["mean_abs_shap"], 0.45, rtol=1e-5)

        # Verify mean_abs_shap for protein_A (present in split 1 only)
        np.testing.assert_allclose(protein_a["mean_abs_shap"], 0.6, rtol=1e-5)

        # Verify mean_abs_shap for protein_D (present in split 2 only)
        np.testing.assert_allclose(protein_d["mean_abs_shap"], 0.1, rtol=1e-5)


def test_filename_contract(logger):
    """Verify the filename pattern matches persistence stage output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create split dir with SHAP file
        split_dir = tmpdir_path / "split_seed0" / "XGBoost"
        cv_dir = split_dir / "cv"
        cv_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "feature": ["protein_1"],
                "mean_abs_shap": [0.5],
                "std_abs_shap": [0.05],
                "median_abs_shap": [0.48],
                "n_folds_nonzero": [5],
            }
        )

        # Filename format from persistence stage
        persistence_filename = "oof_shap_importance__XGBoost.csv"
        df.to_csv(cv_dir / persistence_filename, index=False)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        # Aggregation expects same filename pattern
        result = aggregate_shap_importance([split_dir], "XGBoost", output_dir, logger)

        assert result is not None
        assert len(result) == 1

        # Verify output filename matches expected pattern
        expected_output = output_dir / "importance" / "oof_shap_importance__XGBoost.csv"
        assert expected_output.exists()


def test_aggregate_shap_sorted_by_mean_abs_shap(logger):
    """Test that output is sorted by mean_abs_shap descending."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        split_dir = tmpdir_path / "split_seed0" / "LR_EN"
        cv_dir = split_dir / "cv"
        cv_dir.mkdir(parents=True)

        # Deliberately unsorted input
        df = pd.DataFrame(
            {
                "feature": ["protein_low", "protein_high", "protein_mid"],
                "mean_abs_shap": [0.1, 0.9, 0.5],
                "std_abs_shap": [0.01, 0.09, 0.05],
                "median_abs_shap": [0.09, 0.88, 0.48],
                "n_folds_nonzero": [3, 5, 4],
            }
        )
        df.to_csv(cv_dir / "oof_shap_importance__LR_EN.csv", index=False)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance([split_dir], "LR_EN", output_dir, logger)

        # Verify sorted descending
        assert result is not None
        assert result.iloc[0]["feature"] == "protein_high"
        assert result.iloc[1]["feature"] == "protein_mid"
        assert result.iloc[2]["feature"] == "protein_low"

        # Verify ranks match sort order
        assert result.iloc[0]["rank"] == 1
        assert result.iloc[1]["rank"] == 2
        assert result.iloc[2]["rank"] == 3


def test_aggregate_shap_preserves_n_splits_present(logger):
    """Test that n_splits_present is correctly counted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create 4 splits
        split_dirs = []
        for i in range(4):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            cv_dir = split_dir / "cv"
            cv_dir.mkdir(parents=True)

            # protein_A in all splits, protein_B in first 2, protein_C in first 1
            features = ["protein_A"]
            values = [0.5]
            if i < 2:
                features.append("protein_B")
                values.append(0.3)
            if i < 1:
                features.append("protein_C")
                values.append(0.1)

            df = pd.DataFrame(
                {
                    "feature": features,
                    "mean_abs_shap": values,
                    "std_abs_shap": [0.05] * len(features),
                    "median_abs_shap": [v * 0.9 for v in values],
                    "n_folds_nonzero": [5] * len(features),
                }
            )
            df.to_csv(cv_dir / "oof_shap_importance__LR_EN.csv", index=False)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance(split_dirs, "LR_EN", output_dir, logger)

        # Verify n_splits_present is correct
        assert result is not None

        protein_a = result[result["feature"] == "protein_A"].iloc[0]
        assert protein_a["stability"] == 1.0  # 4/4

        protein_b = result[result["feature"] == "protein_B"].iloc[0]
        assert protein_b["stability"] == 0.5  # 2/4

        protein_c = result[result["feature"] == "protein_C"].iloc[0]
        assert protein_c["stability"] == 0.25  # 1/4
