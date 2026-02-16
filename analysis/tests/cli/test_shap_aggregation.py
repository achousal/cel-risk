"""
Tests for SHAP aggregation (importance, values, metadata, and plots).

Coverage:
- OOF SHAP importance: multi-split, empty, single, different features, filename, sorting, stability
- SHAP values: multi-split pooling, no files, val included
- SHAP metadata: consistent scales, inconsistent scales
- Aggregated SHAP plots: from pooled data, fallback to OOF importance
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ced_ml.cli.aggregation.orchestrator import (
    aggregate_shap_importance,
    aggregate_shap_metadata,
    aggregate_shap_values,
)
from ced_ml.cli.aggregation.plot_generator import generate_aggregated_shap_plots


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

    # Verify columns (includes dual rankings)
    expected_cols = [
        "feature",
        "mean_abs_shap",
        "std_abs_shap",
        "stability",
        "rank",
        "rank_pooled",
        "rank_stability",
    ]
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


# ---------------------------------------------------------------------------
# aggregate_shap_values tests
# ---------------------------------------------------------------------------


def _make_shap_parquet(split_dir: Path, model: str, n_samples: int, features: list[str]):
    """Helper: create a mock test SHAP parquet file."""
    shap_dir = split_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = {feat: rng.standard_normal(n_samples) for feat in features}
    df = pd.DataFrame(data)
    df.to_parquet(shap_dir / f"test_shap_values__{model}.parquet.gz", compression="gzip")
    return df


def test_aggregate_shap_values_multi_split(logger):
    """Test pooling test SHAP parquets across multiple splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        features = ["feat_A", "feat_B", "feat_C"]
        split_dirs = []
        total_samples = 0

        for i in range(3):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            split_dir.mkdir(parents=True)
            n = 10 + i
            _make_shap_parquet(split_dir, "LR_EN", n, features)
            split_dirs.append(split_dir)
            total_samples += n

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_values(split_dirs, "LR_EN", output_dir, logger)

        assert result is not None
        assert len(result) == total_samples
        assert "split_seed" in result.columns
        assert set(result["split_seed"].unique()) == {0, 1, 2}
        for feat in features:
            assert feat in result.columns

        # Verify output file
        out_path = output_dir / "shap" / "test_shap_values_pooled__LR_EN.parquet.gz"
        assert out_path.exists()


def test_aggregate_shap_values_no_files(logger):
    """Test graceful return when no SHAP parquets exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split_dirs = []
        for i in range(2):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            split_dir.mkdir(parents=True)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_values(split_dirs, "LR_EN", output_dir, logger)
        assert result is None
        assert not (output_dir / "shap").exists()


def test_aggregate_shap_values_val_included(logger):
    """Test that val SHAP parquets are pooled when present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        features = ["feat_X", "feat_Y"]
        split_dirs = []

        for i in range(2):
            split_dir = tmpdir_path / f"split_seed{i}" / "RF"
            split_dir.mkdir(parents=True)
            shap_dir = split_dir / "shap"
            shap_dir.mkdir(parents=True)

            # Test parquet
            _make_shap_parquet(split_dir, "RF", 5, features)

            # Val parquet
            rng = np.random.default_rng(i)
            val_df = pd.DataFrame({feat: rng.standard_normal(3) for feat in features})
            val_df.to_parquet(shap_dir / "val_shap_values__RF.parquet.gz", compression="gzip")

            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_values(split_dirs, "RF", output_dir, logger)

        assert result is not None
        # Test pooled exists
        assert (output_dir / "shap" / "test_shap_values_pooled__RF.parquet.gz").exists()
        # Val pooled exists
        assert (output_dir / "shap" / "val_shap_values_pooled__RF.parquet.gz").exists()

        val_pooled = pd.read_parquet(output_dir / "shap" / "val_shap_values_pooled__RF.parquet.gz")
        assert len(val_pooled) == 6  # 3 per split x 2 splits
        assert "split_seed" in val_pooled.columns


# ---------------------------------------------------------------------------
# aggregate_shap_metadata tests
# ---------------------------------------------------------------------------


def _write_shap_metadata(split_dir: Path, model: str, meta: dict):
    """Helper: write a mock SHAP metadata JSON."""
    cv_dir = split_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    with open(cv_dir / f"shap_metadata__{model}.json", "w") as f:
        json.dump(meta, f)


def test_aggregate_shap_metadata_consistent(logger):
    """Test metadata aggregation when all splits have the same scale."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split_dirs = []

        for i in range(3):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            _write_shap_metadata(
                split_dir,
                "LR_EN",
                {
                    "shap_output_scale": "log_odds",
                    "explainer_type": "LinearExplainer",
                    "explained_model_state": "calibrated",
                    "n_features": 50,
                    "n_background": 100,
                    "background_strategy": "kmeans",
                    "raw_dtype": "float64",
                },
            )
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_metadata(split_dirs, "LR_EN", output_dir, logger)

        assert result is not None
        assert result["scale_consistent"] is True
        assert result["explainer_consistent"] is True
        assert result["shap_output_scale"] == "log_odds"
        assert result["explainer_type"] == "LinearExplainer"
        assert result["n_splits_with_shap"] == 3
        assert result["n_splits_total"] == 3

        # Verify output file
        out_path = output_dir / "shap" / "shap_metadata_summary__LR_EN.json"
        assert out_path.exists()
        with open(out_path) as f:
            saved = json.load(f)
        assert saved["scale_consistent"] is True


def test_aggregate_shap_metadata_inconsistent(logger, caplog):
    """Test metadata aggregation warns when scales differ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split_dirs = []

        for i, scale in enumerate(["log_odds", "probability"]):
            split_dir = tmpdir_path / f"split_seed{i}" / "XGBoost"
            _write_shap_metadata(
                split_dir,
                "XGBoost",
                {
                    "shap_output_scale": scale,
                    "explainer_type": "TreeExplainer",
                    "explained_model_state": "raw",
                },
            )
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING):
            result = aggregate_shap_metadata(split_dirs, "XGBoost", output_dir, logger)

        assert result is not None
        assert result["scale_consistent"] is False
        assert set(result["scales_observed"]) == {"log_odds", "probability"}
        assert "inconsistent" in caplog.text.lower()


def test_aggregate_shap_metadata_no_files(logger):
    """Test metadata aggregation returns None when no metadata files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split_dirs = []
        for i in range(2):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            split_dir.mkdir(parents=True)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_metadata(split_dirs, "LR_EN", output_dir, logger)
        assert result is None


# ---------------------------------------------------------------------------
# generate_aggregated_shap_plots tests
# ---------------------------------------------------------------------------


def test_generate_aggregated_shap_plots_from_pooled(logger):
    """Test that bar, beeswarm, and dependence plots are created from pooled SHAP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Create a mock pooled SHAP DataFrame (20 samples x 6 features + split_seed)
        rng = np.random.default_rng(42)
        features = [f"feat_{i}" for i in range(6)]
        data = {feat: rng.standard_normal(20) for feat in features}
        data["split_seed"] = [0] * 10 + [1] * 10
        pooled_df = pd.DataFrame(data)

        metadata = {"shap_output_scale": "log_odds"}

        with (
            patch("ced_ml.plotting.shap_plots.plot_bar_importance") as mock_bar,
            patch("ced_ml.plotting.shap_plots.plot_beeswarm") as mock_bee,
            patch("ced_ml.plotting.shap_plots.plot_dependence") as mock_dep,
            patch("ced_ml.plotting.shap_plots.plot_heatmap") as mock_hm,
        ):

            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=None,
                shap_metadata=metadata,
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                logger=logger,
            )

            # Bar and beeswarm called once each (1 format)
            assert mock_bar.call_count == 1
            assert mock_bee.call_count == 1
            # Scatter called for top 5 features (we have 6, so 5)
            assert mock_dep.call_count == 5
            # Heatmap called once per format
            assert mock_hm.call_count == 1

            # Verify scale passed correctly
            bar_kwargs = mock_bar.call_args
            assert (
                bar_kwargs.kwargs.get("shap_output_scale") == "log_odds"
                or bar_kwargs[1].get("shap_output_scale") == "log_odds"
            )


def test_generate_aggregated_shap_plots_low_overlap_uses_oof_bar(logger):
    """Low overlap pooled SHAP skips unstable distribution plots and uses OOF bar fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Two split seeds with largely disjoint feature coverage
        pooled_df = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, np.nan, np.nan],
                "feat_b": [0.1, 0.2, np.nan, np.nan],
                "feat_c": [np.nan, np.nan, 1.5, 1.8],
                "feat_d": [np.nan, np.nan, 0.4, 0.3],
                "split_seed": [0, 0, 1, 1],
            }
        )

        oof_df = pd.DataFrame(
            {
                "feature": ["feat_a", "feat_b", "feat_c"],
                "mean_abs_shap": [0.5, 0.3, 0.2],
                "std_abs_shap": [0.05, 0.03, 0.02],
                "stability": [0.5, 0.5, 0.5],
                "rank": [1, 2, 3],
            }
        )

        with (
            patch("ced_ml.plotting.shap_plots.plot_bar_importance") as mock_bar,
            patch("ced_ml.plotting.shap_plots.plot_beeswarm") as mock_bee,
            patch("ced_ml.plotting.shap_plots.plot_dependence") as mock_dep,
            patch("ced_ml.plotting.shap_plots.plot_heatmap") as mock_hm,
            patch(
                "ced_ml.cli.aggregation.plot_generator._plot_bar_from_importance_csv"
            ) as mock_fallback,
        ):
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=oof_df,
                shap_metadata={"shap_output_scale": "log_odds"},
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                plot_shap_waterfall=False,
                logger=logger,
            )

            # Stable OOF bar fallback used; unstable pooled summary/distribution skipped.
            assert mock_fallback.call_count == 1
            assert mock_bar.call_count == 0
            assert mock_bee.call_count == 0
            assert mock_dep.call_count == 0
            assert mock_hm.call_count == 0


def test_generate_aggregated_shap_plots_filters_sparse_nan_features(logger):
    """Sparse features are filtered before pooled SHAP plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        pooled_df = pd.DataFrame(
            {
                "feat_dense": [0.1, -0.2, 0.3, -0.4, 0.2],
                "feat_sparse": [np.nan, np.nan, np.nan, np.nan, 0.9],
                "split_seed": [0, 0, 0, 0, 0],
            }
        )

        with (
            patch("ced_ml.plotting.shap_plots.plot_bar_importance") as mock_bar,
            patch("ced_ml.plotting.shap_plots.plot_beeswarm") as mock_bee,
            patch("ced_ml.plotting.shap_plots.plot_dependence") as mock_dep,
            patch("ced_ml.plotting.shap_plots.plot_heatmap") as mock_hm,
        ):
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=None,
                shap_metadata={"shap_output_scale": "log_odds"},
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                plot_shap_waterfall=False,
                logger=logger,
            )

            # feat_sparse has 20% coverage, below distribution threshold (80%),
            # so only feat_dense should appear in beeswarm/scatter/heatmap.
            assert mock_bar.call_count == 1
            assert mock_bee.call_count == 1
            assert mock_dep.call_count == 1
            assert mock_hm.call_count == 1

            bar_values, bar_features = mock_bar.call_args[0][:2]
            assert bar_features == ["feat_dense", "feat_sparse"]
            assert not np.isnan(bar_values).any()

            bee_values, _, bee_features = mock_bee.call_args[0][:3]
            assert bee_features == ["feat_dense"]
            assert not np.isnan(bee_values).any()


def test_generate_aggregated_shap_plots_fallback(logger):
    """Test fallback to OOF importance bar when no pooled data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        oof_df = pd.DataFrame(
            {
                "feature": ["feat_A", "feat_B", "feat_C"],
                "mean_abs_shap": [0.5, 0.3, 0.1],
                "std_abs_shap": [0.05, 0.03, 0.01],
                "stability": [1.0, 1.0, 0.75],
                "rank": [1, 2, 3],
            }
        )

        with patch(
            "ced_ml.cli.aggregation.plot_generator._plot_bar_from_importance_csv"
        ) as mock_fallback:
            generate_aggregated_shap_plots(
                pooled_shap_df=None,
                oof_shap_importance_df=oof_df,
                shap_metadata=None,
                model_name="RF",
                out_dir=out_dir,
                plot_formats=["png"],
                logger=logger,
            )

            assert mock_fallback.call_count == 1


def test_generate_aggregated_shap_plots_no_data(logger):
    """Test that nothing happens when no SHAP data is available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        generate_aggregated_shap_plots(
            pooled_shap_df=None,
            oof_shap_importance_df=None,
            shap_metadata=None,
            model_name="LR_EN",
            out_dir=out_dir,
            plot_formats=["png"],
            logger=logger,
        )

        # No shap plots dir should be created
        assert not (out_dir / "plots" / "shap").exists()


# ---------------------------------------------------------------------------
# Dual ranking tests
# ---------------------------------------------------------------------------


def test_dual_rankings_when_all_stable(mock_shap_splits, logger):
    """When all features appear in every split, rank_pooled == rank_stability."""
    split_dirs, output_dir = mock_shap_splits
    result = aggregate_shap_importance(split_dirs, "LR_EN", output_dir, logger)

    assert result is not None
    assert "rank_pooled" in result.columns
    assert "rank_stability" in result.columns

    # All stability == 1.0, so stability-weighted = mean_abs_shap * 1.0 -> same order
    assert list(result["rank_pooled"]) == list(result["rank_stability"])
    # backward-compat alias
    assert list(result["rank"]) == list(result["rank_pooled"])


def test_dual_rankings_with_partial_stability(logger):
    """Stability-weighted rank can differ from pooled rank when stability varies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Split 0: features A (high SHAP) and B (low SHAP)
        split0 = tmpdir_path / "split_seed0" / "LR_EN"
        (split0 / "cv").mkdir(parents=True)
        pd.DataFrame(
            {
                "feature": ["protein_A", "protein_B"],
                "mean_abs_shap": [0.9, 0.3],
                "std_abs_shap": [0.05, 0.03],
                "median_abs_shap": [0.88, 0.28],
                "n_folds_nonzero": [5, 5],
            }
        ).to_csv(split0 / "cv" / "oof_shap_importance__LR_EN.csv", index=False)

        # Split 1: only feature B (present in both splits -> higher stability)
        split1 = tmpdir_path / "split_seed1" / "LR_EN"
        (split1 / "cv").mkdir(parents=True)
        pd.DataFrame(
            {
                "feature": ["protein_B"],
                "mean_abs_shap": [0.3],
                "std_abs_shap": [0.03],
                "median_abs_shap": [0.28],
                "n_folds_nonzero": [5],
            }
        ).to_csv(split1 / "cv" / "oof_shap_importance__LR_EN.csv", index=False)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_importance([split0, split1], "LR_EN", output_dir, logger)

        assert result is not None
        # By pooled mean: A (0.9) > B (0.3) -> rank_pooled A=1, B=2
        a_row = result[result["feature"] == "protein_A"].iloc[0]
        b_row = result[result["feature"] == "protein_B"].iloc[0]
        assert a_row["rank_pooled"] == 1
        assert b_row["rank_pooled"] == 2

        # Stability-weighted: A = 0.9 * 0.5 = 0.45, B = 0.3 * 1.0 = 0.3
        # A still higher -> rank_stability A=1, B=2 in this case
        assert a_row["rank_stability"] == 1
        assert b_row["rank_stability"] == 2


# ---------------------------------------------------------------------------
# Feature matrix and sample metadata pooling tests
# ---------------------------------------------------------------------------


def _make_shap_features(split_dir: Path, model: str, n_samples: int, features: list[str]):
    """Helper: create a mock test SHAP features parquet file."""
    shap_dir = split_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = {feat: rng.standard_normal(n_samples) for feat in features}
    df = pd.DataFrame(data)
    df.to_parquet(shap_dir / f"test_shap_features__{model}.parquet.gz", compression="gzip")


def _make_shap_sample_meta(
    split_dir: Path, model: str, n_samples: int, include_adjusted: bool = False
):
    """Helper: create a mock test SHAP sample metadata parquet file."""
    shap_dir = split_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    data = {
        "y_true": rng.integers(0, 2, size=n_samples).astype(float),
        "y_prob": rng.random(n_samples),
    }
    if include_adjusted:
        data["y_prob_adjusted"] = rng.random(n_samples)
    df = pd.DataFrame(data)
    df.to_parquet(shap_dir / f"test_shap_sample_meta__{model}.parquet.gz", compression="gzip")


def test_aggregate_shap_pools_features_and_meta(logger):
    """Test that feature matrices and sample metadata are pooled alongside SHAP values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        features = ["feat_A", "feat_B"]
        split_dirs = []

        for i in range(2):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            split_dir.mkdir(parents=True)
            _make_shap_parquet(split_dir, "LR_EN", 5, features)
            _make_shap_features(split_dir, "LR_EN", 5, features)
            _make_shap_sample_meta(split_dir, "LR_EN", 5)
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_values(split_dirs, "LR_EN", output_dir, logger)
        assert result is not None

        # Verify features parquet was pooled
        feat_path = output_dir / "shap" / "test_shap_features_pooled__LR_EN.parquet.gz"
        assert feat_path.exists()
        feat_df = pd.read_parquet(feat_path)
        assert len(feat_df) == 10  # 5 per split x 2 splits
        assert "split_seed" in feat_df.columns

        # Verify sample meta parquet was pooled
        meta_path = output_dir / "shap" / "test_shap_sample_meta_pooled__LR_EN.parquet.gz"
        assert meta_path.exists()
        meta_df = pd.read_parquet(meta_path)
        assert len(meta_df) == 10
        assert "y_true" in meta_df.columns
        assert "y_prob" in meta_df.columns


def test_aggregate_shap_graceful_without_features(logger):
    """Test that aggregation works when feature/meta parquets are absent (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        features = ["feat_A", "feat_B"]
        split_dirs = []

        for i in range(2):
            split_dir = tmpdir_path / f"split_seed{i}" / "LR_EN"
            split_dir.mkdir(parents=True)
            _make_shap_parquet(split_dir, "LR_EN", 5, features)
            # No features or meta parquets
            split_dirs.append(split_dir)

        output_dir = tmpdir_path / "aggregated"
        output_dir.mkdir()

        result = aggregate_shap_values(split_dirs, "LR_EN", output_dir, logger)
        assert result is not None
        assert len(result) == 10

        # Feature/meta parquets should NOT exist
        assert not (output_dir / "shap" / "test_shap_features_pooled__LR_EN.parquet.gz").exists()
        assert not (output_dir / "shap" / "test_shap_sample_meta_pooled__LR_EN.parquet.gz").exists()


# ---------------------------------------------------------------------------
# Aggregate beeswarm with real X_transformed test
# ---------------------------------------------------------------------------


def test_generate_aggregated_shap_plots_uses_x_transformed(logger):
    """Test that beeswarm/dependence use pooled X_transformed when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        rng = np.random.default_rng(42)
        features = [f"feat_{i}" for i in range(6)]
        n = 20
        data = {feat: rng.standard_normal(n) for feat in features}
        data["split_seed"] = [0] * 10 + [1] * 10
        pooled_df = pd.DataFrame(data)

        # Create pooled features parquet in the expected location
        shap_dir = out_dir / "shap"
        shap_dir.mkdir(parents=True)
        feat_data = {feat: rng.standard_normal(n) for feat in features}
        feat_data["split_seed"] = [0] * 10 + [1] * 10
        feat_df = pd.DataFrame(feat_data)
        feat_df.to_parquet(
            shap_dir / "test_shap_features_pooled__LR_EN.parquet.gz", compression="gzip"
        )

        metadata = {"shap_output_scale": "log_odds"}

        with (
            patch("ced_ml.plotting.shap_plots.plot_bar_importance") as mock_bar,
            patch("ced_ml.plotting.shap_plots.plot_beeswarm") as mock_bee,
            patch("ced_ml.plotting.shap_plots.plot_dependence") as mock_dep,
            patch("ced_ml.plotting.shap_plots.plot_heatmap") as mock_hm,
        ):
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=None,
                shap_metadata=metadata,
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                plot_shap_waterfall=False,
                logger=logger,
            )

            assert mock_bar.call_count == 1
            assert mock_bee.call_count == 1
            assert mock_dep.call_count == 5
            assert mock_hm.call_count == 1

            # Verify beeswarm was called with real X_transformed (not shap_values)
            bee_args = mock_bee.call_args[0]
            shap_vals_arg = bee_args[0]
            color_arg = bee_args[1]
            # color_arg should NOT be shap_values (different array)
            assert not np.array_equal(shap_vals_arg, color_arg)


# ---------------------------------------------------------------------------
# Aggregate waterfall test
# ---------------------------------------------------------------------------


def test_generate_aggregated_waterfall_plots(logger):
    """Test that waterfall plots are generated from pooled sample metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        rng = np.random.default_rng(42)
        features = [f"feat_{i}" for i in range(6)]
        n = 20
        data = {feat: rng.standard_normal(n) for feat in features}
        data["split_seed"] = [0] * 10 + [1] * 10
        pooled_df = pd.DataFrame(data)

        # Create sample metadata with y_true and y_prob
        shap_dir = out_dir / "shap"
        shap_dir.mkdir(parents=True)
        y_true = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        y_prob = rng.random(n) * 0.5 + y_true * 0.3  # shift positives higher
        meta_df = pd.DataFrame(
            {
                "y_true": y_true.astype(float),
                "y_prob": y_prob,
                "split_seed": [0] * 10 + [1] * 10,
            }
        )
        meta_df.to_parquet(
            shap_dir / "test_shap_sample_meta_pooled__LR_EN.parquet.gz", compression="gzip"
        )

        metadata = {"shap_output_scale": "log_odds"}

        with patch("ced_ml.plotting.shap_plots.plot_waterfall") as mock_waterfall:
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=None,
                shap_metadata=metadata,
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                plot_shap_summary=False,
                plot_shap_dependence=False,
                plot_shap_waterfall=True,
                logger=logger,
            )

            # Should have generated waterfall plots (up to 4 categories: TP, FP, FN, TN)
            assert mock_waterfall.call_count >= 1
            assert mock_waterfall.call_count <= 4


def test_generate_aggregated_waterfall_skipped_without_meta(logger):
    """Test waterfall gracefully skipped when no sample metadata exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        rng = np.random.default_rng(42)
        features = [f"feat_{i}" for i in range(6)]
        n = 20
        data = {feat: rng.standard_normal(n) for feat in features}
        data["split_seed"] = [0] * 10 + [1] * 10
        pooled_df = pd.DataFrame(data)

        metadata = {"shap_output_scale": "log_odds"}

        with patch("ced_ml.plotting.shap_plots.plot_waterfall") as mock_waterfall:
            generate_aggregated_shap_plots(
                pooled_shap_df=pooled_df,
                oof_shap_importance_df=None,
                shap_metadata=metadata,
                model_name="LR_EN",
                out_dir=out_dir,
                plot_formats=["png"],
                plot_shap_summary=False,
                plot_shap_dependence=False,
                plot_shap_waterfall=True,
                logger=logger,
            )

            # No waterfall since no sample metadata
            assert mock_waterfall.call_count == 0
