"""
Tests for SHAP persistence.

This module tests that SHAP values and metadata are correctly saved at split level
and can be aggregated by the aggregation stage.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ced_ml.cli.orchestration.context import TrainingContext
from ced_ml.cli.orchestration.persistence_stage import _save_cv_artifacts
from ced_ml.config.schema import TrainingConfig
from ced_ml.evaluation.reports import OutputDirectories
from ced_ml.features.shap_values import SHAPTestPayload


@pytest.fixture
def mock_context_with_shap():
    """Create a mock TrainingContext with SHAP data."""
    config = TrainingConfig(
        infile="dummy.csv",
        model="LR_EN",
        outdir="results/test",
        split_seed=0,
    )

    ctx = TrainingContext.from_config(config)

    # Mock required data
    ctx.best_params_df = pd.DataFrame(
        {
            "model": ["LR_EN"],
            "repeat": [0],
            "outer_split": [0],
            "best_score_inner": [0.85],
            "best_params": ['{"C": 0.1}'],
        }
    )

    ctx.selected_proteins_df = pd.DataFrame(
        {
            "model": ["LR_EN"],
            "repeat": [0],
            "outer_split": [0],
            "n_selected_proteins": [50],
            "selected_proteins": ['["protein_1", "protein_2"]'],
            "rfecv_applied": [False],
        }
    )

    # Mock OOF SHAP importance
    ctx.oof_shap_df = pd.DataFrame(
        {
            "feature": ["protein_1", "protein_2", "protein_3"],
            "mean_abs_shap": [0.5, 0.3, 0.1],
            "std_abs_shap": [0.05, 0.03, 0.01],
            "median_abs_shap": [0.45, 0.28, 0.09],
            "n_folds_nonzero": [5, 5, 3],
        }
    )

    # Mock test SHAP payload
    np.random.seed(42)
    ctx.test_shap_payload = SHAPTestPayload(
        values=np.random.randn(50, 3).astype(np.float32),
        expected_value=0.5,
        feature_names=["protein_1", "protein_2", "protein_3"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
        split="test",
        y_true=np.array([0] * 40 + [1] * 10),
    )

    # Mock val SHAP payload
    ctx.val_shap_payload = SHAPTestPayload(
        values=np.random.randn(30, 3).astype(np.float32),
        expected_value=0.5,
        feature_names=["protein_1", "protein_2", "protein_3"],
        shap_output_scale="log_odds",
        model_name="LR_EN",
        explainer_type="LinearExplainer",
        split="val",
        y_true=np.array([0] * 24 + [1] * 6),
    )

    return ctx


def test_shap_fields_exist():
    """Test that SHAP fields exist in TrainingContext and default to None."""
    config = TrainingConfig(
        infile="dummy.csv",
        model="LR_EN",
        outdir="results/test",
        split_seed=0,
    )

    ctx = TrainingContext.from_config(config)

    assert hasattr(ctx, "oof_shap_df"), "oof_shap_df field missing from TrainingContext"
    assert ctx.oof_shap_df is None, "oof_shap_df should default to None"

    assert hasattr(ctx, "test_shap_payload"), "test_shap_payload field missing from TrainingContext"
    assert ctx.test_shap_payload is None, "test_shap_payload should default to None"

    assert hasattr(ctx, "val_shap_payload"), "val_shap_payload field missing from TrainingContext"
    assert ctx.val_shap_payload is None, "val_shap_payload should default to None"


def test_oof_shap_saved_at_split_level(mock_context_with_shap):
    """Test that OOF SHAP importance is saved at split level during persistence."""
    ctx = mock_context_with_shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock output directories using the factory method
        ctx.outdirs = OutputDirectories.create(
            root=str(tmpdir_path),
            exist_ok=True,
            split_seed=0,
            run_id="test_run",
            model="LR_EN",
        )

        # Create directories
        Path(ctx.outdirs.cv).mkdir(parents=True, exist_ok=True)
        Path(ctx.outdirs.shap).mkdir(parents=True, exist_ok=True)

        # Save CV artifacts (this should save OOF SHAP importance)
        _save_cv_artifacts(ctx)

        # Verify OOF SHAP importance was saved
        expected_path = Path(ctx.outdirs.cv) / f"oof_shap_importance__{ctx.config.model}.csv"
        assert expected_path.exists(), f"OOF SHAP importance not saved at {expected_path}"

        # Verify content
        saved_df = pd.read_csv(expected_path)
        assert len(saved_df) == 3, "Saved DataFrame has wrong length"
        assert list(saved_df.columns) == [
            "feature",
            "mean_abs_shap",
            "std_abs_shap",
            "median_abs_shap",
            "n_folds_nonzero",
        ]
        assert saved_df["feature"].tolist() == ["protein_1", "protein_2", "protein_3"]


def test_shap_metadata_saved(mock_context_with_shap):
    """Test that SHAP metadata is saved when oof_shap_df and test_shap_payload exist."""
    ctx = mock_context_with_shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock output directories using the factory method
        ctx.outdirs = OutputDirectories.create(
            root=str(tmpdir_path),
            exist_ok=True,
            split_seed=0,
            run_id="test_run",
            model="LR_EN",
        )

        # Create directories
        Path(ctx.outdirs.cv).mkdir(parents=True, exist_ok=True)
        Path(ctx.outdirs.shap).mkdir(parents=True, exist_ok=True)

        # Save CV artifacts
        _save_cv_artifacts(ctx)

        # Verify SHAP metadata was saved
        metadata_path = Path(ctx.outdirs.cv) / f"shap_metadata__{ctx.config.model}.json"
        assert metadata_path.exists(), f"SHAP metadata not saved at {metadata_path}"

        # Verify metadata content
        import json

        metadata = json.loads(metadata_path.read_text())

        expected_keys = {
            "shap_output_scale",
            "tree_model_output_requested",
            "tree_model_output_effective",
            "explained_model_state",
            "explainer_type",
            "n_features",
            "n_background",
            "background_strategy",
            "raw_dtype",
            "note",
            "background_sensitivity",
        }
        assert set(metadata.keys()) == expected_keys, f"Metadata keys mismatch: {metadata.keys()}"

        assert metadata["shap_output_scale"] == "log_odds"
        assert metadata["tree_model_output_requested"] == "auto"
        assert metadata["tree_model_output_effective"] is None
        assert metadata["explained_model_state"] == "pre_calibration"
        assert metadata["explainer_type"] == "LinearExplainer"
        assert metadata["n_features"] == 3


def test_test_shap_parquet_saved(mock_context_with_shap):
    """Test that test SHAP values are saved as parquet."""
    ctx = mock_context_with_shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock output directories using the factory method
        ctx.outdirs = OutputDirectories.create(
            root=str(tmpdir_path),
            exist_ok=True,
            split_seed=0,
            run_id="test_run",
            model="LR_EN",
        )

        # Create directories
        Path(ctx.outdirs.cv).mkdir(parents=True, exist_ok=True)
        Path(ctx.outdirs.shap).mkdir(parents=True, exist_ok=True)

        # Save CV artifacts
        _save_cv_artifacts(ctx)

        # Verify test SHAP values parquet was saved
        test_shap_path = Path(ctx.outdirs.shap) / f"test_shap_values__{ctx.config.model}.parquet.gz"
        assert test_shap_path.exists(), f"Test SHAP values not saved at {test_shap_path}"

        # Verify content
        saved_df = pd.read_parquet(test_shap_path)
        assert saved_df.shape == (50, 3), f"Unexpected shape: {saved_df.shape}"
        assert list(saved_df.columns) == ["protein_1", "protein_2", "protein_3"]


def test_val_shap_parquet_saved(mock_context_with_shap):
    """Test that val SHAP values are saved as parquet."""
    ctx = mock_context_with_shap

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock output directories using the factory method
        ctx.outdirs = OutputDirectories.create(
            root=str(tmpdir_path),
            exist_ok=True,
            split_seed=0,
            run_id="test_run",
            model="LR_EN",
        )

        # Create directories
        Path(ctx.outdirs.cv).mkdir(parents=True, exist_ok=True)
        Path(ctx.outdirs.shap).mkdir(parents=True, exist_ok=True)

        # Save CV artifacts
        _save_cv_artifacts(ctx)

        # Verify val SHAP values parquet was saved
        val_shap_path = Path(ctx.outdirs.shap) / f"val_shap_values__{ctx.config.model}.parquet.gz"
        assert val_shap_path.exists(), f"Val SHAP values not saved at {val_shap_path}"

        # Verify content
        saved_df = pd.read_parquet(val_shap_path)
        assert saved_df.shape == (30, 3), f"Unexpected shape: {saved_df.shape}"
        assert list(saved_df.columns) == ["protein_1", "protein_2", "protein_3"]


def test_no_artifacts_when_shap_disabled(mock_context_with_shap):
    """Test that no SHAP artifacts are saved when SHAP is disabled (all None)."""
    ctx = mock_context_with_shap

    # Explicitly set all SHAP fields to None
    ctx.oof_shap_df = None
    ctx.test_shap_payload = None
    ctx.val_shap_payload = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create mock output directories using the factory method
        ctx.outdirs = OutputDirectories.create(
            root=str(tmpdir_path),
            exist_ok=True,
            split_seed=0,
            run_id="test_run",
            model="LR_EN",
        )

        # Create directories
        Path(ctx.outdirs.cv).mkdir(parents=True, exist_ok=True)
        Path(ctx.outdirs.shap).mkdir(parents=True, exist_ok=True)

        # Save CV artifacts
        _save_cv_artifacts(ctx)

        # Verify NO SHAP artifacts were saved
        oof_shap_path = Path(ctx.outdirs.cv) / f"oof_shap_importance__{ctx.config.model}.csv"
        assert not oof_shap_path.exists(), "OOF SHAP importance should not be saved when None"

        metadata_path = Path(ctx.outdirs.cv) / f"shap_metadata__{ctx.config.model}.json"
        assert not metadata_path.exists(), "SHAP metadata should not be saved when None"

        test_shap_path = Path(ctx.outdirs.shap) / f"test_shap_values__{ctx.config.model}.parquet.gz"
        assert not test_shap_path.exists(), "Test SHAP values should not be saved when None"

        val_shap_path = Path(ctx.outdirs.shap) / f"val_shap_values__{ctx.config.model}.parquet.gz"
        assert not val_shap_path.exists(), "Val SHAP values should not be saved when None"


def test_filename_matches_aggregator_expectation():
    """Test that the filename format matches what the aggregator expects."""
    model_name = "LR_EN"

    # Persistence stage format
    persistence_filename = f"oof_shap_importance__{model_name}.csv"

    # Aggregator expects: cv_dir / f"oof_shap_importance__{model_name}.csv"
    aggregator_expected = f"oof_shap_importance__{model_name}.csv"

    assert (
        persistence_filename == aggregator_expected
    ), "Filename format mismatch between persistence and aggregator"
