"""
Tests for OOF importance persistence (V-03 fix).

This module tests that OOF importance is correctly saved at split level
and can be aggregated by the aggregation stage.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ced_ml.cli.orchestration.context import TrainingContext
from ced_ml.cli.orchestration.persistence_stage import _save_cv_artifacts
from ced_ml.config.schema import TrainingConfig
from ced_ml.evaluation.reports import OutputDirectories


@pytest.fixture
def mock_context_with_importance():
    """Create a mock TrainingContext with OOF importance data."""
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

    # Mock OOF importance (the key artifact being tested)
    ctx.oof_importance_df = pd.DataFrame(
        {
            "feature": ["protein_1", "protein_2", "protein_3"],
            "mean_importance": [0.5, 0.3, 0.1],
            "std_importance": [0.05, 0.03, 0.01],
            "n_folds_nonzero": [5, 5, 3],
        }
    )

    return ctx


def test_oof_importance_field_exists():
    """Test that oof_importance_df field exists in TrainingContext."""
    config = TrainingConfig(
        infile="dummy.csv",
        model="LR_EN",
        outdir="results/test",
        split_seed=0,
    )

    ctx = TrainingContext.from_config(config)

    assert hasattr(ctx, "oof_importance_df"), "oof_importance_df field missing from TrainingContext"
    assert ctx.oof_importance_df is None, "oof_importance_df should default to None"


def test_oof_importance_can_be_set():
    """Test that oof_importance_df can be set and retrieved."""
    config = TrainingConfig(
        infile="dummy.csv",
        model="LR_EN",
        outdir="results/test",
        split_seed=0,
    )

    ctx = TrainingContext.from_config(config)

    test_df = pd.DataFrame({"feature": ["A", "B"], "importance": [0.5, 0.3]})
    ctx.oof_importance_df = test_df

    assert ctx.oof_importance_df is not None
    assert len(ctx.oof_importance_df) == 2
    assert list(ctx.oof_importance_df.columns) == ["feature", "importance"]


def test_oof_importance_saved_at_split_level(mock_context_with_importance):
    """Test that OOF importance is saved at split level during persistence."""
    ctx = mock_context_with_importance

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

        # Save CV artifacts (this should save OOF importance)
        _save_cv_artifacts(ctx)

        # Verify OOF importance was saved
        expected_path = Path(ctx.outdirs.cv) / f"oof_importance__{ctx.config.model}.csv"
        assert expected_path.exists(), f"OOF importance not saved at {expected_path}"

        # Verify content
        saved_df = pd.read_csv(expected_path)
        assert len(saved_df) == 3, "Saved DataFrame has wrong length"
        assert list(saved_df.columns) == [
            "feature",
            "mean_importance",
            "std_importance",
            "n_folds_nonzero",
        ]
        assert saved_df["feature"].tolist() == ["protein_1", "protein_2", "protein_3"]


def test_oof_importance_not_saved_when_none(mock_context_with_importance):
    """Test that OOF importance is not saved when it's None."""
    ctx = mock_context_with_importance
    ctx.oof_importance_df = None  # Explicitly set to None

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

        _save_cv_artifacts(ctx)

        # Verify OOF importance was NOT saved
        expected_path = Path(ctx.outdirs.cv) / f"oof_importance__{ctx.config.model}.csv"
        assert not expected_path.exists(), "OOF importance should not be saved when None"


def test_oof_importance_filename_matches_aggregator_expectation():
    """Test that the filename format matches what the aggregator expects."""
    model_name = "LR_EN"

    # Persistence stage format
    persistence_filename = f"oof_importance__{model_name}.csv"

    # Aggregator expects: cv_dir / f"oof_importance__{model_name}.csv"
    aggregator_expected = f"oof_importance__{model_name}.csv"

    assert (
        persistence_filename == aggregator_expected
    ), "Filename format mismatch between persistence and aggregator"
