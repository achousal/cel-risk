"""
Unit tests for aggregate-splits auto-discovery functionality.

Tests the resolve_results_dir_from_run_id function to ensure it correctly
discovers and returns all models when --run-id is used without --model.
"""

import tempfile
from pathlib import Path

import pytest

from ced_ml.cli.aggregate_splits import resolve_results_dir_from_run_id


@pytest.fixture
def mock_results_dir(tmp_path, monkeypatch):
    """Create a mock results directory structure."""
    results_root = tmp_path / "results"
    results_root.mkdir()

    run_dir = results_root / "run_20260131_232604"
    run_dir.mkdir()

    # Create multiple model directories
    for model in ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]:
        model_dir = run_dir / model
        model_dir.mkdir()

    # Create special directories that should be ignored
    (run_dir / "consensus").mkdir()
    (run_dir / ".hidden").mkdir()

    # Mock _find_results_root to return our temp results dir
    import ced_ml.cli.aggregate_splits as agg_module

    original_find = agg_module._find_results_root

    def mock_find():
        return results_root

    monkeypatch.setattr(agg_module, "_find_results_root", mock_find)

    yield results_root, run_dir

    # Restore original
    monkeypatch.setattr(agg_module, "_find_results_root", original_find)


class TestResolveResultsDirFromRunId:
    """Test resolve_results_dir_from_run_id function."""

    def test_single_model_specified(self, mock_results_dir):
        """Test that specifying a model returns only that model's path."""
        results_root, run_dir = mock_results_dir

        result = resolve_results_dir_from_run_id(
            run_id="20260131_232604", model="LR_EN", return_all_models=False
        )

        assert isinstance(result, str)
        assert Path(result).name == "LR_EN"
        assert "run_20260131_232604" in result

    def test_all_models_return_all_true(self, mock_results_dir):
        """Test that return_all_models=True returns dict of all models."""
        results_root, run_dir = mock_results_dir

        result = resolve_results_dir_from_run_id(
            run_id="20260131_232604", model=None, return_all_models=True
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {"LR_EN", "LinSVM_cal", "RF", "XGBoost"}

        # Verify paths are correct
        for model_name, model_path in result.items():
            assert Path(model_path).name == model_name
            assert "run_20260131_232604" in model_path

    def test_all_models_ignores_special_dirs(self, mock_results_dir):
        """Test that special directories are ignored."""
        results_root, run_dir = mock_results_dir

        result = resolve_results_dir_from_run_id(
            run_id="20260131_232604", model=None, return_all_models=True
        )

        # Should not include consensus or .hidden
        assert "consensus" not in result
        assert ".hidden" not in result
        assert "investigations" not in result

    def test_single_model_with_return_all_true(self, mock_results_dir):
        """Test that specifying model with return_all_models=True returns dict with one entry."""
        results_root, run_dir = mock_results_dir

        result = resolve_results_dir_from_run_id(
            run_id="20260131_232604", model="RF", return_all_models=True
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {"RF"}

    def test_nonexistent_model_raises(self, mock_results_dir):
        """Test that requesting a nonexistent model raises FileNotFoundError."""
        results_root, run_dir = mock_results_dir

        with pytest.raises(FileNotFoundError, match="Results directory not found for model"):
            resolve_results_dir_from_run_id(
                run_id="20260131_232604", model="NONEXISTENT", return_all_models=False
            )

    def test_nonexistent_run_raises(self, mock_results_dir):
        """Test that requesting a nonexistent run raises FileNotFoundError."""
        results_root, run_dir = mock_results_dir

        with pytest.raises(FileNotFoundError, match="No results found for run"):
            resolve_results_dir_from_run_id(
                run_id="99999999_999999", model=None, return_all_models=True
            )

    def test_multiple_models_without_specification_raises(self, mock_results_dir):
        """Test that multiple models without return_all_models=True raises ValueError."""
        results_root, run_dir = mock_results_dir

        with pytest.raises(ValueError, match="Multiple models found for run"):
            resolve_results_dir_from_run_id(
                run_id="20260131_232604", model=None, return_all_models=False
            )

    def test_auto_detect_latest_run_id(self, mock_results_dir):
        """Test that run_id=None auto-detects the latest run."""
        results_root, run_dir = mock_results_dir

        # Create an older run
        older_run = results_root / "run_20260130_000000"
        older_run.mkdir()
        (older_run / "LR_EN").mkdir()

        result = resolve_results_dir_from_run_id(
            run_id=None, model="LR_EN", return_all_models=False
        )

        # Should auto-detect the latest run (20260131_232604)
        assert "run_20260131_232604" in result
        assert "run_20260130_000000" not in result
