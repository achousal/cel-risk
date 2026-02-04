"""Tests for aggregation discovery module."""

import logging

import pytest
from ced_ml.cli.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Create temporary results directory structure."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create split directories (production expects: results_dir/splits/split_seed*)
    splits_dir = results_dir / "splits"
    splits_dir.mkdir()
    for i in [0, 1, 5, 10]:
        (splits_dir / f"split_seed{i}").mkdir()

    # Create ensemble directories (production expects: results_dir/ENSEMBLE/splits/split_seed*)
    ensemble_dir = results_dir / "ENSEMBLE"
    ensemble_dir.mkdir()
    ensemble_splits_dir = ensemble_dir / "splits"
    ensemble_splits_dir.mkdir()
    for i in [0, 1, 5, 10]:
        (ensemble_splits_dir / f"split_seed{i}").mkdir()

    # Create a file (should be ignored)
    (splits_dir / "split_seed99.txt").touch()

    return results_dir


def test_discover_split_dirs_basic(tmp_results_dir):
    """Test basic split directory discovery."""
    dirs = discover_split_dirs(tmp_results_dir)
    assert len(dirs) == 4
    assert all(d.is_dir() for d in dirs)
    # Check sorted order
    assert [int(d.name.replace("split_seed", "")) for d in dirs] == [0, 1, 5, 10]


def test_discover_split_dirs_with_logger(tmp_results_dir, caplog):
    """Test split directory discovery with logger."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        dirs = discover_split_dirs(tmp_results_dir, logger=logger)
    assert len(dirs) == 4
    assert "Found 4 splits in" in caplog.text


def test_discover_split_dirs_empty(tmp_path):
    """Test discovery with no split directories."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Production code expects splits/ subdirectory
    splits_dir = results_dir / "splits"
    splits_dir.mkdir()
    dirs = discover_split_dirs(results_dir)
    assert len(dirs) == 0


def test_discover_ensemble_dirs_basic(tmp_results_dir):
    """Test basic ensemble directory discovery."""
    dirs = discover_ensemble_dirs(tmp_results_dir)
    assert len(dirs) == 4
    assert all(d.is_dir() for d in dirs)
    # Check sorted order
    assert [int(d.name.replace("split_seed", "")) for d in dirs] == [0, 1, 5, 10]


def test_discover_ensemble_dirs_no_ensemble(tmp_path):
    """Test discovery when no ENSEMBLE directory exists."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    dirs = discover_ensemble_dirs(results_dir)
    assert len(dirs) == 0


def test_discover_ensemble_dirs_with_logger(tmp_results_dir, caplog):
    """Test ensemble directory discovery with logger."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        dirs = discover_ensemble_dirs(tmp_results_dir, logger=logger)
    assert len(dirs) == 4
    assert "Discovered 4 ENSEMBLE split directories" in caplog.text


def test_discover_ensemble_dirs_empty_ensemble(tmp_path):
    """Test discovery with empty ENSEMBLE directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    ensemble_dir = results_dir / "ENSEMBLE"
    ensemble_dir.mkdir()
    # Production code expects splits/ subdirectory
    (ensemble_dir / "splits").mkdir()
    dirs = discover_ensemble_dirs(results_dir)
    assert len(dirs) == 0
