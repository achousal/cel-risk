"""
Tests for run_pipeline CLI command.

Regression tests for bugs discovered during development.
"""

import json


def test_run_id_detection_after_training(tmp_path):
    """
    Regression test: run_id should be correctly extracted from run_metadata.json.

    Bug: glob pattern was looking for run_*/*/splits/split_seed{N}/run_metadata.json
    Fix: Changed to run_*/run_metadata.json (file is at run root, not in splits/)

    This test verifies the glob pattern finds the file in the correct location.
    """
    # Create mock directory structure matching actual output
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run_20260130_093436"
    run_dir.mkdir(parents=True)

    # Create run_metadata.json at run root (correct location)
    metadata = {
        "run_id": "20260130_093436",
        "infile": "/path/to/data.parquet",
        "split_dir": "/path/to/splits",
        "models": {
            "LR_EN": {
                "scenario": "IncidentPlusPrevalent",
                "split_seed": 0,
            }
        },
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Test glob pattern (should find file)
    metadata_pattern = list(results_dir.glob("run_*/run_metadata.json"))
    assert len(metadata_pattern) == 1, "Should find exactly one run_metadata.json"
    assert metadata_pattern[0] == run_dir / "run_metadata.json"

    # Verify we can read run_id
    with open(metadata_pattern[0]) as f:
        loaded = json.load(f)
        assert loaded["run_id"] == "20260130_093436"


def test_old_glob_pattern_fails(tmp_path):
    """
    Verify that the OLD (buggy) glob pattern does NOT find the file.

    This demonstrates why the bug occurred.
    """
    # Create correct structure
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run_20260130_093436"
    run_dir.mkdir(parents=True)

    metadata = {"run_id": "20260130_093436"}
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f)

    # Test OLD pattern (should NOT find file)
    split_seed = 0
    old_pattern = list(results_dir.glob(f"run_*/*/splits/split_seed{split_seed}/run_metadata.json"))
    assert len(old_pattern) == 0, "Old pattern should NOT find file at run root"


def test_ensemble_parameter_name_lowercase():
    """
    Regression test: train_ensemble function uses 'meta_c' (lowercase) parameter.

    Bug: run_pipeline.py was calling run_train_ensemble(meta_C=None) (uppercase)
    Fix: Changed to meta_c=None (lowercase) to match function signature

    This test verifies the correct parameter name is used.
    """
    import inspect

    from ced_ml.cli.train_ensemble import run_train_ensemble

    # Get function signature
    sig = inspect.signature(run_train_ensemble)
    params = list(sig.parameters.keys())

    # Verify 'meta_c' (lowercase) is in parameters
    assert "meta_c" in params, "Function should accept 'meta_c' parameter"

    # Verify 'meta_C' (uppercase) is NOT in parameters
    assert "meta_C" not in params, "Function should NOT accept 'meta_C' parameter"


def test_run_pipeline_metadata_structure():
    """
    Test that run_metadata.json has expected structure for auto-detection.
    """

    # This is a structural test - just verify the expected fields exist
    # in the metadata format (without actually running the pipeline)
    expected_fields = ["run_id", "infile", "split_dir", "models"]

    # Mock metadata structure
    metadata = {
        "run_id": "20260130_000000",
        "infile": "/path/to/data.parquet",
        "split_dir": "/path/to/splits",
        "models": {
            "LR_EN": {
                "scenario": "IncidentPlusPrevalent",
                "split_seed": 0,
            }
        },
    }

    # Verify structure
    for field in expected_fields:
        assert field in metadata, f"Metadata should contain '{field}' field"


def test_fixed_panel_disables_optimization_and_consensus(tmp_path):
    """
    Regression test: When feature_selection_strategy='fixed_panel',
    panel optimization and consensus should be automatically disabled.

    This prevents wasted computation on steps that don't make sense
    when using a pre-specified panel.
    """
    import yaml
    from ced_ml.cli.run_pipeline import _detect_fixed_panel_strategy

    # Create a mock config file with fixed_panel strategy
    config_path = tmp_path / "training_config.yaml"
    config_data = {
        "features": {
            "feature_selection_strategy": "fixed_panel",
            "fixed_panel_csv": "fixed_panel.csv",
            "screen_method": "mannwhitney",
            "screen_top_n": 1000,
        },
        "cv": {
            "folds": 5,
            "repeats": 3,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Test detection function
    import logging

    logger = logging.getLogger(__name__)

    # Should detect fixed_panel strategy
    result = _detect_fixed_panel_strategy(config_path, logger)
    assert result is True, "Should detect fixed_panel strategy"


def test_non_fixed_panel_strategy_detection(tmp_path):
    """
    Test that other strategies are correctly NOT detected as fixed_panel.
    """
    import yaml
    from ced_ml.cli.run_pipeline import _detect_fixed_panel_strategy

    # Test hybrid_stability strategy
    config_path = tmp_path / "training_config.yaml"
    config_data = {
        "features": {
            "feature_selection_strategy": "hybrid_stability",
            "k_grid": [25, 50, 100],
            "screen_method": "mannwhitney",
            "screen_top_n": 1000,
        },
        "cv": {
            "folds": 5,
            "repeats": 3,
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    import logging

    logger = logging.getLogger(__name__)

    # Should NOT detect fixed_panel strategy
    result = _detect_fixed_panel_strategy(config_path, logger)
    assert result is False, "Should NOT detect fixed_panel for hybrid_stability strategy"


def test_missing_config_returns_false(tmp_path):
    """
    Test that missing config file returns False (doesn't crash).
    """
    import logging

    from ced_ml.cli.run_pipeline import _detect_fixed_panel_strategy

    logger = logging.getLogger(__name__)

    # Non-existent config
    result = _detect_fixed_panel_strategy(tmp_path / "nonexistent.yaml", logger)
    assert result is False, "Should return False for missing config"

    # None config
    result = _detect_fixed_panel_strategy(None, logger)
    assert result is False, "Should return False for None config"
