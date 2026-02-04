"""
Tests for config management tools.

Tests:
- Config validation
- Config diff
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from ced_ml.cli.config_tools import (
    diff_configs,
    validate_config_file,
)


class TestValidateConfigFile:
    """Test config validation."""

    def test_validate_valid_splits_config(self):
        """Test validating a valid splits config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "splits_config.yaml"

            # Create valid config
            config_dict = {
                "mode": "development",
                "scenarios": ["IncidentOnly"],
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            is_valid, errors, warnings = validate_config_file(
                config_file=config_file,
                command="save-splits",
                strict=False,
                log_level=None,
            )

            assert is_valid is True
            assert len(errors) == 0

    def test_validate_invalid_splits_config(self):
        """Test validating an invalid splits config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "splits_config.yaml"

            # Create invalid config (val + test >= 1.0)
            config_dict = {
                "mode": "development",
                "val_size": 0.6,
                "test_size": 0.6,
            }

            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            is_valid, errors, warnings = validate_config_file(
                config_file=config_file,
                command="save-splits",
                strict=False,
                log_level=None,
            )

            assert is_valid is False
            assert len(errors) > 0


class TestDiffConfigs:
    """Test config diff functionality."""

    def test_diff_identical_configs(self):
        """Test diffing identical configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict = {
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                log_level=None,
            )

            assert len(diff_result["only_in_first"]) == 0
            assert len(diff_result["only_in_second"]) == 0
            assert len(diff_result["different_values"]) == 0

    def test_diff_different_configs(self):
        """Test diffing different configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict1 = {
                "n_splits": 10,
                "val_size": 0.25,
                "test_size": 0.25,
            }

            config_dict2 = {
                "n_splits": 20,
                "val_size": 0.25,
                "test_size": 0.30,
                "new_param": "value",
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict1, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict2, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                log_level=None,
            )

            # Different values
            assert "n_splits" in diff_result["different_values"]
            assert "test_size" in diff_result["different_values"]

            # Only in second
            assert "new_param" in diff_result["only_in_second"]

    def test_diff_nested_configs(self):
        """Test diffing nested config structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = Path(tmpdir) / "config1.yaml"
            config2 = Path(tmpdir) / "config2.yaml"

            config_dict1 = {
                "cv": {
                    "folds": 5,
                    "repeats": 10,
                },
                "features": {
                    "screen_top_n": 1000,
                },
            }

            config_dict2 = {
                "cv": {
                    "folds": 10,
                    "repeats": 10,
                },
                "features": {
                    "screen_top_n": 2000,
                },
            }

            with open(config1, "w") as f:
                yaml.dump(config_dict1, f)

            with open(config2, "w") as f:
                yaml.dump(config_dict2, f)

            diff_result = diff_configs(
                config_file1=config1,
                config_file2=config2,
                log_level=None,
            )

            # Check nested differences
            assert "cv.folds" in diff_result["different_values"]
            assert "features.screen_top_n" in diff_result["different_values"]
            assert "cv.repeats" in diff_result["same"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
