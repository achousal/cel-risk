"""
Tests for CLI utility functions.

Tests the shared utility functions used across CLI commands.
"""

from pathlib import Path

import pytest

from ced_ml.cli.utils.config_merge import (
    load_config_file,
    merge_config_with_cli,
    merge_nested_config,
)
from ced_ml.cli.utils.seed_parsing import parse_seed_list, parse_seed_range
from ced_ml.cli.utils.validation import (
    validate_model_name,
    validate_mutually_exclusive,
    validate_run_id_format,
)


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_mutually_exclusive_both_none(self):
        """Should not raise when both arguments are None."""
        validate_mutually_exclusive("--arg1", None, "--arg2", None)

    def test_validate_mutually_exclusive_one_provided(self):
        """Should not raise when only one argument is provided."""
        validate_mutually_exclusive("--arg1", "value1", "--arg2", None)
        validate_mutually_exclusive("--arg1", None, "--arg2", "value2")

    def test_validate_mutually_exclusive_both_provided(self):
        """Should raise when both arguments are provided."""
        with pytest.raises(ValueError, match="--arg1 and --arg2 are mutually exclusive"):
            validate_mutually_exclusive("--arg1", "value1", "--arg2", "value2")

    def test_validate_mutually_exclusive_custom_message(self):
        """Should use custom error message when provided."""
        with pytest.raises(ValueError, match="Custom error message"):
            validate_mutually_exclusive(
                "--arg1",
                "value1",
                "--arg2",
                "value2",
                error_message="Custom error message",
            )

    def test_validate_run_id_format_valid_timestamp(self):
        """Should accept valid timestamp format."""
        assert validate_run_id_format("20260127_115115")

    def test_validate_run_id_format_valid_custom(self):
        """Should accept valid custom format."""
        assert validate_run_id_format("production_v1")
        assert validate_run_id_format("test_run_123")

    def test_validate_run_id_format_invalid(self):
        """Should reject invalid format."""
        assert not validate_run_id_format("invalid-run-id!")
        assert not validate_run_id_format("run id with spaces")
        assert not validate_run_id_format("run@id")

    def test_validate_model_name_valid(self):
        """Should accept valid model names."""
        assert validate_model_name("LR_EN")
        assert validate_model_name("XGBoost")
        assert validate_model_name("RF")

    def test_validate_model_name_invalid(self):
        """Should reject invalid model names."""
        assert not validate_model_name("invalid-model!")
        assert not validate_model_name("model with spaces")


class TestSeedParsing:
    """Tests for seed parsing utilities."""

    def test_parse_seed_list_basic(self):
        """Should parse comma-separated seeds."""
        assert parse_seed_list("0,1,2") == [0, 1, 2]

    def test_parse_seed_list_with_spaces(self):
        """Should handle spaces around seeds."""
        assert parse_seed_list("72, 73, 74") == [72, 73, 74]

    def test_parse_seed_list_single_seed(self):
        """Should handle single seed."""
        assert parse_seed_list("42") == [42]

    def test_parse_seed_list_empty_string(self):
        """Should raise on empty string."""
        with pytest.raises(ValueError, match="Empty seed string"):
            parse_seed_list("")

    def test_parse_seed_list_invalid_format(self):
        """Should raise on invalid format."""
        with pytest.raises(ValueError, match="Invalid seed format"):
            parse_seed_list("0,1,invalid")

    def test_parse_seed_range_basic(self):
        """Should generate consecutive seeds."""
        assert parse_seed_range(0, 3) == [0, 1, 2]

    def test_parse_seed_range_custom_start(self):
        """Should generate seeds from custom start."""
        assert parse_seed_range(72, 5) == [72, 73, 74, 75, 76]

    def test_parse_seed_range_invalid_count(self):
        """Should raise on invalid count."""
        with pytest.raises(ValueError, match="Seed count must be at least 1"):
            parse_seed_range(0, 0)
        with pytest.raises(ValueError, match="Seed count must be at least 1"):
            parse_seed_range(0, -1)


class TestConfigMerge:
    """Tests for config merging utilities."""

    def test_merge_config_with_cli_cli_priority(self, tmp_path):
        """CLI args should take precedence over config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("cv_folds: 5\nmin_size: 10\n")

        cli_kwargs = {"cv_folds": 10, "min_size": None}
        param_keys = ["cv_folds", "min_size"]

        merged = merge_config_with_cli(config_path, cli_kwargs, param_keys)

        assert merged["cv_folds"] == 10  # From CLI
        assert merged["min_size"] == 10  # From config

    def test_merge_config_with_cli_no_config(self):
        """Should handle missing config file."""
        cli_kwargs = {"cv_folds": 10, "min_size": 5}
        param_keys = ["cv_folds", "min_size"]

        merged = merge_config_with_cli(None, cli_kwargs, param_keys)

        assert merged["cv_folds"] == 10
        assert merged["min_size"] == 5

    def test_merge_config_with_cli_neither_provided(self, tmp_path):
        """Should leave as None when neither CLI nor config provided."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("cv_folds: 5\n")

        cli_kwargs = {"cv_folds": None, "min_size": None}
        param_keys = ["cv_folds", "min_size"]

        merged = merge_config_with_cli(config_path, cli_kwargs, param_keys)

        assert merged["cv_folds"] == 5  # From config
        assert merged["min_size"] is None  # Not in either

    def test_load_config_file_success(self, tmp_path):
        """Should load valid YAML config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("models:\n  - LR_EN\n  - RF\n")

        config = load_config_file(config_path)

        assert config["models"] == ["LR_EN", "RF"]

    def test_load_config_file_missing(self):
        """Should return empty dict for missing file."""
        config = load_config_file(None)
        assert config == {}

        config = load_config_file(Path("/nonexistent/path.yaml"))
        assert config == {}

    def test_load_config_file_invalid_yaml(self, tmp_path):
        """Should raise on invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError, match="Failed to load config file"):
            load_config_file(config_path)

    def test_merge_nested_config_basic(self):
        """Should merge nested config sections."""
        config_params = {"composite_ranking": {"oof_weight": 0.6, "essentiality_weight": 0.3}}
        cli_kwargs = {"oof_weight": 0.7}
        param_keys = ["oof_weight", "essentiality_weight"]

        merged = merge_nested_config(config_params, cli_kwargs, "composite_ranking", param_keys)

        assert merged["oof_weight"] == 0.7  # From CLI
        assert merged["essentiality_weight"] == 0.3  # From config

    def test_merge_nested_config_missing_section(self):
        """Should handle missing nested section."""
        config_params = {}
        cli_kwargs = {"oof_weight": 0.7}
        param_keys = ["oof_weight", "essentiality_weight"]

        merged = merge_nested_config(config_params, cli_kwargs, "composite_ranking", param_keys)

        assert merged["oof_weight"] == 0.7  # From CLI
        assert merged["essentiality_weight"] is None  # Not in either
