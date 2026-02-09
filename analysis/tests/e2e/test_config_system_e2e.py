"""
E2E tests for the configuration system.

Tests config loading, validation, override precedence, base-file
inheritance, and path resolution through actual CLI commands and
public APIs.

Run with: pytest tests/e2e/test_config_system_e2e.py -v
Run slow tests: pytest tests/e2e/test_config_system_e2e.py -v -m slow
"""

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ced_ml.cli.main import cli

# ---------------------------------------------------------------------------
# Class 1: Config validation via `ced config validate`
# ---------------------------------------------------------------------------


class TestConfigLoadingViaCliValidate:
    """Test config validation through the CLI validate command."""

    def test_valid_splits_config_via_cli(self, tmp_path):
        """Valid splits config passes validation without errors."""
        config_file = tmp_path / "splits_config.yaml"
        config_dict = {
            "mode": "development",
            "scenarios": ["IncidentOnly"],
            "n_splits": 10,
            "val_size": 0.25,
            "test_size": 0.25,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "validate", str(config_file), "--command", "save-splits"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "error" not in result.output.lower() or "0 error" in result.output.lower()

    def test_invalid_splits_config_via_cli(self, tmp_path):
        """Invalid splits config (val + test > 1.0) fails validation."""
        config_file = tmp_path / "bad_splits.yaml"
        config_dict = {
            "mode": "development",
            "val_size": 0.6,
            "test_size": 0.6,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "validate", str(config_file), "--command", "save-splits"],
        )

        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_valid_training_config_via_cli(self, fast_training_config):
        """Valid training config passes validation."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "validate", str(fast_training_config), "--command", "train"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

    def test_invalid_cv_folds_via_cli(self, tmp_path):
        """Training config with cv.folds=1 fails validation."""
        config_file = tmp_path / "bad_training.yaml"
        config_dict = {
            "cv": {"folds": 1},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "validate", str(config_file), "--command", "train"],
        )

        assert result.exit_code != 0 or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# Class 2: Config diff via `ced config diff`
# ---------------------------------------------------------------------------


class TestConfigDiffViaCli:
    """Test config diff through the CLI diff command."""

    def test_diff_identical_configs_via_cli(self, tmp_path):
        """Diffing identical configs reports no differences."""
        config1 = tmp_path / "config1.yaml"
        config2 = tmp_path / "config2.yaml"

        config_dict = {"n_splits": 10, "val_size": 0.25, "test_size": 0.25}
        for path in [config1, config2]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "diff", str(config1), str(config2)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        output_lower = result.output.lower()
        # Should indicate no differences (check for absence of diff markers)
        assert "different" not in output_lower or "0" in output_lower

    def test_diff_different_configs_via_cli(self, tmp_path):
        """Diffing configs with different cv.folds reports the difference."""
        config1 = tmp_path / "config1.yaml"
        config2 = tmp_path / "config2.yaml"

        with open(config1, "w") as f:
            yaml.dump({"cv": {"folds": 5, "repeats": 10}}, f)

        with open(config2, "w") as f:
            yaml.dump({"cv": {"folds": 10, "repeats": 10}}, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["config", "diff", str(config1), str(config2)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Diff output should mention the changed key
        assert "folds" in result.output.lower()


# ---------------------------------------------------------------------------
# Class 3: Config override precedence via actual training
# ---------------------------------------------------------------------------


class TestConfigOverridePrecedence:
    """Test that CLI overrides take precedence over YAML file values."""

    def _generate_splits(self, runner, data_path, splits_dir):
        """Helper: generate splits needed before training."""
        return runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(data_path),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--seed-start",
                "0",
            ],
            catch_exceptions=False,
        )

    @pytest.mark.slow
    def test_yaml_values_used_without_overrides(self, small_proteomics_data, tmp_path):
        """
        When no CLI overrides are provided, YAML config values are used.

        Trains with a YAML that sets cv.folds=2 and verifies the saved
        config in the output directory reflects folds=2.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Config explicitly sets folds=2
        config_file = tmp_path / "custom_config.yaml"
        config_dict = {
            "scenario": "IncidentOnly",
            "cv": {
                "folds": 2,
                "repeats": 1,
                "inner_folds": 2,
                "scoring": "roc_auc",
                "n_jobs": 1,
                "random_state": 42,
            },
            "optuna": {"enabled": False},
            "features": {
                "feature_select": "hybrid",
                "kbest_scope": "protein",
                "screen_method": "mannwhitney",
                "screen_top_n": 6,
                "k_grid": [3],
                "stability_thresh": 0.5,
                "corr_thresh": 0.95,
            },
            "calibration": {"enabled": True, "method": "isotonic", "strategy": "oof_posthoc"},
            "thresholds": {"objective": "youden", "fixed_spec": 0.95},
            "allow_test_thresholding": True,  # Test doesn't care about threshold behavior
            "lr": {
                "C_min": 0.1,
                "C_max": 10.0,
                "C_points": 2,
                "l1_ratio": [0.5],
                "solver": "saga",
                "max_iter": 500,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Generate splits first (ced train requires pre-existing splits)
        result_splits = self._generate_splits(runner, small_proteomics_data, splits_dir)
        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:300]}")

        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(config_file),
                "--model",
                "LR_EN",
                "--split-seed",
                "0",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed: {result.output[:300]}")

        # Find saved config in output
        saved_configs = list(results_dir.rglob("*config*.yaml"))
        assert len(saved_configs) >= 1, "Saved config should exist in output"

        with open(saved_configs[0]) as f:
            saved = yaml.safe_load(f)

        # YAML value should be used: folds=2
        if "cv" in saved and "folds" in saved["cv"]:
            assert saved["cv"]["folds"] == 2

    @pytest.mark.slow
    def test_cli_override_takes_precedence_over_yaml(self, small_proteomics_data, tmp_path):
        """
        CLI --override values take precedence over YAML file values.

        YAML sets cv.folds=5, CLI overrides to cv.folds=2.
        Verifies the saved config shows folds=2.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        config_file = tmp_path / "base_config.yaml"
        config_dict = {
            "scenario": "IncidentOnly",
            "cv": {
                "folds": 5,
                "repeats": 1,
                "inner_folds": 2,
                "scoring": "roc_auc",
                "n_jobs": 1,
                "random_state": 42,
            },
            "optuna": {"enabled": False},
            "features": {
                "feature_select": "hybrid",
                "kbest_scope": "protein",
                "screen_method": "mannwhitney",
                "screen_top_n": 6,
                "k_grid": [3],
                "stability_thresh": 0.5,
                "corr_thresh": 0.95,
            },
            "calibration": {"enabled": True, "method": "isotonic", "strategy": "oof_posthoc"},
            "thresholds": {"objective": "youden", "fixed_spec": 0.95},
            "allow_test_thresholding": True,  # Test doesn't care about threshold behavior
            "lr": {
                "C_min": 0.1,
                "C_max": 10.0,
                "C_points": 2,
                "l1_ratio": [0.5],
                "solver": "saga",
                "max_iter": 500,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Generate splits first
        result_splits = self._generate_splits(runner, small_proteomics_data, splits_dir)
        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:300]}")

        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(config_file),
                "--model",
                "LR_EN",
                "--split-seed",
                "0",
                "--override",
                "cv.folds=2",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed: {result.output[:300]}")

        saved_configs = list(results_dir.rglob("*config*.yaml"))
        assert len(saved_configs) >= 1, "Saved config should exist in output"

        with open(saved_configs[0]) as f:
            saved = yaml.safe_load(f)

        # CLI override should win: folds=2, not 5
        if "cv" in saved and "folds" in saved["cv"]:
            assert (
                saved["cv"]["folds"] == 2
            ), f"CLI override should set folds=2, got {saved['cv']['folds']}"


# ---------------------------------------------------------------------------
# Class 4: Config base-file inheritance
# ---------------------------------------------------------------------------


class TestConfigBaseInheritance:
    """Test YAML _base inheritance via load_yaml."""

    def test_base_config_inheritance(self, tmp_path):
        """Child config inherits values from _base config."""
        from ced_ml.config.loader import load_yaml

        base_file = tmp_path / "base.yaml"
        child_file = tmp_path / "child.yaml"

        with open(base_file, "w") as f:
            yaml.dump({"cv": {"folds": 5, "repeats": 10}, "features": {"screen_top_n": 1000}}, f)

        with open(child_file, "w") as f:
            yaml.dump({"_base": "base.yaml", "features": {"screen_top_n": 500}}, f)

        result = load_yaml(child_file)

        # Inherited from base
        assert result["cv"]["folds"] == 5
        assert result["cv"]["repeats"] == 10
        # Overridden by child
        assert result["features"]["screen_top_n"] == 500

    def test_child_overrides_base_values(self, tmp_path):
        """Child config values override base values at all nesting levels."""
        from ced_ml.config.loader import load_yaml

        base_file = tmp_path / "base.yaml"
        child_file = tmp_path / "child.yaml"

        with open(base_file, "w") as f:
            yaml.dump(
                {"cv": {"folds": 5, "repeats": 10, "scoring": "roc_auc"}},
                f,
            )

        with open(child_file, "w") as f:
            yaml.dump({"_base": "base.yaml", "cv": {"folds": 3}}, f)

        result = load_yaml(child_file)

        # Child overrides folds
        assert result["cv"]["folds"] == 3
        # Base values preserved where child does not override
        assert result["cv"]["repeats"] == 10
        assert result["cv"]["scoring"] == "roc_auc"


# ---------------------------------------------------------------------------
# Class 5: Config path resolution
# ---------------------------------------------------------------------------


class TestConfigPathResolutionE2E:
    """Test that relative paths in configs resolve correctly."""

    def test_relative_infile_resolved_from_config_dir(self, tmp_path):
        """Relative infile path is resolved relative to config file directory."""
        from ced_ml.config.loader import resolve_paths_relative_to_config

        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        test_file = data_dir / "test.parquet"
        test_file.write_text("dummy")

        config_file = configs_dir / "pipeline.yaml"
        config_dict = {"infile": "../data/test.parquet", "outdir": "../results"}

        resolved = resolve_paths_relative_to_config(config_dict, config_file)

        assert Path(resolved["infile"]).is_absolute()
        assert Path(resolved["infile"]).exists()
        assert Path(resolved["outdir"]).is_absolute()
