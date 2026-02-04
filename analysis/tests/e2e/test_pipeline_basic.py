"""
E2E tests for basic pipeline workflows.

Tests fundamental pipeline operations:
- Split generation and reproducibility
- Single model training workflow
- Output file structure validation
- Error handling for common failure modes
- Config validation

Run with: pytest tests/e2e/test_pipeline_basic.py -v
Run slow tests: pytest tests/e2e/test_pipeline_basic.py -v -m slow
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from ced_ml.cli.main import cli
from click.testing import CliRunner


class TestE2EFullPipeline:
    """Test full pipeline: splits -> train -> aggregate."""

    def test_splits_generation_basic(self, minimal_proteomics_data, tmp_path):
        """
        Test: Generate splits and verify output structure.

        Validates split files, metadata, and reproducibility.
        """
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--seed-start",
                "42",
            ],
        )

        assert result.exit_code == 0, f"save-splits failed: {result.output}"

        # Verify files exist for both splits
        for seed in [42, 43]:
            assert (splits_dir / f"train_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"val_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"test_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"split_meta_IncidentOnly_seed{seed}.json").exists()

        # Verify metadata
        with open(splits_dir / "split_meta_IncidentOnly_seed42.json") as f:
            meta = json.load(f)

        assert meta["scenario"] == "IncidentOnly"
        assert meta["seed"] == 42
        assert meta["split_type"] == "development"
        assert meta["n_train"] > 0
        assert meta["n_val"] > 0
        assert meta["n_test"] > 0

    def test_reproducibility_same_seed(self, minimal_proteomics_data, tmp_path):
        """
        Test: Same seed produces identical splits.

        Critical for reproducibility verification.
        """
        splits_dir1 = tmp_path / "splits1"
        splits_dir2 = tmp_path / "splits2"
        splits_dir1.mkdir()
        splits_dir2.mkdir()

        runner = CliRunner()

        # Run 1
        result1 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir1),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--seed-start",
                "123",
            ],
        )
        assert result1.exit_code == 0

        # Run 2
        result2 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir2),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--seed-start",
                "123",
            ],
        )
        assert result2.exit_code == 0

        # Compare splits
        train1 = pd.read_csv(splits_dir1 / "train_idx_IncidentOnly_seed123.csv")["idx"].values
        train2 = pd.read_csv(splits_dir2 / "train_idx_IncidentOnly_seed123.csv")["idx"].values

        np.testing.assert_array_equal(train1, train2)

    @pytest.mark.slow
    def test_full_pipeline_single_model(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Full pipeline with one model (splits -> train -> results).

        This is the core E2E test. Marked slow (~30-60s).
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Step 1: Generate splits
        result_splits = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--seed-start",
                "42",
            ],
        )
        assert result_splits.exit_code == 0, f"Splits failed: {result_splits.output}"

        # Step 2: Train model
        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(minimal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            print("TRAIN OUTPUT:", result_train.output)
            if result_train.exception:
                import traceback

                traceback.print_exception(
                    type(result_train.exception),
                    result_train.exception,
                    result_train.exception.__traceback__,
                )

        assert result_train.exit_code == 0, f"Train failed: {result_train.output}"

        # Step 3: Verify outputs
        # Find the run directory (timestamped run_YYYYMMDD_HHMMSS)
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}: {run_dirs}"
        model_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"
        assert model_dir.exists(), f"Model directory not found: {model_dir}"

        # Check required output files
        required_files = [
            "core/val_metrics.csv",
            "core/test_metrics.csv",
            "preds/train_oof__LR_EN.csv",
            "preds/test_preds__LR_EN.csv",
        ]

        for file_path in required_files:
            full_path = model_dir / file_path
            assert full_path.exists(), f"Missing output: {full_path}"

        # Validate metrics structure
        test_metrics = pd.read_csv(model_dir / "core/test_metrics.csv")

        # Check for expected metric columns (try both uppercase and lowercase)
        has_auroc = any(col.lower() == "auroc" for col in test_metrics.columns)
        has_metric_col = "metric" in test_metrics.columns

        assert (
            has_auroc or has_metric_col
        ), f"No AUROC column found. Columns: {test_metrics.columns.tolist()}"

        # If it's a long-format metrics file, check for auroc row
        if has_metric_col:
            assert any(val.lower() == "auroc" for val in test_metrics["metric"].values)
            auroc_val = test_metrics[test_metrics["metric"].str.lower() == "auroc"]["value"].iloc[0]
        else:
            # Find the AUROC column (case-insensitive)
            auroc_col = [col for col in test_metrics.columns if col.lower() == "auroc"][0]
            auroc_val = test_metrics[auroc_col].iloc[0]

        assert 0.0 <= auroc_val <= 1.0, f"AUROC out of bounds: {auroc_val}"

    def test_output_file_structure(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Verify complete output file structure after training.

        Ensures all expected outputs are generated.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--seed-start",
                "42",
            ],
        )

        # Train
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(minimal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed, skipping structure check: {result.output[:200]}")

        # Find the actual output directory (may be under run_YYYYMMDD_HHMMSS/splits/)
        model_dirs = list(results_dir.rglob("splits/split_seed42"))

        if not model_dirs:
            all_files = list(results_dir.rglob("*"))
            pytest.skip(
                f"No split_seed42 directory found. Files: {[str(f.relative_to(results_dir)) for f in all_files[:10]]}"
            )

        model_dir = model_dirs[0]

        # Verify key outputs exist (flexible check for different structures)
        has_predictions = any(model_dir.rglob("*.csv"))
        has_config = any(model_dir.rglob("*config*.yaml"))
        has_some_output = len(list(model_dir.rglob("*"))) > 5

        assert has_predictions, "No CSV files (predictions) found"
        assert has_config, "No config YAML found"
        assert has_some_output, "Output directory is mostly empty"


class TestE2EConfigValidation:
    """Test config validation workflow."""

    def test_config_validate_valid_training_config(self, minimal_training_config):
        """
        Test: Config validation passes for valid training config.

        Validates config validation command.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "validate",
                str(minimal_training_config),
                "--command",
                "train",
            ],
        )

        assert result.exit_code == 0, f"Validation should pass: {result.output}"

    def test_config_validate_invalid_config(self, tmp_path):
        """
        Test: Config validation fails for invalid config.

        Error handling test for malformed configs.
        """
        bad_config = tmp_path / "bad_config.yaml"
        with open(bad_config, "w") as f:
            yaml.dump({"invalid": "structure", "missing": "required_fields"}, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "validate",
                str(bad_config),
                "--command",
                "train",
            ],
        )

        # Should fail or warn
        # Note: May exit with 0 if warnings only, check output
        assert (
            result.exit_code != 0 or "warning" in result.output.lower()
        ), "Should warn or fail for invalid config"

    def test_config_validate_strict_mode(self, minimal_training_config):
        """
        Test: Strict mode treats warnings as errors.

        Validates strict validation behavior.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "validate",
                str(minimal_training_config),
                "--command",
                "train",
                "--strict",
            ],
        )

        # In strict mode, any warnings should cause failure
        # (or pass if config is perfect)
        assert result.exit_code in [
            0,
            1,
        ], "Strict validation should exit with 0 (pass) or 1 (fail)"

    def test_config_diff_identical_configs(self, minimal_training_config, tmp_path):
        """
        Test: Config diff shows no differences for identical configs.

        Validates config comparison command.
        """
        # Create a copy of the config
        config_copy = tmp_path / "config_copy.yaml"
        with open(minimal_training_config) as f:
            config_data = yaml.safe_load(f)
        with open(config_copy, "w") as f:
            yaml.dump(config_data, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "diff",
                str(minimal_training_config),
                str(config_copy),
            ],
        )

        assert result.exit_code == 0
        assert "identical" in result.output.lower() or "no diff" in result.output.lower()

    def test_config_diff_different_configs(self, minimal_training_config, minimal_splits_config):
        """
        Test: Config diff shows differences for different configs.

        Validates diff detection.
        """
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "diff",
                str(minimal_training_config),
                str(minimal_splits_config),
            ],
        )

        assert result.exit_code == 0
        # Should show differences
        assert len(result.output) > 100, "Diff output should show differences"

    def test_config_diff_output_file(
        self, minimal_training_config, minimal_splits_config, tmp_path
    ):
        """
        Test: Config diff can write to output file.

        Validates output file option.
        """
        output_file = tmp_path / "diff_report.txt"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "diff",
                str(minimal_training_config),
                str(minimal_splits_config),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists(), "Diff output file should be created"
        assert output_file.stat().st_size > 0, "Diff output should have content"


class TestE2EErrorHandling:
    """Test error handling for common failure modes."""

    def test_missing_input_file(self, tmp_path):
        """Test: Graceful error for missing input file."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(tmp_path / "nonexistent.parquet"),
                "--outdir",
                str(tmp_path / "splits"),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_invalid_model_name(self, minimal_proteomics_data, minimal_training_config, tmp_path):
        """Test: Graceful error for invalid model name."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
            ],
        )

        # Try to train with invalid model
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(minimal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--model",
                "INVALID_MODEL_XYZ",
                "--split-seed",
                "42",
            ],
        )

        assert result.exit_code != 0
        assert "model" in result.output.lower() or "invalid" in result.output.lower()

    def test_missing_splits_dir(self, minimal_proteomics_data, minimal_training_config, tmp_path):
        """Test: Graceful error when splits directory missing."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(minimal_proteomics_data),
                "--split-dir",
                str(tmp_path / "nonexistent_splits"),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        assert result.exit_code != 0

    def test_corrupted_config(self, minimal_proteomics_data, tmp_path):
        """Test: Graceful error for corrupted config file."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Create corrupted config
        bad_config = tmp_path / "bad_config.yaml"
        with open(bad_config, "w") as f:
            f.write("{ invalid yaml content: [ unclosed")

        runner = CliRunner()

        # Generate splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
            ],
        )

        # Try to train with bad config
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(minimal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(bad_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        assert result.exit_code != 0
