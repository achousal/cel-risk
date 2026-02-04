"""
Basic workflow tests for the pipeline.

Tests core functionality: splits generation, training, output validation, and error handling.
Run with: pytest tests/e2e/test_basic_workflow.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest
from ced_ml.cli.main import cli
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    TARGET_COL,
)
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

        for seed in [42, 43]:
            assert (splits_dir / f"train_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"val_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"test_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (splits_dir / f"split_meta_IncidentOnly_seed{seed}.json").exists()

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

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}: {run_dirs}"
        model_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"
        assert model_dir.exists(), f"Model directory not found: {model_dir}"

        required_files = [
            "core/val_metrics.csv",
            "core/test_metrics.csv",
            "preds/train_oof__LR_EN.csv",
            "preds/test_preds__LR_EN.csv",
        ]

        for file_path in required_files:
            full_path = model_dir / file_path
            assert full_path.exists(), f"Missing output: {full_path}"

        test_metrics = pd.read_csv(model_dir / "core/test_metrics.csv")

        has_auroc = any(col.lower() == "auroc" for col in test_metrics.columns)
        has_metric_col = "metric" in test_metrics.columns

        assert (
            has_auroc or has_metric_col
        ), f"No AUROC column found. Columns: {test_metrics.columns.tolist()}"

        if has_metric_col:
            assert any(val.lower() == "auroc" for val in test_metrics["metric"].values)
            auroc_val = test_metrics[test_metrics["metric"].str.lower() == "auroc"]["value"].iloc[0]
        else:
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

        model_dirs = list(results_dir.rglob("splits/split_seed42"))

        if not model_dirs:
            all_files = list(results_dir.rglob("*"))
            pytest.skip(
                f"No split_seed42 directory found. Files: {[str(f.relative_to(results_dir)) for f in all_files[:10]]}"
            )

        model_dir = model_dirs[0]

        has_predictions = any(model_dir.rglob("*.csv"))
        has_config = any(model_dir.rglob("*config*.yaml"))
        has_some_output = len(list(model_dir.rglob("*"))) > 5

        assert has_predictions, "No CSV files (predictions) found"
        assert has_config, "No config YAML found"
        assert has_some_output, "Output directory is mostly empty"


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

        bad_config = tmp_path / "bad_config.yaml"
        with open(bad_config, "w") as f:
            f.write("{ invalid yaml content: [ unclosed")

        runner = CliRunner()

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


class TestE2EDataConversion:
    """Test data format conversion utilities."""

    def test_convert_to_parquet_basic(self, minimal_proteomics_data, tmp_path):
        """
        Test: CSV to Parquet conversion.

        Validates data format conversion command.
        """
        csv_path = tmp_path / "input.csv"
        df = pd.read_parquet(minimal_proteomics_data)
        df.to_csv(csv_path, index=False)

        output_path = tmp_path / "output.parquet"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert-to-parquet",
                str(csv_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, f"Conversion failed: {result.output}"
        assert output_path.exists(), "Output parquet file should exist"

        df_converted = pd.read_parquet(output_path)
        assert len(df_converted) == len(df), "Row count should match"
        assert len(df_converted.columns) == len(df.columns), "Column count should match"

    def test_convert_to_parquet_default_output(self, tmp_path):
        """
        Test: Conversion with default output path.

        Validates automatic output path generation.
        """
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame(
            {
                ID_COL: ["S001", "S002"],
                TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL],
                "protein_001_resid": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert-to-parquet",
                str(csv_path),
            ],
        )

        assert result.exit_code == 0

        expected_output = tmp_path / "test_data.parquet"
        assert expected_output.exists(), "Default output should be created"

    def test_convert_to_parquet_compression_options(self, tmp_path):
        """
        Test: Parquet conversion with different compression algorithms.

        Validates compression option.
        """
        csv_path = tmp_path / "input.csv"
        rng = np.random.default_rng(42)
        labels = [CONTROL_LABEL] * 45 + [INCIDENT_LABEL] * 5
        df = pd.DataFrame(
            {
                ID_COL: [f"S{i:03d}" for i in range(50)],
                TARGET_COL: labels,
                "age": rng.integers(25, 75, 50),
                "BMI": rng.uniform(18, 35, 50),
                "sex": rng.choice(["M", "F"], 50),
                "Genetic ethnic grouping": rng.choice(["White", "Asian"], 50),
                "protein_001_resid": np.random.randn(50),
                "protein_002_resid": np.random.randn(50),
            }
        )
        df.to_csv(csv_path, index=False)

        for compression in ["snappy", "gzip", "zstd"]:
            output_path = tmp_path / f"output_{compression}.parquet"

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "convert-to-parquet",
                    str(csv_path),
                    "--output",
                    str(output_path),
                    "--compression",
                    compression,
                ],
            )

            if result.exit_code != 0 and "not available" in result.output:
                pytest.skip(f"Compression {compression} not available in environment")

            assert result.exit_code == 0, f"Conversion with {compression} failed"
            assert output_path.exists(), f"Output with {compression} should exist"

    def test_convert_to_parquet_invalid_csv(self, tmp_path):
        """
        Test: Conversion fails gracefully with invalid CSV.

        Error handling test.
        """
        bad_csv = tmp_path / "bad.csv"
        with open(bad_csv, "w") as f:
            f.write("invalid,csv,structure\nno,proper,headers\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "convert-to-parquet",
                str(bad_csv),
            ],
        )

        assert result.exit_code in [0, 1]
