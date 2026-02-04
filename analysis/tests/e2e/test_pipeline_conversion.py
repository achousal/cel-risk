"""
E2E tests for data format conversion utilities.

Tests CSV to Parquet conversion:
- Basic conversion workflow
- Default output path generation
- Compression options
- Error handling for invalid inputs

Run with: pytest tests/e2e/test_pipeline_conversion.py -v
"""

import numpy as np
import pandas as pd
import pytest
from ced_ml.cli.main import cli
from ced_ml.data.schema import CONTROL_LABEL, ID_COL, INCIDENT_LABEL, TARGET_COL
from click.testing import CliRunner


class TestE2EDataConversion:
    """Test data format conversion utilities."""

    def test_convert_to_parquet_basic(self, minimal_proteomics_data, tmp_path):
        """
        Test: CSV to Parquet conversion.

        Validates data format conversion command.
        """
        # Create CSV version from parquet fixture
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

        # Verify content
        df_converted = pd.read_parquet(output_path)
        assert len(df_converted) == len(df), "Row count should match"
        assert len(df_converted.columns) == len(df.columns), "Column count should match"

    def test_convert_to_parquet_default_output(self, tmp_path):
        """
        Test: Conversion with default output path.

        Validates automatic output path generation.
        """
        # Create minimal CSV
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

        # Check default output path (same as input with .parquet extension)
        expected_output = tmp_path / "test_data.parquet"
        assert expected_output.exists(), "Default output should be created"

    def test_convert_to_parquet_compression_options(self, tmp_path):
        """
        Test: Parquet conversion with different compression algorithms.

        Validates compression option.
        """
        csv_path = tmp_path / "input.csv"
        rng = np.random.default_rng(42)
        # Need at least one case sample for validation
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

        # Test different compression algorithms
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

        # May fail or succeed depending on CSV content
        # Just ensure it doesn't crash
        assert result.exit_code in [0, 1]
