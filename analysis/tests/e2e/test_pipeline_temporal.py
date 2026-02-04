"""
E2E tests for temporal validation workflows.

Tests temporal split generation and chronological ordering:
- Temporal splits with date-based ordering
- Validation of train/test temporal separation

Run with: pytest tests/e2e/test_pipeline_temporal.py -v
"""

import pandas as pd
import pytest
import yaml
from ced_ml.cli.main import cli
from click.testing import CliRunner


class TestE2ETemporalValidation:
    """Test temporal validation workflow."""

    def test_temporal_splits_generation(self, temporal_proteomics_data, tmp_path):
        """
        Test: Generate temporal splits with chronological ordering.

        Validates temporal split logic.
        """
        splits_dir = tmp_path / "splits_temporal"
        splits_dir.mkdir()

        # Create temporal config
        config = {
            "mode": "development",
            "scenarios": ["IncidentOnly"],
            "n_splits": 1,
            "temporal_split": True,
            "temporal_col": "sample_date",
            "val_size": 0.15,
            "test_size": 0.15,
        }

        config_path = tmp_path / "temporal_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--config",
                str(config_path),
            ],
        )

        # Temporal splits should work now
        if result.exit_code != 0:
            pytest.fail(f"Temporal splits failed: {result.output}")

        # If implemented, verify chronological ordering
        df = pd.read_parquet(temporal_proteomics_data)
        train_idx = pd.read_csv(splits_dir / "train_idx_IncidentOnly_seed0.csv")["idx"].values
        test_idx = pd.read_csv(splits_dir / "test_idx_IncidentOnly_seed0.csv")["idx"].values

        # Train should have earliest dates
        train_dates = df.loc[train_idx, "sample_date"]
        test_dates = df.loc[test_idx, "sample_date"]

        assert train_dates.max() < test_dates.min(), "Temporal ordering violated"
