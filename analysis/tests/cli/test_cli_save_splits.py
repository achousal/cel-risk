"""
Integration tests for save-splits CLI.

Tests the full pipeline from CLI invocation through split generation
to verify D.1.4 CLI integration is complete.
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
    PREVALENT_LABEL,
    TARGET_COL,
)
from click.testing import CliRunner


@pytest.fixture
def toy_proteomics_csv(tmp_path):
    """Create a minimal proteomics CSV for testing."""
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_proteins = 10

    # Create data with enough samples for stratification
    # Use balanced groups to ensure sufficient samples in each stratum
    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_samples)],
        TARGET_COL: ([CONTROL_LABEL] * 800 + [INCIDENT_LABEL] * 100 + [PREVALENT_LABEL] * 100),
        "age": rng.integers(30, 70, n_samples),  # Narrower age range for better stratification
        "BMI": rng.uniform(18, 35, n_samples),
        "sex": rng.choice(["M", "F"], n_samples),
        "Genetic_ethnic_grouping": rng.choice(
            ["White", "Asian"], n_samples  # Fewer categories for better stratification
        ),
    }

    # Add protein columns
    for i in range(n_proteins):
        data[f"PROT_{i:03d}_resid"] = rng.standard_normal(n_samples)

    df = pd.DataFrame(data)
    csv_path = tmp_path / "toy_proteomics.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


class TestSaveSplitsCLIBasic:
    """Basic CLI functionality tests."""

    def test_help_command(self):
        """Test that --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["save-splits", "--help"])
        assert result.exit_code == 0
        assert "Generate train/val/test splits" in result.output
        assert "--infile" in result.output
        assert "--outdir" in result.output

    def test_missing_infile_raises(self):
        """Test that missing --infile raises error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["save-splits", "--outdir", "tmp"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Error" in result.output


class TestSaveSplitsDevelopmentMode:
    """Test development mode (TRAIN/VAL/TEST only)."""

    def test_basic_development_split(self, toy_proteomics_csv, tmp_path):
        """Test basic three-way split generation."""
        outdir = tmp_path / "splits_dev"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
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
            ],
        )

        if result.exit_code != 0:
            print("STDOUT:", result.output)
            if result.exception:
                import traceback

                print(
                    "EXCEPTION:",
                    traceback.format_exception(
                        type(result.exception),
                        result.exception,
                        result.exception.__traceback__,
                    ),
                )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check output files exist (new format with scenario in filename)
        assert (outdir / "train_idx_IncidentOnly_seed0.csv").exists()
        assert (outdir / "val_idx_IncidentOnly_seed0.csv").exists()
        assert (outdir / "test_idx_IncidentOnly_seed0.csv").exists()
        assert (outdir / "split_meta_IncidentOnly_seed0.json").exists()

        # Validate splits
        train_idx = pd.read_csv(outdir / "train_idx_IncidentOnly_seed0.csv")["idx"].values
        val_idx = pd.read_csv(outdir / "val_idx_IncidentOnly_seed0.csv")["idx"].values
        test_idx = pd.read_csv(outdir / "test_idx_IncidentOnly_seed0.csv")["idx"].values

        # No overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # All non-negative integers
        assert all(train_idx >= 0)
        assert all(val_idx >= 0)
        assert all(test_idx >= 0)

        # Validate metadata
        with open(outdir / "split_meta_IncidentOnly_seed0.json") as f:
            meta = json.load(f)

        assert meta["scenario"] == "IncidentOnly"
        assert meta["seed"] == 0
        assert meta["split_type"] == "development"
        assert meta["n_train"] == len(train_idx)
        assert meta["n_val"] == len(val_idx)
        assert meta["n_test"] == len(test_idx)
        assert 0 <= meta["prevalence_train"] <= 1
        assert 0 <= meta["prevalence_val"] <= 1
        assert 0 <= meta["prevalence_test"] <= 1

    def test_incident_plus_prevalent_scenario(self, toy_proteomics_csv, tmp_path):
        """Test IncidentPlusPrevalent scenario."""
        outdir = tmp_path / "splits_inc_prev"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentPlusPrevalent",
                "--n-splits",
                "1",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--prevalent-train-only",
                "--prevalent-train-frac",
                "0.5",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Files should exist (new format with scenario in filename)
        assert (outdir / "train_idx_IncidentPlusPrevalent_seed0.csv").exists()
        assert (outdir / "val_idx_IncidentPlusPrevalent_seed0.csv").exists()
        assert (outdir / "test_idx_IncidentPlusPrevalent_seed0.csv").exists()

        # Load original data to verify prevalent handling
        df = pd.read_csv(toy_proteomics_csv)
        val_idx = pd.read_csv(outdir / "val_idx_IncidentPlusPrevalent_seed0.csv")["idx"].values
        test_idx = pd.read_csv(outdir / "test_idx_IncidentPlusPrevalent_seed0.csv")["idx"].values

        # VAL and TEST should only have Incident + Controls (no prevalent)
        val_labels = df.loc[val_idx, TARGET_COL].values
        test_labels = df.loc[test_idx, TARGET_COL].values

        assert PREVALENT_LABEL not in val_labels
        assert PREVALENT_LABEL not in test_labels

        # TRAIN may have prevalent (not verified with assertion due to small sample size)

    def test_control_downsampling(self, toy_proteomics_csv, tmp_path):
        """Test control downsampling works."""
        outdir = tmp_path / "splits_downsample"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
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
                "--train-control-per-case",
                "5.0",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check that TRAIN has reduced controls
        df = pd.read_csv(toy_proteomics_csv)
        train_idx = pd.read_csv(outdir / "train_idx_IncidentOnly_seed0.csv")["idx"].values
        train_labels = df.loc[train_idx, TARGET_COL].values

        n_cases = (train_labels == INCIDENT_LABEL).sum()
        n_controls = (train_labels == CONTROL_LABEL).sum()

        # Ratio should be approximately 1:5 (allowing some variance due to stratification)
        if n_cases > 0:
            ratio = n_controls / n_cases
            assert ratio <= 6.0, f"Control ratio too high: {ratio:.2f}"

    def test_repeated_splits(self, toy_proteomics_csv, tmp_path):
        """Test repeated splits with different seeds."""
        outdir = tmp_path / "splits_repeated"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "3",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--seed-start",
                "100",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Check that all 3 splits were created (new format with scenario in filename)
        for seed in [100, 101, 102]:
            assert (outdir / f"train_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (outdir / f"val_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (outdir / f"test_idx_IncidentOnly_seed{seed}.csv").exists()
            assert (outdir / f"split_meta_IncidentOnly_seed{seed}.json").exists()

        # Verify splits are different
        train_100 = pd.read_csv(outdir / "train_idx_IncidentOnly_seed100.csv")["idx"].values
        train_101 = pd.read_csv(outdir / "train_idx_IncidentOnly_seed101.csv")["idx"].values

        # Different seeds should produce different splits
        assert not np.array_equal(sorted(train_100), sorted(train_101))


class TestSaveSplitsHoldoutMode:
    """Test holdout mode (TRAIN/VAL/TEST + HOLDOUT)."""

    def test_holdout_creation(self, toy_proteomics_csv, tmp_path):
        """Test holdout set creation."""
        outdir = tmp_path / "splits_holdout"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
                "--mode",
                "holdout",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--holdout-size",
                "0.20",  # Reduced to 20% to leave more for dev splits
            ],
        )

        if result.exit_code != 0:
            print(f"CLI FAILED with exit code {result.exit_code}")
            print(f"Output:\n{result.output}")
            if result.exception:
                import traceback

                print("Exception:")
                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )

        assert (
            result.exit_code == 0
        ), f"CLI failed with code {result.exit_code}: {result.output[:500]}"

        # Check holdout files (scenario-specific naming per M6 fix)
        assert (outdir / "HOLDOUT_idx_IncidentOnly.csv").exists()
        assert (outdir / "HOLDOUT_meta_IncidentOnly.json").exists()

        # Check development split files (new format with scenario in filename)
        assert (outdir / "train_idx_IncidentOnly_seed0.csv").exists()
        assert (outdir / "val_idx_IncidentOnly_seed0.csv").exists()
        assert (outdir / "test_idx_IncidentOnly_seed0.csv").exists()

        # Load indices
        holdout_idx = pd.read_csv(outdir / "HOLDOUT_idx_IncidentOnly.csv")["idx"].values
        train_idx = pd.read_csv(outdir / "train_idx_IncidentOnly_seed0.csv")["idx"].values
        val_idx = pd.read_csv(outdir / "val_idx_IncidentOnly_seed0.csv")["idx"].values
        test_idx = pd.read_csv(outdir / "test_idx_IncidentOnly_seed0.csv")["idx"].values

        # Holdout should be disjoint from development sets
        dev_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(set(holdout_idx) & set(dev_idx)) == 0

        # Validate holdout metadata
        with open(outdir / "HOLDOUT_meta_IncidentOnly.json") as f:
            meta = json.load(f)

        assert meta["split_type"] == "holdout"
        assert meta["n_holdout"] == len(holdout_idx)
        assert "NEVER use this set during development" in meta["note"]


class TestSaveSplitsErrorHandling:
    """Test error handling and validation."""

    def test_overwrite_protection(self, toy_proteomics_csv, tmp_path):
        """Test that overwrite protection works."""
        outdir = tmp_path / "splits_overwrite"
        outdir.mkdir()

        runner = CliRunner()

        # First run should succeed
        result1 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
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
            ],
        )
        assert result1.exit_code == 0

        # Second run without --overwrite should succeed (idempotent behavior)
        # Refactored code allows re-running with same config (logs warning, no error)
        result2 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
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
            ],
        )
        # Should succeed (idempotent - same config is allowed to avoid breaking workflows)
        assert result2.exit_code == 0

    def test_invalid_scenario_raises(self, toy_proteomics_csv, tmp_path):
        """Test that invalid scenario name raises error."""
        outdir = tmp_path / "splits_invalid"
        outdir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir),
                "--scenarios",
                "InvalidScenario",
            ],
        )

        assert result.exit_code != 0


class TestSaveSplitsReproducibility:
    """Test reproducibility with fixed seeds."""

    def test_same_seed_produces_identical_splits(self, toy_proteomics_csv, tmp_path):
        """Test that same seed produces identical splits."""
        outdir1 = tmp_path / "splits_repro_1"
        outdir2 = tmp_path / "splits_repro_2"
        outdir1.mkdir()
        outdir2.mkdir()

        runner = CliRunner()

        # Run 1
        result1 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir1),
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
        assert result1.exit_code == 0

        # Run 2 with same seed
        result2 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(toy_proteomics_csv),
                "--outdir",
                str(outdir2),
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
        assert result2.exit_code == 0

        # Compare indices (new format with scenario in filename)
        train1 = pd.read_csv(outdir1 / "train_idx_IncidentOnly_seed42.csv")["idx"].values
        train2 = pd.read_csv(outdir2 / "train_idx_IncidentOnly_seed42.csv")["idx"].values

        np.testing.assert_array_equal(train1, train2)

        # Compare metadata (excluding timestamps if any)
        with open(outdir1 / "split_meta_IncidentOnly_seed42.json") as f:
            meta1 = json.load(f)
        with open(outdir2 / "split_meta_IncidentOnly_seed42.json") as f:
            meta2 = json.load(f)

        # Check key fields match
        for key in [
            "n_train",
            "n_val",
            "n_test",
            "split_id_train",
            "split_id_val",
            "split_id_test",
        ]:
            assert meta1[key] == meta2[key], f"Mismatch in {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
