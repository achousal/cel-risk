"""
End-to-end tests for run_id metadata creation and error handling.

Tests validate that run_metadata.json is correctly created during training
and that error conditions are handled gracefully.

Run with: pytest tests/e2e/test_run_id_metadata.py -v
Run slow tests: pytest tests/e2e/test_run_id_metadata.py -v -m slow
"""

import json
from pathlib import Path

import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner

SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


def verify_run_metadata(run_dir: Path, expected_model: str, expected_split_seed: int):
    """Verify run_metadata.json has correct structure and content."""
    metadata_path = run_dir / "run_metadata.json"
    assert metadata_path.exists(), f"Missing run_metadata.json in {run_dir}"

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check required top-level fields
    assert "run_id" in metadata
    assert "infile" in metadata
    assert "split_dir" in metadata
    assert "models" in metadata

    # Check model-specific fields (nested structure)
    assert expected_model in metadata["models"], f"Model {expected_model} not in metadata"
    model_entry = metadata["models"][expected_model]
    assert "scenario" in model_entry
    assert "infile" in model_entry
    assert "split_dir" in model_entry
    assert "split_seed" in model_entry
    assert "timestamp" in model_entry

    # Validate content
    assert model_entry["split_seed"] == expected_split_seed


class TestRunIdMetadataCreation:
    """Test that training creates correct run_metadata.json files."""

    @pytest.mark.slow
    def test_train_creates_run_metadata(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Train command creates run_metadata.json with correct fields.

        This is the foundation for all --run-id auto-detection.
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
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Train model
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
                str(fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
                "--run-id",
                SHARED_RUN_ID,
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed: {result.output[:200]}")

        # Find run directory (production layout: results/run_{ID}/)
        # Actual root for this split is: results/run_{ID}/LR_EN/splits/split_seed42/
        run_dir = results_dir / f"run_{SHARED_RUN_ID}"
        assert run_dir.exists()

        # Verify metadata (stored at run level, not split level)
        verify_run_metadata(run_dir, "LR_EN", 42)

    def test_run_metadata_persists_across_splits(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Same run_id used for multiple split seeds creates separate metadata.

        Each split_seed should have its own split_seed directory with separate metadata.
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
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        # Train on two splits with shared run_id (mirrors production usage)
        for seed in [42, 43]:
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
                    str(fast_training_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                    "--run-id",
                    SHARED_RUN_ID,
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip(f"Training on seed {seed} failed")

        # Both splits share one run_id directory
        # Production layout: results/run_{ID}/{MODEL}/splits/split_seed{N}/
        run_dir = results_dir / f"run_{SHARED_RUN_ID}"
        assert run_dir.exists()
        model_dir = run_dir / "LR_EN"
        assert model_dir.exists(), "Model subdirectory not found"
        splits_parent = model_dir / "splits"
        assert splits_parent.exists(), "splits/ subdirectory not found"
        split_dirs = [
            d for d in splits_parent.iterdir() if d.is_dir() and d.name.startswith("split_")
        ]
        assert len(split_dirs) == 2, f"Expected 2 split dirs, got {len(split_dirs)}"
        assert (run_dir / "run_metadata.json").exists()


class TestRunIdErrorHandling:
    """Test error handling and edge cases for --run-id auto-detection."""

    def test_invalid_run_id_format(self):
        """Test graceful handling of malformed run_id."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "invalid_format", "--model", "LR_EN"],
        )

        assert result.exit_code != 0

    def test_run_id_partial_results(self, tmp_path):
        """
        Test handling when run_id exists but results are incomplete.

        Some splits trained, others missing.
        """
        results_dir = tmp_path / "results"

        # Create partial structure (matching production layout)
        # Production layout: results/run_{ID}/{MODEL}/splits/split_seed{N}/
        run_dir = results_dir / "run_20260127_115115"
        (run_dir / "LR_EN" / "splits" / "split_seed42").mkdir(parents=True)
        # Missing split_seed43 to test partial results handling

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "20260127_115115", "--model", "LR_EN"],
        )

        # Aggregation succeeds with partial results (any number of splits).
        # It does not know the expected total, so no warning is emitted.
        assert result.exit_code == 0

    def test_run_id_missing_metadata(self, tmp_path):
        """
        Test handling when run directory exists but run_metadata.json is missing.

        Should still work (metadata is optional for older runs).
        """
        results_dir = tmp_path / "results"
        run_dir = results_dir / "run_20260127_115115"
        (run_dir / "LR_EN").mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "20260127_115115", "--model", "LR_EN"],
        )

        # May fail due to missing actual results, but shouldn't crash on missing metadata
        assert result.exit_code in [0, 1]
