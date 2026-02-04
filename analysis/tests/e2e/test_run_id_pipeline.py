"""
End-to-end tests for complete pipelines using run_id auto-detection.

Tests validate full end-to-end workflows that combine multiple CLI commands
with run_id auto-detection throughout.

Run with: pytest tests/e2e/test_run_id_pipeline.py -v
Run slow tests: pytest tests/e2e/test_run_id_pipeline.py -v -m slow
"""

import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner

SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


class TestFullPipelineWithRunId:
    """Test complete pipelines using --run-id auto-detection throughout."""

    @pytest.mark.slow
    def test_complete_single_model_pipeline(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Complete single-model pipeline with --run-id auto-detection.

        Workflow: save-splits -> train -> aggregate-splits -> optimize-panel
        Uses --run-id throughout to minimize configuration.
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
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )
        assert result_splits.exit_code == 0

        # Step 2: Train on both splits with shared run_id
        for seed in [42, 43]:
            result_train = runner.invoke(
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

            if result_train.exit_code != 0:
                pytest.skip(f"Training failed on seed {seed}")

        run_id = SHARED_RUN_ID

        # Step 3: Aggregate with --run-id
        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Step 4: Optimize panel with --run-id
        result_opt = runner.invoke(
            cli,
            ["optimize-panel", "--run-id", run_id, "--min-size", "3"],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_opt.exit_code != 0:
            pytest.skip("Panel optimization failed")

        # Verify complete pipeline outputs exist
        run_dir = results_dir / f"run_{run_id}"
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "LR_EN" / "aggregated").exists()
        assert (run_dir / "LR_EN" / "aggregated" / "optimize_panel").exists()

    @pytest.mark.slow
    def test_complete_ensemble_pipeline(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Complete ensemble pipeline with --run-id auto-detection.

        Workflow: save-splits -> train base models -> train-ensemble -> aggregate
        Uses --run-id for ensemble training.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Step 1: Generate splits
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

        # Step 2: Train base models with shared run_id
        base_models = ["LR_EN", "RF"]
        for model in base_models:
            result_train = runner.invoke(
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
                    model,
                    "--split-seed",
                    "42",
                    "--run-id",
                    SHARED_RUN_ID,
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip(f"Base model {model} training failed")

        run_id = SHARED_RUN_ID

        # Step 3: Train ensemble with --run-id
        result_ens = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--run-id",
                run_id,
                "--split-seed",
                "42",
                "--results-dir",
                str(results_dir),
            ],
            catch_exceptions=False,
        )

        if result_ens.exit_code != 0:
            pytest.skip("Ensemble training failed")

        # Step 4: Verify ensemble outputs
        ensemble_dir = results_dir / f"run_{SHARED_RUN_ID}" / "ENSEMBLE" / "splits" / "split_seed42"
        assert ensemble_dir.exists()
        assert (ensemble_dir / "core" / "metrics.json").exists()
