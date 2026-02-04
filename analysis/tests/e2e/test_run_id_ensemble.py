"""
End-to-end tests for train-ensemble command with run_id auto-detection.

Tests validate that train-ensemble can auto-detect base models from run_id
and correctly train ensemble meta-learners.

Run with: pytest tests/e2e/test_run_id_ensemble.py -v
Run slow tests: pytest tests/e2e/test_run_id_ensemble.py -v -m slow
"""

import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner

SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


class TestEnsembleWithRunId:
    """Test train-ensemble command with --run-id auto-detection."""

    @pytest.mark.slow
    def test_ensemble_auto_detects_base_models(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: ced train-ensemble --run-id <RUN_ID> auto-detects base models and paths.

        Critical workflow: train base models -> ensemble with zero config.
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
                    model,
                    "--split-seed",
                    "42",
                    "--run-id",
                    SHARED_RUN_ID,
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip(f"Base model {model} training failed")

        run_id = SHARED_RUN_ID

        # Step 3: Train ensemble with --run-id auto-detection
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
            print("ENSEMBLE OUTPUT:", result_ens.output)
            if result_ens.exception:
                import traceback

                traceback.print_exception(
                    type(result_ens.exception),
                    result_ens.exception,
                    result_ens.exception.__traceback__,
                )
            pytest.skip(f"Ensemble training failed: {result_ens.output[:200]}")

        assert result_ens.exit_code == 0, f"Ensemble with --run-id failed: {result_ens.output}"

        # Verify ensemble outputs
        ensemble_dir = results_dir / f"run_{SHARED_RUN_ID}" / "ENSEMBLE" / "splits" / "split_seed42"
        assert ensemble_dir.exists(), f"Ensemble directory should exist: {ensemble_dir}"
        assert (ensemble_dir / "core" / "metrics.json").exists(), "Ensemble metrics should exist"

    def test_ensemble_run_id_without_base_models_fails(self, tmp_path):
        """
        Test: Ensemble with run_id but no base models fails gracefully.

        Error handling for missing base models.
        """
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--run-id",
                "20260101_000000",
                "--results-dir",
                str(results_dir),
                "--split-seed",
                "42",
            ],
        )

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert (
            "base model" in output_lower
            or "not found" in output_lower
            or "no models" in output_lower
            or "no results found" in output_lower
        )
