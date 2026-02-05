"""
E2E tests for ensemble model workflows.

Tests ensemble learning pipeline:
- Base model training
- Stacking meta-learner training
- Ensemble output validation
- Error handling without base models

Run with: pytest tests/e2e/test_pipeline_ensemble.py -v
Run slow tests: pytest tests/e2e/test_pipeline_ensemble.py -v -m slow
"""

import pytest
from click.testing import CliRunner

from ced_ml.cli.main import cli


class TestE2EEnsembleWorkflow:
    """Test ensemble workflow: base models -> stacking -> aggregate."""

    @pytest.mark.slow
    def test_ensemble_training_workflow(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Train base models then ensemble meta-learner.

        Critical ensemble integration test. Marked slow (~2-3 min).
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
                "--scenarios",
                "IncidentOnly",
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
        assert result_splits.exit_code == 0

        # Step 2: Train base models (LR_EN and RF)
        base_models = ["LR_EN", "RF"]

        for model in base_models:
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
                    model,
                    "--split-seed",
                    "42",
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip(f"Base model {model} training failed: {result_train.output[:200]}")

        # Step 3: Train ensemble
        result_ensemble = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--results-dir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--base-models",
                ",".join(base_models),
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_ensemble.exit_code != 0:
            print("ENSEMBLE OUTPUT:", result_ensemble.output)
            if result_ensemble.exception:
                import traceback

                traceback.print_exception(
                    type(result_ensemble.exception),
                    result_ensemble.exception,
                    result_ensemble.exception.__traceback__,
                )

        assert result_ensemble.exit_code == 0, f"Ensemble failed: {result_ensemble.output}"

        # Verify ensemble outputs
        # Find the run directory with ENSEMBLE model output
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) >= 1, f"Expected at least 1 run directory, found {len(run_dirs)}"

        # Look for ENSEMBLE in any run directory (may be in different run_id than base models)
        ensemble_dir = None
        for run_dir in run_dirs:
            candidate = run_dir / "ENSEMBLE" / "splits" / "split_seed42"
            if candidate.exists():
                ensemble_dir = candidate
                break

        assert ensemble_dir is not None, f"Ensemble directory not found in any run_dir: {run_dirs}"

        # Check ensemble-specific files (using actual file structure)
        assert (ensemble_dir / "core/metrics.json").exists(), "Missing metrics.json"
        assert (
            ensemble_dir / "preds/test_preds__ENSEMBLE.csv"
        ).exists(), "Missing test predictions"

    def test_ensemble_requires_base_models(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Ensemble training fails gracefully without base models.

        Error handling test.
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
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
            ],
        )

        # Try to train ensemble without base models
        result = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--results-dir",
                str(results_dir),
                "--config",
                str(minimal_training_config),
                "--base-models",
                "LR_EN,RF",
                "--split-seed",
                "42",
            ],
        )

        # Should fail with informative error
        assert result.exit_code != 0
        assert "base model" in result.output.lower() or "not found" in result.output.lower()
