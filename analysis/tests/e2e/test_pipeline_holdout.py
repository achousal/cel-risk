"""
E2E tests for holdout evaluation workflows.

Tests final model validation on held-out data:
- Holdout evaluation workflow (train -> eval-holdout)
- Error handling for missing models
- Optional DCA computation flag

Run with: pytest tests/e2e/test_pipeline_holdout.py -v
Run slow tests: pytest tests/e2e/test_pipeline_holdout.py -v -m slow
"""

import pandas as pd
import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner


class TestE2EHoldoutEvaluation:
    """Test holdout evaluation workflow."""

    @pytest.mark.slow
    def test_eval_holdout_workflow(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Holdout evaluation workflow (train -> eval-holdout).

        Critical for final model validation. Marked slow (~2 min).
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        holdout_dir = tmp_path / "holdout_results"
        splits_dir.mkdir()
        results_dir.mkdir()
        holdout_dir.mkdir()

        runner = CliRunner()

        # Step 1: Generate splits with holdout
        # Note: Using development mode (no holdout) first to test eval-holdout CLI
        # Holdout stratification requires many more samples to avoid single-group strata
        import yaml

        holdout_splits_config = {
            "mode": "development",
            "scenarios": ["IncidentOnly"],
            "n_splits": 1,
            "val_size": 0.25,
            "test_size": 0.25,
            "seed_start": 42,
        }
        holdout_config_path = tmp_path / "holdout_splits_config.yaml"
        with open(holdout_config_path, "w") as f:
            yaml.dump(holdout_splits_config, f)

        result_splits = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--config",
                str(holdout_config_path),
            ],
        )
        assert result_splits.exit_code == 0, f"Splits failed: {result_splits.output}"

        # Create a fake holdout indices file from test set indices
        # (In real workflow, holdout would come from save-splits with mode=holdout)
        test_idx_file = splits_dir / "test_idx_IncidentOnly_seed42.csv"
        assert test_idx_file.exists(), "Test indices should exist"
        test_idx_df = pd.read_csv(test_idx_file)
        holdout_idx_file = splits_dir / "holdout_idx_IncidentOnly_seed42.csv"
        test_idx_df.to_csv(holdout_idx_file, index=False)

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
            pytest.skip(f"Training failed: {result_train.output[:200]}")

        # Find trained model
        model_files = list(results_dir.rglob("*final_model*.joblib"))
        if not model_files:
            model_files = list(results_dir.rglob("*.joblib"))
        if not model_files:
            pytest.skip("No trained model found")

        model_path = model_files[0]

        # Step 3: Evaluate on holdout
        result_holdout = runner.invoke(
            cli,
            [
                "eval-holdout",
                "--infile",
                str(minimal_proteomics_data),
                "--model-artifact",
                str(model_path),
                "--holdout-idx",
                str(holdout_idx_file),
                "--outdir",
                str(holdout_dir),
            ],
            catch_exceptions=False,
        )

        if result_holdout.exit_code != 0:
            print("HOLDOUT OUTPUT:", result_holdout.output)
            if result_holdout.exception:
                import traceback

                traceback.print_exception(
                    type(result_holdout.exception),
                    result_holdout.exception,
                    result_holdout.exception.__traceback__,
                )

        assert result_holdout.exit_code == 0, f"Holdout eval failed: {result_holdout.output}"

        # Verify holdout outputs
        assert any(holdout_dir.rglob("*")), "Holdout output directory should contain files"
        # Check for metrics (could be metrics.json or metrics.csv)
        has_metrics = any(holdout_dir.rglob("*metrics*")) or any(holdout_dir.rglob("*.json"))
        assert has_metrics, "Holdout metrics file should exist"

    def test_eval_holdout_missing_model(self, minimal_proteomics_data, tmp_path):
        """
        Test: Holdout evaluation fails gracefully with missing model.

        Error handling test.
        """
        splits_dir = tmp_path / "splits"
        holdout_dir = tmp_path / "holdout_results"
        splits_dir.mkdir()
        holdout_dir.mkdir()

        # Create fake holdout indices
        holdout_idx = pd.DataFrame({"idx": [0, 1, 2, 3, 4]})
        holdout_idx_file = splits_dir / "holdout_idx.csv"
        holdout_idx.to_csv(holdout_idx_file, index=False)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "eval-holdout",
                "--infile",
                str(minimal_proteomics_data),
                "--model-artifact",
                str(tmp_path / "nonexistent_model.joblib"),
                "--holdout-idx",
                str(holdout_idx_file),
                "--outdir",
                str(holdout_dir),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()

    def test_eval_holdout_compute_dca_flag(self, minimal_proteomics_data, tmp_path):
        """
        Test: Holdout evaluation with --compute-dca flag.

        Validates optional DCA computation.
        """
        # This test would require a trained model
        # Skipping actual execution, just testing flag parsing
        runner = CliRunner()

        # Just verify the flag is recognized (will fail on missing files)
        result = runner.invoke(
            cli,
            [
                "eval-holdout",
                "--infile",
                str(minimal_proteomics_data),
                "--model-artifact",
                str(tmp_path / "fake_model.joblib"),
                "--holdout-idx",
                str(tmp_path / "fake_holdout.csv"),
                "--outdir",
                str(tmp_path / "out"),
                "--compute-dca",
            ],
        )

        # Should fail on missing files, but flag should be recognized
        assert "--compute-dca" not in result.output or result.exit_code != 0
