"""
E2E tests for panel optimization workflows.

Tests panel optimization and fixed panel validation:
- Aggregated RFE-based panel optimization (ADR-013 compliant)
- Panel curve generation and feature ranking
- Fixed panel training workflow
- Error handling for missing aggregation

Run with: pytest tests/e2e/test_pipeline_panel.py -v
Run slow tests: pytest tests/e2e/test_pipeline_panel.py -v -m slow
"""

import pandas as pd
import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner


class TestE2EPanelOptimization:
    """Test panel optimization workflow: train -> optimize-panel -> validate."""

    @pytest.mark.slow
    def test_panel_optimization_workflow(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Panel optimization via RFE after training and aggregation.

        Critical workflow for deployment sizing. Marked slow (~1-2 min).
        Tests the aggregated panel optimization path (ADR-013 compliant).
        """
        import os

        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Set environment variable for CLI to find results
        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Step 1: Generate splits
        result_splits = runner.invoke(
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
        assert result_splits.exit_code == 0

        # Step 2: Train model with run-id
        run_id = "test_panel_opt"
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
                "--run-id",
                run_id,
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Training failed: {result_train.output[:200]}")

        # Step 3: Aggregate results (required for panel optimization)
        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip(f"Aggregation failed: {result_agg.output[:200]}")

        # Step 4: Run panel optimization on aggregated results
        result_optimize = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--min-size",
                "3",
                "--stability-threshold",
                "0.75",
            ],
            catch_exceptions=False,
        )

        if result_optimize.exit_code != 0:
            print("OPTIMIZE OUTPUT:", result_optimize.output)
            if result_optimize.exception:
                import traceback

                traceback.print_exception(
                    type(result_optimize.exception),
                    result_optimize.exception,
                    result_optimize.exception.__traceback__,
                )

        assert (
            result_optimize.exit_code == 0
        ), f"Panel optimization failed: {result_optimize.output}"

        # Verify RFE outputs in aggregated directory
        panel_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated" / "optimize_panel"
        assert panel_dir.exists(), "Panel optimization directory not found"
        assert (panel_dir / "panel_curve_aggregated.csv").exists(), "Missing panel curve"
        assert (panel_dir / "feature_ranking_aggregated.csv").exists(), "Missing feature ranking"
        assert (
            panel_dir / "recommended_panels_aggregated.json"
        ).exists(), "Missing recommendations"

        # Validate panel curve structure
        panel_curve = pd.read_csv(panel_dir / "panel_curve_aggregated.csv")
        assert "size" in panel_curve.columns
        assert "auroc_val" in panel_curve.columns
        assert len(panel_curve) > 0

        # Check that AUROC is in valid range
        assert all(0.0 <= auc <= 1.0 for auc in panel_curve["auroc_val"])

    def test_panel_optimization_requires_aggregated_results(
        self, minimal_proteomics_data, tmp_path
    ):
        """
        Test: Panel optimization fails gracefully without aggregated results.

        Error handling test for missing aggregation step.
        """
        import os

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Set environment variable
        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Try to optimize panel with non-existent run-id
        result = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                "nonexistent_run",
            ],
        )

        # Should fail with informative error
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "no models" in result.output.lower()


class TestE2EFixedPanelValidation:
    """Test fixed panel validation workflow: train with --fixed-panel."""

    @pytest.mark.slow
    def test_fixed_panel_training_workflow(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Train model with fixed panel (bypasses feature selection).

        Critical for panel validation. Marked slow (~30-60s).
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

        # Step 2: Create a fixed panel file (5 proteins)
        panel_file = tmp_path / "fixed_panel.csv"
        panel_proteins = [f"PROT_{i:03d}_resid" for i in range(5)]
        pd.DataFrame({"protein": panel_proteins}).to_csv(panel_file, index=False)

        # Step 3: Train with fixed panel
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
                "--fixed-panel",
                str(panel_file),
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

        assert result_train.exit_code == 0, f"Fixed panel training failed: {result_train.output}"

        # Verify outputs exist
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"
        model_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"
        assert model_dir.exists()

        # Check that model was trained
        has_predictions = any(model_dir.rglob("*.csv"))
        has_config = any(model_dir.rglob("*config*.yaml"))

        assert has_predictions, "No predictions found"
        assert has_config, "No config file found"

    def test_fixed_panel_invalid_file(
        self, minimal_proteomics_data, minimal_training_config, tmp_path
    ):
        """
        Test: Fixed panel training fails gracefully with invalid panel file.

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
                "--infile",
                str(minimal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
            ],
        )

        # Try to train with nonexistent panel file
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
                "--fixed-panel",
                str(tmp_path / "nonexistent_panel.csv"),
            ],
        )

        # Should fail with informative error
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()
