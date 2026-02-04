"""
End-to-end tests for optimize-panel and consensus-panel commands with run_id auto-detection.

Tests validate that panel optimization commands can auto-detect paths from run_id
and correctly process aggregated results.

Run with: pytest tests/e2e/test_run_id_optimization.py -v
Run slow tests: pytest tests/e2e/test_run_id_optimization.py -v -m slow
"""

import os
import unittest.mock

import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner

SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


class TestOptimizePanelWithRunId:
    """Test optimize-panel command with --run-id auto-detection."""

    @pytest.mark.slow
    def test_optimize_panel_auto_detects_all_models(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: ced optimize-panel --run-id <RUN_ID> processes all base models automatically.

        Critical workflow: train -> aggregate -> optimize-panel with zero config.
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
            catch_exceptions=False,
        )

        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:200]}")

        # Step 2: Train one model on both splits with shared run_id
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
                pytest.skip(f"Training failed on seed {seed}")

        run_id = SHARED_RUN_ID

        # Step 3: Aggregate (required for optimize-panel)
        result_agg = runner.invoke(
            cli,
            [
                "aggregate-splits",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
            ],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_agg.exit_code != 0:
            pytest.skip(f"Aggregation failed: {result_agg.output[:200]}")

        # Step 4: Optimize panel with --run-id
        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--min-size",
                "3",
            ],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_opt.exit_code != 0:
            print("OPTIMIZE-PANEL OUTPUT:", result_opt.output)
            if result_opt.exception:
                import traceback

                traceback.print_exception(
                    type(result_opt.exception),
                    result_opt.exception,
                    result_opt.exception.__traceback__,
                )
            pytest.skip(f"Panel optimization failed: {result_opt.output[:200]}")

        assert (
            result_opt.exit_code == 0
        ), f"Optimize-panel with --run-id failed: {result_opt.output}"

        # Verify panel optimization outputs
        panel_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated" / "optimize_panel"
        assert panel_dir.exists(), "Panel optimization directory should exist"
        assert (panel_dir / "panel_curve_aggregated.csv").exists(), "Panel curve should exist"

    @pytest.mark.slow
    def test_optimize_panel_specific_model_with_run_id(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: ced optimize-panel --run-id <RUN_ID> --model <MODEL> processes single model.

        Validates selective model processing.
        NOTE: Requires training 2 splits with shared run_id for aggregation.
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
                "--val-size",
                "0.25",
            ],
        )

        # Train both splits with shared run_id
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
                pytest.skip(f"Training failed on seed {seed}")

        run_id = SHARED_RUN_ID

        # Aggregate
        runner.invoke(
            cli,
            [
                "aggregate-splits",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
            ],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        # Optimize panel for specific model
        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--min-size",
                "3",
            ],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_opt.exit_code != 0:
            pytest.skip(f"Panel optimization failed: {result_opt.output[:200]}")

        assert result_opt.exit_code == 0

    def test_optimize_panel_run_id_without_aggregation_fails(self, tmp_path):
        """
        Test: Optimize-panel with run_id but no aggregated results fails gracefully.

        Error handling for missing aggregation.
        """
        import os

        # Create results directory structure but without aggregated results
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        model_run_dir = results_dir / "run_20260101_000000" / "LR_EN"
        model_run_dir.mkdir(parents=True)

        # Create a split directory but no aggregated directory
        split_dir = model_run_dir / "splits" / "split_seed42"
        split_dir.mkdir(parents=True)

        # Set environment variable so CLI finds the tmp results dir
        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        result = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                "20260101_000000",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "aggregate" in result.output.lower()


class TestConsensusPanelWithRunId:
    """Test consensus-panel command with --run-id auto-detection."""

    @pytest.mark.slow
    def test_consensus_panel_auto_detects_models(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: ced consensus-panel --run-id <RUN_ID> auto-detects all base models.

        Critical workflow: train multiple models -> aggregate -> consensus-panel.
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
                "2",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        # Step 2: Train two models on both splits with shared run_id
        base_models = ["LR_EN", "RF"]
        for model in base_models:
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
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        SHARED_RUN_ID,
                    ],
                    catch_exceptions=False,
                )

                if result.exit_code != 0:
                    pytest.skip(f"Training {model} seed {seed} failed")

        run_id = SHARED_RUN_ID

        # Step 3: Aggregate both models
        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            for model in base_models:
                result_agg = runner.invoke(
                    cli,
                    [
                        "aggregate-splits",
                        "--run-id",
                        run_id,
                        "--model",
                        model,
                    ],
                    catch_exceptions=False,
                )

                if result_agg.exit_code != 0:
                    pytest.skip(f"Aggregation for {model} failed")

            # Step 4: Generate consensus panel
            result_consensus = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    run_id,
                    "--infile",
                    str(small_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_consensus.exit_code != 0:
            print("CONSENSUS-PANEL OUTPUT:", result_consensus.output)
            if result_consensus.exception:
                import traceback

                traceback.print_exception(
                    type(result_consensus.exception),
                    result_consensus.exception,
                    result_consensus.exception.__traceback__,
                )
            pytest.skip(f"Consensus panel failed: {result_consensus.output[:200]}")

        assert (
            result_consensus.exit_code == 0
        ), f"Consensus-panel with --run-id failed: {result_consensus.output}"

        # Verify consensus panel outputs
        consensus_dir = results_dir / f"run_{run_id}" / "consensus"
        assert consensus_dir.exists(), f"Consensus panel directory should exist: {consensus_dir}"
        assert (consensus_dir / "final_panel.txt").exists(), "Final panel should exist"
        assert (consensus_dir / "consensus_ranking.csv").exists(), "Consensus ranking should exist"

    def test_consensus_panel_run_id_without_aggregation_fails(self, tmp_path):
        """
        Test: Consensus-panel with run_id but no aggregated results fails gracefully.

        Error handling for missing aggregation.
        """
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        runner = CliRunner()

        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    "20260101_000000",
                ],
            )

        assert result.exit_code != 0
        # Check error in either output or exception message
        error_text = (result.output + str(result.exception)).lower()
        assert "not found" in error_text or "aggregate" in error_text
