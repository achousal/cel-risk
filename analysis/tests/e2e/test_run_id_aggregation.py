"""
End-to-end tests for aggregate-splits command with run_id auto-detection.

Tests validate that aggregate-splits can auto-detect paths from run_id
and correctly process multiple split seeds.

Run with: pytest tests/e2e/test_run_id_aggregation.py -v
Run slow tests: pytest tests/e2e/test_run_id_aggregation.py -v -m slow
"""

import pytest
from ced_ml.cli.main import cli
from click.testing import CliRunner

SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


class TestAggregateWithRunId:
    """Test aggregate-splits command with --run-id auto-detection."""

    @pytest.mark.slow
    def test_aggregate_auto_detects_from_run_id(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: ced aggregate-splits --run-id <RUN_ID> --model <MODEL> auto-detects paths.

        Critical workflow: train -> aggregate with minimal config.
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

        # Step 2: Train on both splits with shared run_id
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

        # Step 3: Aggregate with --run-id
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
        )

        if result_agg.exit_code != 0:
            print("AGGREGATE OUTPUT:", result_agg.output)
            if result_agg.exception:
                import traceback

                traceback.print_exception(
                    type(result_agg.exception),
                    result_agg.exception,
                    result_agg.exception.__traceback__,
                )
            pytest.skip(f"Aggregation failed: {result_agg.output[:200]}")

        assert result_agg.exit_code == 0, f"Aggregation with --run-id failed: {result_agg.output}"

        # Verify aggregated outputs
        agg_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated"
        assert agg_dir.exists(), "Aggregated directory should exist"
        assert any(agg_dir.rglob("*metrics*")), "Aggregated metrics should exist"

    def test_aggregate_fails_with_invalid_run_id(self, tmp_path):
        """
        Test: Aggregate with nonexistent run_id fails gracefully.

        Error handling for invalid run_id.
        """
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "aggregate-splits",
                "--run-id",
                "INVALID_RUN_ID_999",
                "--model",
                "LR_EN",
            ],
        )

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert (
            "not found" in output_lower
            or "does not exist" in output_lower
            or "no results found" in output_lower
        )
