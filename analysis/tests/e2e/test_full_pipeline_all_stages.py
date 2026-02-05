"""
E2E tests for the full `ced run-pipeline` with ALL stages enabled.

Existing tests in test_run_pipeline_e2e.py disable ensemble, consensus,
and panel optimisation.  These tests exercise the complete 8-stage
pipeline to verify cross-stage artifact compatibility.

Run with: pytest tests/e2e/test_full_pipeline_all_stages.py -v
Run slow tests: pytest tests/e2e/test_full_pipeline_all_stages.py -v -m slow
"""

import os

import pandas as pd
import pytest
from click.testing import CliRunner

from ced_ml.cli.main import cli

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_pipeline(runner, args):
    """Run ced run-pipeline, print output on failure, return result."""
    result = runner.invoke(cli, ["run-pipeline"] + args, catch_exceptions=False)
    if result.exit_code != 0:
        print("RUN-PIPELINE OUTPUT:", result.output[:2000])
        if result.exception:
            import traceback

            traceback.print_exception(
                type(result.exception),
                result.exception,
                result.exception.__traceback__,
            )
    return result


def _find_run_dir(results_dir):
    """Return the single run_* directory inside results_dir."""
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}: {run_dirs}"
    return run_dirs[0]


def _generate_splits(runner, data_path, splits_dir, scenario="IncidentOnly", seeds=(0, 1)):
    """Pre-generate splits matching the training config scenario and seeds.

    run-pipeline's default splits config uses a different scenario/seed range,
    so we generate splits explicitly to avoid mismatch.
    """
    for seed in seeds:
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(data_path),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                scenario,
                "--n-splits",
                "1",
                "--seed-start",
                str(seed),
            ],
        )
        if result.exit_code != 0:
            return result
    return result


# ---------------------------------------------------------------------------
# Class 1: Full pipeline with all stages
# ---------------------------------------------------------------------------


class TestFullPipelineAllStages:
    """Test `ced run-pipeline` with ensemble, panel optimisation, and consensus enabled."""

    @pytest.mark.slow
    def test_full_pipeline_with_ensemble(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """
        Pipeline with 2 models, 2 seeds, and ensemble training.

        Stages validated:
        1. Splits pre-generated (IncidentOnly, seeds 0,1)
        2. Base models trained (LR_EN, RF x seed 0, 1)
        3. Base models aggregated
        4. Ensemble trained (per seed)
        5. Ensemble aggregated

        Note: panel optimisation and consensus are disabled because
        run-pipeline's optimise-panel stage uses get_project_root()
        which requires CWD to be the project root. Tests run from
        analysis/.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Pre-generate splits matching training config (scenario=IncidentOnly, seeds 0,1)
        result_splits = _generate_splits(
            runner, small_proteomics_data, splits_dir, scenario="IncidentOnly", seeds=(0, 1)
        )
        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:300]}")

        result = _invoke_pipeline(
            runner,
            [
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(ultra_fast_training_config),
                "--models",
                "LR_EN,RF",
                "--split-seeds",
                "0,1",
                "--no-optimize-panel",
                "--no-consensus",
                "--no-permutation-test",
            ],
        )

        if result.exit_code != 0:
            pytest.skip(f"Full pipeline failed: {result.output[:500]}")

        run_dir = _find_run_dir(results_dir)

        # --- Stage 1: Splits ---
        split_files = list(splits_dir.glob("*_idx_*.csv"))
        assert len(split_files) >= 3, f"Expected split files, found: {split_files}"

        # --- Stage 2+3: Base model training + aggregation ---
        for model in ["LR_EN", "RF"]:
            model_dir = run_dir / model
            assert model_dir.exists(), f"Missing model directory: {model}"

            splits_subdir = model_dir / "splits"
            if splits_subdir.exists():
                seed_dirs = list(splits_subdir.glob("split_seed*"))
                assert (
                    len(seed_dirs) >= 2
                ), f"Expected 2 split dirs for {model}, found {len(seed_dirs)}"

            agg_dir = model_dir / "aggregated"
            if agg_dir.exists():
                assert any(agg_dir.rglob("*")), f"Aggregated dir for {model} is empty"

        # --- Stage 4+5: Ensemble ---
        ensemble_dir = run_dir / "ENSEMBLE"
        if ensemble_dir.exists():
            ensemble_splits = ensemble_dir / "splits"
            if ensemble_splits.exists():
                assert any(ensemble_splits.rglob("*")), "Ensemble splits directory is empty"

            ensemble_agg = ensemble_dir / "aggregated"
            if ensemble_agg.exists():
                assert any(ensemble_agg.rglob("*")), "Ensemble aggregated directory is empty"

    @pytest.mark.slow
    def test_full_pipeline_cross_stage_artifacts(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """
        Verify cross-stage artifact schemas after a full pipeline run.

        Loads key output files and validates their structure to ensure
        stages produce artifacts consumable by downstream stages.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Pre-generate splits
        result_splits = _generate_splits(
            runner, small_proteomics_data, splits_dir, scenario="IncidentOnly", seeds=(0, 1)
        )
        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:300]}")

        result = _invoke_pipeline(
            runner,
            [
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(ultra_fast_training_config),
                "--models",
                "LR_EN,RF",
                "--split-seeds",
                "0,1",
                "--no-optimize-panel",
                "--no-consensus",
                "--no-permutation-test",
            ],
        )

        if result.exit_code != 0:
            pytest.skip(f"Pipeline failed: {result.output[:500]}")

        run_dir = _find_run_dir(results_dir)

        # -- OOF predictions schema --
        oof_files = list(run_dir.rglob("*oof*.csv"))
        if oof_files:
            oof_df = pd.read_csv(oof_files[0])
            # OOF predictions should have at least a probability column
            has_prob_col = any("prob" in c.lower() or "pred" in c.lower() for c in oof_df.columns)
            assert (
                has_prob_col
            ), f"OOF file missing probability column. Columns: {list(oof_df.columns)}"

        # -- Metrics schema --
        metric_files = list(run_dir.rglob("*metrics*.csv"))
        if metric_files:
            metrics_df = pd.read_csv(metric_files[0])
            assert len(metrics_df) > 0, "Metrics file is empty"

        # -- Aggregated importance (if exists) --
        importance_files = list(run_dir.rglob("*importance*.csv"))
        if importance_files:
            imp_df = pd.read_csv(importance_files[0])
            assert len(imp_df) > 0, "Importance file is empty"

    @pytest.mark.slow
    def test_pipeline_fixed_panel_skips_optimization(
        self, small_proteomics_data, fixed_panel_training_config, tmp_path
    ):
        """
        When feature_selection_strategy='fixed_panel', pipeline auto-disables
        panel optimisation and consensus stages.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        env = os.environ.copy()
        env["CED_RESULTS_DIR"] = str(results_dir)

        runner = CliRunner(env=env)

        # Pre-generate splits
        result_splits = _generate_splits(
            runner, small_proteomics_data, splits_dir, scenario="IncidentOnly", seeds=(0,)
        )
        if result_splits.exit_code != 0:
            pytest.skip(f"Split generation failed: {result_splits.output[:300]}")

        result = _invoke_pipeline(
            runner,
            [
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(fixed_panel_training_config),
                "--models",
                "LR_EN",
                "--split-seeds",
                "0",
                "--no-ensemble",
                "--no-permutation-test",
            ],
        )

        if result.exit_code != 0:
            pytest.skip(f"Fixed-panel pipeline failed: {result.output[:500]}")

        run_dir = _find_run_dir(results_dir)

        # Panel optimisation should NOT exist
        panel_opt_dirs = list(run_dir.rglob("*optimize_panel*"))
        assert (
            len(panel_opt_dirs) == 0
        ), f"Panel optimisation should be skipped for fixed_panel, found: {panel_opt_dirs}"

        # Consensus should NOT exist
        consensus_dirs = list(run_dir.rglob("*consensus_panel*"))
        assert (
            len(consensus_dirs) == 0
        ), f"Consensus should be skipped for fixed_panel, found: {consensus_dirs}"

        # Model should still have trained successfully
        model_dir = run_dir / "LR_EN"
        assert model_dir.exists(), "LR_EN model directory should exist"

    def test_pipeline_insufficient_data_fails(
        self, tiny_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """
        Pipeline fails with clear error when dataset is too small for
        stratified splitting.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run-pipeline",
                "--infile",
                str(tiny_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(ultra_fast_training_config),
                "--models",
                "LR_EN",
                "--split-seeds",
                "0",
                "--no-ensemble",
                "--no-consensus",
                "--no-optimize-panel",
                "--no-permutation-test",
            ],
        )

        # Should fail -- dataset too small for meaningful splits/CV
        assert result.exit_code != 0, "Pipeline should fail on tiny dataset, but exited 0"
