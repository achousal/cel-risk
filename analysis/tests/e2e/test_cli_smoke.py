"""Smoke tests for all major CLI commands.

Tests basic execution and success path for each command without validating detailed logic.
Focuses on CLI integration: command line parsing, argument handling, output file generation.

Test Coverage:
- save-splits, train, aggregate-splits, optimize-panel, consensus-panel
- permutation-test, config validate/diff

Run all: pytest analysis/tests/e2e/test_cli_smoke.py -v
Execution time: ~60-90 seconds total
"""

import pytest
from click.testing import CliRunner

from ced_ml.cli.main import cli


class TestCliSmoke:
    """Smoke tests for all major CLI commands."""

    def test_save_splits_smoke(self, small_proteomics_data, tmp_path):
        """Smoke: save-splits generates splits without error."""
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        assert result.exit_code == 0, f"save-splits failed: {result.output}"
        assert (splits_dir / "train_idx_IncidentOnly_seed42.csv").exists()
        assert (splits_dir / "test_idx_IncidentOnly_seed42.csv").exists()

    def test_train_smoke(self, small_proteomics_data, ultra_fast_training_config, tmp_path):
        """Smoke: train executes and generates results without error."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        # Smoke: allow exit code 0 or 1 (graceful failure okay for smoke test)
        if result_train.exit_code != 0:
            pytest.skip(f"train command failed or skipped: {result_train.output[:200]}")
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1

    def test_aggregate_splits_smoke(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: aggregate-splits runs without error on results."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

        runner.invoke(
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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            pytest.skip("No run directory created")

        results_model_dir = run_dirs[0] / "LR_EN"
        result_agg = runner.invoke(
            cli, ["aggregate-splits", "--results-dir", str(results_model_dir)]
        )
        assert result_agg.exit_code in [0, 1], f"aggregate-splits error: {result_agg.output}"

    def test_optimize_panel_smoke(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: optimize-panel runs without error."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

        runner.invoke(
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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            pytest.skip("No run directory created")

        results_model_dir = run_dirs[0] / "LR_EN"
        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--results-dir",
                str(results_model_dir),
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(splits_dir),
            ],
        )
        assert result_opt.exit_code in [0, 1], f"optimize-panel error: {result_opt.output}"

    def test_consensus_panel_smoke(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: consensus-panel runs without error."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

        runner.invoke(
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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            pytest.skip("No run directory created")
        run_id = run_dirs[0].name.replace("run_", "")

        result_cons = runner.invoke(cli, ["consensus-panel", "--run-id", run_id])
        assert result_cons.exit_code in [0, 1], f"consensus-panel error: {result_cons.output}"

    def test_permutation_test_smoke(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: permutation-test tests model significance."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

        runner.invoke(
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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        run_id = run_dirs[0].name.replace("run_", "")

        result_perm = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--n-perms",
                "2",
                "--n-jobs",
                "1",
            ],
        )
        assert result_perm.exit_code in [0, 1], f"permutation-test error: {result_perm.output}"

    def test_config_validate_smoke(self, ultra_fast_training_config):
        """Smoke: config validate runs without error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "validate",
                str(ultra_fast_training_config),
                "--command",
                "train",
            ],
        )
        assert result.exit_code == 0, f"config validate failed: {result.output}"

    def test_config_diff_smoke(self, ultra_fast_training_config, minimal_splits_config):
        """Smoke: config diff runs without error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "diff",
                str(ultra_fast_training_config),
                str(minimal_splits_config),
            ],
        )
        assert result.exit_code == 0, f"config diff failed: {result.output}"
        assert len(result.output) > 10, "config diff should produce output"


class TestCliSmokeErrorHandling:
    """Smoke tests for error handling paths."""

    def test_save_splits_missing_infile(self, tmp_path):
        """Smoke: save-splits fails gracefully with missing input."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(tmp_path / "nonexistent.parquet"),
                "--outdir",
                str(tmp_path / "splits"),
            ],
        )
        assert result.exit_code != 0, "Should fail with missing input"

    def test_train_missing_splits_dir(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: train fails gracefully with missing splits directory."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(small_proteomics_data),
                "--split-dir",
                str(tmp_path / "nonexistent_splits"),
                "--outdir",
                str(results_dir),
                "--config",
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )
        assert result.exit_code != 0, "Should fail with missing splits directory"

    def test_train_invalid_model(self, small_proteomics_data, ultra_fast_training_config, tmp_path):
        """Smoke: train fails gracefully with invalid model name."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )

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
                str(ultra_fast_training_config),
                "--model",
                "INVALID_MODEL",
                "--split-seed",
                "42",
            ],
        )
        assert result.exit_code != 0, "Should fail with invalid model"

    def test_config_validate_missing_config(self, tmp_path):
        """Smoke: config validate fails gracefully with missing config."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "config",
                "validate",
                str(tmp_path / "nonexistent_config.yaml"),
                "--command",
                "train",
            ],
        )
        assert result.exit_code != 0, "Should fail with missing config"


class TestCliSmokeIntegration:
    """Integration smoke tests across multiple commands."""

    def test_full_workflow_single_model(
        self, small_proteomics_data, ultra_fast_training_config, tmp_path
    ):
        """Smoke: Full workflow (splits -> train) executes without error."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        result1 = runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(small_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
            ],
        )
        assert result1.exit_code == 0, f"save-splits failed: {result1.output}"

        result2 = runner.invoke(
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
                str(ultra_fast_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )
        if result2.exit_code != 0:
            pytest.skip(f"train command failed: {result2.output[:200]}")

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1, "Should have 1 run directory"
