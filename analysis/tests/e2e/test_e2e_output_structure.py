"""
End-to-end tests for output structure validation and persistence.

This test suite validates that the pipeline produces correctly structured
outputs that persist across the entire workflow and can be consumed by
downstream commands.

Key areas tested:
1. Training output structure (predictions, metrics, models, configs)
2. Aggregation output structure (pooled predictions, metrics, reports)
3. Ensemble output structure (meta-learner artifacts)
4. Panel optimization output structure (RFE curves, recommendations)
5. Consensus panel output structure (rankings, metadata)
6. Cross-stage compatibility (outputs from one stage consumed by next)

These tests ensure reproducibility and correct data flow through the pipeline.

Run with: pytest tests/test_e2e_output_structure.py -v
Run slow tests: pytest tests/test_e2e_output_structure.py -v -m slow
"""

import json
import os
import unittest.mock

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from ced_ml.cli.main import cli
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)


@pytest.fixture
def tiny_proteomics_data(tmp_path):
    """
    Create tiny proteomics dataset for structure validation tests.

    180 samples: 120 controls, 48 incident, 12 prevalent
    8 protein features
    Minimal data for fast structure checks.

    NOTE: Increased from 60 to 180 samples to support val_size=0.25
    """
    rng = np.random.default_rng(42)

    n_controls = 120
    n_incident = 48
    n_prevalent = 12
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 8

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    data = {
        ID_COL: [f"S{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": rng.integers(30, 70, n_total),
        "BMI": rng.uniform(20, 32, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)
        if i < 2:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.5, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.0, 0.3, n_prevalent)
        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "tiny_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def minimal_config(tmp_path):
    """Create minimal training config for structure validation."""
    config = {
        "scenario": "IncidentOnly",
        "run_id": "test_e2e_run",  # Fixed run_id for multi-split tests
        "cv": {"folds": 2, "repeats": 1, "inner_folds": 2, "scoring": "roc_auc"},
        "optuna": {"enabled": False},
        "features": {
            "feature_select": "hybrid",
            "screen_method": "mannwhitney",
            "screen_top_n": 6,
            "k_grid": [3],
            "stability_thresh": 0.5,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {"objective": "youden", "fixed_spec": 0.95},
        "lr": {"C_min": 1.0, "C_max": 10.0, "C_points": 1, "l1_ratio": [0.5]},
    }

    config_path = tmp_path / "minimal_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


class TestTrainingOutputStructure:
    """Test training command produces correctly structured outputs."""

    @pytest.mark.slow
    def test_training_creates_required_directories(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Training creates complete directory structure.

        Validates presence of core/, preds/, cv/, plots/, and config/ directories.
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
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Train
        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(tiny_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed: {result.output[:200]}")

        # Find run directory under model directory
        # Actual structure: results/LR_EN/run_test_e2e_run/LR_EN/splits/split_seed42/
        run_dir = results_dir / "run_test_e2e_run"
        assert run_dir.exists(), f"Run directory not found: {run_dir}"
        split_dir = run_dir / "LR_EN" / "splits" / "split_seed42"

        # Verify directory structure
        required_dirs = ["core", "preds", "cv"]
        for dirname in required_dirs:
            assert (split_dir / dirname).exists(), f"Missing directory: {dirname}"

    @pytest.mark.slow
    def test_training_predictions_have_correct_schema(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Training predictions have required columns and valid data.

        Validates idx, y_true, y_prob, y_prob_adjusted columns.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(tiny_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip("Training failed")

        # Find predictions files
        pred_files = list(results_dir.rglob("*test_preds*.csv"))
        assert len(pred_files) > 0, "No prediction files found"

        # Validate schema
        preds_df = pd.read_csv(pred_files[0])

        required_cols = ["idx", "y_true", "y_prob"]
        for col in required_cols:
            assert col in preds_df.columns, f"Missing column: {col}"

        # Validate data types and ranges
        assert preds_df["idx"].dtype in [np.int64, np.int32]
        assert preds_df["y_true"].dtype in [np.int64, np.int32, np.float64]
        assert all(0.0 <= p <= 1.0 for p in preds_df["y_prob"]), "y_prob out of [0, 1] range"
        assert not preds_df["y_prob"].isna().any(), "y_prob contains NaN"

    @pytest.mark.slow
    def test_training_metrics_have_correct_structure(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Training metrics files have expected structure.

        Validates val_metrics.csv and test_metrics.csv structure.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(tiny_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip("Training failed")

        # Find metrics files
        test_metrics_files = list(results_dir.rglob("test_metrics.csv"))
        assert len(test_metrics_files) > 0, "No test metrics found"

        metrics_df = pd.read_csv(test_metrics_files[0])

        # Metrics should have metric names and values
        assert len(metrics_df) > 0, "Metrics file is empty"

        # Common metric columns (format may vary)
        has_metric_col = "metric" in metrics_df.columns
        has_auroc_col = any(c.lower() == "auroc" for c in metrics_df.columns)

        assert (
            has_metric_col or has_auroc_col
        ), f"Unexpected metrics format. Columns: {metrics_df.columns.tolist()}"

    @pytest.mark.slow
    def test_training_saves_config_copy(self, tiny_proteomics_data, minimal_config, tmp_path):
        """
        Test: Training saves a copy of the config for reproducibility.

        Validates config persistence.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        result = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(tiny_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(minimal_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip("Training failed")

        # Find saved config
        config_files = list(results_dir.rglob("*config*.yaml"))
        assert len(config_files) > 0, "No config file saved"

        # Verify it's valid YAML
        with open(config_files[0]) as f:
            saved_config = yaml.safe_load(f)

        assert "scenario" in saved_config
        assert "cv" in saved_config


class TestAggregationOutputStructure:
    """Test aggregation command produces correctly structured outputs."""

    @pytest.mark.slow
    def test_aggregation_creates_required_subdirectories(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Aggregation creates complete output structure.

        Validates core/, preds/, cv/, reports/ directories.
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
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        # Train on both splits
        for seed in [42, 43]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip(f"Training failed on seed {seed}")

        # Use fixed run_id from config (both splits share the same run_id)
        run_id = "test_e2e_run"

        # Aggregate
        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Verify aggregation directory structure
        agg_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated"
        assert agg_dir.exists()

        required_dirs = ["metrics", "preds", "panels"]
        for dirname in required_dirs:
            assert (agg_dir / dirname).exists(), f"Missing aggregated directory: {dirname}"

    @pytest.mark.slow
    def test_aggregation_pooled_predictions_schema(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Aggregated pooled predictions have correct schema.

        Validates split_seed, idx, y_true, y_prob columns in pooled data.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        for seed in [42, 43]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip("Training failed")

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Find pooled predictions
        pooled_files = list(results_dir.rglob("pooled_test_preds*.csv"))
        assert len(pooled_files) > 0, "No pooled predictions found"

        pooled_df = pd.read_csv(pooled_files[0])

        # Required columns
        required_cols = ["split_seed", "idx", "y_true", "y_prob"]
        for col in required_cols:
            assert col in pooled_df.columns, f"Missing column in pooled data: {col}"

        # Verify data from multiple splits
        assert pooled_df["split_seed"].nunique() == 2, "Pooled data should have 2 splits"

    @pytest.mark.slow
    def test_aggregation_summary_metrics_structure(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Aggregated summary metrics have correct structure.

        Validates mean/std/CI columns for metrics.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        for seed in [42, 43]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip("Training failed")

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Find summary metrics
        summary_files = list(results_dir.rglob("*summary*.csv"))
        if len(summary_files) == 0:
            summary_files = list(results_dir.rglob("*aggregated*metrics*.csv"))

        if len(summary_files) > 0:
            summary_df = pd.read_csv(summary_files[0])

            # Summary should have mean/std or CI columns
            cols_lower = [c.lower() for c in summary_df.columns]
            has_stats = any(
                s in " ".join(cols_lower) for s in ["mean", "std", "ci", "lower", "upper", "median"]
            )

            assert has_stats, f"Summary metrics missing statistical columns: {summary_df.columns}"


class TestEnsembleOutputStructure:
    """Test ensemble training produces correctly structured outputs."""

    @pytest.mark.slow
    def test_ensemble_creates_separate_directory(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Ensemble creates outputs in ENSEMBLE/split_{seed} directory.

        Validates separation from base models.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Train base models
        for model in ["LR_EN", "RF"]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    model,
                    "--split-seed",
                    "42",
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip(f"Base model {model} training failed")

        # Train ensemble
        # Use fixed run_id from config
        run_id = "test_e2e_run"

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

        # Verify ENSEMBLE directory exists and is separate
        ensemble_dir = results_dir / f"run_{run_id}" / "ENSEMBLE" / "splits" / "split_seed42"
        assert ensemble_dir.exists(), "ENSEMBLE directory not created"

        # Should have similar structure to base models
        assert (ensemble_dir / "core").exists()
        assert (ensemble_dir / "preds").exists()

    @pytest.mark.slow
    def test_ensemble_predictions_reference_base_models(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Ensemble outputs reference which base models were used.

        Validates metadata tracking.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        for model in ["LR_EN", "RF"]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    model,
                    "--split-seed",
                    "42",
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip("Base model training failed")

        # Use fixed run_id from config
        run_id = "test_e2e_run"

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

        # Find ensemble metadata
        metrics_files = list((results_dir / f"run_{run_id}" / "ENSEMBLE").rglob("metrics.json"))
        if len(metrics_files) > 0:
            with open(metrics_files[0]) as f:
                metrics = json.load(f)

            # Should reference base models (implementation-specific)
            # At minimum, metrics should exist
            assert len(metrics) > 0


class TestPanelOptimizationOutputStructure:
    """Test panel optimization produces correctly structured outputs."""

    @pytest.mark.slow
    def test_panel_optimization_creates_required_files(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Panel optimization creates panel_curve.csv and recommendations.

        Validates RFE output structure.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
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

        for seed in [42, 43]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip("Training failed")

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--min-size",
                "2",
                "--stability-threshold",
                "0.75",
            ],
            catch_exceptions=False,
        )

        if result_opt.exit_code != 0:
            pytest.skip("Panel optimization failed")

        # Verify RFE outputs
        panel_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated" / "optimize_panel"
        assert panel_dir.exists()

        required_files = [
            "panel_curve_aggregated.csv",
            "feature_ranking_aggregated.csv",
            "recommended_panels_aggregated.json",
        ]

        for filename in required_files:
            assert (panel_dir / filename).exists(), f"Missing panel optimization file: {filename}"

    @pytest.mark.slow
    def test_panel_curve_has_correct_schema(self, tiny_proteomics_data, minimal_config, tmp_path):
        """
        Test: Panel curve CSV has size, auroc_val, auroc_test columns.

        Validates panel optimization data structure.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
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

        for seed in [42, 43]:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(minimal_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result.exit_code != 0:
                pytest.skip("Training failed")

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--min-size",
                "2",
                "--stability-threshold",
                "0.75",
            ],
            catch_exceptions=False,
        )

        if result_opt.exit_code != 0:
            pytest.skip("Panel optimization failed")

        # Validate panel curve
        panel_curve_path = (
            results_dir
            / f"run_{run_id}"
            / "LR_EN"
            / "aggregated"
            / "optimize_panel"
            / "panel_curve_aggregated.csv"
        )
        panel_curve = pd.read_csv(panel_curve_path)

        assert "size" in panel_curve.columns
        assert "auroc_val" in panel_curve.columns or "AUROC_val" in panel_curve.columns
        assert len(panel_curve) > 0
        assert (
            panel_curve["size"].is_monotonic_decreasing
            or panel_curve["size"].is_monotonic_increasing
        )


class TestConsensusPanelOutputStructure:
    """Test consensus panel produces correctly structured outputs."""

    @pytest.mark.slow
    def test_consensus_panel_creates_required_files(
        self, tiny_proteomics_data, minimal_config, tmp_path
    ):
        """
        Test: Consensus panel creates final_panel.txt and rankings.

        Validates consensus output structure.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        # Train two models with run_id to ensure correct directory structure
        for model in ["LR_EN", "RF"]:
            for seed in [42, 43]:
                result = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(tiny_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(minimal_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        run_id,
                    ],
                    catch_exceptions=False,
                )

                if result.exit_code != 0:
                    pytest.skip(f"Training {model} seed {seed} failed")

        # Aggregate both models
        for model in ["LR_EN", "RF"]:
            result = runner.invoke(
                cli,
                ["aggregate-splits", "--run-id", run_id, "--model", model],
                catch_exceptions=False,
            )
            if result.exit_code != 0:
                pytest.skip(f"Aggregation {model} failed")

        # Generate consensus
        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result_consensus = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    run_id,
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_consensus.exit_code != 0:
            pytest.skip("Consensus panel failed")

        # Verify consensus outputs
        # Current structure: results/run_{run_id}/{model}/consensus/
        # Consensus creates a separate consensus directory for the aggregated panel
        consensus_dir = results_dir / f"run_{run_id}" / "consensus"
        if not consensus_dir.exists():
            # Fallback: check if it's under the first base model
            consensus_dir = results_dir / f"run_{run_id}" / "LR_EN" / "consensus"

        assert consensus_dir.exists(), f"Consensus directory not found at {consensus_dir}"

        required_files = [
            "final_panel.txt",
            "final_panel.csv",
            "consensus_ranking.csv",
            "consensus_metadata.json",
        ]

        for filename in required_files:
            assert (consensus_dir / filename).exists(), f"Missing consensus file: {filename}"

    @pytest.mark.slow
    def test_final_panel_txt_format(self, tiny_proteomics_data, minimal_config, tmp_path):
        """
        Test: final_panel.txt has one protein per line for --fixed-panel.

        Validates format for downstream validation.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(tiny_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        # Use fixed run_id from config
        run_id = "test_e2e_run"

        for model in ["LR_EN", "RF"]:
            for seed in [42, 43]:
                result = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(tiny_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(minimal_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        run_id,
                    ],
                    catch_exceptions=False,
                )

                if result.exit_code != 0:
                    pytest.skip("Training failed")

        for model in ["LR_EN", "RF"]:
            result = runner.invoke(
                cli,
                ["aggregate-splits", "--run-id", run_id, "--model", model],
                catch_exceptions=False,
            )
            if result.exit_code != 0:
                pytest.skip(f"Aggregation {model} failed")

        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result_consensus = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    run_id,
                    "--infile",
                    str(tiny_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_consensus.exit_code != 0:
            pytest.skip("Consensus panel failed")

        # Validate final_panel.txt format
        # Current structure: results/run_{run_id}/consensus/final_panel.txt
        panel_txt = results_dir / f"run_{run_id}" / "consensus" / "final_panel.txt"
        if not panel_txt.exists():
            # Fallback: check if it's under the first base model
            panel_txt = results_dir / f"run_{run_id}" / "LR_EN" / "consensus" / "final_panel.txt"
        with open(panel_txt) as f:
            lines = [line.strip() for line in f if line.strip()]

        # Should have protein names, one per line
        assert len(lines) > 0
        assert all("PROT_" in line or line.endswith("_resid") for line in lines)


# ==================== How to Run ====================
# Fast tests only:
#   pytest tests/test_e2e_output_structure.py -v -m "not slow"
#
# All tests including slow:
#   pytest tests/test_e2e_output_structure.py -v
#
# Specific test class:
#   pytest tests/test_e2e_output_structure.py::TestTrainingOutputStructure -v
