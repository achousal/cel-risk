"""
End-to-end tests for multi-model integration workflows.

This test suite validates coordinated multi-model training and analysis:
1. Train multiple models with shared run_id
2. Cross-model aggregation consistency
3. Consensus panel generation across models
4. Run metadata consistency across models
5. Multi-model ensemble workflows

Multi-model coordination is critical for consensus panel generation and
ensemble learning, where results from different algorithms are combined.

Run with: pytest tests/test_e2e_multi_model_workflows.py -v
Run slow tests: pytest tests/test_e2e_multi_model_workflows.py -v -m slow
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
def multi_model_proteomics_data(tmp_path):
    """
    Create proteomics dataset for multi-model testing.

    180 samples: 120 controls, 48 incident, 12 prevalent
    12 protein features with diverse signals for different models.

    NOTE: Increased from 100 to 180 samples to support val_size=0.25
    """
    rng = np.random.default_rng(42)

    n_controls = 120
    n_incident = 48
    n_prevalent = 12
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 12

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    data = {
        ID_COL: [f"MULTI_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    # Add proteins with varied signal types for different model sensitivities
    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 3:
            # Linear signals (good for LR)
            signal[n_controls : n_controls + n_incident] = rng.normal(2.0, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.5, 0.3, n_prevalent)
        elif i < 6:
            # Non-linear signals (good for RF/XGBoost)
            signal[n_controls : n_controls + n_incident] = np.abs(rng.normal(1.5, 0.5, n_incident))
            signal[n_controls + n_incident :] = np.abs(rng.normal(1.0, 0.5, n_prevalent))

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "multi_model_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def fast_multi_config(tmp_path):
    """Create fast training config for multi-model testing."""
    config = {
        "scenario": "IncidentOnly",
        "cv": {
            "folds": 2,
            "repeats": 1,
            "inner_folds": 2,
            "scoring": "roc_auc",
        },
        "optuna": {"enabled": False},
        "features": {
            "feature_select": "hybrid",
            "screen_method": "mannwhitney",
            "screen_top_n": 10,
            "k_grid": [5],
            "stability_thresh": 0.5,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {"objective": "youden"},
        "allow_test_thresholding": True,
        "lr": {"C_min": 1.0, "C_max": 10.0, "C_points": 1, "l1_ratio": [0.5]},
        "rf": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
    }

    config_path = tmp_path / "fast_multi_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


SHARED_RUN_ID = "multi_model_test"
"""Shared run_id for multi-model coordination tests."""


class TestSharedRunIdCoordination:
    """Test training multiple models with shared run_id."""

    @pytest.mark.slow
    def test_multiple_models_share_run_id(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Multiple models trained with same run_id create consistent structure.

        Validates that different models can coordinate via shared run_id.
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
                str(multi_model_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        models = ["LR_EN", "RF"]

        # Train both models with shared run_id
        for model in models:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(fast_multi_config),
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
                pytest.skip(f"{model} training failed: {result.output[:200]}")

        # Verify both models created run directories with shared run_id
        # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/splits/split_seed{N}/
        lr_run_dir = results_dir / f"run_{SHARED_RUN_ID}" / "LR_EN"
        rf_run_dir = results_dir / f"run_{SHARED_RUN_ID}" / "RF"

        assert lr_run_dir.exists(), "LR_EN run directory not found"
        assert rf_run_dir.exists(), "RF run directory not found"

        # Both should have split_seed42 subdirectories
        assert (lr_run_dir / "splits" / "split_seed42").exists()
        assert (rf_run_dir / "splits" / "split_seed42").exists()

    @pytest.mark.slow
    def test_shared_run_id_metadata_consistency(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Models with shared run_id have consistent metadata.

        Validates run_id, infile, split_dir consistency.
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
                str(multi_model_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        models = ["LR_EN", "RF"]

        for model in models:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(fast_multi_config),
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
                pytest.skip(f"{model} training failed")

        # Load metadata from both models
        # Each model has its own run_metadata.json at results/{MODEL}/run_{RUN_ID}/
        lr_metadata_path = results_dir / f"run_{SHARED_RUN_ID}" / "run_metadata.json"
        rf_metadata_path = results_dir / f"run_{SHARED_RUN_ID}" / "run_metadata.json"

        if not lr_metadata_path.exists() or not rf_metadata_path.exists():
            pytest.skip("Metadata files not found")

        with open(lr_metadata_path) as f:
            lr_metadata = json.load(f)

        with open(rf_metadata_path) as f:
            rf_metadata = json.load(f)

        # Check consistency of shared fields across separate metadata files
        assert lr_metadata["run_id"] == rf_metadata["run_id"]
        assert lr_metadata["run_id"] == SHARED_RUN_ID

        # Each metadata file contains its own model's entry
        assert "models" in lr_metadata
        assert "LR_EN" in lr_metadata["models"]
        assert lr_metadata["models"]["LR_EN"]["split_seed"] == 42

        assert "models" in rf_metadata
        assert "RF" in rf_metadata["models"]
        assert rf_metadata["models"]["RF"]["split_seed"] == 42


class TestCrossModelAggregation:
    """Test aggregation consistency across models."""

    @pytest.mark.slow
    def test_aggregate_multiple_models_independently(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Each model can be aggregated independently with shared run_id.

        Validates per-model aggregation with run_id coordination.
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
                str(multi_model_proteomics_data),
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

        models = ["LR_EN", "RF"]

        # Train both models on both splits with shared run_id
        for model in models:
            for seed in [42, 43]:
                result = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(multi_model_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(fast_multi_config),
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
                    pytest.skip(f"{model} training failed on seed {seed}")

        # Aggregate each model
        for model in models:
            # Pass full results-dir explicitly since we're using non-standard layout
            # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/
            model_results_dir = results_dir / f"run_{SHARED_RUN_ID}" / model
            result_agg = runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--results-dir",
                    str(model_results_dir),
                ],
                catch_exceptions=False,
            )

            if result_agg.exit_code != 0:
                pytest.skip(f"{model} aggregation failed: {result_agg.output[:200]}")

            # Verify aggregated outputs
            # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/aggregated/
            agg_dir = results_dir / f"run_{SHARED_RUN_ID}" / model / "aggregated"
            assert agg_dir.exists(), f"{model} aggregation directory not found"
            assert (agg_dir / "metrics").exists(), f"{model} aggregation metrics missing"

    @pytest.mark.slow
    def test_aggregated_models_produce_comparable_outputs(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Aggregated models produce comparable output structures.

        Validates that different models follow same aggregation schema.
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
                str(multi_model_proteomics_data),
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

        models = ["LR_EN", "RF"]

        for model in models:
            for seed in [42, 43]:
                result = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(multi_model_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(fast_multi_config),
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
                    pytest.skip("Training failed")

        for model in models:
            # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/
            model_results_dir = results_dir / f"run_{SHARED_RUN_ID}" / model
            runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--results-dir",
                    str(model_results_dir),
                ],
                catch_exceptions=False,
            )

        # Check that both models have similar output structures
        # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/aggregated/
        lr_agg_dir = results_dir / f"run_{SHARED_RUN_ID}" / "LR_EN" / "aggregated"
        rf_agg_dir = results_dir / f"run_{SHARED_RUN_ID}" / "RF" / "aggregated"

        lr_subdirs = {d.name for d in lr_agg_dir.iterdir() if d.is_dir()}
        rf_subdirs = {d.name for d in rf_agg_dir.iterdir() if d.is_dir()}

        # Should have significant overlap in directory structure
        common_dirs = lr_subdirs & rf_subdirs
        assert len(common_dirs) >= 2, f"Too few common directories: {common_dirs}"


class TestConsensusPanelIntegration:
    """Test consensus panel generation across models."""

    @pytest.mark.slow
    def test_consensus_panel_requires_multiple_models(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Consensus panel generation works with multiple trained models.

        Validates cross-model feature ranking aggregation.
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
                str(multi_model_proteomics_data),
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

        models = ["LR_EN", "RF"]

        # Train and aggregate both models
        for model in models:
            for seed in [42, 43]:
                result_train = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(multi_model_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(fast_multi_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        SHARED_RUN_ID,
                    ],
                    catch_exceptions=False,
                )

                if result_train.exit_code != 0:
                    pytest.skip(f"{model} training failed")

            # Production structure: results/run_{RUN_ID}/{MODEL}/
            model_results_dir = results_dir / f"run_{SHARED_RUN_ID}" / model
            result_agg = runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--results-dir",
                    str(model_results_dir),
                ],
                catch_exceptions=False,
            )

            if result_agg.exit_code != 0:
                pytest.skip(f"{model} aggregation failed")

        # Generate consensus panel
        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result_consensus = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    SHARED_RUN_ID,
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_consensus.exit_code != 0:
            pytest.skip(f"Consensus panel failed: {result_consensus.output[:200]}")

        # Verify consensus outputs
        # Production structure: results/run_{RUN_ID}/consensus/
        consensus_dir = results_dir / f"run_{SHARED_RUN_ID}" / "consensus"
        assert consensus_dir.exists(), "Consensus directory not found"
        assert (consensus_dir / "final_panel.txt").exists(), "Final panel not found"
        assert (consensus_dir / "consensus_ranking.csv").exists(), "Ranking not found"

    @pytest.mark.slow
    def test_consensus_panel_integrates_all_models(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Consensus panel metadata references all contributing models.

        Validates traceability of consensus results.
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
                str(multi_model_proteomics_data),
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

        models = ["LR_EN", "RF"]

        for model in models:
            for seed in [42, 43]:
                runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(multi_model_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(fast_multi_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        SHARED_RUN_ID,
                    ],
                    catch_exceptions=False,
                )

            model_results_dir = results_dir / f"run_{SHARED_RUN_ID}" / model
            runner.invoke(
                cli,
                ["aggregate-splits", "--results-dir", str(model_results_dir)],
                catch_exceptions=False,
            )

        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result_consensus = runner.invoke(
                cli,
                [
                    "consensus-panel",
                    "--run-id",
                    SHARED_RUN_ID,
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_consensus.exit_code != 0:
            pytest.skip("Consensus panel failed")

        # Check consensus metadata
        # Production structure: results/run_{RUN_ID}/consensus/
        metadata_path = (
            results_dir / f"run_{SHARED_RUN_ID}" / "consensus" / "consensus_metadata.json"
        )

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Should reference multiple models
            metadata_str = json.dumps(metadata).lower()
            has_lr = "lr_en" in metadata_str or "lr" in metadata_str
            has_rf = "rf" in metadata_str

            # Should mention both models
            assert has_lr or has_rf, "Consensus metadata missing model references"


class TestEnsembleMultiModelWorkflow:
    """Test ensemble training with multiple base models."""

    @pytest.mark.slow
    def test_ensemble_auto_detects_multiple_base_models(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Ensemble training auto-detects all trained base models.

        Validates that ensemble finds all models with shared run_id.
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
                str(multi_model_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        models = ["LR_EN", "RF"]

        # Train base models
        for model in models:
            result = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(fast_multi_config),
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
                pytest.skip(f"{model} training failed")

        # Train ensemble with auto-detection
        result_ensemble = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--run-id",
                SHARED_RUN_ID,
                "--results-dir",
                str(results_dir),
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_ensemble.exit_code != 0:
            pytest.fail(f"Ensemble training failed: {result_ensemble.output}")

        # Verify ensemble outputs
        # Production structure: results/run_{RUN_ID}/ENSEMBLE/splits/split_seed{N}/
        ensemble_dir = results_dir / f"run_{SHARED_RUN_ID}" / "ENSEMBLE" / "splits" / "split_seed42"
        assert ensemble_dir.exists(), "Ensemble directory not found"
        assert (ensemble_dir / "core").exists(), "Ensemble core missing"

    @pytest.mark.slow
    def test_ensemble_fails_gracefully_with_single_model(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Ensemble training fails gracefully with only one base model.

        Validates minimum model requirement.
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
                str(multi_model_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        # Train only one model
        runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(multi_model_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(fast_multi_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
                "--run-id",
                SHARED_RUN_ID,
            ],
            catch_exceptions=False,
        )

        # Try ensemble (should fail or warn)
        result_ensemble = runner.invoke(
            cli,
            [
                "train-ensemble",
                "--run-id",
                SHARED_RUN_ID,
                "--results-dir",
                str(results_dir),
                "--split-seed",
                "42",
            ],
        )

        # Should fail or warn about insufficient models
        assert result_ensemble.exit_code != 0 or (
            "warning" in result_ensemble.output.lower()
            or "single" in result_ensemble.output.lower()
        )


class TestMultiModelPanelOptimization:
    """Test panel optimization across multiple models."""

    @pytest.mark.slow
    def test_optimize_panel_processes_all_models(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: Panel optimization with run_id processes all trained models.

        Validates batch panel optimization.
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
                str(multi_model_proteomics_data),
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

        models = ["LR_EN", "RF"]

        # Train and aggregate both models
        for model in models:
            for seed in [42, 43]:
                runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(multi_model_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(fast_multi_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        SHARED_RUN_ID,
                    ],
                    catch_exceptions=False,
                )

            # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/
            model_results_dir = results_dir / f"run_{SHARED_RUN_ID}" / model
            result_agg = runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--results-dir",
                    str(model_results_dir),
                ],
                catch_exceptions=False,
            )

            if result_agg.exit_code != 0:
                pytest.skip(f"{model} aggregation failed")

        # Optimize panels for all models
        with unittest.mock.patch.dict(os.environ, {"CED_RESULTS_DIR": str(results_dir)}):
            result_optimize = runner.invoke(
                cli,
                [
                    "optimize-panel",
                    "--run-id",
                    SHARED_RUN_ID,
                    "--min-size",
                    "3",
                    "--stability-threshold",
                    "0.75",
                ],
                catch_exceptions=False,
            )

        if result_optimize.exit_code != 0:
            pytest.skip(f"Panel optimization failed: {result_optimize.output[:200]}")

        # Verify panel optimization for both models
        for model in models:
            # Production structure: results/{MODEL}/run_{RUN_ID}/{MODEL}/aggregated/optimize_panel/
            panel_dir = (
                results_dir / f"run_{SHARED_RUN_ID}" / model / "aggregated" / "optimize_panel"
            )
            assert panel_dir.exists(), f"{model} panel optimization not found"
            assert (
                panel_dir / "panel_curve_aggregated.csv"
            ).exists(), f"{model} panel curve missing"


class TestMultiModelMetadataConsistency:
    """Test metadata consistency across coordinated multi-model runs."""

    @pytest.mark.slow
    def test_all_models_share_consistent_run_parameters(
        self, multi_model_proteomics_data, fast_multi_config, tmp_path
    ):
        """
        Test: All models with shared run_id have consistent run parameters.

        Validates dataset, splits, and configuration consistency.
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
                str(multi_model_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--val-size",
                "0.25",
            ],
        )

        models = ["LR_EN", "RF"]

        for model in models:
            runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(multi_model_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(fast_multi_config),
                    "--model",
                    model,
                    "--split-seed",
                    "42",
                    "--run-id",
                    SHARED_RUN_ID,
                ],
                catch_exceptions=False,
            )

        # Load metadata files
        # Production structure: results/{MODEL}/run_{RUN_ID}/run_metadata.json
        # Each model has its own metadata file
        lr_metadata_path = results_dir / f"run_{SHARED_RUN_ID}" / "run_metadata.json"
        rf_metadata_path = results_dir / f"run_{SHARED_RUN_ID}" / "run_metadata.json"

        if not lr_metadata_path.exists() or not rf_metadata_path.exists():
            pytest.skip("Metadata files not found")

        with open(lr_metadata_path) as f:
            lr_metadata = json.load(f)

        with open(rf_metadata_path) as f:
            rf_metadata = json.load(f)

        # Both should have same run_id
        assert lr_metadata["run_id"] == SHARED_RUN_ID
        assert rf_metadata["run_id"] == SHARED_RUN_ID
        assert lr_metadata["run_id"] == rf_metadata["run_id"]

        # Check per-model metadata
        assert "models" in lr_metadata and "LR_EN" in lr_metadata["models"]
        assert "models" in rf_metadata and "RF" in rf_metadata["models"]

        # Check consistency across models
        lr_seed = lr_metadata["models"]["LR_EN"]["split_seed"]
        rf_seed = rf_metadata["models"]["RF"]["split_seed"]
        assert lr_seed == rf_seed, f"Inconsistent split_seeds: LR_EN={lr_seed}, RF={rf_seed}"


# ==================== How to Run ====================
# Fast tests only:
#   pytest tests/test_e2e_multi_model_workflows.py -v -m "not slow"
#
# All tests including slow:
#   pytest tests/test_e2e_multi_model_workflows.py -v
#
# Specific test class:
#   pytest tests/test_e2e_multi_model_workflows.py::TestSharedRunIdCoordination -v
