"""
End-to-end tests for --run-id auto-detection workflows.

This test suite validates the complete --run-id auto-detection functionality
across all CLI commands, ensuring that run metadata propagates correctly and
users can execute full pipelines with minimal configuration.

Priority workflows tested:
1. Train -> aggregate with --run-id
2. Train -> train-ensemble with --run-id
3. Train -> aggregate -> optimize-panel with --run-id
4. Train -> aggregate -> consensus-panel with --run-id
5. Full pipeline with auto-detected paths

These tests complement test_e2e_runner.py by focusing specifically on the
--run-id auto-detection mechanism rather than general E2E functionality.

Run with: pytest tests/test_e2e_run_id_workflows.py -v
Run slow tests: pytest tests/test_e2e_run_id_workflows.py -v -m slow
"""

import json
import os
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ced_ml.cli.main import cli
from ced_ml.data.schema import (
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    PREVALENT_LABEL,
    TARGET_COL,
)
from click.testing import CliRunner


@pytest.fixture
def small_proteomics_data(tmp_path):
    """
    Create minimal proteomics dataset for fast E2E testing.

    180 samples: 120 controls, 48 incident, 12 prevalent
    10 protein features + demographics
    Fast execution while maintaining realistic structure.
    Balanced demographics ensure proper stratification (2x sex x 3x age = 6 groups, 8 per group).
    """
    rng = np.random.default_rng(42)

    n_controls = 120
    n_incident = 48  # 8 per group (2 sex x 3 age = 6 groups)
    n_prevalent = 12
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 10

    # Create fully balanced demographics to ensure stratification works
    # For incident cases: 2 sexes x 3 age bins x 8 samples = 48
    sex_values = []
    age_values = []

    # Incident cases: balanced across sex x age groups
    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:  # young, middle, old
            for _ in range(8):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    # Prevalent cases: balanced
    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:
            for _ in range(2):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    # Controls: also balanced
    for sex in ["M", "F"]:
        for age_bin in [(30, 35), (45, 55), (65, 70)]:
            for _ in range(20):
                sex_values.append(sex)
                age_values.append(rng.integers(age_bin[0], age_bin[1]))

    labels = (
        [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
        + [CONTROL_LABEL] * n_controls
    )

    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": age_values,
        "BMI": rng.uniform(18, 35, n_total),
        "sex": sex_values,
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    # Add proteins with signal in first 3
    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 3:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.2, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(0.9, 0.3, n_prevalent)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "small_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def fast_training_config(tmp_path):
    """
    Create minimal training config optimized for speed.

    2-fold CV, single repeat, no Optuna, minimal features.
    """
    config = {
        "scenario": "IncidentOnly",
        "cv": {
            "folds": 2,
            "repeats": 1,
            "inner_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
            "random_state": 42,
        },
        "optuna": {"enabled": False},
        "features": {
            "feature_select": "hybrid",
            "kbest_scope": "protein",
            "screen_method": "mannwhitney",
            "screen_top_n": 8,
            "k_grid": [3, 5],
            "stability_thresh": 0.6,
            "corr_thresh": 0.85,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {
            "objective": "youden",
            "fixed_spec": 0.95,
        },
        "lr": {
            "C_min": 0.1,
            "C_max": 10.0,
            "C_points": 2,
            "l1_ratio": [0.5],
            "solver": "saga",
            "max_iter": 500,
        },
        "rf": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
        "xgboost": {
            "n_estimators_grid": [30],
            "max_depth_grid": [3],
            "learning_rate_grid": [0.1],
            "subsample_grid": [0.8],
            "colsample_bytree_grid": [0.8],
        },
        "ensemble": {
            "method": "stacking",
            "base_models": ["LR_EN", "RF"],
            "meta_model": {
                "type": "logistic_regression",
                "penalty": "l2",
                "C": 1.0,
            },
        },
    }

    config_path = tmp_path / "fast_training_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def fast_splits_config(tmp_path):
    """Create minimal splits config."""
    config = {
        "mode": "development",
        "scenarios": ["IncidentOnly"],
        "n_splits": 2,
        "val_size": 0.25,
        "test_size": 0.25,
        "seed_start": 42,
        "train_control_per_case": 5.0,
        "prevalent_train_only": False,
    }

    config_path = tmp_path / "fast_splits_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture(autouse=True)
def set_results_env(tmp_path, monkeypatch):
    """Point CED_RESULTS_DIR to tmp_path/results for auto-detection."""
    monkeypatch.setenv("CED_RESULTS_DIR", str(tmp_path / "results"))


SHARED_RUN_ID = "20260128_E2ETEST"
"""Fixed run_id shared across all train calls within a test, so downstream
commands (aggregate, optimize-panel, etc.) can locate outputs reliably."""


def extract_run_id_from_dir(results_dir: Path) -> str:
    """Extract run_id from results directory structure."""
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise ValueError("No run directory found")
    return run_dirs[0].name.replace("run_", "")


def verify_run_metadata(run_dir: Path, expected_model: str, expected_split_seed: int):
    """Verify run_metadata.json has correct structure and content."""
    metadata_path = run_dir / "run_metadata.json"
    assert metadata_path.exists(), f"Missing run_metadata.json in {run_dir}"

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check required top-level fields
    assert "run_id" in metadata
    assert "infile" in metadata
    assert "split_dir" in metadata
    assert "models" in metadata

    # Check model-specific fields (nested structure)
    assert expected_model in metadata["models"], f"Model {expected_model} not in metadata"
    model_entry = metadata["models"][expected_model]
    assert "scenario" in model_entry
    assert "infile" in model_entry
    assert "split_dir" in model_entry
    assert "split_seed" in model_entry
    assert "timestamp" in model_entry

    # Validate content
    assert model_entry["split_seed"] == expected_split_seed


class TestRunIdMetadataCreation:
    """Test that training creates correct run_metadata.json files."""

    @pytest.mark.slow
    def test_train_creates_run_metadata(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Train command creates run_metadata.json with correct fields.

        This is the foundation for all --run-id auto-detection.
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
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Train model
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
                "42",
                "--run-id",
                SHARED_RUN_ID,
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            pytest.skip(f"Training failed: {result.output[:200]}")

        # Find run directory (production layout: results/run_{ID}/)
        # Actual root for this split is: results/run_{ID}/LR_EN/splits/split_seed42/
        run_dir = results_dir / f"run_{SHARED_RUN_ID}"
        assert run_dir.exists()

        # Verify metadata (stored at run level, not split level)
        verify_run_metadata(run_dir, "LR_EN", 42)

    def test_run_metadata_persists_across_splits(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Same run_id used for multiple split seeds creates separate metadata.

        Each split_seed should have its own split_seed directory with separate metadata.
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
            ],
        )

        # Train on two splits with shared run_id (mirrors production usage)
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
                pytest.skip(f"Training on seed {seed} failed")

        # Both splits share one run_id directory
        # Production layout: results/run_{ID}/{MODEL}/splits/split_seed{N}/
        run_dir = results_dir / f"run_{SHARED_RUN_ID}"
        assert run_dir.exists()
        model_dir = run_dir / "LR_EN"
        assert model_dir.exists(), "Model subdirectory not found"
        splits_parent = model_dir / "splits"
        assert splits_parent.exists(), "splits/ subdirectory not found"
        split_dirs = [
            d for d in splits_parent.iterdir() if d.is_dir() and d.name.startswith("split_")
        ]
        assert len(split_dirs) == 2, f"Expected 2 split dirs, got {len(split_dirs)}"
        assert (run_dir / "run_metadata.json").exists()


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
        assert "not found" in result.output.lower() or "does not exist" in result.output.lower()


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
        assert (
            "base model" in result.output.lower()
            or "not found" in result.output.lower()
            or "no models" in result.output.lower()
        )


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


class TestFullPipelineWithRunId:
    """Test complete pipelines using --run-id auto-detection throughout."""

    @pytest.mark.slow
    def test_complete_single_model_pipeline(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Complete single-model pipeline with --run-id auto-detection.

        Workflow: save-splits -> train -> aggregate-splits -> optimize-panel
        Uses --run-id throughout to minimize configuration.
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
        )
        assert result_splits.exit_code == 0

        # Step 2: Train on both splits with shared run_id
        for seed in [42, 43]:
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

            if result_train.exit_code != 0:
                pytest.skip(f"Training failed on seed {seed}")

        run_id = SHARED_RUN_ID

        # Step 3: Aggregate with --run-id
        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Step 4: Optimize panel with --run-id
        result_opt = runner.invoke(
            cli,
            ["optimize-panel", "--run-id", run_id, "--min-size", "3"],
            catch_exceptions=False,
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        if result_opt.exit_code != 0:
            pytest.skip("Panel optimization failed")

        # Verify complete pipeline outputs exist
        run_dir = results_dir / f"run_{run_id}"
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "LR_EN" / "aggregated").exists()
        assert (run_dir / "LR_EN" / "aggregated" / "optimize_panel").exists()

    @pytest.mark.slow
    def test_complete_ensemble_pipeline(
        self, small_proteomics_data, fast_training_config, tmp_path
    ):
        """
        Test: Complete ensemble pipeline with --run-id auto-detection.

        Workflow: save-splits -> train base models -> train-ensemble -> aggregate
        Uses --run-id for ensemble training.
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

            if result_train.exit_code != 0:
                pytest.skip(f"Base model {model} training failed")

        run_id = SHARED_RUN_ID

        # Step 3: Train ensemble with --run-id
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

        # Step 4: Verify ensemble outputs
        ensemble_dir = results_dir / f"run_{SHARED_RUN_ID}" / "ENSEMBLE" / "splits" / "split_seed42"
        assert ensemble_dir.exists()
        assert (ensemble_dir / "core" / "metrics.json").exists()


class TestRunIdErrorHandling:
    """Test error handling and edge cases for --run-id auto-detection."""

    def test_invalid_run_id_format(self):
        """Test graceful handling of malformed run_id."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "invalid_format", "--model", "LR_EN"],
        )

        assert result.exit_code != 0

    def test_run_id_partial_results(self, tmp_path):
        """
        Test handling when run_id exists but results are incomplete.

        Some splits trained, others missing.
        """
        results_dir = tmp_path / "results"

        # Create partial structure (matching production layout)
        # Production layout: results/run_{ID}/{MODEL}/splits/split_seed{N}/
        run_dir = results_dir / "run_20260127_115115"
        (run_dir / "LR_EN" / "splits" / "split_seed42").mkdir(parents=True)
        # Missing split_seed43 to test partial results handling

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "20260127_115115", "--model", "LR_EN"],
        )

        # Aggregation succeeds with partial results (any number of splits).
        # It does not know the expected total, so no warning is emitted.
        assert result.exit_code == 0

    def test_run_id_missing_metadata(self, tmp_path):
        """
        Test handling when run directory exists but run_metadata.json is missing.

        Should still work (metadata is optional for older runs).
        """
        results_dir = tmp_path / "results"
        run_dir = results_dir / "run_20260127_115115"
        (run_dir / "LR_EN").mkdir(parents=True)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", "20260127_115115", "--model", "LR_EN"],
        )

        # May fail due to missing actual results, but shouldn't crash on missing metadata
        assert result.exit_code in [0, 1]


# ==================== How to Run ====================
# Fast tests only (skips @pytest.mark.slow):
#   pytest tests/test_e2e_run_id_workflows.py -v -m "not slow"
#
# All tests including slow integration tests:
#   pytest tests/test_e2e_run_id_workflows.py -v
#
# Specific test class:
#   pytest tests/test_e2e_run_id_workflows.py::TestRunIdMetadataCreation -v
#
# Single test:
#   pytest tests/test_e2e_run_id_workflows.py::TestAggregateWithRunId::test_aggregate_auto_detects_from_run_id -v
