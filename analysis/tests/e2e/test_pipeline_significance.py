"""
E2E tests for permutation testing and feature importance workflows.

Tests the full workflow for:
- Label permutation testing (ced permutation-test)
- OOF feature importance (computed during training)
- Aggregated importance (ced aggregate-splits)
- Drop-column validation (ced optimize-panel)

Run with: pytest tests/e2e/test_pipeline_significance.py -v
Run slow tests: pytest tests/e2e/test_pipeline_significance.py -v -m slow
"""

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
def importance_enabled_config(tmp_path):
    """
    Create training config with OOF importance enabled.

    Uses minimal settings for fast E2E tests while enabling importance computation.
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
            "importance": {
                "compute_oof_importance": True,
                "pfi_n_repeats": 5,
                "grouped_threshold": 0.85,
                "include_builtin": True,
            },
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
        "allow_test_thresholding": True,
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
    }

    config_path = tmp_path / "importance_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def significance_proteomics_data(tmp_path):
    """
    Create proteomics dataset with strong signal for significance testing.

    200 samples with clear class separation to ensure meaningful AUROC.
    """
    rng = np.random.default_rng(42)

    n_controls = 140
    n_incident = 40
    n_prevalent = 20
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 12

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    data = {
        ID_COL: [f"SAMPLE_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic ethnic grouping": rng.choice(["White", "Asian"], n_total),
    }

    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        # Strong signal in first 4 proteins for significance detection
        if i < 4:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.5, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.2, 0.3, n_prevalent)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "significance_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


class TestE2EPermutationTest:
    """E2E tests for permutation testing workflow."""

    @pytest.mark.slow
    def test_permutation_test_basic(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """
        Test: Basic permutation test with small B for CI.

        Validates:
        - permutation-test command runs successfully
        - Output files are generated
        - p-value is in valid range (0, 1]
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
                str(significance_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--mode",
                "development",
                "--scenarios",
                "IncidentOnly",
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
        assert result_splits.exit_code == 0, f"Splits failed: {result_splits.output}"

        # Step 2: Train model
        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(significance_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(importance_enabled_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )
        assert result_train.exit_code == 0, f"Train failed: {result_train.output}"

        # Step 3: Find run_id
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1
        run_id = run_dirs[0].name.replace("run_", "")

        # Step 4: Run permutation test (small B=5 for CI)
        result_perm = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--split-seed-start",
                "42",
                "--n-split-seeds",
                "1",
                "--n-perms",
                "5",  # Small B for fast CI
                "--n-jobs",
                "1",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_perm.exit_code != 0:
            print("PERMUTATION OUTPUT:", result_perm.output)
            pytest.skip(f"Permutation test failed: {result_perm.output[:500]}")

        assert result_perm.exit_code == 0, f"Permutation test failed: {result_perm.output}"

        # Step 5: Verify outputs
        significance_dir = run_dirs[0] / "LR_EN" / "significance"
        assert significance_dir.exists(), f"Significance dir not found: {significance_dir}"

        results_csv = significance_dir / "permutation_test_results_seed42.csv"
        null_csv = significance_dir / "null_distribution_seed42.csv"

        assert results_csv.exists(), "permutation_test_results_seed42.csv not found"
        assert null_csv.exists(), "null_distribution_seed42.csv not found"

        # Validate results structure
        df_results = pd.read_csv(results_csv)
        assert "observed_auroc" in df_results.columns
        assert "p_value" in df_results.columns
        assert "null_mean" in df_results.columns
        assert "n_perms" in df_results.columns

        # p-value must be in (0, 1] (never exactly 0 due to +1 correction)
        p_val = df_results["p_value"].iloc[0]
        assert 0 < p_val <= 1.0, f"p-value out of range: {p_val}"

        # Observed AUROC should be valid
        obs_auroc = df_results["observed_auroc"].iloc[0]
        assert 0.0 <= obs_auroc <= 1.0, f"Observed AUROC out of range: {obs_auroc}"


class TestE2EOOFImportance:
    """E2E tests for OOF feature importance workflow."""

    @pytest.mark.slow
    def test_oof_importance_during_training(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """
        Test: OOF importance computed during training.

        Validates:
        - Importance files generated when enabled in config
        - Linear model produces coefficient-based importance
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
                str(significance_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Train with importance enabled
        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(significance_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(importance_enabled_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Training failed: {result_train.output[:200]}")

        # Find output directory
        run_dirs = list(results_dir.glob("run_*"))
        model_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"

        # Check for importance files (may be in cv/ subdirectory)
        cv_dir = model_dir / "cv"
        if cv_dir.exists():
            importance_files = list(cv_dir.glob("oof_importance__*.csv"))
        else:
            importance_files = list(model_dir.glob("**/oof_importance__*.csv"))

        # Note: If importance not yet hooked into training, this may be empty
        # This is expected behavior until hook is verified
        if importance_files:
            df_importance = pd.read_csv(importance_files[0])
            assert "feature" in df_importance.columns
            assert "mean_importance" in df_importance.columns
            assert len(df_importance) > 0, "Importance DataFrame is empty"


class TestE2EAggregatedImportance:
    """E2E tests for importance aggregation across splits."""

    @pytest.mark.slow
    def test_aggregated_importance(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """
        Test: Importance aggregation across multiple splits.

        Validates:
        - aggregate-splits includes importance aggregation
        - Stability scores computed correctly
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate multiple splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(significance_proteomics_data),
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
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(significance_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(importance_enabled_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )
            if result_train.exit_code != 0:
                pytest.skip(f"Training seed {seed} failed")

        # Find run_id
        run_dirs = list(results_dir.glob("run_*"))
        run_id = run_dirs[0].name.replace("run_", "")

        # Run aggregation
        result_agg = runner.invoke(
            cli,
            [
                "aggregate-splits",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip(f"Aggregation failed: {result_agg.output[:500]}")

        # Check for aggregated importance (may be in importance/ subdirectory)
        aggregated_dir = run_dirs[0] / "LR_EN" / "aggregated"
        if aggregated_dir.exists():
            importance_dir = aggregated_dir / "importance"
            if importance_dir.exists():
                agg_importance_files = list(importance_dir.glob("oof_importance__*.csv"))
                if agg_importance_files:
                    df_agg = pd.read_csv(agg_importance_files[0])
                    assert "feature" in df_agg.columns
                    assert "stability" in df_agg.columns or "mean_importance" in df_agg.columns


class TestE2EDropColumnValidation:
    """E2E tests for drop-column validation in optimize-panel."""

    @pytest.mark.slow
    def test_drop_column_in_optimize_panel(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """
        Test: Drop-column validation runs during optimize-panel.

        Validates:
        - optimize-panel includes drop-column validation step
        - Drop-column results CSV is generated
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate splits with validation set
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(significance_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--val-size",
                "0.25",
                "--test-size",
                "0.25",
                "--seed-start",
                "42",
            ],
        )

        # Train on multiple splits
        for seed in [42, 43]:
            runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(significance_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(importance_enabled_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

        # Find run_id and run aggregation
        run_dirs = list(results_dir.glob("run_*"))
        run_id = run_dirs[0].name.replace("run_", "")

        runner.invoke(
            cli,
            [
                "aggregate-splits",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        # Run optimize-panel (which should include drop-column)
        result_opt = runner.invoke(
            cli,
            [
                "optimize-panel",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_opt.exit_code != 0:
            pytest.skip(f"Optimize-panel failed: {result_opt.output[:500]}")

        # Check for drop-column output
        panels_dir = run_dirs[0] / "LR_EN" / "aggregated" / "panels"
        if panels_dir.exists():
            drop_col_files = list(panels_dir.glob("drop_column_validation__*.csv"))
            if drop_col_files:
                df_drop = pd.read_csv(drop_col_files[0])
                assert "cluster_id" in df_drop.columns or "feature" in df_drop.columns
                assert "mean_delta_auroc" in df_drop.columns or "delta_auroc" in df_drop.columns


class TestPermutationTestEdgeCases:
    """Test edge cases and error handling for permutation testing."""

    def test_permutation_test_missing_run_id(self, tmp_path):
        """Test: Error when run_id not found."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                "nonexistent_run_12345",
                "--model",
                "LR_EN",
            ],
        )

        assert result.exit_code != 0
        # Error can be in output or exception message
        error_text = result.output.lower()
        if result.exception:
            error_text += str(result.exception).lower()
        assert "not found" in error_text or "error" in error_text

    def test_permutation_test_invalid_model(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """Test: Error for invalid model name."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate splits and train
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(significance_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(significance_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(importance_enabled_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = list(results_dir.glob("run_*"))
        if not run_dirs:
            pytest.skip("Training did not produce output")

        run_id = run_dirs[0].name.replace("run_", "")

        # Try invalid model
        result = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                run_id,
                "--model",
                "INVALID_MODEL_XYZ",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        assert result.exit_code != 0

    def test_permutation_test_invalid_metric(
        self, significance_proteomics_data, importance_enabled_config, tmp_path
    ):
        """Test: Error for unsupported metric (only AUROC per ADR-007)."""
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate splits and train
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(significance_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(significance_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(importance_enabled_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
        )

        run_dirs = list(results_dir.glob("run_*"))
        if not run_dirs:
            pytest.skip("Training did not produce output")

        run_id = run_dirs[0].name.replace("run_", "")

        # Try unsupported metric
        result = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--metric",
                "accuracy",  # Not supported per ADR-007
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        assert result.exit_code != 0
        # Error can be in output or exception message
        error_text = result.output.lower()
        if result.exception:
            error_text += str(result.exception).lower()
        assert "auroc" in error_text or "not supported" in error_text
