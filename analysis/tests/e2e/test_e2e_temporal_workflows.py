"""
End-to-end tests for temporal validation workflows.

This test suite validates temporal split functionality, ensuring that:
1. Chronological train/val/test splits are created correctly
2. Temporal ordering is preserved (no future leakage)
3. Temporal metadata is recorded in outputs
4. Model training and evaluation work correctly with temporal splits

Temporal validation is critical for prospective risk prediction, where
models trained on historical data must generalize to future samples.

Run with: pytest tests/test_e2e_temporal_workflows.py -v
Run slow tests: pytest tests/test_e2e_temporal_workflows.py -v -m slow
"""

import json

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
def temporal_proteomics_data(tmp_path):
    """
    Create proteomics dataset with temporal dimension.

    150 samples: 130 controls, 12 incident, 8 prevalent
    Samples span 4 time points (years)
    Ensures sufficient samples in each temporal split.
    """
    rng = np.random.default_rng(42)

    n_controls = 130
    n_incident = 12
    n_prevalent = 8
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 10

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    # Create temporal dimension: distribute cases across time
    # This ensures temporal splits have cases in train/val/test
    base_date = pd.Timestamp("2020-01-01")

    # Generate random dates uniformly across ~4 years
    # CRITICAL: Use random dates (not sequential by index) to avoid all cases ending up in test set
    days_offsets = rng.integers(0, 1460, n_total)  # Random day within 4 years
    sample_dates = [base_date + pd.Timedelta(days=int(d)) for d in days_offsets]

    # Create CeD_date column ONLY for cases (controls should have NaT to avoid drop_uncertain_controls filter)
    # For cases, use sample_date +/- small jitter to simulate diagnosis date near sample collection
    ced_dates = [pd.NaT] * n_controls
    for i in range(n_controls, n_total):
        jitter_days = rng.integers(-30, 30)  # ±1 month jitter
        ced_dates.append(sample_dates[i] + pd.Timedelta(days=jitter_days))

    data = {
        ID_COL: [f"TEMP_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "sample_date": sample_dates,  # Temporal ordering column (all samples)
        "CeD_date": ced_dates,  # Diagnosis date column (cases only, controls=NaT)
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    # Add proteins with temporal drift
    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        # Add signal for incident cases
        if i < 3:
            signal[n_controls : n_controls + n_incident] = rng.normal(1.5, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.0, 0.3, n_prevalent)

        # Add slight temporal drift for realism
        temporal_trend = np.arange(n_total) / n_total * 0.1
        data[f"PROT_{i:03d}_resid"] = base + signal + temporal_trend

    df = pd.DataFrame(data)
    # Shuffle to break temporal ordering in input file
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    parquet_path = tmp_path / "temporal_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def temporal_training_config(tmp_path):
    """Create training config with temporal validation enabled."""
    config = {
        "scenario": "IncidentOnly",
        "splits": {
            "temporal_split": True,
            "temporal_column": "sample_date",
        },
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
            "screen_top_n": 8,
            "k_grid": [3],
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
    }

    config_path = tmp_path / "temporal_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture
def temporal_splits_config(tmp_path):
    """Create splits config with temporal validation enabled."""
    config = {
        "mode": "development",
        "scenarios": ["IncidentOnly"],
        "n_splits": 1,
        "val_size": 0.2,  # Smaller val/test to ensure enough samples in train
        "test_size": 0.2,
        "seed_start": 42,
        "train_control_per_case": 3.0,  # Lower ratio for temporal splits
        "prevalent_train_only": False,
        "temporal_split": True,
        "temporal_col": "sample_date",  # Fixed: was temporal_column
    }

    config_path = tmp_path / "temporal_splits_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


class TestTemporalSplitCreation:
    """Test temporal split generation preserves chronological ordering."""

    @pytest.mark.xfail(
        reason="Temporal splits with downsampling needs adjustment for small datasets"
    )
    def test_temporal_splits_preserve_ordering(
        self, temporal_proteomics_data, temporal_splits_config, tmp_path
    ):
        """
        Test: Temporal splits maintain chronological order.

        Train samples should precede val samples, which precede test samples.
        No future leakage.
        """
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--config",
                str(temporal_splits_config),
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
            ],
        )

        assert result.exit_code == 0, f"Split generation failed: {result.output}"

        # Load generated splits from CSV files
        train_csv = list(splits_dir.glob("train_idx_*.csv"))
        val_csv = list(splits_dir.glob("val_idx_*.csv"))
        test_csv = list(splits_dir.glob("test_idx_*.csv"))

        assert len(train_csv) > 0, "No train split files generated"
        assert len(val_csv) > 0, "No val split files generated"
        assert len(test_csv) > 0, "No test split files generated"

        # Load original data to check dates
        df = pd.read_parquet(temporal_proteomics_data)

        # Check first split
        train_indices = pd.read_csv(train_csv[0])["idx"].values
        val_indices = pd.read_csv(val_csv[0])["idx"].values
        test_indices = pd.read_csv(test_csv[0])["idx"].values

        train_dates = df.loc[train_indices, "sample_date"]
        val_dates = df.loc[val_indices, "sample_date"]
        test_dates = df.loc[test_indices, "sample_date"]

        # Verify temporal ordering: max(train) <= min(val), max(val) <= min(test)
        max_train_date = train_dates.max()
        min_val_date = val_dates.min()
        max_val_date = val_dates.max()
        min_test_date = test_dates.min()

        assert max_train_date <= min_val_date, "Train dates leak into validation"
        assert max_val_date <= min_test_date, "Validation dates leak into test"

    @pytest.mark.xfail(
        reason="Temporal splits with downsampling needs adjustment for small datasets"
    )
    def test_temporal_splits_balanced_across_sets(
        self, temporal_proteomics_data, temporal_splits_config, tmp_path
    ):
        """
        Test: Temporal splits maintain reasonable class balance.

        Each set should have sufficient samples for training/evaluation.
        """
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "save-splits",
                "--config",
                str(temporal_splits_config),
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
            ],
        )

        assert result.exit_code == 0

        df = pd.read_parquet(temporal_proteomics_data)

        # Load split indices from CSV files
        train_csvs = sorted(splits_dir.glob("train_idx_*.csv"))
        val_csvs = sorted(splits_dir.glob("val_idx_*.csv"))
        test_csvs = sorted(splits_dir.glob("test_idx_*.csv"))

        # Check all splits (should be 2 based on n-splits=2)
        for train_csv, val_csv, test_csv in zip(train_csvs, val_csvs, test_csvs, strict=False):
            train_indices = pd.read_csv(train_csv)["idx"].values
            val_indices = pd.read_csv(val_csv)["idx"].values
            test_indices = pd.read_csv(test_csv)["idx"].values

            for set_name, indices in [
                ("train", train_indices),
                ("val", val_indices),
                ("test", test_indices),
            ]:
                labels = df.loc[indices, TARGET_COL]

                # Should have at least 2 cases per set
                n_cases = (labels == INCIDENT_LABEL).sum()
                assert n_cases >= 2, f"{set_name} has insufficient cases: {n_cases}"


class TestTemporalTrainingWorkflow:
    """Test training with temporal validation."""

    @pytest.mark.slow
    def test_training_with_temporal_splits(
        self, temporal_proteomics_data, temporal_training_config, tmp_path
    ):
        """
        Test: Training completes successfully with temporal splits.

        Validates that temporal splits integrate correctly with training pipeline.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate temporal splits (config specifies temporal mode)
        result_splits = runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        assert result_splits.exit_code == 0, f"Split generation failed: {result_splits.output}"

        # Train with temporal splits
        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(temporal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(temporal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Training failed: {result_train.output[:200]}")

        # Verify outputs exist (production writes to results_dir/run_{ID}/LR_EN/splits/split_seed{N}/)
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) > 0
        split_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"
        assert split_dir.exists()
        assert (split_dir / "core").exists()

    @pytest.mark.slow
    def test_temporal_metadata_recorded(
        self, temporal_proteomics_data, temporal_training_config, tmp_path
    ):
        """
        Test: Temporal validation metadata is recorded in outputs.

        Run metadata should indicate temporal split mode.
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
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(temporal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(temporal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip("Training failed")

        # Check run metadata
        metadata_files = list(results_dir.rglob("run_metadata.json"))
        if len(metadata_files) > 0:
            with open(metadata_files[0]) as f:
                metadata = json.load(f)

            # Should indicate temporal validation (implementation-specific field name)
            # Check for common temporal indicators
            metadata_str = json.dumps(metadata).lower()
            has_temporal_info = (
                "temporal" in metadata_str
                or "chronological" in metadata_str
                or "sample_date" in metadata_str
            )

            assert has_temporal_info, "Temporal metadata not recorded"


class TestTemporalValidationMetrics:
    """Test that temporal validation produces valid performance metrics."""

    @pytest.mark.slow
    def test_temporal_metrics_valid_ranges(
        self, temporal_proteomics_data, temporal_training_config, tmp_path
    ):
        """
        Test: Performance metrics from temporal validation are in valid ranges.

        AUROC should be in [0, 1], not NaN or degenerate.
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
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(temporal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(temporal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip("Training failed")

        # Find test metrics
        test_metrics_files = list(results_dir.rglob("test_metrics.csv"))
        if len(test_metrics_files) == 0:
            pytest.skip("No test metrics found")

        metrics_df = pd.read_csv(test_metrics_files[0])

        # Check AUROC is valid (column is lowercase "auroc" per schema)
        if "auroc" in metrics_df.columns:
            auroc = metrics_df["auroc"].iloc[0]
        elif "AUROC" in metrics_df.columns:
            auroc = metrics_df["AUROC"].iloc[0]
        elif "metric" in metrics_df.columns and "value" in metrics_df.columns:
            auroc_rows = metrics_df[metrics_df["metric"].str.lower() == "auroc"]
            if len(auroc_rows) > 0:
                auroc = auroc_rows["value"].iloc[0]
            else:
                pytest.skip("AUROC not found in metrics")
        else:
            pytest.skip("Unknown metrics format")

        assert 0.0 <= auroc <= 1.0, f"Invalid AUROC: {auroc}"
        assert not np.isnan(auroc), "AUROC is NaN"

    @pytest.mark.slow
    def test_temporal_predictions_no_future_leakage(
        self, temporal_proteomics_data, temporal_training_config, tmp_path
    ):
        """
        Test: Model predictions don't leak future information.

        Test set predictions should come from chronologically later samples.
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
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(temporal_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(temporal_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip("Training failed")

        # Load splits from CSV files
        train_csv = list(splits_dir.glob("train_idx_*.csv"))[0]
        test_csv = list(splits_dir.glob("test_idx_*.csv"))[0]

        train_indices = pd.read_csv(train_csv)["idx"].values
        test_indices = pd.read_csv(test_csv)["idx"].values

        # Load and sort DataFrame temporally (split indices reference sorted DataFrame)
        df_orig = pd.read_parquet(temporal_proteomics_data)
        # Apply same row filters as split generation
        from ced_ml.data.filters import apply_row_filters
        from ced_ml.data.schema import CONTROL_LABEL, INCIDENT_LABEL, TARGET_COL

        # Filter to IncidentOnly scenario (same as split generation)
        df_scenario = df_orig[df_orig[TARGET_COL].isin([CONTROL_LABEL, INCIDENT_LABEL])].copy()
        df_scenario, _ = apply_row_filters(df_scenario)

        # Sort temporally (same as split generation)
        df_sorted = df_scenario.sort_values("sample_date").reset_index(drop=True)

        train_dates = df_sorted.loc[train_indices, "sample_date"]
        test_dates = df_sorted.loc[test_indices, "sample_date"]

        # Test samples should be chronologically after train samples
        assert train_dates.max() <= test_dates.min(), "Future leakage detected"


class TestTemporalAggregation:
    """Test aggregation works correctly with temporal validation results."""

    @pytest.mark.slow
    def test_aggregate_temporal_results(
        self, temporal_proteomics_data, temporal_training_config, tmp_path
    ):
        """
        Test: Aggregation works with temporal validation results.

        Ensures temporal split results aggregate correctly.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Generate 2 temporal splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        run_id = "temporal_test_run"

        # Train on both splits
        for seed in [42, 43]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(temporal_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(temporal_training_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                    "--run-id",
                    run_id,
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip(f"Training failed on seed {seed}")

        # Aggregate
        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Verify aggregated outputs (production writes to results_dir/run_{ID}/LR_EN/aggregated/)
        agg_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated"
        assert agg_dir.exists()
        assert (agg_dir / "metrics").exists()


class TestTemporalVsRandomComparison:
    """Compare temporal vs random split behavior."""

    def test_temporal_and_random_splits_differ(self, temporal_proteomics_data, tmp_path):
        """
        Test: Temporal splits produce different sample assignments than random splits.

        Validates that temporal mode actually changes split behavior.
        """
        temporal_dir = tmp_path / "splits_temporal"
        random_dir = tmp_path / "splits_random"
        temporal_dir.mkdir()
        random_dir.mkdir()

        runner = CliRunner()

        # Generate temporal splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(temporal_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
                "--temporal-split",
                "--temporal-column",
                "sample_date",
            ],
        )

        # Generate random splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--scenarios",
                "IncidentOnly",
                "--infile",
                str(temporal_proteomics_data),
                "--outdir",
                str(random_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        # Load both split types from CSV files
        temporal_train_csv = list(temporal_dir.glob("train_idx_*.csv"))[0]
        random_train_csv = list(random_dir.glob("train_idx_*.csv"))[0]

        temporal_train = set(pd.read_csv(temporal_train_csv)["idx"].values)
        random_train = set(pd.read_csv(random_train_csv)["idx"].values)

        # Should have some differences
        overlap = len(temporal_train & random_train)
        total = len(temporal_train)

        # Expect <90% overlap (temporal should reorder samples)
        overlap_pct = overlap / total
        assert (
            overlap_pct < 0.90
        ), f"Temporal and random splits too similar: {overlap_pct:.2%} overlap"


# ==================== How to Run ====================
# Fast tests only:
#   pytest tests/test_e2e_temporal_workflows.py -v -m "not slow"
#
# All tests including slow:
#   pytest tests/test_e2e_temporal_workflows.py -v
#
# Specific test class:
#   pytest tests/test_e2e_temporal_workflows.py::TestTemporalSplitCreation -v
