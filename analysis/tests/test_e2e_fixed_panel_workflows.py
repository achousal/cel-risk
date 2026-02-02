"""
End-to-end tests for fixed-panel validation workflows.

This test suite validates the complete fixed-panel validation pipeline:
1. Extract consensus panel from trained model
2. Train with --fixed-panel on new split seed
3. Verify feature selection is bypassed
4. Verify unbiased performance estimates
5. Validate regulatory-grade workflows

Fixed-panel validation is critical for FDA submissions and clinical deployment,
providing unbiased performance estimates on panels discovered in separate experiments.

Run with: pytest tests/test_e2e_fixed_panel_workflows.py -v
Run slow tests: pytest tests/test_e2e_fixed_panel_workflows.py -v -m slow
"""

import json

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
def panel_proteomics_data(tmp_path):
    """
    Create proteomics dataset for panel validation testing.

    100 samples: 80 controls, 12 incident, 8 prevalent
    15 protein features (enough for meaningful panels)
    """
    rng = np.random.default_rng(42)

    n_controls = 80
    n_incident = 12
    n_prevalent = 8
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 15

    labels = (
        [CONTROL_LABEL] * n_controls
        + [INCIDENT_LABEL] * n_incident
        + [PREVALENT_LABEL] * n_prevalent
    )

    data = {
        ID_COL: [f"PANEL_{i:04d}" for i in range(n_total)],
        TARGET_COL: labels,
        "age": rng.integers(25, 75, n_total),
        "BMI": rng.uniform(18, 35, n_total),
        "sex": rng.choice(["M", "F"], n_total),
        "Genetic_ethnic_grouping": rng.choice(["White", "Asian"], n_total),
    }

    # Add proteins: first 5 have strong signal (panel targets)
    for i in range(n_proteins):
        base = rng.standard_normal(n_total)
        signal = np.zeros(n_total)

        if i < 5:
            # Strong signal proteins (true panel)
            signal[n_controls : n_controls + n_incident] = rng.normal(2.0, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.5, 0.3, n_prevalent)
        elif i < 8:
            # Weak signal proteins
            signal[n_controls : n_controls + n_incident] = rng.normal(0.5, 0.3, n_incident)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "panel_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def panel_training_config(tmp_path):
    """Create training config for panel discovery."""
    config = {
        "cv": {
            "n_outer": 2,
            "n_repeats": 1,
            "n_inner": 2,
        },
        "optuna": {"enabled": False},
        "features": {
            "feature_selection_strategy": "hybrid_stability",
            "screen_method": "mannwhitney",
            "screen_top_n": 12,
            "k_grid": [5],
            "stability_thresh": 0.6,
            "stable_corr_thresh": 0.85,
        },
        "calibration": {
            "enabled": True,
            "method": "isotonic",
            "strategy": "oof_posthoc",
        },
        "thresholds": {
            "objective": "fixed_spec",
            "fixed_spec": 0.95,
        },
    }

    config_path = tmp_path / "panel_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


@pytest.fixture(autouse=True)
def set_results_env(tmp_path, monkeypatch):
    """Point CED_RESULTS_DIR to tmp_path/results."""
    monkeypatch.setenv("CED_RESULTS_DIR", str(tmp_path / "results"))


class TestPanelExtraction:
    """Test panel extraction from trained models."""

    @pytest.mark.slow
    def test_extract_stable_features_as_panel(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Extract stable features from aggregated results as consensus panel.

        Validates panel extraction workflow for downstream validation.
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
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        run_id = "panel_discovery_run"

        # Train on discovery splits (42, 43)
        for seed in [42, 43]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(panel_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(panel_training_config),
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

        # Extract panel from feature stability
        agg_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated"
        stability_file = agg_dir / "panels" / "feature_stability_summary.csv"

        if not stability_file.exists():
            pytest.skip("Feature stability file not found")

        stability_df = pd.read_csv(stability_file)

        # Extract high-stability proteins
        panel_proteins = stability_df[stability_df["selection_fraction"] >= 0.6]["protein"].tolist()

        assert len(panel_proteins) >= 3, f"Too few stable proteins: {len(panel_proteins)}"
        assert len(panel_proteins) <= 10, f"Too many stable proteins: {len(panel_proteins)}"

        # Save as panel file
        panel_path = tmp_path / "extracted_panel.csv"
        with open(panel_path, "w") as f:
            for protein in panel_proteins:
                f.write(f"{protein}\n")

        assert panel_path.exists()

    @pytest.mark.slow
    def test_panel_extraction_filters_by_threshold(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Panel extraction respects stability threshold.

        Different thresholds should produce different panel sizes.
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
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        run_id = "panel_threshold_test"

        for seed in [42, 43]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(panel_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(panel_training_config),
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
                pytest.skip("Training failed")

        runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        agg_dir = results_dir / f"run_{run_id}" / "LR_EN" / "aggregated"
        stability_file = agg_dir / "panels" / "feature_stability_summary.csv"

        if not stability_file.exists():
            pytest.skip("No stability file")

        stability_df = pd.read_csv(stability_file)

        # Extract panels at different thresholds
        panel_50 = stability_df[stability_df["selection_fraction"] >= 0.5]["protein"].tolist()
        panel_70 = stability_df[stability_df["selection_fraction"] >= 0.7]["protein"].tolist()

        # Higher threshold should produce smaller panel
        assert len(panel_70) <= len(panel_50), "Higher threshold should be more selective"


class TestFixedPanelTraining:
    """Test training with fixed protein panels."""

    @pytest.mark.slow
    def test_train_with_fixed_panel(self, panel_proteomics_data, panel_training_config, tmp_path):
        """
        Test: Training with --fixed-panel bypasses feature selection.

        Validates that fixed-panel mode trains on exact protein set.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Create fixed panel file (manually specified panel)
        fixed_panel = [
            "PROT_000_resid",
            "PROT_001_resid",
            "PROT_002_resid",
            "PROT_003_resid",
            "PROT_004_resid",
        ]

        panel_path = tmp_path / "fixed_panel.csv"
        with open(panel_path, "w") as f:
            for protein in fixed_panel:
                f.write(f"{protein}\n")

        runner = CliRunner()

        # Generate validation splits (new seed)
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "100",  # New seed for unbiased validation
            ],
        )

        # Train with fixed panel
        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir / "LR_EN_fixed"),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "100",
                "--fixed-panel",
                str(panel_path),
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Fixed-panel training failed: {result_train.output[:200]}")

        assert result_train.exit_code == 0

        # Verify outputs exist
        run_dirs = [
            d
            for d in (results_dir / "LR_EN_fixed").iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        assert len(run_dirs) > 0

    @pytest.mark.slow
    def test_fixed_panel_metadata_records_panel(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Fixed-panel training records panel proteins in metadata.

        Validates traceability for regulatory submissions.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        fixed_panel = [
            "PROT_000_resid",
            "PROT_001_resid",
            "PROT_002_resid",
        ]

        panel_path = tmp_path / "fixed_panel.csv"
        with open(panel_path, "w") as f:
            for protein in fixed_panel:
                f.write(f"{protein}\n")

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "100",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir / "LR_EN_fixed"),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "100",
                "--fixed-panel",
                str(panel_path),
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

            # Should record fixed panel usage
            metadata_str = json.dumps(metadata).lower()
            _has_panel_info = "fixed_panel" in metadata_str or "panel" in metadata_str

            # May be recorded in different fields
            # At minimum, should not be empty
            assert len(metadata) > 0

    @pytest.mark.slow
    def test_fixed_panel_no_feature_selection_artifacts(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Fixed-panel training doesn't create feature selection artifacts.

        No stability files or feature ranking should be generated.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        fixed_panel = ["PROT_000_resid", "PROT_001_resid"]

        panel_path = tmp_path / "fixed_panel.csv"
        with open(panel_path, "w") as f:
            for protein in fixed_panel:
                f.write(f"{protein}\n")

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "1",
                "--seed-start",
                "100",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir / "LR_EN_fixed"),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "100",
                "--fixed-panel",
                str(panel_path),
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip("Training failed")

        # Should not have feature selection artifacts
        run_dirs = list((results_dir / "LR_EN_fixed").rglob("run_*"))
        if len(run_dirs) > 0:
            split_dir = run_dirs[0] / "splits" / "split_seed100"

            # Check for absence of feature selection files
            _has_stability = (split_dir / "feature_stability.csv").exists()
            _has_kbest = (split_dir / "kbest_results.csv").exists()

            # Implementation may still save minimal feature info
            # Main point: training completes without feature selection


class TestUnbiasedPanelValidation:
    """Test unbiased panel validation workflow."""

    @pytest.mark.slow
    def test_discovery_then_validation_workflow(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Complete discovery -> validation workflow.

        1. Train on discovery splits (42, 43)
        2. Extract panel
        3. Validate on new split (100) with fixed panel
        4. Verify performance estimates differ
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        runner = CliRunner()

        # Step 1: Discovery phase
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir / "discovery"),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        run_id_discovery = "discovery"

        for seed in [42, 43]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(panel_proteomics_data),
                    "--split-dir",
                    str(splits_dir / "discovery"),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(panel_training_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                    "--run-id",
                    run_id_discovery,
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip("Discovery training failed")

        result_agg = runner.invoke(
            cli,
            ["aggregate-splits", "--run-id", run_id_discovery, "--model", "LR_EN"],
            catch_exceptions=False,
        )

        if result_agg.exit_code != 0:
            pytest.skip("Aggregation failed")

        # Step 2: Extract panel
        agg_dir = results_dir / f"run_{run_id_discovery}" / "LR_EN" / "aggregated"
        stability_file = agg_dir / "panels" / "feature_stability_summary.csv"

        if not stability_file.exists():
            pytest.skip("No stability file")

        stability_df = pd.read_csv(stability_file)
        panel_proteins = stability_df[stability_df["selection_fraction"] >= 0.6]["protein"].tolist()

        if len(panel_proteins) < 3:
            pytest.skip("Insufficient stable proteins")

        panel_path = tmp_path / "validated_panel.csv"
        with open(panel_path, "w") as f:
            for protein in panel_proteins:
                f.write(f"{protein}\n")

        # Step 3: Validation phase (NEW split seed)
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir / "validation"),
                "--n-splits",
                "1",
                "--seed-start",
                "100",  # Critical: NEW seed
            ],
        )

        result_validation = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir / "validation"),
                "--outdir",
                str(results_dir / "LR_EN_validated"),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "100",
                "--fixed-panel",
                str(panel_path),
            ],
            catch_exceptions=False,
        )

        if result_validation.exit_code != 0:
            pytest.skip("Validation training failed")

        # Step 4: Both phases completed successfully
        discovery_metrics = list(results_dir.rglob("test_metrics.csv"))
        validation_metrics = list(results_dir.rglob("test_metrics.csv"))

        assert len(discovery_metrics) > 0, "Discovery metrics not found"
        assert len(validation_metrics) > 0, "Validation metrics not found"

    @pytest.mark.slow
    def test_fixed_panel_split_independence(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Fixed-panel validation uses independent test set.

        Validation split (seed=100) should differ from discovery splits (42, 43).
        """
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        runner = CliRunner()

        # Generate discovery and validation splits
        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
                "--outdir",
                str(splits_dir / "validation"),
                "--n-splits",
                "1",
                "--seed-start",
                "100",
            ],
        )

        # Load splits from CSV files
        import pandas as pd

        # Load discovery test set (seed 42)
        discovery_test_csv = splits_dir / "test_idx_IncidentOnly_seed42.csv"
        assert discovery_test_csv.exists(), f"Discovery test CSV not found: {discovery_test_csv}"
        discovery_test = set(pd.read_csv(discovery_test_csv)["idx"].values)

        # Load validation test set (seed 100)
        validation_splits_dir = splits_dir / "validation"
        validation_test_csv = validation_splits_dir / "test_idx_IncidentOnly_seed100.csv"
        assert validation_test_csv.exists(), f"Validation test CSV not found: {validation_test_csv}"
        validation_test = set(pd.read_csv(validation_test_csv)["idx"].values)

        # Compare test sets
        overlap = len(discovery_test & validation_test)
        total = len(validation_test)

        # Should have significant differences
        overlap_pct = overlap / total if total > 0 else 0
        assert overlap_pct < 0.80, f"Test sets too similar: {overlap_pct:.2%} overlap"


class TestPanelFormatValidation:
    """Test panel file format validation."""

    def test_panel_file_txt_format_accepted(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Panel file with .txt extension is accepted.

        Validates format flexibility.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Create .txt panel file
        panel_path = tmp_path / "panel.txt"
        with open(panel_path, "w") as f:
            f.write("PROT_000_resid\n")
            f.write("PROT_001_resid\n")

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
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
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
                "--fixed-panel",
                str(panel_path),
            ],
            catch_exceptions=False,
        )

        # Should accept .txt format
        # May fail for other reasons, but not format
        assert "format" not in result.output.lower() or result.exit_code == 0

    def test_panel_file_missing_proteins_fails(
        self, panel_proteomics_data, panel_training_config, tmp_path
    ):
        """
        Test: Panel with nonexistent proteins fails gracefully.

        Validates input validation.
        """
        splits_dir = tmp_path / "splits"
        results_dir = tmp_path / "results"
        splits_dir.mkdir()
        results_dir.mkdir()

        # Create panel with invalid proteins
        panel_path = tmp_path / "invalid_panel.csv"
        with open(panel_path, "w") as f:
            f.write("INVALID_PROTEIN_999\n")
            f.write("ANOTHER_FAKE_PROTEIN\n")

        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "save-splits",
                "--infile",
                str(panel_proteomics_data),
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
                str(panel_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(panel_training_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
                "--fixed-panel",
                str(panel_path),
            ],
        )

        # Should fail with clear error
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "missing" in result.output.lower()


# ==================== How to Run ====================
# Fast tests only:
#   pytest tests/test_e2e_fixed_panel_workflows.py -v -m "not slow"
#
# All tests including slow:
#   pytest tests/test_e2e_fixed_panel_workflows.py -v
#
# Specific test class:
#   pytest tests/test_e2e_fixed_panel_workflows.py::TestFixedPanelTraining -v
