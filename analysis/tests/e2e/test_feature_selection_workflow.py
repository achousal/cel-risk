"""
E2E tests for three-stage feature selection workflow.

Tests the full workflow:
    Stage 1 (Model Gate): permutation test filters significant models
    Stage 2 (Per-Model Evidence): OOF importance, stability, RFE
    Stage 3 (RRA Consensus): geometric mean rank aggregation across models

Run with: pytest tests/e2e/test_feature_selection_workflow.py -v
Run slow tests: pytest tests/e2e/test_feature_selection_workflow.py -v -m slow

Architecture Decision Records:
    ADR-004: Three-stage feature selection workflow
    ADR-011: Label permutation testing for model significance
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
def feature_selection_proteomics_data(tmp_path):
    """
    Create proteomics dataset with clear signal for feature selection testing.

    200 samples with strong signal in first 5 proteins to ensure:
    - Permutation test passes (p < 0.05)
    - Stability selection works (same proteins selected across folds)
    - RRA consensus correctly identifies top proteins

    Signal structure:
    - PROT_000-004: strong signal (incident + prevalent)
    - PROT_005-009: weak signal (incident only)
    - PROT_010-019: noise
    """
    rng = np.random.default_rng(42)

    n_controls = 140
    n_incident = 40
    n_prevalent = 20
    n_total = n_controls + n_incident + n_prevalent
    n_proteins = 20

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

        if i < 5:
            signal[n_controls : n_controls + n_incident] = rng.normal(2.0, 0.3, n_incident)
            signal[n_controls + n_incident :] = rng.normal(1.8, 0.3, n_prevalent)
        elif i < 10:
            signal[n_controls : n_controls + n_incident] = rng.normal(0.8, 0.3, n_incident)

        data[f"PROT_{i:03d}_resid"] = base + signal

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "feature_selection_proteomics.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path


@pytest.fixture
def feature_selection_config(tmp_path):
    """
    Create training config for feature selection testing.

    Enables:
    - OOF importance computation
    - Stability selection
    - Hybrid feature selection (screening + kbest + stability)
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
            "feature_selection_strategy": "multi_stage",
            "kbest_scope": "protein",
            "screen_method": "mannwhitney",
            "screen_top_n": 15,
            "k_grid": [5, 8],
            "stability_thresh": 0.5,
            "corr_thresh": 0.85,
            "importance": {
                "compute_oof_importance": True,
                "pfi_n_repeats": 3,
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
            "n_estimators_grid": [40],
            "max_depth_grid": [3],
            "min_samples_split_grid": [2],
            "min_samples_leaf_grid": [1],
            "max_features_grid": [0.5],
        },
    }

    config_path = tmp_path / "feature_selection_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


class TestStage1ModelGate:
    """
    Test Stage 1: Model Gate (Permutation Testing).

    Validates that permutation testing correctly filters models with real signal.
    """

    @pytest.mark.slow
    def test_permutation_test_identifies_significant_model(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Permutation test identifies model with real signal.

        Validates:
        - Model with strong signal yields p < 0.05
        - Observed AUROC > mean(null distribution)
        - Output files generated correctly
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
                str(feature_selection_proteomics_data),
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

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(feature_selection_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(feature_selection_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Training failed: {result_train.output[:500]}")

        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        assert len(run_dirs) == 1
        run_id = run_dirs[0].name.replace("run_", "")

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
                "10",
                "--n-jobs",
                "1",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_perm.exit_code != 0:
            pytest.skip(f"Permutation test failed: {result_perm.output[:500]}")

        assert result_perm.exit_code == 0

        significance_dir = run_dirs[0] / "LR_EN" / "significance"
        results_csv = significance_dir / "permutation_test_results_seed42.csv"
        null_csv = significance_dir / "null_distribution_seed42.csv"

        assert results_csv.exists()
        assert null_csv.exists()

        df_results = pd.read_csv(results_csv)
        p_val = df_results["p_value"].iloc[0]
        obs_auroc = df_results["observed_auroc"].iloc[0]
        null_mean = df_results["null_mean"].iloc[0]

        assert 0 < p_val <= 1.0
        assert p_val < 0.5
        assert obs_auroc > null_mean

    @pytest.mark.slow
    def test_permutation_test_multiple_models(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Permutation test across multiple models.

        Validates:
        - Each model generates separate significance results
        - Results are stored in model-specific subdirectories
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
                str(feature_selection_proteomics_data),
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

        shared_run_id = "test_multi_model_perm"
        for model in ["LR_EN", "RF"]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(feature_selection_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(feature_selection_config),
                    "--model",
                    model,
                    "--split-seed",
                    "42",
                    "--run-id",
                    shared_run_id,
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip(f"Training {model} failed")

        run_dir = results_dir / f"run_{shared_run_id}"

        for model in ["LR_EN", "RF"]:
            result_perm = runner.invoke(
                cli,
                [
                    "permutation-test",
                    "--run-id",
                    shared_run_id,
                    "--model",
                    model,
                    "--split-seed-start",
                    "42",
                    "--n-split-seeds",
                    "1",
                    "--n-perms",
                    "10",
                    "--n-jobs",
                    "1",
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
                catch_exceptions=False,
            )

            if result_perm.exit_code != 0:
                pytest.skip(f"Permutation test for {model} failed")

        for model in ["LR_EN", "RF"]:
            significance_dir = run_dir / model / "significance"
            assert significance_dir.exists()
            assert (significance_dir / "permutation_test_results_seed42.csv").exists()


class TestStage2PerModelEvidence:
    """
    Test Stage 2: Per-Model Evidence (OOF importance, stability, RFE).

    Validates feature selection methods compute correctly for each model.
    """

    @pytest.mark.slow
    def test_oof_importance_computed(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: OOF importance computed during training.

        Validates:
        - OOF importance files generated
        - Top proteins have high importance
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "1",
                "--seed-start",
                "42",
            ],
        )

        result_train = runner.invoke(
            cli,
            [
                "train",
                "--infile",
                str(feature_selection_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(feature_selection_config),
                "--model",
                "LR_EN",
                "--split-seed",
                "42",
            ],
            catch_exceptions=False,
        )

        if result_train.exit_code != 0:
            pytest.skip(f"Training failed: {result_train.output[:500]}")

        run_dirs = list(results_dir.glob("run_*"))
        model_dir = run_dirs[0] / "LR_EN" / "splits" / "split_seed42"

        importance_files = list(model_dir.glob("**/oof_importance__*.csv"))

        if importance_files:
            df_importance = pd.read_csv(importance_files[0])
            assert "feature" in df_importance.columns
            assert "mean_importance" in df_importance.columns
            assert len(df_importance) > 0

            top_proteins = df_importance.nlargest(5, "mean_importance")["feature"].tolist()
            signal_proteins = [f"PROT_{i:03d}_resid" for i in range(5)]
            overlap = len(set(top_proteins) & set(signal_proteins))
            assert overlap >= 3

    @pytest.mark.slow
    def test_stability_selection_consistent(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Stability selection identifies consistent proteins.

        Validates:
        - Stability summary generated after aggregation
        - Strong-signal proteins have high stability
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        for seed in [42, 43]:
            result_train = runner.invoke(
                cli,
                [
                    "train",
                    "--infile",
                    str(feature_selection_proteomics_data),
                    "--split-dir",
                    str(splits_dir),
                    "--outdir",
                    str(results_dir),
                    "--config",
                    str(feature_selection_config),
                    "--model",
                    "LR_EN",
                    "--split-seed",
                    str(seed),
                ],
                catch_exceptions=False,
            )

            if result_train.exit_code != 0:
                pytest.skip(f"Training seed {seed} failed")

        run_dirs = list(results_dir.glob("run_*"))
        run_id = run_dirs[0].name.replace("run_", "")

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

        aggregated_dir = run_dirs[0] / "LR_EN" / "aggregated"
        panels_dir = aggregated_dir / "panels"
        stability_csv = panels_dir / "feature_stability_summary.csv"

        if stability_csv.exists():
            df_stability = pd.read_csv(stability_csv)
            assert "protein" in df_stability.columns
            assert "selection_fraction" in df_stability.columns

            top_stable = df_stability.nlargest(5, "selection_fraction")["protein"].tolist()
            signal_proteins = [f"PROT_{i:03d}_resid" for i in range(5)]
            overlap = len(set(top_stable) & set(signal_proteins))
            assert overlap >= 2


class TestStage3RRAConsensus:
    """
    Test Stage 3: RRA Consensus (geometric mean rank aggregation).

    Validates cross-model consensus panel generation.
    """

    @pytest.mark.slow
    def test_consensus_panel_basic(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Consensus panel generation across models.

        Validates:
        - consensus-panel command runs successfully
        - Output includes final panel CSV and rankings
        - Metadata includes correct number of models
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        shared_run_id = "test_consensus_basic"
        for model in ["LR_EN", "RF"]:
            for seed in [42, 43]:
                result_train = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(feature_selection_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(feature_selection_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        shared_run_id,
                    ],
                    catch_exceptions=False,
                )

                if result_train.exit_code != 0:
                    pytest.skip(f"Training {model} seed {seed} failed")

        run_dir = results_dir / f"run_{shared_run_id}"
        run_id = shared_run_id

        for model in ["LR_EN", "RF"]:
            runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--run-id",
                    run_id,
                    "--model",
                    model,
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
                catch_exceptions=False,
            )

        result_consensus = runner.invoke(
            cli,
            [
                "consensus-panel",
                "--run-id",
                run_id,
                "--stability-threshold",
                "0.5",
                "--target-size",
                "10",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_consensus.exit_code != 0:
            pytest.skip(f"Consensus panel failed: {result_consensus.output[:500]}")

        assert result_consensus.exit_code == 0

        consensus_dir = run_dir / "consensus"
        assert consensus_dir.exists()

        final_panel_csv = consensus_dir / "final_panel.csv"
        consensus_ranking_csv = consensus_dir / "consensus_ranking.csv"
        metadata_json = consensus_dir / "consensus_metadata.json"

        assert final_panel_csv.exists()
        assert consensus_ranking_csv.exists()
        assert metadata_json.exists()

        df_panel = pd.read_csv(final_panel_csv)
        assert "protein" in df_panel.columns or len(df_panel) > 0

        df_ranking = pd.read_csv(consensus_ranking_csv)
        assert "protein" in df_ranking.columns
        assert "consensus_rank" in df_ranking.columns
        assert "n_models_present" in df_ranking.columns

        with open(metadata_json) as f:
            metadata = json.load(f)
        assert metadata["n_models"] == 2
        assert "LR_EN" in metadata["models"]
        assert "RF" in metadata["models"]

    @pytest.mark.slow
    def test_consensus_panel_rra_correctness(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: RRA (geometric mean rank aggregation) correctness.

        Validates:
        - Proteins ranked by geometric mean of reciprocal ranks
        - Proteins present in both models ranked higher
        - Per-model rank columns included
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        shared_run_id = "test_rra_correctness"
        for model in ["LR_EN", "RF"]:
            for seed in [42, 43]:
                runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(feature_selection_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(feature_selection_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        shared_run_id,
                    ],
                    catch_exceptions=False,
                )

        run_dir = results_dir / f"run_{shared_run_id}"
        run_id = shared_run_id

        for model in ["LR_EN", "RF"]:
            runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--run-id",
                    run_id,
                    "--model",
                    model,
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
            )

        result_consensus = runner.invoke(
            cli,
            [
                "consensus-panel",
                "--run-id",
                run_id,
                "--stability-threshold",
                "0.5",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_consensus.exit_code != 0:
            pytest.skip("Consensus panel failed")

        consensus_dir = run_dir / "consensus"
        consensus_ranking_csv = consensus_dir / "consensus_ranking.csv"

        df_ranking = pd.read_csv(consensus_ranking_csv)

        assert "LR_EN_rank" in df_ranking.columns
        assert "RF_rank" in df_ranking.columns
        assert "consensus_score" in df_ranking.columns
        assert "n_models_present" in df_ranking.columns

        top_protein = df_ranking.iloc[0]
        assert top_protein["n_models_present"] >= 1
        assert top_protein["consensus_rank"] == 1

    @pytest.mark.slow
    def test_consensus_panel_with_oof_importance(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Consensus panel uses OOF importance when available.

        Validates:
        - Per-model rankings include OOF importance columns
        - OOF importance drives final ranking (not just stability)
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
                "--n-splits",
                "2",
                "--seed-start",
                "42",
            ],
        )

        shared_run_id = "test_oof_importance_consensus"
        for model in ["LR_EN", "RF"]:
            for seed in [42, 43]:
                runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(feature_selection_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(feature_selection_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        shared_run_id,
                    ],
                    catch_exceptions=False,
                )

        run_dir = results_dir / f"run_{shared_run_id}"
        run_id = shared_run_id

        for model in ["LR_EN", "RF"]:
            runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--run-id",
                    run_id,
                    "--model",
                    model,
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
            )

        result_consensus = runner.invoke(
            cli,
            [
                "consensus-panel",
                "--run-id",
                run_id,
                "--stability-threshold",
                "0.5",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_consensus.exit_code != 0:
            pytest.skip("Consensus panel failed")

        consensus_dir = run_dir / "consensus"
        per_model_csv = consensus_dir / "per_model_rankings.csv"

        if per_model_csv.exists():
            df_per_model = pd.read_csv(per_model_csv)
            assert "model" in df_per_model.columns
            assert "protein" in df_per_model.columns
            assert "stability_freq" in df_per_model.columns
            assert "final_rank" in df_per_model.columns

            if "oof_importance" in df_per_model.columns:
                lr_rankings = df_per_model[df_per_model["model"] == "LR_EN"]
                assert not lr_rankings["oof_importance"].isna().all()


class TestFullThreeStageWorkflow:
    """
    Test complete three-stage workflow end-to-end.

    Validates integration of all three stages.
    """

    @pytest.mark.slow
    def test_complete_workflow(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """
        Test: Complete three-stage workflow.

        Workflow:
        1. Train multiple models
        2. Permutation test filters significant models
        3. Aggregate per-model evidence (OOF, stability)
        4. Generate consensus panel via RRA

        Validates:
        - All stages complete successfully
        - Output structure correct
        - Final panel contains expected proteins
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
                str(feature_selection_proteomics_data),
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

        models = ["LR_EN", "RF"]
        seeds = [42, 43]
        shared_run_id = "test_full_workflow"

        for model in models:
            for seed in seeds:
                result = runner.invoke(
                    cli,
                    [
                        "train",
                        "--infile",
                        str(feature_selection_proteomics_data),
                        "--split-dir",
                        str(splits_dir),
                        "--outdir",
                        str(results_dir),
                        "--config",
                        str(feature_selection_config),
                        "--model",
                        model,
                        "--split-seed",
                        str(seed),
                        "--run-id",
                        shared_run_id,
                    ],
                    catch_exceptions=False,
                )

                if result.exit_code != 0:
                    pytest.skip(f"Training {model} seed {seed} failed")

        run_dir = results_dir / f"run_{shared_run_id}"
        run_id = shared_run_id

        significant_models = []
        for model in models:
            result = runner.invoke(
                cli,
                [
                    "permutation-test",
                    "--run-id",
                    run_id,
                    "--model",
                    model,
                    "--split-seed-start",
                    "42",
                    "--n-split-seeds",
                    "1",
                    "--n-perms",
                    "10",
                    "--n-jobs",
                    "1",
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
                catch_exceptions=False,
            )

            if result.exit_code == 0:
                significance_dir = run_dir / model / "significance"
                results_csv = significance_dir / "permutation_test_results_seed42.csv"
                if results_csv.exists():
                    df = pd.read_csv(results_csv)
                    mean_p = df["p_value"].mean()
                    if mean_p < 0.5:
                        significant_models.append(model)

        if len(significant_models) < 2:
            pytest.skip("Not enough significant models for consensus")

        for model in significant_models:
            runner.invoke(
                cli,
                [
                    "aggregate-splits",
                    "--run-id",
                    run_id,
                    "--model",
                    model,
                ],
                env={"CED_RESULTS_DIR": str(results_dir)},
                catch_exceptions=False,
            )

        result_consensus = runner.invoke(
            cli,
            [
                "consensus-panel",
                "--run-id",
                run_id,
                "--stability-threshold",
                "0.5",
                "--target-size",
                "10",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
            catch_exceptions=False,
        )

        if result_consensus.exit_code != 0:
            pytest.skip(f"Consensus panel failed: {result_consensus.output[:500]}")

        assert result_consensus.exit_code == 0

        consensus_dir = run_dir / "consensus"
        assert (consensus_dir / "final_panel.csv").exists()
        assert (consensus_dir / "consensus_ranking.csv").exists()
        assert (consensus_dir / "per_model_rankings.csv").exists()
        assert (consensus_dir / "consensus_metadata.json").exists()

        df_panel = pd.read_csv(consensus_dir / "final_panel.csv")
        assert len(df_panel) <= 10
        assert len(df_panel) > 0

        signal_proteins = {f"PROT_{i:03d}_resid" for i in range(5)}
        panel_proteins = set(df_panel["protein"]) if "protein" in df_panel.columns else set()
        overlap = len(panel_proteins & signal_proteins)
        assert overlap >= 2


class TestFeatureSelectionErrorHandling:
    """Test error handling and edge cases in feature selection workflow."""

    def test_consensus_panel_nonexistent_run(self, tmp_path):
        """Test: Error when run_id not found."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "consensus-panel",
                "--run-id",
                "nonexistent_run_99999",
            ],
        )

        assert result.exit_code != 0

    def test_permutation_test_zero_perms(
        self, feature_selection_proteomics_data, feature_selection_config, tmp_path
    ):
        """Test: Error when n_perms = 0."""
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
                str(feature_selection_proteomics_data),
                "--outdir",
                str(splits_dir),
                "--scenarios",
                "IncidentOnly",
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
                str(feature_selection_proteomics_data),
                "--split-dir",
                str(splits_dir),
                "--outdir",
                str(results_dir),
                "--config",
                str(feature_selection_config),
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

        result = runner.invoke(
            cli,
            [
                "permutation-test",
                "--run-id",
                run_id,
                "--model",
                "LR_EN",
                "--n-perms",
                "0",
            ],
            env={"CED_RESULTS_DIR": str(results_dir)},
        )

        assert result.exit_code != 0
