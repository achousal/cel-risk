"""Tests for StackingEnsemble aggregation support.

Tests cover:
- ENSEMBLE predictions directory structure
- ENSEMBLE directory discovery
- ENSEMBLE predictions collection
- ENSEMBLE metrics collection
- Model comparison report generation
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def test_ensemble_predictions_directory_structure():
    """Test that ENSEMBLE predictions are saved to correct subdirectories for aggregation."""
    rng = np.random.default_rng(42)
    from ced_ml.cli.train_ensemble import run_train_ensemble

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir(parents=True)

        # Create mock base model results (with run_* directory and splits/)
        for model in ["LR_EN", "RF"]:
            preds_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            preds_dir.mkdir(parents=True)

            # Mock OOF predictions
            oof_df = pd.DataFrame(
                {
                    "idx": np.arange(100),
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                    "y_prob_repeat1": rng.uniform(0, 1, 100),
                }
            )
            oof_df.to_csv(preds_dir / f"train_oof__{model}.csv", index=False)

            # Mock val predictions
            val_df = pd.DataFrame(
                {
                    "idx": np.arange(50),
                    "y_true": rng.integers(0, 2, 50),
                    "y_prob": rng.uniform(0, 1, 50),
                }
            )
            val_df.to_csv(preds_dir / f"val_preds__{model}.csv", index=False)

            # Mock test predictions
            test_df = pd.DataFrame(
                {
                    "idx": np.arange(50, 100),
                    "y_true": rng.integers(0, 2, 50),
                    "y_prob": rng.uniform(0, 1, 50),
                }
            )
            test_df.to_csv(preds_dir / f"test_preds__{model}.csv", index=False)

        # Run ensemble training
        result = run_train_ensemble(
            results_dir=str(results_dir),
            base_models=["LR_EN", "RF"],
            split_seed=0,
            log_level=None,
        )

        assert result is not None

        # Check that ENSEMBLE predictions are in the expected directory
        ensemble_dir = results_dir / "run_test" / "ENSEMBLE" / "splits" / "split_seed0"

        # Val predictions should be in preds/ directory
        val_preds_path = ensemble_dir / "preds" / "val_preds__ENSEMBLE.csv"
        assert val_preds_path.exists(), f"Val predictions not found at {val_preds_path}"

        # Test predictions should be in preds/ directory
        test_preds_path = ensemble_dir / "preds" / "test_preds__ENSEMBLE.csv"
        assert test_preds_path.exists(), f"Test predictions not found at {test_preds_path}"

        # OOF predictions should be in preds/ directory
        oof_preds_path = ensemble_dir / "preds" / "train_oof__ENSEMBLE.csv"
        assert oof_preds_path.exists(), f"OOF predictions not found at {oof_preds_path}"

        # Verify that predictions can be loaded and have correct structure
        val_preds = pd.read_csv(val_preds_path)
        assert "y_prob" in val_preds.columns
        assert "y_true" in val_preds.columns

        test_preds = pd.read_csv(test_preds_path)
        assert "y_prob" in test_preds.columns
        assert "y_true" in test_preds.columns


class TestEnsembleAggregationSupport:
    """Test ensemble discovery and aggregation functions for aggregate_splits."""

    def test_discover_ensemble_dirs_empty(self, tmp_path):
        """Test that empty directory returns empty list."""
        from ced_ml.cli.discovery import discover_ensemble_dirs

        result = discover_ensemble_dirs(tmp_path)
        assert result == []

    def test_discover_ensemble_dirs_no_ensemble_folder(self, tmp_path):
        """Test that missing ENSEMBLE folder returns empty list."""
        from ced_ml.cli.discovery import discover_ensemble_dirs

        (tmp_path / "some_other_folder").mkdir()
        result = discover_ensemble_dirs(tmp_path)
        assert result == []

    def test_discover_ensemble_dirs_split_underscore_format(self, tmp_path):
        """Test discovery of split_seedX format directories."""
        from ced_ml.cli.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed1").mkdir()
        (ensemble_dir / "split_seed2").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3
        # Check they're sorted by seed
        assert result[0].name == "split_seed0"
        assert result[1].name == "split_seed1"
        assert result[2].name == "split_seed2"

    def test_discover_ensemble_dirs_split_seed_format(self, tmp_path):
        """Test discovery of split_seed{X} format directories."""
        from ced_ml.cli.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed5").mkdir()
        (ensemble_dir / "split_seed10").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3
        # Check they're sorted by seed
        assert result[0].name == "split_seed0"
        assert result[1].name == "split_seed5"
        assert result[2].name == "split_seed10"

    def test_discover_ensemble_dirs_mixed_formats(self, tmp_path):
        """Test discovery handles only split_seed format directories."""
        from ced_ml.cli.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed1").mkdir()
        (ensemble_dir / "split_seed2").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3

    def test_collect_ensemble_predictions_empty(self, tmp_path):
        """Test collecting predictions from empty directories."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")
        assert result.empty

    def test_collect_ensemble_predictions_with_data(self, tmp_path):
        """Test collecting predictions with actual data."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.discovery import discover_ensemble_dirs

        # Setup directory structure
        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        split_dir = ensemble_dir / "split_seed0"
        split_dir.mkdir()
        preds_dir = split_dir / "preds"
        preds_dir.mkdir(parents=True)

        # Create test predictions
        pd.DataFrame(
            {
                "idx": [0, 1, 2],
                "y_true": [0, 0, 1],
                "y_prob": [0.1, 0.2, 0.9],
            }
        ).to_csv(preds_dir / "test_preds__ENSEMBLE.csv", index=False)

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")

        assert len(result) == 3
        assert "model" in result.columns
        assert result["model"].iloc[0] == "ENSEMBLE"
        assert "split_seed" in result.columns
        assert result["split_seed"].iloc[0] == 0

    def test_collect_ensemble_predictions_multiple_splits(self, tmp_path):
        """Test collecting predictions from multiple splits."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.discovery import discover_ensemble_dirs

        # Setup directory structure with two splits
        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)

        for seed in [0, 1]:
            split_dir = ensemble_dir / f"split_seed{seed}"
            split_dir.mkdir()
            preds_dir = split_dir / "preds"
            preds_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "idx": [0, 1],
                    "y_true": [0, 1],
                    "y_prob": [0.2 + seed * 0.1, 0.8 + seed * 0.05],
                }
            ).to_csv(preds_dir / "test_preds__ENSEMBLE.csv", index=False)

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")

        assert len(result) == 4  # 2 samples * 2 splits
        assert result["split_seed"].nunique() == 2

    def test_generate_model_comparison_report(self, tmp_path):
        """Test model comparison report generation."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        test_metrics = {
            "LR_EN": {"AUROC": 0.85, "PR_AUC": 0.12, "Brier": 0.08},
            "RF": {"AUROC": 0.82, "PR_AUC": 0.10, "Brier": 0.09},
            "ENSEMBLE": {"AUROC": 0.87, "PR_AUC": 0.14, "Brier": 0.07},
        }
        val_metrics = {
            "LR_EN": {"AUROC": 0.84},
            "RF": {"AUROC": 0.81},
            "ENSEMBLE": {"AUROC": 0.86},
        }
        threshold_info = {
            "LR_EN": {"youden_threshold": 0.3},
            "RF": {"youden_threshold": 0.25},
            "ENSEMBLE": {"youden_threshold": 0.28},
        }

        result = generate_model_comparison_report(
            test_metrics, val_metrics, threshold_info, tmp_path
        )

        assert len(result) == 3
        assert "is_ensemble" in result.columns

        # Check ENSEMBLE is correctly flagged
        ensemble_row = result[result["model"] == "ENSEMBLE"]
        assert ensemble_row["is_ensemble"].iloc[0] == True  # noqa: E712

        # Check non-ensemble models are not flagged
        lr_row = result[result["model"] == "LR_EN"]
        assert lr_row["is_ensemble"].iloc[0] == False  # noqa: E712

        # Check report was saved
        report_path = tmp_path / "metrics" / "model_comparison.csv"
        assert report_path.exists()

    def test_generate_model_comparison_report_sorted_by_auroc(self, tmp_path):
        """Test that model comparison report is sorted by test AUROC."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        test_metrics = {
            "RF": {"AUROC": 0.82},
            "LR_EN": {"AUROC": 0.85},
            "ENSEMBLE": {"AUROC": 0.87},
        }

        result = generate_model_comparison_report(test_metrics, {}, {}, tmp_path)

        # Should be sorted by AUROC descending
        assert result.iloc[0]["model"] == "ENSEMBLE"
        assert result.iloc[1]["model"] == "LR_EN"
        assert result.iloc[2]["model"] == "RF"

    def test_generate_model_comparison_report_empty(self, tmp_path):
        """Test model comparison report with empty metrics."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        result = generate_model_comparison_report({}, {}, {}, tmp_path)
        assert result.empty
