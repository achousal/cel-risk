"""Tests for consensus panel CLI auto-discovery and loading."""

import pandas as pd
import pytest

from ced_ml.cli.consensus_panel import (
    discover_models_with_aggregated_results,
    load_model_stability,
)


@pytest.fixture
def mock_aggregated_structure(tmp_path):
    """
    Create a mock results directory structure with aggregated stability results:
        results/
            run_20260127_115115/
                LR_EN/
                    aggregated/
                        panels/
                            feature_stability_summary.csv
                        optimize_panel/
                            feature_ranking_aggregated.csv
                RF/
                    aggregated/
                        panels/
                            feature_stability_summary.csv
                XGBoost/
                    (no aggregated dir - not run yet)
                ENSEMBLE/
                    aggregated/
                        panels/
                            feature_stability_summary.csv
    """
    results_root = tmp_path / "results"

    # LR_EN with full aggregated results including RFE
    lr_en_agg = results_root / "run_20260127_115115" / "LR_EN" / "aggregated"
    lr_en_reports = lr_en_agg / "panels"
    lr_en_reports.mkdir(parents=True)

    # Create stability CSV
    stability_df = pd.DataFrame(
        {
            "protein": ["P1", "P2", "P3", "P4", "P5"],
            "selection_fraction": [0.95, 0.90, 0.85, 0.75, 0.60],
            "n_splits_selected": [10, 9, 8, 7, 6],
        }
    )
    stability_df.to_csv(lr_en_reports / "feature_stability_summary.csv", index=False)

    # Create RFE ranking
    lr_en_rfe = lr_en_agg / "optimize_panel"
    lr_en_rfe.mkdir(parents=True)
    rfe_df = pd.DataFrame(
        {
            "protein": ["P5", "P4", "P3", "P2", "P1"],
            "elimination_order": [0, 1, 2, 3, 4],
        }
    )
    rfe_df.to_csv(lr_en_rfe / "feature_ranking_aggregated.csv", index=False)

    # RF with stability only (no RFE)
    rf_agg = results_root / "run_20260127_115115" / "RF" / "aggregated"
    rf_reports = rf_agg / "panels"
    rf_reports.mkdir(parents=True)

    # Create stability CSV with slightly different rankings
    stability_df_rf = pd.DataFrame(
        {
            "protein": ["P1", "P3", "P2", "P5", "P4"],
            "selection_fraction": [0.92, 0.88, 0.85, 0.80, 0.70],
            "n_splits_selected": [10, 9, 8, 7, 6],
        }
    )
    stability_df_rf.to_csv(rf_reports / "feature_stability_summary.csv", index=False)

    # XGBoost without aggregated
    xgb_dir = results_root / "run_20260127_115115" / "XGBoost"
    xgb_dir.mkdir(parents=True)

    # ENSEMBLE with stability (should be skipped by default)
    ens_agg = results_root / "run_20260127_115115" / "ENSEMBLE" / "aggregated"
    ens_reports = ens_agg / "panels"
    ens_reports.mkdir(parents=True)
    stability_df.to_csv(ens_reports / "feature_stability_summary.csv", index=False)

    return results_root


class TestDiscoverModelsWithAggregatedResults:
    """Tests for discover_models_with_aggregated_results function."""

    def test_discovers_all_base_models(self, mock_aggregated_structure):
        """Discovers all base models with aggregated stability results."""
        discovered = discover_models_with_aggregated_results(
            run_id="20260127_115115",
            results_dir=mock_aggregated_structure,
        )

        # Should find LR_EN and RF (have stability results)
        # Should NOT find XGBoost (no aggregated) or ENSEMBLE (skipped)
        assert len(discovered) == 2
        assert "LR_EN" in discovered
        assert "RF" in discovered
        assert "XGBoost" not in discovered
        assert "ENSEMBLE" not in discovered

    def test_skip_ensemble_default(self, mock_aggregated_structure):
        """ENSEMBLE is skipped by default."""
        discovered = discover_models_with_aggregated_results(
            run_id="20260127_115115",
            results_dir=mock_aggregated_structure,
            skip_ensemble=True,
        )

        assert "ENSEMBLE" not in discovered

    def test_include_ensemble_optional(self, mock_aggregated_structure):
        """ENSEMBLE can be included if requested."""
        discovered = discover_models_with_aggregated_results(
            run_id="20260127_115115",
            results_dir=mock_aggregated_structure,
            skip_ensemble=False,
        )

        assert "ENSEMBLE" in discovered

    def test_model_filter(self, mock_aggregated_structure):
        """Model filter limits discovery to specific model."""
        discovered = discover_models_with_aggregated_results(
            run_id="20260127_115115",
            results_dir=mock_aggregated_structure,
            model_filter="LR_EN",
        )

        assert len(discovered) == 1
        assert "LR_EN" in discovered

    def test_returns_aggregated_paths(self, mock_aggregated_structure):
        """Returns paths to aggregated directories."""
        discovered = discover_models_with_aggregated_results(
            run_id="20260127_115115",
            results_dir=mock_aggregated_structure,
        )

        expected_lr = mock_aggregated_structure / "run_20260127_115115" / "LR_EN" / "aggregated"
        assert discovered["LR_EN"] == expected_lr

    def test_nonexistent_run_id_raises(self, mock_aggregated_structure):
        """Nonexistent run_id raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No results found for run"):
            discover_models_with_aggregated_results(
                run_id="NONEXISTENT",
                results_dir=mock_aggregated_structure,
            )


class TestLoadModelStability:
    """Tests for load_model_stability function."""

    def test_loads_stability_data(self, mock_aggregated_structure):
        """Loads stability data from aggregated directory."""
        aggregated_dir = mock_aggregated_structure / "run_20260127_115115" / "LR_EN" / "aggregated"

        stability_df = load_model_stability(aggregated_dir)

        assert len(stability_df) == 5
        assert "protein" in stability_df.columns
        assert "selection_fraction" in stability_df.columns
        assert stability_df.iloc[0]["protein"] == "P1"
        assert stability_df.iloc[0]["selection_fraction"] == 0.95

    def test_stability_threshold_filtering(self, mock_aggregated_structure):
        """Filters proteins below stability threshold."""
        aggregated_dir = mock_aggregated_structure / "run_20260127_115115" / "LR_EN" / "aggregated"

        stability_df = load_model_stability(aggregated_dir, stability_threshold=0.80)

        # Only proteins with selection_fraction >= 0.80
        assert len(stability_df) == 3  # P1, P2, P3
        assert all(stability_df["selection_fraction"] >= 0.80)

    def test_missing_file_raises(self, tmp_path):
        """Missing stability file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Feature stability"):
            load_model_stability(tmp_path)
