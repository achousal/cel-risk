"""
Tests for optimize_panel auto-discovery by run_id.
"""

import pytest

from ced_ml.cli.optimize_panel import discover_models_by_run_id


@pytest.fixture
def mock_results_structure(tmp_path):
    """
    Create a mock results directory structure matching production layout:
        results/
            run_20260127_115115/
                LR_EN/
                    aggregated/
                        panels/feature_stability_summary.csv
                RF/
                    aggregated/
                        panels/feature_stability_summary.csv
                XGBoost/
                    (no aggregated dir)
                ENSEMBLE/
                    aggregated/
                        panels/feature_stability_summary.csv
            run_20260127_120000/
                LR_EN/
                    (no aggregated dir)
    """
    results_root = tmp_path / "results"

    # LR_EN with two runs, only first has aggregated
    lr_en_run1 = results_root / "run_20260127_115115" / "LR_EN" / "aggregated"
    lr_en_feature_reports = lr_en_run1 / "panels"
    lr_en_feature_reports.mkdir(parents=True)
    (lr_en_feature_reports / "feature_stability_summary.csv").write_text(
        "feature,stability\\nP1,0.8\\n"
    )

    lr_en_run2 = results_root / "run_20260127_120000" / "LR_EN"
    lr_en_run2.mkdir(parents=True)

    # RF with aggregated
    rf_run1 = results_root / "run_20260127_115115" / "RF" / "aggregated"
    rf_feature_reports = rf_run1 / "panels"
    rf_feature_reports.mkdir(parents=True)
    (rf_feature_reports / "feature_stability_summary.csv").write_text(
        "feature,stability\\nP2,0.75\\n"
    )

    # XGBoost without aggregated
    xgb_run1 = results_root / "run_20260127_115115" / "XGBoost"
    xgb_run1.mkdir(parents=True)

    # ENSEMBLE with aggregated
    ens_run1 = results_root / "run_20260127_115115" / "ENSEMBLE" / "aggregated"
    ens_feature_reports = ens_run1 / "panels"
    ens_feature_reports.mkdir(parents=True)
    (ens_feature_reports / "feature_stability_summary.csv").write_text(
        "feature,stability\\nP3,0.9\\n"
    )

    return results_root


def test_discover_all_models(mock_results_structure):
    """Test discovering all models with aggregated results for a run_id."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_dir=mock_results_structure,
    )

    # Should find LR_EN and RF, but exclude ENSEMBLE and XGBoost (no aggregated)
    assert len(discovered) == 2
    assert "LR_EN" in discovered
    assert "RF" in discovered
    assert "ENSEMBLE" not in discovered  # Excluded by default
    assert "XGBoost" not in discovered  # No aggregated dir

    # Verify paths are correct (should point to aggregated directories)
    assert (
        discovered["LR_EN"]
        == mock_results_structure / "run_20260127_115115" / "LR_EN" / "aggregated"
    )
    assert discovered["RF"] == mock_results_structure / "run_20260127_115115" / "RF" / "aggregated"


def test_discover_with_model_filter(mock_results_structure):
    """Test discovering models with a filter."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_dir=mock_results_structure,
        model_filter="LR_EN",
    )

    assert len(discovered) == 1
    assert "LR_EN" in discovered
    assert "RF" not in discovered
    assert "ENSEMBLE" not in discovered


def test_discover_nonexistent_run_id(mock_results_structure):
    """Test discovering with a run_id that doesn't exist."""
    discovered = discover_models_by_run_id(
        run_id="99999999_999999",
        results_dir=mock_results_structure,
    )

    assert len(discovered) == 0


def test_discover_run_id_without_aggregation(mock_results_structure):
    """Test discovering run_id where no models have aggregated results."""
    discovered = discover_models_by_run_id(
        run_id="20260127_120000",
        results_dir=mock_results_structure,
    )

    # Only LR_EN has this run_id, but it has no aggregated dir
    assert len(discovered) == 0


def test_discover_invalid_results_dir():
    """Test error handling for nonexistent results directory."""
    with pytest.raises(FileNotFoundError, match="Results directory not found"):
        discover_models_by_run_id(
            run_id="20260127_115115",
            results_dir="/nonexistent/path",
        )


def test_discover_empty_results_root(tmp_path):
    """Test discovering in an empty results directory."""
    empty_results = tmp_path / "results"
    empty_results.mkdir()

    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_dir=empty_results,
    )

    assert len(discovered) == 0


def test_discover_with_partial_structure(tmp_path):
    """Test discovering when some models have incomplete structure."""
    results_root = tmp_path / "results"

    # Model with run dir but no aggregated subdir
    (results_root / "run_20260127_115115" / "Model1").mkdir(parents=True)

    # Model with aggregated dir but missing required feature stability file
    (results_root / "run_20260127_115115" / "Model2" / "aggregated").mkdir(parents=True)

    # Model with complete aggregated structure
    model3_feature_reports = (
        results_root / "run_20260127_115115" / "Model3" / "aggregated" / "panels"
    )
    model3_feature_reports.mkdir(parents=True)
    (model3_feature_reports / "feature_stability_summary.csv").write_text(
        "feature,stability\\nP1,0.8\\n"
    )

    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_dir=results_root,
    )

    # Only Model3 should be discovered (has required feature stability file)
    assert len(discovered) == 1
    assert "Model3" in discovered
    assert "Model1" not in discovered  # No aggregated dir
    assert "Model2" not in discovered  # Missing feature stability file


def test_discover_case_sensitive_model_filter(mock_results_structure):
    """Test that model filter is case-sensitive."""
    discovered = discover_models_by_run_id(
        run_id="20260127_115115",
        results_dir=mock_results_structure,
        model_filter="lr_en",  # lowercase
    )

    # Should not match "LR_EN"
    assert len(discovered) == 0
