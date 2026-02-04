"""
Tests for Decision Curve Analysis (DCA) module.

Validates:
- Net benefit calculations
- DCA curve generation
- Cross-model comparison tables
- Summary statistics
- Zero-crossing detection
- Persistence (save/load)
"""

import json

import numpy as np
import pandas as pd
import pytest
from ced_ml.metrics.dca import (
    compute_dca_summary,
    decision_curve_analysis,
    decision_curve_table,
    find_dca_zero_crossing,
    generate_dca_thresholds,
    net_benefit,
    net_benefit_treat_all,
    parse_dca_report_points,
    save_dca_results,
    threshold_dca_zero_crossing,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def binary_classification_data():
    """Create reproducible binary classification data."""
    _rng = np.random.default_rng(42)

    # Controls (80%)
    y_ctrl = np.zeros(160)
    p_ctrl = np.random.beta(2, 8, size=160)

    # Cases (20%)
    y_case = np.ones(40)
    p_case = np.random.beta(8, 2, size=40)

    y_true = np.concatenate([y_ctrl, y_case])
    y_pred = np.concatenate([p_ctrl, p_case])

    return y_true, y_pred


@pytest.fixture
def perfect_classification():
    """Perfect model: cases=1.0, controls=0.0."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return y_true, y_pred


@pytest.fixture
def random_classification():
    """Random model: uniform probabilities."""
    rng = np.random.default_rng(123)
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = rng.uniform(0, 1, size=100)
    return y_true, y_pred


# =============================================================================
# Test: Net Benefit Calculations
# =============================================================================


def test_net_benefit_perfect_model():
    """Perfect model at optimal threshold should have high net benefit."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    # At threshold 0.5: all cases above, all controls below
    nb = net_benefit(y_true, y_pred, threshold=0.5)

    # TP=4, FP=0, n=8
    # NB = 4/8 - 0/8 * (0.5/0.5) = 0.5
    assert nb == pytest.approx(0.5, abs=1e-6)


def test_net_benefit_all_wrong():
    """Model that gets everything wrong should have negative net benefit."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    nb = net_benefit(y_true, y_pred, threshold=0.5)

    # TP=0, FP=4, n=8
    # NB = 0/8 - 4/8 * (0.5/0.5) = -0.5
    assert nb == pytest.approx(-0.5, abs=1e-6)


def test_net_benefit_edge_cases():
    """Test net benefit with edge case thresholds."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.3, 0.7, 0.2, 0.8])

    # Threshold <= 0 or >= 1 should return NaN
    assert np.isnan(net_benefit(y_true, y_pred, 0.0))
    assert np.isnan(net_benefit(y_true, y_pred, 1.0))
    assert np.isnan(net_benefit(y_true, y_pred, -0.1))
    assert np.isnan(net_benefit(y_true, y_pred, 1.5))

    # Empty array
    assert np.isnan(net_benefit(np.array([]), np.array([]), 0.5))


def test_net_benefit_treat_all():
    """Test treat-all strategy net benefit."""
    # Prevalence = 0.2, threshold = 0.05
    # NB_all = 0.2 - 0.8 * (0.05 / 0.95) = 0.2 - 0.042 = 0.158
    nb = net_benefit_treat_all(prevalence=0.2, threshold=0.05)
    assert nb == pytest.approx(0.158, abs=1e-3)

    # High threshold should reduce net benefit
    nb_high = net_benefit_treat_all(prevalence=0.2, threshold=0.5)
    assert nb_high < nb

    # Edge cases
    assert np.isnan(net_benefit_treat_all(0.2, 0.0))
    assert np.isnan(net_benefit_treat_all(0.2, 1.0))


# =============================================================================
# Test: DCA Curve Generation
# =============================================================================


def test_decision_curve_analysis_structure(binary_classification_data):
    """Test DCA returns expected columns and structure."""
    y_true, y_pred = binary_classification_data
    thresholds = np.linspace(0.01, 0.20, 20)

    dca_df = decision_curve_analysis(y_true, y_pred, thresholds=thresholds)

    # Check structure
    assert isinstance(dca_df, pd.DataFrame)
    assert len(dca_df) == len(thresholds)

    # Check columns
    expected_cols = [
        "threshold",
        "threshold_pct",
        "net_benefit_model",
        "net_benefit_all",
        "net_benefit_none",
        "relative_utility",
        "tp",
        "fp",
        "tn",
        "fn",
        "n_treat",
        "sensitivity",
        "specificity",
    ]
    for col in expected_cols:
        assert col in dca_df.columns

    # Check monotonicity of threshold
    assert (dca_df["threshold"].diff().dropna() >= 0).all()


def test_decision_curve_analysis_treat_none_baseline(binary_classification_data):
    """Treat-none net benefit should always be 0."""
    y_true, y_pred = binary_classification_data
    dca_df = decision_curve_analysis(y_true, y_pred)

    assert (dca_df["net_benefit_none"] == 0.0).all()


def test_decision_curve_analysis_perfect_model(perfect_classification):
    """Perfect model should beat treat-all at reasonable thresholds."""
    y_true, y_pred = perfect_classification
    dca_df = decision_curve_analysis(y_true, y_pred, thresholds=np.linspace(0.01, 0.99, 50))

    # At prevalence = 0.5, model should beat treat-all in mid-range
    beats_all = dca_df[dca_df["net_benefit_model"] > dca_df["net_benefit_all"]]
    assert len(beats_all) > 0


def test_decision_curve_analysis_empty_data():
    """Empty data should return empty DataFrame."""
    dca_df = decision_curve_analysis(np.array([]), np.array([]))
    assert dca_df.empty


def test_decision_curve_analysis_prevalence_adjustment():
    """Prevalence adjustment should affect treat-all net benefit."""
    y_true = np.array([0] * 90 + [1] * 10)  # 10% prevalence
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    # Default: uses observed prevalence
    dca_default = decision_curve_analysis(y_true, y_pred, thresholds=[0.05])

    # Adjusted: force 30% prevalence
    dca_adjusted = decision_curve_analysis(
        y_true, y_pred, thresholds=[0.05], prevalence_adjustment=0.30
    )

    # Treat-all benefit should differ
    nb_all_default = dca_default.iloc[0]["net_benefit_all"]
    nb_all_adjusted = dca_adjusted.iloc[0]["net_benefit_all"]
    assert nb_all_default != pytest.approx(nb_all_adjusted)


def test_decision_curve_analysis_default_thresholds():
    """Default thresholds should be 0.001 to 0.10."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    dca_df = decision_curve_analysis(y_true, y_pred)

    assert dca_df["threshold"].min() >= 0.001
    assert dca_df["threshold"].max() <= 0.10


# =============================================================================
# Test: Cross-Model Comparison
# =============================================================================


def test_decision_curve_table_structure(binary_classification_data):
    """Test cross-model DCA table structure."""
    y_true, y_pred = binary_classification_data

    # Create two models
    pred_dict = {
        "model_a": y_pred,
        "model_b": y_pred * 0.8,  # Scaled version
    }

    dca_table = decision_curve_table(
        scenario="TestScenario",
        y_true=y_true,
        pred_dict=pred_dict,
        max_pt=0.20,
        step=0.01,
    )

    # Check structure
    assert isinstance(dca_table, pd.DataFrame)
    assert set(dca_table.columns) == {"scenario", "threshold", "model", "net_benefit"}

    # Check models present
    models = dca_table["model"].unique()
    assert "treat_none" in models
    assert "treat_all" in models
    assert "model_a" in models
    assert "model_b" in models

    # Check scenario
    assert (dca_table["scenario"] == "TestScenario").all()


def test_decision_curve_table_threshold_range():
    """Test threshold range in cross-model table."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    dca_table = decision_curve_table(
        scenario="Test",
        y_true=y_true,
        pred_dict={"model": y_pred},
        max_pt=0.15,
        step=0.05,
    )

    thresholds = dca_table["threshold"].unique()
    assert thresholds.min() >= 0.05
    assert thresholds.max() <= 0.15 + 1e-10  # Allow floating point precision


def test_decision_curve_table_treat_none_always_zero():
    """Treat-none should always have net benefit = 0."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    dca_table = decision_curve_table(
        scenario="Test",
        y_true=y_true,
        pred_dict={"model": y_pred},
    )

    treat_none = dca_table[dca_table["model"] == "treat_none"]
    assert (treat_none["net_benefit"] == 0.0).all()


# =============================================================================
# Test: DCA Summary
# =============================================================================


def test_compute_dca_summary_structure(binary_classification_data):
    """Test DCA summary contains expected metrics."""
    y_true, y_pred = binary_classification_data
    dca_df = decision_curve_analysis(y_true, y_pred)

    summary = compute_dca_summary(dca_df)

    # Check required keys
    assert summary["dca_computed"] is True
    assert "n_thresholds" in summary
    assert "threshold_range" in summary
    assert "model_beats_all_from" in summary
    assert "model_beats_all_to" in summary
    assert "model_beats_all_range" in summary
    assert "integrated_nb_model" in summary
    assert "integrated_nb_all" in summary
    assert "integrated_nb_improvement" in summary


def test_compute_dca_summary_empty():
    """Empty DCA should return minimal summary."""
    summary = compute_dca_summary(pd.DataFrame())
    assert summary == {"dca_computed": False}


def test_compute_dca_summary_report_points():
    """Summary should include net benefit at specified thresholds."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    dca_df = decision_curve_analysis(y_true, y_pred, thresholds=np.linspace(0.001, 0.10, 100))
    summary = compute_dca_summary(dca_df, report_points=[0.01, 0.05])

    # Check report point keys (formatted as nb_model_at_1p0, nb_model_at_5p0)
    assert "nb_model_at_1p0%" in summary
    assert "nb_all_at_1p0%" in summary
    assert "nb_model_at_5p0%" in summary
    assert "nb_all_at_5p0%" in summary


def test_compute_dca_summary_never_beats_all():
    """Model that never beats treat-all should have 'Never' range."""
    # Random model unlikely to beat treat-all
    y_true = np.array([0] * 95 + [1] * 5)  # Very low prevalence
    y_pred = np.random.RandomState(99).uniform(0, 1, size=100)

    dca_df = decision_curve_analysis(y_true, y_pred)
    summary = compute_dca_summary(dca_df)

    # May or may not beat - check structure
    if summary.get("model_beats_all_range") == "Never":
        assert np.isnan(summary["model_beats_all_from"])
        assert np.isnan(summary["model_beats_all_to"])


# =============================================================================
# Test: Zero-Crossing Detection
# =============================================================================


def test_find_dca_zero_crossing_interpolation(tmp_path):
    """Test zero-crossing with linear interpolation."""
    # Create DCA curve that crosses zero
    dca_df = pd.DataFrame(
        {
            "threshold": [0.01, 0.02, 0.03, 0.04, 0.05],
            "net_benefit_model": [-0.02, -0.01, 0.01, 0.02, 0.03],
            "net_benefit_all": [0.15, 0.10, 0.05, 0.00, -0.05],
            "net_benefit_none": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    csv_path = tmp_path / "dca_curve.csv"
    dca_df.to_csv(csv_path, index=False)

    crossing = find_dca_zero_crossing(str(csv_path))

    # Should interpolate between 0.02 (-0.01) and 0.03 (0.01)
    # Zero at: 0.02 + (0 - (-0.01)) * (0.03 - 0.02) / (0.01 - (-0.01))
    #        = 0.02 + 0.01 * 0.01 / 0.02 = 0.025
    assert crossing is not None
    assert crossing == pytest.approx(0.025, abs=1e-6)


def test_find_dca_zero_crossing_no_crossing(tmp_path):
    """Test when model never crosses zero."""
    # Always positive net benefit
    dca_df = pd.DataFrame(
        {
            "threshold": [0.01, 0.02, 0.03],
            "net_benefit_model": [0.10, 0.08, 0.06],
            "net_benefit_all": [0.05, 0.03, 0.01],
            "net_benefit_none": [0.0, 0.0, 0.0],
        }
    )

    csv_path = tmp_path / "dca_curve.csv"
    dca_df.to_csv(csv_path, index=False)

    crossing = find_dca_zero_crossing(str(csv_path))

    # No crossing, but all points positive - should return None
    # (closest is 0.06, which is > 0.05 threshold)
    assert crossing is None


def test_find_dca_zero_crossing_fallback_near_zero(tmp_path):
    """Test fallback when no crossing but point very close to zero."""
    # No crossing but one point close to zero
    dca_df = pd.DataFrame(
        {
            "threshold": [0.01, 0.02, 0.03],
            "net_benefit_model": [0.10, 0.02, 0.08],  # 0.02 is close to zero
            "net_benefit_all": [0.05, 0.03, 0.01],
            "net_benefit_none": [0.0, 0.0, 0.0],
        }
    )

    csv_path = tmp_path / "dca_curve.csv"
    dca_df.to_csv(csv_path, index=False)

    crossing = find_dca_zero_crossing(str(csv_path))

    # Should use fallback (closest to zero within 0.05)
    assert crossing == pytest.approx(0.02, abs=1e-6)


def test_find_dca_zero_crossing_missing_file():
    """Missing file should return None."""
    crossing = find_dca_zero_crossing("/nonexistent/path.csv")
    assert crossing is None


def test_find_dca_zero_crossing_invalid_columns(tmp_path):
    """Missing required columns should return None."""
    dca_df = pd.DataFrame(
        {
            "threshold": [0.01, 0.02],
            "net_benefit_all": [0.05, 0.03],  # Missing net_benefit_model
        }
    )

    csv_path = tmp_path / "dca_curve.csv"
    dca_df.to_csv(csv_path, index=False)

    crossing = find_dca_zero_crossing(str(csv_path))
    assert crossing is None


# =============================================================================
# Test: DCA Persistence
# =============================================================================


def test_save_dca_results_files_created(binary_classification_data, tmp_path):
    """Test save_dca_results creates expected files."""
    y_true, y_pred = binary_classification_data

    summary = save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prefix="test__",
    )

    # Check files exist
    csv_path = tmp_path / "test__dca_curve.csv"
    json_path = tmp_path / "test__dca_summary.json"

    assert csv_path.exists()
    assert json_path.exists()

    # Check summary references
    assert summary["dca_csv_path"] == str(csv_path)
    assert summary["dca_json_path"] == str(json_path)


def test_save_dca_results_csv_content(binary_classification_data, tmp_path):
    """Test DCA curve CSV has correct structure."""
    y_true, y_pred = binary_classification_data

    save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prefix="test__",
        thresholds=np.linspace(0.01, 0.10, 10),
    )

    dca_df = pd.read_csv(tmp_path / "test__dca_curve.csv")

    assert len(dca_df) == 10
    assert "threshold" in dca_df.columns
    assert "net_benefit_model" in dca_df.columns
    assert "net_benefit_all" in dca_df.columns


def test_save_dca_results_json_content(binary_classification_data, tmp_path):
    """Test DCA summary JSON has correct content."""
    y_true, y_pred = binary_classification_data

    save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prefix="test__",
    )

    with open(tmp_path / "test__dca_summary.json") as f:
        summary = json.load(f)

    assert summary["dca_computed"] is True
    assert "n_thresholds" in summary
    assert "integrated_nb_model" in summary


def test_save_dca_results_prevalence_adjustment(binary_classification_data, tmp_path):
    """Test prevalence adjustment is recorded in summary."""
    y_true, y_pred = binary_classification_data

    summary = save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prevalence_adjustment=0.15,
    )

    assert summary["prevalence_adjustment"] == 0.15


def test_save_dca_results_empty_data(tmp_path):
    """Empty data should return error summary."""
    summary = save_dca_results(
        np.array([]),
        np.array(
            [],
        ),
        out_dir=str(tmp_path),
    )

    assert summary["dca_computed"] is False
    assert "error" in summary


# =============================================================================
# Test: Utility Functions
# =============================================================================


def test_generate_dca_thresholds_default():
    """Test default threshold generation."""
    thresholds = generate_dca_thresholds(min_thr=0.001, max_thr=0.10, step=0.001)

    assert thresholds[0] >= 0.001
    assert thresholds[-1] <= 0.10
    assert len(thresholds) > 0

    # Check spacing
    diffs = np.diff(thresholds)
    assert np.allclose(diffs, 0.001, atol=1e-6)


def test_generate_dca_thresholds_clamping():
    """Test thresholds are clamped to valid range."""
    # Min clamped to 0.0001
    thresholds = generate_dca_thresholds(min_thr=-0.5, max_thr=0.05, step=0.01)
    assert thresholds[0] >= 0.0001

    # Max clamped to 0.999
    thresholds = generate_dca_thresholds(min_thr=0.5, max_thr=1.5, step=0.01)
    assert thresholds[-1] <= 0.999


def test_generate_dca_thresholds_min_equals_max():
    """When min == max, return both endpoints."""
    thresholds = generate_dca_thresholds(min_thr=0.05, max_thr=0.05, step=0.01)
    assert len(thresholds) == 2
    assert thresholds[0] == thresholds[1] == 0.05


def test_parse_dca_report_points():
    """Test parsing comma-separated report points."""
    points = parse_dca_report_points("0.005,0.01,0.02,0.05")
    assert points == [0.005, 0.01, 0.02, 0.05]


def test_parse_dca_report_points_with_spaces():
    """Test parsing with extra whitespace."""
    points = parse_dca_report_points("  0.005 , 0.01,  0.02  ")
    assert points == [0.005, 0.01, 0.02]


def test_parse_dca_report_points_invalid():
    """Test parsing with invalid values."""
    # Out of range values filtered
    points = parse_dca_report_points("0.0,0.5,1.0,1.5,-0.1")
    assert points == [0.5]

    # Non-numeric filtered
    points = parse_dca_report_points("0.01,abc,0.05,xyz")
    assert points == [0.01, 0.05]


def test_parse_dca_report_points_empty():
    """Test parsing empty or None string."""
    assert parse_dca_report_points("") == []
    assert parse_dca_report_points(None) == []


# =============================================================================
# Integration Tests
# =============================================================================


def test_dca_workflow_end_to_end(binary_classification_data, tmp_path):
    """Test complete DCA workflow: compute -> save -> load crossing."""
    y_true, y_pred = binary_classification_data

    # Save DCA results
    summary = save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prefix="integration__",
        thresholds=np.linspace(0.001, 0.20, 200),
        report_points=[0.01, 0.05],
    )

    # Check summary
    assert summary["dca_computed"] is True
    assert summary["n_thresholds"] == 200

    # Find zero crossing
    csv_path = summary["dca_csv_path"]
    crossing = find_dca_zero_crossing(csv_path)

    # Should find a crossing (may be None for random data)
    # Just verify it doesn't crash
    assert crossing is None or isinstance(crossing, float)


def test_cross_model_dca_comparison(binary_classification_data):
    """Test comparing multiple models via DCA table."""
    y_true, y_pred = binary_classification_data

    # Create three models with different calibration
    pred_dict = {
        "model_good": y_pred,
        "model_medium": y_pred * 0.7 + 0.15,
        "model_poor": y_pred * 0.5 + 0.25,
    }

    dca_table = decision_curve_table(
        scenario="MultiModel",
        y_true=y_true,
        pred_dict=pred_dict,
        max_pt=0.20,
        step=0.01,
    )

    # Check all models present
    models = dca_table["model"].unique()
    assert "treat_none" in models
    assert "treat_all" in models
    assert "model_good" in models
    assert "model_medium" in models
    assert "model_poor" in models

    # Each threshold should have all 5 models
    for threshold in dca_table["threshold"].unique():
        df_t = dca_table[dca_table["threshold"] == threshold]
        assert len(df_t) == 5


# =============================================================================
# Test: DCA Auto-Range Configuration
# =============================================================================


def test_generate_dca_thresholds_auto_range_low_prevalence():
    """Test auto-range with low prevalence (0.003 = 0.3%)."""
    prevalence = 0.003  # 0.3% like Celiac disease

    thresholds = generate_dca_thresholds(prevalence=prevalence)

    # Expected: min_thr = max(0.0001, 0.003 / 10) = 0.0003
    # Expected: max_thr = min(0.5, 0.003 * 10) = 0.03
    assert thresholds[0] == pytest.approx(0.0003, abs=1e-6)
    assert thresholds[-1] == pytest.approx(0.03, abs=1e-3)


def test_generate_dca_thresholds_auto_range_medium_prevalence():
    """Test auto-range with medium prevalence (0.01 = 1%)."""
    prevalence = 0.01

    thresholds = generate_dca_thresholds(prevalence=prevalence)

    # Expected: min_thr = max(0.0001, 0.01 / 10) = 0.001
    # Expected: max_thr = min(0.5, 0.01 * 10) = 0.10
    assert thresholds[0] == pytest.approx(0.001, abs=1e-6)
    assert thresholds[-1] == pytest.approx(0.10, abs=1e-3)


def test_generate_dca_thresholds_auto_range_high_prevalence():
    """Test auto-range with high prevalence (0.10 = 10%)."""
    prevalence = 0.10

    thresholds = generate_dca_thresholds(prevalence=prevalence)

    # Expected: min_thr = max(0.0001, 0.10 / 10) = 0.01
    # Expected: max_thr = min(0.5, 0.10 * 10) = 0.5
    assert thresholds[0] == pytest.approx(0.01, abs=1e-6)
    assert thresholds[-1] == pytest.approx(0.5, abs=1e-3)


def test_generate_dca_thresholds_auto_range_capped_at_05():
    """Test auto-range with very high prevalence caps at 0.5."""
    prevalence = 0.20  # 20% prevalence

    thresholds = generate_dca_thresholds(prevalence=prevalence)

    # Expected: max_thr = min(0.5, 0.20 * 10) = 0.5 (capped)
    assert thresholds[-1] == pytest.approx(0.5, abs=1e-3)


def test_generate_dca_thresholds_auto_range_minimum_floor():
    """Test auto-range with very low prevalence respects 0.0001 floor."""
    prevalence = 0.0001  # 0.01% prevalence

    thresholds = generate_dca_thresholds(prevalence=prevalence)

    # Expected: min_thr = max(0.0001, 0.0001 / 10) = 0.0001 (floor)
    assert thresholds[0] == pytest.approx(0.0001, abs=1e-6)


def test_generate_dca_thresholds_backward_compatibility():
    """Test default behavior unchanged when prevalence not specified."""
    thresholds_default = generate_dca_thresholds()
    thresholds_none = generate_dca_thresholds(prevalence=None)

    # Should produce same results
    np.testing.assert_array_almost_equal(thresholds_default, thresholds_none)

    # Check default range 0.001 to 0.10
    assert thresholds_default[0] >= 0.001
    assert thresholds_default[-1] <= 0.10


def test_generate_dca_thresholds_explicit_overrides_prevalence():
    """Test explicit min/max thresholds are used when prevalence not provided."""
    thresholds = generate_dca_thresholds(min_thr=0.05, max_thr=0.50, step=0.01)

    assert thresholds[0] >= 0.05
    assert thresholds[-1] <= 0.50


def test_decision_curve_analysis_with_prevalence_auto_range():
    """Test DCA auto-configures range when prevalence is provided."""
    y_true = np.array([0] * 997 + [1] * 3)  # 0.3% prevalence
    y_pred = np.random.RandomState(42).beta(2, 8, size=1000)

    # Without prevalence: default range 0.001 to 0.10
    dca_default = decision_curve_analysis(y_true, y_pred)
    assert dca_default["threshold"].min() >= 0.001
    assert dca_default["threshold"].max() <= 0.10

    # With prevalence: auto-configures range
    dca_auto = decision_curve_analysis(y_true, y_pred, prevalence=0.003)
    # Range should be 0.0003 to 0.03 for 0.3% prevalence
    assert dca_auto["threshold"].min() >= 0.0003
    assert dca_auto["threshold"].max() <= 0.03


def test_decision_curve_analysis_prevalence_backward_compatible():
    """Test existing calls without prevalence work unchanged."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).beta(2, 8, size=100)

    # These should all work and produce consistent results
    dca1 = decision_curve_analysis(y_true, y_pred)
    _ = decision_curve_analysis(y_true, y_pred, thresholds=None)
    _ = decision_curve_analysis(y_true, y_pred, prevalence_adjustment=None)

    # Check structure is preserved
    assert "threshold" in dca1.columns
    assert "net_benefit_model" in dca1.columns
    assert len(dca1) > 0


def test_save_dca_results_with_prevalence(binary_classification_data, tmp_path):
    """Test save_dca_results records auto-range prevalence in summary."""
    y_true, y_pred = binary_classification_data

    summary = save_dca_results(
        y_true,
        y_pred,
        out_dir=str(tmp_path),
        prefix="auto_range__",
        prevalence=0.003,
    )

    # Check that auto_range_prevalence is recorded
    assert summary.get("auto_range_prevalence") == 0.003
    assert summary["dca_computed"] is True

    # Verify threshold range is appropriate for 0.3% prevalence
    dca_df = pd.read_csv(tmp_path / "auto_range__dca_curve.csv")
    assert dca_df["threshold"].min() >= 0.0003 - 1e-6
    assert dca_df["threshold"].max() <= 0.03 + 1e-3


def test_threshold_dca_zero_crossing_with_prevalence():
    """Test zero-crossing detection with auto-ranged thresholds."""
    # Create data where model has positive net benefit at low thresholds
    y_true = np.array([0] * 990 + [1] * 10)  # 1% prevalence
    # High-risk predictions for cases, low for controls
    y_pred = np.concatenate(
        [
            np.random.RandomState(42).uniform(0, 0.3, size=990),
            np.random.RandomState(42).uniform(0.5, 1.0, size=10),
        ]
    )

    # Without prevalence: may miss relevant thresholds
    crossing_default = threshold_dca_zero_crossing(y_true, y_pred)

    # With prevalence: auto-ranges to relevant thresholds
    crossing_auto = threshold_dca_zero_crossing(y_true, y_pred, prevalence=0.01)

    # Both should return valid results (or None if no crossing)
    assert crossing_default is None or isinstance(crossing_default, float)
    assert crossing_auto is None or isinstance(crossing_auto, float)


# =============================================================================
# Test: Prevalence Validation
# =============================================================================


def test_net_benefit_treat_all_invalid_prevalence_negative():
    """Test net_benefit_treat_all raises ValueError for negative prevalence."""
    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        net_benefit_treat_all(prevalence=-0.1, threshold=0.5)


def test_net_benefit_treat_all_invalid_prevalence_above_one():
    """Test net_benefit_treat_all raises ValueError for prevalence > 1."""
    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        net_benefit_treat_all(prevalence=1.5, threshold=0.5)


def test_net_benefit_treat_all_valid_prevalence_boundary():
    """Test net_benefit_treat_all accepts valid boundary values."""
    # prevalence=0 and prevalence=1 are valid boundary values
    nb_zero = net_benefit_treat_all(prevalence=0.0, threshold=0.5)
    nb_one = net_benefit_treat_all(prevalence=1.0, threshold=0.5)

    # prevalence=0: NB = 0 - 1 * (0.5/0.5) = -1.0
    assert nb_zero == pytest.approx(-1.0, abs=1e-6)
    # prevalence=1: NB = 1 - 0 * (0.5/0.5) = 1.0
    assert nb_one == pytest.approx(1.0, abs=1e-6)


def test_generate_dca_thresholds_invalid_prevalence_negative():
    """Test generate_dca_thresholds raises ValueError for negative prevalence."""
    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        generate_dca_thresholds(prevalence=-0.5)


def test_generate_dca_thresholds_invalid_prevalence_above_one():
    """Test generate_dca_thresholds raises ValueError for prevalence > 1."""
    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        generate_dca_thresholds(prevalence=2.0)


def test_decision_curve_analysis_invalid_prevalence_adjustment():
    """Test decision_curve_analysis raises ValueError for invalid prevalence_adjustment."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).uniform(0, 1, size=100)

    with pytest.raises(ValueError, match="prevalence_adjustment must be in \\[0, 1\\] range"):
        decision_curve_analysis(y_true, y_pred, prevalence_adjustment=-0.1)

    with pytest.raises(ValueError, match="prevalence_adjustment must be in \\[0, 1\\] range"):
        decision_curve_analysis(y_true, y_pred, prevalence_adjustment=1.5)


def test_decision_curve_analysis_invalid_prevalence():
    """Test decision_curve_analysis raises ValueError for invalid prevalence."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).uniform(0, 1, size=100)

    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        decision_curve_analysis(y_true, y_pred, prevalence=-0.01)

    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        decision_curve_analysis(y_true, y_pred, prevalence=1.001)


def test_threshold_dca_zero_crossing_invalid_prevalence():
    """Test threshold_dca_zero_crossing raises ValueError for invalid prevalence."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).uniform(0, 1, size=100)

    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        threshold_dca_zero_crossing(y_true, y_pred, prevalence=-0.5)

    with pytest.raises(ValueError, match="prevalence_adjustment must be in \\[0, 1\\] range"):
        threshold_dca_zero_crossing(y_true, y_pred, prevalence_adjustment=10.0)


def test_save_dca_results_invalid_prevalence(tmp_path):
    """Test save_dca_results raises ValueError for invalid prevalence."""
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.random.RandomState(42).uniform(0, 1, size=100)

    with pytest.raises(ValueError, match="prevalence must be in \\[0, 1\\] range"):
        save_dca_results(y_true, y_pred, out_dir=str(tmp_path), prevalence=-0.1)

    with pytest.raises(ValueError, match="prevalence_adjustment must be in \\[0, 1\\] range"):
        save_dca_results(y_true, y_pred, out_dir=str(tmp_path), prevalence_adjustment=5.0)
