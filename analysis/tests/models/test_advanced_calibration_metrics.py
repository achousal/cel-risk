"""
Tests for advanced calibration assessment metrics added to models.calibration.

Coverage areas:
- ICI (Integrated Calibration Index)
- CalibrationQuantiles (E50 / E90 / ICI)
- Spiegelhalter's z-test for calibration
- Adaptive Expected Calibration Error (Adaptive ECE)
- Brier Score Decomposition (Murphy 1973)
- All scalar wrappers for bootstrap compatibility
"""

import numpy as np
import pytest

from ced_ml.models.calibration import (
    BrierDecomposition,
    CalibrationQuantiles,
    SpiegelhalterResult,
    adaptive_ece_metric,
    adaptive_expected_calibration_error,
    brier_reliability_metric,
    brier_resolution_metric,
    brier_score_decomposition,
    calibration_error_quantiles,
    ici_metric,
    integrated_calibration_index,
    spiegelhalter_z_metric,
    spiegelhalter_z_test,
)

# ---------------------------------------------------------------------------
# ICI (Integrated Calibration Index)
# ---------------------------------------------------------------------------


class TestIntegratedCalibrationIndex:
    """Tests for integrated_calibration_index."""

    def test_well_calibrated_data_low_ici(self):
        """Well-calibrated predictions should have a low ICI."""
        rng = np.random.default_rng(100)
        n = 800
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        ici = integrated_calibration_index(y, p)
        assert np.isfinite(ici)
        assert 0.0 <= ici <= 0.1, f"Expected low ICI for calibrated data, got {ici}"

    def test_output_in_unit_interval(self):
        """ICI must be in [0, 1] for valid inputs."""
        rng = np.random.default_rng(101)
        n = 500
        p = rng.uniform(0.01, 0.99, n)
        y = rng.binomial(1, p)
        ici = integrated_calibration_index(y, p)
        assert 0.0 <= ici <= 1.0

    def test_empty_returns_nan(self):
        """Empty arrays should return NaN."""
        assert np.isnan(integrated_calibration_index(np.array([]), np.array([])))

    def test_nan_in_inputs_handled(self):
        """NaN values in inputs should be filtered and not raise."""
        rng = np.random.default_rng(102)
        n = 500
        p = rng.uniform(0.05, 0.5, n).astype(float)
        y = rng.binomial(1, p).astype(float)
        p[::20] = np.nan
        y[::30] = np.nan
        ici = integrated_calibration_index(y, p)
        assert isinstance(ici, float)

    def test_too_few_unique_predictions_returns_nan(self):
        """Fewer than 10 unique prediction values should return NaN."""
        y = np.array([0, 1] * 10)
        p = np.array([0.1, 0.9] * 10)  # only 2 unique values
        ici = integrated_calibration_index(y, p)
        assert np.isnan(ici)

    def test_ici_metric_alias(self):
        """ici_metric should return the same value as integrated_calibration_index."""
        rng = np.random.default_rng(103)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        assert ici_metric(y, p) == integrated_calibration_index(y, p)


# ---------------------------------------------------------------------------
# CalibrationQuantiles (E50 / E90 / ICI)
# ---------------------------------------------------------------------------


class TestCalibrationErrorQuantiles:
    """Tests for calibration_error_quantiles."""

    def test_returns_calibration_quantiles_dataclass(self):
        """Should return a CalibrationQuantiles instance."""
        rng = np.random.default_rng(110)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        q = calibration_error_quantiles(y, p)
        assert isinstance(q, CalibrationQuantiles)

    def test_e50_le_e90(self):
        """E50 must be <= E90 by definition."""
        rng = np.random.default_rng(111)
        n = 600
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        q = calibration_error_quantiles(y, p)
        assert q.e50 <= q.e90

    def test_ici_consistent_with_standalone_function(self):
        """ICI field should match integrated_calibration_index exactly."""
        rng = np.random.default_rng(112)
        n = 600
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        q = calibration_error_quantiles(y, p)
        ici = integrated_calibration_index(y, p)
        assert abs(q.ici - ici) < 1e-10

    def test_all_fields_finite_for_valid_data(self):
        """All fields should be finite for valid data with enough unique predictions."""
        rng = np.random.default_rng(113)
        n = 600
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        q = calibration_error_quantiles(y, p)
        assert np.isfinite(q.e50)
        assert np.isfinite(q.e90)
        assert np.isfinite(q.ici)

    def test_empty_returns_nan_fields(self):
        """Empty arrays should return NaN in all fields."""
        q = calibration_error_quantiles(np.array([]), np.array([]))
        assert np.isnan(q.e50)
        assert np.isnan(q.e90)
        assert np.isnan(q.ici)

    def test_too_few_unique_returns_nan(self):
        """Too few unique predictions should yield NaN fields."""
        y = np.array([0, 1] * 10)
        p = np.array([0.2, 0.8] * 10)  # only 2 unique values
        q = calibration_error_quantiles(y, p)
        assert np.isnan(q.e50)
        assert np.isnan(q.e90)
        assert np.isnan(q.ici)


# ---------------------------------------------------------------------------
# Spiegelhalter's z-test
# ---------------------------------------------------------------------------


class TestSpiegelhalterZTest:
    """Tests for spiegelhalter_z_test."""

    def test_returns_spiegelhalter_result_dataclass(self):
        """Should return a SpiegelhalterResult instance."""
        rng = np.random.default_rng(120)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        result = spiegelhalter_z_test(y, p)
        assert isinstance(result, SpiegelhalterResult)

    def test_well_calibrated_data_not_rejected(self):
        """Well-calibrated data should typically not reject the null (p > 0.05)."""
        rng = np.random.default_rng(121)
        n = 1000
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        result = spiegelhalter_z_test(y, p)
        assert result.is_calibrated
        assert result.p_value > 0.05

    def test_is_calibrated_flag_consistent_with_p_value(self):
        """is_calibrated must equal (p_value > 0.05) exactly."""
        rng = np.random.default_rng(122)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        result = spiegelhalter_z_test(y, p)
        assert result.is_calibrated == (result.p_value > 0.05)

    def test_p_value_in_unit_interval(self):
        """p-value must be in [0, 1]."""
        rng = np.random.default_rng(123)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        result = spiegelhalter_z_test(y, p)
        assert 0.0 <= result.p_value <= 1.0

    def test_severely_miscalibrated_rejected(self):
        """Severely miscalibrated predictions should have a large |z| and p < 0.05."""
        rng = np.random.default_rng(124)
        n = 1000
        y = rng.binomial(1, 0.1, n)
        p = np.full(n, 0.9)
        result = spiegelhalter_z_test(y, p)
        assert abs(result.z_statistic) > 3.0
        assert result.p_value < 0.05
        assert not result.is_calibrated

    def test_single_class_returns_nan(self):
        """Single class in y_true should return NaN fields."""
        rng = np.random.default_rng(125)
        p = rng.uniform(0.1, 0.9, 100)
        y = np.zeros(100)
        result = spiegelhalter_z_test(y, p)
        assert np.isnan(result.z_statistic)
        assert np.isnan(result.p_value)

    def test_empty_returns_nan(self):
        """Empty arrays should return NaN fields."""
        result = spiegelhalter_z_test(np.array([]), np.array([]))
        assert np.isnan(result.z_statistic)
        assert np.isnan(result.p_value)

    def test_constant_prediction_returns_nan(self):
        """Constant predictions at 0.5 produce zero variance -> NaN."""
        rng = np.random.default_rng(126)
        y = rng.binomial(1, 0.5, 200)
        p = np.full(200, 0.5)
        result = spiegelhalter_z_test(y, p)
        # weights = 1 - 2*0.5 = 0 for all, so variance = 0
        assert np.isnan(result.z_statistic)

    def test_spiegelhalter_z_metric_alias(self):
        """spiegelhalter_z_metric should return z_statistic."""
        rng = np.random.default_rng(127)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        result = spiegelhalter_z_test(y, p)
        assert spiegelhalter_z_metric(y, p) == result.z_statistic

    def test_nan_in_inputs_handled(self):
        """NaN values should be filtered before computing the statistic."""
        rng = np.random.default_rng(128)
        n = 500
        p = rng.uniform(0.05, 0.5, n).astype(float)
        y = rng.binomial(1, p).astype(float)
        p[::25] = np.nan
        y[::35] = np.nan
        result = spiegelhalter_z_test(y, p)
        assert isinstance(result.z_statistic, float)


# ---------------------------------------------------------------------------
# Adaptive ECE
# ---------------------------------------------------------------------------


class TestAdaptiveExpectedCalibrationError:
    """Tests for adaptive_expected_calibration_error."""

    def test_well_calibrated_low_value(self):
        """Well-calibrated predictions should have a low Adaptive ECE."""
        rng = np.random.default_rng(130)
        n = 1000
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        aece = adaptive_expected_calibration_error(y, p)
        assert np.isfinite(aece)
        assert 0.0 <= aece <= 0.15, f"Expected low AECE for calibrated data, got {aece}"

    def test_output_in_unit_interval(self):
        """AECE must be in [0, 1] for valid inputs."""
        rng = np.random.default_rng(131)
        n = 500
        p = rng.uniform(0.01, 0.99, n)
        y = rng.binomial(1, p)
        aece = adaptive_expected_calibration_error(y, p)
        assert 0.0 <= aece <= 1.0

    def test_empty_returns_nan(self):
        """Empty arrays should return NaN."""
        assert np.isnan(adaptive_expected_calibration_error(np.array([]), np.array([])))

    def test_invalid_strategy_raises(self):
        """Unsupported strategy should raise ValueError."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])
        with pytest.raises(ValueError, match="strategy must be 'quantile'"):
            adaptive_expected_calibration_error(y, p, strategy="uniform")

    def test_nan_in_inputs_handled(self):
        """NaN values should be filtered before computation."""
        rng = np.random.default_rng(132)
        n = 500
        p = rng.uniform(0.05, 0.5, n).astype(float)
        y = rng.binomial(1, p).astype(float)
        p[::20] = np.nan
        y[::30] = np.nan
        aece = adaptive_expected_calibration_error(y, p)
        assert np.isfinite(aece)
        assert 0.0 <= aece <= 1.0

    def test_extreme_imbalance_low_prevalence(self):
        """Adaptive ECE should handle ~0.5% prevalence without error."""
        rng = np.random.default_rng(133)
        n = 10000
        p = rng.uniform(0.001, 0.02, n)
        y = rng.binomial(1, p)
        aece = adaptive_expected_calibration_error(y, p, min_events_per_bin=5)
        assert np.isfinite(aece)
        assert 0.0 <= aece <= 1.0

    def test_adaptive_ece_metric_alias(self):
        """adaptive_ece_metric should match adaptive_expected_calibration_error defaults."""
        rng = np.random.default_rng(134)
        n = 500
        p = rng.uniform(0.05, 0.5, n)
        y = rng.binomial(1, p)
        assert adaptive_ece_metric(y, p) == adaptive_expected_calibration_error(y, p)

    def test_bin_merging_produces_finite_result(self):
        """With very low prevalence, merging should produce a finite ECE."""
        rng = np.random.default_rng(135)
        n = 2000
        p = rng.uniform(0.001, 0.01, n)
        y = rng.binomial(1, p)
        aece = adaptive_expected_calibration_error(y, p, n_bins=10, min_events_per_bin=5)
        assert np.isfinite(aece)


# ---------------------------------------------------------------------------
# Brier Score Decomposition
# ---------------------------------------------------------------------------


class TestBrierScoreDecomposition:
    """Tests for brier_score_decomposition."""

    def test_returns_brier_decomposition_dataclass(self):
        """Should return a BrierDecomposition instance."""
        rng = np.random.default_rng(140)
        p = rng.uniform(0, 1, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert isinstance(decomp, BrierDecomposition)

    def test_identity_holds_exactly(self):
        """reliability - resolution + uncertainty == brier_score to float precision."""
        rng = np.random.default_rng(141)
        p = rng.uniform(0, 1, 1000)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        identity_err = abs(
            decomp.reliability - decomp.resolution + decomp.uncertainty - decomp.brier_score
        )
        assert identity_err < 1e-12, f"Identity violated by {identity_err:.2e}"

    def test_identity_holds_for_imbalanced_data(self):
        """Identity must hold for very imbalanced data (low prevalence)."""
        rng = np.random.default_rng(142)
        n = 5000
        p = rng.uniform(0.001, 0.02, n)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        identity_err = abs(
            decomp.reliability - decomp.resolution + decomp.uncertainty - decomp.brier_score
        )
        assert identity_err < 1e-12

    def test_all_fields_finite_for_valid_data(self):
        """All fields should be finite for valid binary data."""
        rng = np.random.default_rng(143)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert np.isfinite(decomp.reliability)
        assert np.isfinite(decomp.resolution)
        assert np.isfinite(decomp.uncertainty)
        assert np.isfinite(decomp.brier_score)

    def test_reliability_nonnegative(self):
        """Reliability component must be >= 0 (it is a mean squared deviation)."""
        rng = np.random.default_rng(144)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert decomp.reliability >= 0.0

    def test_resolution_nonnegative(self):
        """Resolution component must be >= 0 (it is a mean squared deviation)."""
        rng = np.random.default_rng(145)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert decomp.resolution >= 0.0

    def test_uncertainty_equals_o_bar_times_complement(self):
        """Uncertainty = o_bar * (1 - o_bar)."""
        rng = np.random.default_rng(146)
        n = 500
        p = rng.uniform(0.05, 0.95, n)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        o_bar = np.mean(y)
        expected_unc = o_bar * (1.0 - o_bar)
        assert abs(decomp.uncertainty - expected_unc) < 1e-12

    def test_empty_returns_nan_fields(self):
        """Empty arrays should return NaN in all fields."""
        decomp = brier_score_decomposition(np.array([]), np.array([]))
        assert np.isnan(decomp.reliability)
        assert np.isnan(decomp.resolution)
        assert np.isnan(decomp.uncertainty)
        assert np.isnan(decomp.brier_score)

    def test_nan_in_inputs_handled(self):
        """NaN values should be filtered before decomposition."""
        rng = np.random.default_rng(147)
        p = rng.uniform(0.05, 0.95, 500).astype(float)
        y = rng.binomial(1, p).astype(float)
        p[::20] = np.nan
        y[::25] = np.nan
        decomp = brier_score_decomposition(y, p)
        assert np.isfinite(decomp.brier_score)

    def test_perfect_discrimination_high_resolution(self):
        """A model with near-perfect separation should have high resolution."""
        y = np.array([0] * 100 + [1] * 100)
        p = np.array([0.01] * 100 + [0.99] * 100)
        decomp = brier_score_decomposition(y, p, n_bins=10)
        o_bar = 0.5
        max_resolution = o_bar * (1.0 - o_bar)
        assert decomp.resolution > 0.2 * max_resolution

    def test_scalar_wrapper_reliability(self):
        """brier_reliability_metric should return the reliability field."""
        rng = np.random.default_rng(148)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert brier_reliability_metric(y, p) == decomp.reliability

    def test_scalar_wrapper_resolution(self):
        """brier_resolution_metric should return the resolution field."""
        rng = np.random.default_rng(149)
        p = rng.uniform(0.05, 0.95, 500)
        y = rng.binomial(1, p)
        decomp = brier_score_decomposition(y, p)
        assert brier_resolution_metric(y, p) == decomp.resolution
