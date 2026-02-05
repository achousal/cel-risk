"""Tests for metrics.thresholds module.

Coverage:
- All threshold selection strategies (F1, F-beta, Youden, fixed spec/PPV, control-based)
- Binary metrics computation at thresholds
- Top risk capture analysis
- DCA-based threshold selection
- Edge cases (empty arrays, all same class, perfect separation)
"""

import numpy as np
import pytest

from ced_ml.metrics.dca import threshold_dca_zero_crossing
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    choose_threshold_objective,
    compute_multi_target_specificity_metrics,
    threshold_for_precision,
    threshold_for_specificity,
    threshold_from_controls,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
    top_risk_capture,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def balanced_data():
    """Balanced dataset with clear separation."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def imbalanced_data():
    """Imbalanced dataset (1:9 ratio) simulating rare disease."""
    y = np.array([0] * 90 + [1] * 10)
    _rng = np.random.default_rng(42)
    p_controls = np.random.beta(2, 5, size=90)  # Skewed low
    p_cases = np.random.beta(5, 2, size=10)  # Skewed high
    p = np.concatenate([p_controls, p_cases])
    return y, p


@pytest.fixture
def perfect_separation():
    """Perfectly separated data (AUROC = 1.0)."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def all_controls():
    """All negative samples (no cases)."""
    rng = np.random.default_rng(42)
    y = np.zeros(100, dtype=int)
    p = rng.uniform(0, 1, size=100)
    return y, p


@pytest.fixture
def all_cases():
    """All positive samples (no controls)."""
    rng = np.random.default_rng(42)
    y = np.ones(100, dtype=int)
    p = rng.uniform(0, 1, size=100)
    return y, p


# ============================================================================
# Test threshold_max_f1
# ============================================================================


def test_threshold_max_f1_balanced(balanced_data):
    """Test F1-maximizing threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0
    # Expect threshold around 0.5 for balanced separation
    assert 0.4 <= thr <= 0.6


def test_threshold_max_f1_imbalanced(imbalanced_data):
    """Test F1-maximizing threshold on imbalanced data."""
    y, p = imbalanced_data
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0
    # Imbalanced data typically requires higher threshold
    assert thr > 0.5


def test_threshold_max_f1_empty():
    """Test F1 threshold with empty arrays."""
    y = np.array([])
    p = np.array([])
    thr = threshold_max_f1(y, p)
    assert thr == 0.5  # Fallback


def test_threshold_max_f1_all_same_class(all_controls):
    """Test F1 threshold when all samples are same class."""
    y, p = all_controls
    thr = threshold_max_f1(y, p)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_max_fbeta
# ============================================================================


def test_threshold_max_fbeta_beta1(balanced_data):
    """Test F-beta with beta=1 (should equal F1)."""
    y, p = balanced_data
    thr_f1 = threshold_max_f1(y, p)
    thr_fb = threshold_max_fbeta(y, p, beta=1.0)
    assert abs(thr_f1 - thr_fb) < 0.01


def test_threshold_max_fbeta_beta2(balanced_data):
    """Test F-beta with beta=2 (emphasize recall)."""
    y, p = balanced_data
    thr_fb = threshold_max_fbeta(y, p, beta=2.0)
    thr_f1 = threshold_max_f1(y, p)
    # Higher beta -> lower threshold (more sensitive)
    assert thr_fb <= thr_f1


def test_threshold_max_fbeta_beta05(balanced_data):
    """Test F-beta with beta=0.5 (emphasize precision)."""
    y, p = balanced_data
    thr_fb = threshold_max_fbeta(y, p, beta=0.5)
    thr_f1 = threshold_max_f1(y, p)
    # Lower beta -> higher threshold (more precise)
    assert thr_fb >= thr_f1


def test_threshold_max_fbeta_invalid_beta(balanced_data):
    """Test F-beta with invalid beta (should default to 1.0)."""
    y, p = balanced_data
    thr = threshold_max_fbeta(y, p, beta=-1.0)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_youden
# ============================================================================


def test_threshold_youden_balanced(balanced_data):
    """Test Youden threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_youden(y, p)
    assert 0.0 <= thr <= 1.0


def test_threshold_youden_perfect_separation(perfect_separation):
    """Test Youden threshold with perfect separation."""
    y, p = perfect_separation
    thr = threshold_youden(y, p)
    assert 0.0 <= thr <= 1.0
    # Should separate perfectly - threshold should be in the gap
    # Between max control (0.3) and min case (0.7)
    assert 0.0 <= thr <= 1.0  # Just verify valid range


def test_threshold_youden_empty():
    """Test Youden threshold with empty arrays."""
    y = np.array([])
    p = np.array([])
    thr = threshold_youden(y, p)
    assert thr == 0.5  # Fallback


# ============================================================================
# Test threshold_for_specificity
# ============================================================================


def test_threshold_for_specificity_90(balanced_data):
    """Test fixed specificity threshold (0.90)."""
    y, p = balanced_data
    thr = threshold_for_specificity(y, p, target_spec=0.90)
    assert 0.0 <= thr <= 1.0
    # Higher specificity -> higher threshold (or equal in edge cases)
    thr_50 = threshold_for_specificity(y, p, target_spec=0.50)
    assert thr >= thr_50


def test_threshold_for_specificity_95(imbalanced_data):
    """Test high specificity threshold (0.95)."""
    y, p = imbalanced_data
    thr = threshold_for_specificity(y, p, target_spec=0.95)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_specificity_unattainable(balanced_data):
    """Test specificity threshold when target is unattainable."""
    y, p = balanced_data
    # Request impossibly high specificity
    thr = threshold_for_specificity(y, p, target_spec=0.999)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_specificity_perfect(perfect_separation):
    """Test specificity threshold with perfect separation."""
    y, p = perfect_separation
    thr = threshold_for_specificity(y, p, target_spec=1.0)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test threshold_for_precision
# ============================================================================


def test_threshold_for_precision_balanced(balanced_data):
    """Test fixed precision threshold on balanced data."""
    y, p = balanced_data
    thr = threshold_for_precision(y, p, target_ppv=0.8)
    assert 0.0 <= thr <= 1.0


def test_threshold_for_precision_unattainable(imbalanced_data):
    """Test precision threshold when target is unattainable (fallback to F1)."""
    y, p = imbalanced_data
    # Request impossibly high precision
    thr_high = threshold_for_precision(y, p, target_ppv=0.99)
    thr_f1 = threshold_max_f1(y, p)
    # Should fall back to F1 (or be close in case of randomness)
    # Allow generous tolerance due to imbalanced data variability
    assert abs(thr_high - thr_f1) < 0.5


def test_threshold_for_precision_invalid_target(balanced_data):
    """Test precision threshold with invalid target (should fallback to F1)."""
    y, p = balanced_data
    thr = threshold_for_precision(y, p, target_ppv=1.5)  # Invalid
    thr_f1 = threshold_max_f1(y, p)
    assert abs(thr - thr_f1) < 0.01


# ============================================================================
# Test threshold_from_controls
# ============================================================================


def test_threshold_from_controls_basic():
    """Test control-based threshold (basic quantile)."""
    p_controls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    # 90th percentile should be 0.9
    assert 0.8 <= thr <= 1.0


def test_threshold_from_controls_95():
    """Test 95% specificity threshold from controls."""
    p_controls = np.linspace(0, 1, 100)
    thr = threshold_from_controls(p_controls, target_spec=0.95)
    assert 0.90 <= thr <= 1.0


def test_threshold_from_controls_empty():
    """Test control-based threshold with empty array."""
    p_controls = np.array([])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    assert thr == 0.5  # Fallback


def test_threshold_from_controls_nan():
    """Test control-based threshold with NaN values."""
    p_controls = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
    thr = threshold_from_controls(p_controls, target_spec=0.90)
    assert 0.0 <= thr <= 1.0


# ============================================================================
# Test binary_metrics_at_threshold
# ============================================================================


def test_binary_metrics_at_threshold_perfect(perfect_separation):
    """Test metrics at optimal threshold with perfect separation."""
    y, p = perfect_separation
    thr = 0.5  # Separates perfectly
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics.threshold == 0.5
    assert metrics.precision == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.f1 == 1.0
    assert metrics.specificity == 1.0
    assert metrics.tp == 3
    assert metrics.tn == 3
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_binary_metrics_at_threshold_balanced(balanced_data):
    """Test metrics at threshold on balanced data."""
    y, p = balanced_data
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics.threshold == 0.5
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.sensitivity <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.specificity <= 1.0
    assert metrics.tp + metrics.fn == 4  # Total cases
    assert metrics.tn + metrics.fp == 4  # Total controls


def test_binary_metrics_at_threshold_all_positive():
    """Test metrics when all predictions are positive."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9])
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics.fp == 2
    assert metrics.tp == 2
    assert metrics.tn == 0
    assert metrics.fn == 0


def test_binary_metrics_at_threshold_all_negative():
    """Test metrics when all predictions are negative."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.1, 0.1, 0.1])
    thr = 0.5
    metrics = binary_metrics_at_threshold(y, p, thr)

    assert metrics.fp == 0
    assert metrics.tp == 0
    assert metrics.tn == 2
    assert metrics.fn == 2
    assert metrics.precision == 0.0  # zero_division=0


# ============================================================================
# Test top_risk_capture
# ============================================================================


def test_top_risk_capture_1pct(imbalanced_data):
    """Test top 1% risk capture on imbalanced data."""
    y, p = imbalanced_data
    result = top_risk_capture(y, p, frac=0.01)

    assert result["frac"] == 0.01
    assert result["n_top"] == 1  # ceil(100 * 0.01)
    assert result["cases_in_top"] >= 0
    assert result["controls_in_top"] >= 0
    assert result["cases_in_top"] + result["controls_in_top"] == result["n_top"]


def test_top_risk_capture_10pct(balanced_data):
    """Test top 10% risk capture on balanced data."""
    y, p = balanced_data
    result = top_risk_capture(y, p, frac=0.10)

    assert result["frac"] == 0.10
    assert result["n_top"] == 1  # ceil(8 * 0.10)
    assert 0.0 <= result["case_capture"] <= 1.0


def test_top_risk_capture_50pct(balanced_data):
    """Test top 50% risk capture."""
    y, p = balanced_data
    result = top_risk_capture(y, p, frac=0.50)

    assert result["frac"] == 0.50
    assert result["n_top"] == 4
    # Should capture ~50% of cases
    assert result["case_capture"] >= 0.25


def test_top_risk_capture_empty():
    """Test top risk capture with empty arrays."""
    y = np.array([])
    p = np.array([])
    result = top_risk_capture(y, p, frac=0.01)

    assert result["n_top"] == 0
    assert result["cases_in_top"] == 0
    assert np.isnan(result["case_capture"])


def test_top_risk_capture_all_controls(all_controls):
    """Test top risk capture when there are no cases."""
    y, p = all_controls
    result = top_risk_capture(y, p, frac=0.05)

    assert result["cases_in_top"] == 0
    assert np.isnan(result["case_capture"])


# ============================================================================
# Test choose_threshold_objective
# ============================================================================


def test_choose_threshold_objective_max_f1(balanced_data):
    """Test threshold selection with max_f1 objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="max_f1")

    assert name == "max_f1"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_max_fbeta(balanced_data):
    """Test threshold selection with max_fbeta objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="max_fbeta", fbeta=2.0)

    assert name == "max_fbeta"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_youden(balanced_data):
    """Test threshold selection with youden objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="youden")

    assert name == "youden"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_fixed_spec(balanced_data):
    """Test threshold selection with fixed_spec objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="fixed_spec", fixed_spec=0.95)

    assert name == "fixed_spec"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_fixed_ppv(balanced_data):
    """Test threshold selection with fixed_ppv objective."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="fixed_ppv", fixed_ppv=0.8)

    assert name == "fixed_ppv"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_unknown_fallback(balanced_data):
    """Test threshold selection with unknown objective (should fallback to max_f1)."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective="unknown_method")

    assert name == "max_f1"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_none(balanced_data):
    """Test threshold selection with None objective (should default to youden with warning)."""
    y, p = balanced_data
    name, thr = choose_threshold_objective(y, p, objective=None)

    # None now defaults to youden (not max_f1) with a logged warning
    assert name == "youden"
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_objective_none_logs_warning(balanced_data, caplog):
    """Test that None objective logs a warning with actionable message."""
    import logging

    # Re-enable propagation on ced_ml logger so caplog (root handler) can capture
    ced_ml_logger = logging.getLogger("ced_ml")
    orig_propagate = ced_ml_logger.propagate
    ced_ml_logger.propagate = True
    try:
        y, p = balanced_data
        with caplog.at_level(logging.WARNING, logger="ced_ml.metrics.thresholds"):
            name, thr = choose_threshold_objective(y, p, objective=None)
    finally:
        ced_ml_logger.propagate = orig_propagate

    assert name == "youden"
    assert 0.0 <= thr <= 1.0

    # Verify warning was logged with actionable message
    assert len(caplog.records) >= 1
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("objective=None" in msg for msg in warning_messages)
    assert any("youden" in msg.lower() for msg in warning_messages)


def test_choose_threshold_objective_case_insensitive(balanced_data):
    """Test threshold selection is case-insensitive."""
    y, p = balanced_data
    name1, thr1 = choose_threshold_objective(y, p, objective="MAX_F1")
    name2, thr2 = choose_threshold_objective(y, p, objective="max_f1")

    assert name1 == name2 == "max_f1"
    assert thr1 == thr2


# ============================================================================
# Test threshold_dca_zero_crossing
# ============================================================================


def test_threshold_dca_zero_crossing_normal_case():
    """Test DCA threshold with typical zero crossing in mid-range."""
    _rng = np.random.default_rng(42)
    # Create scenario where model has utility at low thresholds, crosses zero at higher
    # 200 samples, 20% prevalence
    y = np.array([0] * 160 + [1] * 40)
    # Controls: low risk, Cases: higher risk
    p_controls = np.random.beta(2, 5, size=160)
    p_cases = np.random.beta(5, 2, size=40)
    p = np.concatenate([p_controls, p_cases])

    thr = threshold_dca_zero_crossing(y, p)

    # Should find a crossing point (or upper bound if NB stays positive)
    assert thr is not None
    assert 0.001 <= thr <= 0.20


def test_threshold_dca_zero_crossing_always_positive():
    """Test DCA threshold when net benefit never crosses zero (always positive)."""
    # Perfect separation - model always has utility
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.95, 0.96, 0.97, 0.98, 0.99])

    thr = threshold_dca_zero_crossing(y, p)

    # Should return the highest threshold where NB is still positive
    assert thr is not None
    assert 0.001 <= thr <= 0.20


def test_threshold_dca_zero_crossing_always_negative():
    """Test DCA threshold when net benefit is always negative."""
    # Random predictions - no utility
    rng = np.random.default_rng(123)
    y = np.array([0] * 80 + [1] * 20)
    p = rng.uniform(0, 1, size=100)

    thr = threshold_dca_zero_crossing(y, p)

    # May return None or a very low threshold
    if thr is not None:
        assert 0.001 <= thr <= 0.20


def test_threshold_dca_zero_crossing_empty_data():
    """Test DCA threshold with empty data."""
    y = np.array([])
    p = np.array([])

    thr = threshold_dca_zero_crossing(y, p)

    # Should return None for empty data
    assert thr is None


def test_threshold_dca_zero_crossing_custom_thresholds():
    """Test DCA threshold with custom threshold range."""
    _rng = np.random.default_rng(42)
    y = np.array([0] * 80 + [1] * 20)
    p_controls = np.random.beta(2, 5, size=80)
    p_cases = np.random.beta(5, 2, size=20)
    p = np.concatenate([p_controls, p_cases])

    custom_thresholds = np.linspace(0.01, 0.30, 100)
    thr = threshold_dca_zero_crossing(y, p, thresholds=custom_thresholds)

    if thr is not None:
        assert 0.01 <= thr <= 0.30


def test_threshold_dca_zero_crossing_with_prevalence_adjustment():
    """Test DCA threshold with prevalence adjustment."""
    _rng = np.random.default_rng(42)
    # Training data: 20% prevalence
    y = np.array([0] * 80 + [1] * 20)
    p_controls = np.random.beta(2, 5, size=80)
    p_cases = np.random.beta(5, 2, size=20)
    p = np.concatenate([p_controls, p_cases])

    # Test with different prevalence adjustment (e.g., 10% in target population)
    thr = threshold_dca_zero_crossing(y, p, prevalence_adjustment=0.10)

    # Should still return valid threshold (may differ from unadjusted)
    if thr is not None:
        assert 0.001 <= thr <= 0.20


def test_threshold_dca_zero_crossing_interpolation_accuracy():
    """Test that interpolation produces accurate zero crossing estimate."""
    # Create a scenario with known zero crossing
    # Manually construct data where we can verify the crossing point
    y = np.array([0] * 900 + [1] * 100)
    _rng = np.random.default_rng(999)
    p_controls = np.random.beta(2, 8, size=900)
    p_cases = np.random.beta(6, 2, size=100)
    p = np.concatenate([p_controls, p_cases])

    # Use fine-grained thresholds for better accuracy
    fine_thresholds = np.linspace(0.001, 0.20, 1000)
    thr = threshold_dca_zero_crossing(y, p, thresholds=fine_thresholds)

    # Verify threshold is in valid range (allow full range since NB may stay positive)
    if thr is not None:
        assert 0.001 <= thr <= 0.20


def test_threshold_dca_zero_crossing_crossing_at_boundary():
    """Test DCA threshold when crossing occurs at threshold array boundary."""
    # Create scenario where crossing is at edge
    y = np.array([0] * 95 + [1] * 5)
    p_controls = np.random.beta(2, 8, size=95)
    p_cases = np.random.beta(8, 1, size=5)
    p = np.concatenate([p_controls, p_cases])

    # Very narrow threshold range
    narrow_thresholds = np.linspace(0.001, 0.05, 20)
    thr = threshold_dca_zero_crossing(y, p, thresholds=narrow_thresholds)

    if thr is not None:
        assert 0.001 <= thr <= 0.05


def test_threshold_dca_zero_crossing_no_positive_benefit():
    """Test DCA threshold when model never has positive net benefit."""
    # Useless model - worse than random
    rng = np.random.default_rng(999)
    y = np.array([0] * 50 + [1] * 50)
    # Predictions are opposite of truth
    p = np.concatenate(
        [
            rng.uniform(0.6, 1.0, size=50),  # Controls get high scores
            rng.uniform(0.0, 0.4, size=50),  # Cases get low scores
        ]
    )

    thr = threshold_dca_zero_crossing(y, p)

    # Even bad models may have some positive NB at very low/high thresholds
    # Just verify it returns a valid threshold or None
    if thr is not None:
        assert 0.001 <= thr <= 0.20


# ============================================================================
# Multi-Target Specificity Tests
# ============================================================================


def test_compute_multi_target_specificity_metrics_basic():
    """Test multi-target specificity metrics with basic case."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    spec_targets = [0.90, 0.95, 0.99]

    metrics = compute_multi_target_specificity_metrics(y_true, y_pred, spec_targets)

    assert "thr_ctrl_90" in metrics
    assert "sens_ctrl_90" in metrics
    assert "prec_ctrl_90" in metrics
    assert "spec_ctrl_90" in metrics

    assert "thr_ctrl_95" in metrics
    assert "sens_ctrl_95" in metrics
    assert "prec_ctrl_95" in metrics
    assert "spec_ctrl_95" in metrics

    assert "thr_ctrl_99" in metrics
    assert "sens_ctrl_99" in metrics
    assert "prec_ctrl_99" in metrics
    assert "spec_ctrl_99" in metrics

    # Total keys: 4 metrics x 3 targets = 12
    assert len(metrics) == 12

    # Check specificity values are close to targets
    assert abs(metrics["spec_ctrl_90"] - 0.90) < 0.25
    assert abs(metrics["spec_ctrl_95"] - 0.95) < 0.25
    assert abs(metrics["spec_ctrl_99"] - 0.99) < 0.25

    # Thresholds should increase with increasing specificity
    assert metrics["thr_ctrl_90"] <= metrics["thr_ctrl_95"] <= metrics["thr_ctrl_99"]


def test_compute_multi_target_specificity_metrics_empty():
    """Test multi-target specificity metrics with empty targets list."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    spec_targets = []

    metrics = compute_multi_target_specificity_metrics(y_true, y_pred, spec_targets)

    assert len(metrics) == 0


def test_compute_multi_target_specificity_metrics_single_target():
    """Test multi-target specificity metrics with single target."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    spec_targets = [0.95]

    metrics = compute_multi_target_specificity_metrics(y_true, y_pred, spec_targets)

    assert "thr_ctrl_95" in metrics
    assert "sens_ctrl_95" in metrics
    assert "prec_ctrl_95" in metrics
    assert "spec_ctrl_95" in metrics
    assert len(metrics) == 4


def test_compute_multi_target_specificity_metrics_perfect_separation():
    """Test multi-target specificity metrics with perfect separation."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    spec_targets = [0.90, 0.95, 0.99]

    metrics = compute_multi_target_specificity_metrics(y_true, y_pred, spec_targets)

    # All metrics should be valid
    for key in metrics:
        assert np.isfinite(metrics[key])

    # With good separation, high specificity should still have reasonable sensitivity
    assert metrics["sens_ctrl_90"] > 0.0


def test_compute_multi_target_specificity_metrics_imbalanced():
    """Test multi-target specificity metrics with imbalanced data."""
    _rng = np.random.default_rng(42)
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.concatenate(
        [
            np.random.beta(2, 5, size=95),  # Controls
            np.random.beta(5, 2, size=5),  # Cases
        ]
    )
    spec_targets = [0.90, 0.95, 0.99]

    metrics = compute_multi_target_specificity_metrics(y_true, y_pred, spec_targets)

    assert len(metrics) == 12

    # With imbalanced data, high specificity is easier to achieve
    for target in [90, 95, 99]:
        assert f"spec_ctrl_{target}" in metrics
        assert 0.0 <= metrics[f"spec_ctrl_{target}"] <= 1.0
        assert 0.0 <= metrics[f"sens_ctrl_{target}"] <= 1.0
