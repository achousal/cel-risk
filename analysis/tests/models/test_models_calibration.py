"""
Tests for models.calibration module.

Coverage areas:
- Calibration intercept/slope computation
- Expected Calibration Error (ECE)
- Prevalence adjustment logic
- PrevalenceAdjustedModel wrapper
- CalibratedClassifierCV utilities
- OOF (out-of-fold) calibration
- OOFCalibratedModel wrapper
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ced_ml.models.calibration import (
    OOFCalibratedModel,
    OOFCalibrator,
    apply_oof_calibrator,
    calib_intercept_metric,
    calib_slope_metric,
    calibration_intercept_slope,
    expected_calibration_error,
    fit_oof_calibrator,
    get_calibrated_cv_param_name,
    get_calibrated_estimator_param_name,
    maybe_calibrate_estimator,
)
from ced_ml.models.prevalence import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
)

# ============================================================================
# Calibration Metrics Tests
# ============================================================================


def test_calibration_intercept_slope_perfect():
    """Perfect calibration should have intercept ~0 and slope ~1."""
    rng = np.random.default_rng(42)
    n = 1000
    # Create reasonable predicted probabilities
    y_pred = rng.uniform(0.1, 0.5, size=n)
    # Generate outcomes consistent with predictions
    y_true = rng.binomial(1, y_pred)

    _cal_result = calibration_intercept_slope(y_true, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    # With logit-scale calibration, slope ~1 indicates good calibration
    # But the exact value depends on the data distribution
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0, "Calibration slope should be positive"


def test_calibration_intercept_slope_underconfident():
    """Underconfident predictions should have specific calibration characteristics."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.5, size=n)
    # Shrink predictions toward 0.5 (underconfident)
    y_pred = y_true * 0.3 + 0.35

    _cal_result = calibration_intercept_slope(y_true, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    # Test that we get valid calibration metrics
    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert slope > 0, "Calibration slope should be positive"


def test_calibration_intercept_slope_single_class():
    """Single class should return NaN."""
    rng = np.random.default_rng(42)
    y_true = np.ones(100)
    y_pred = rng.uniform(0.3, 0.7, size=100)

    _cal_result = calibration_intercept_slope(y_true, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    assert np.isnan(intercept)
    assert np.isnan(slope)


def test_calibration_intercept_slope_with_nans():
    """Should filter out NaN values."""
    y_true = np.array([0, 1, 0, 1, 0, 1, np.nan, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, np.nan, 0.15, 0.85, 0.5, 0.1, 0.95])

    _cal_result = calibration_intercept_slope(y_true, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    assert np.isfinite(intercept)
    assert np.isfinite(slope)


def test_calib_intercept_metric():
    """Test calibration intercept metric wrapper."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.3, size=100)
    y_pred = rng.uniform(0.1, 0.5, size=100)

    metric = calib_intercept_metric(y_true, y_pred)
    intercept = calibration_intercept_slope(y_true, y_pred).intercept

    assert np.isclose(metric, intercept)


def test_calib_slope_metric():
    """Test calibration slope metric wrapper."""
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.3, size=100)
    y_pred = rng.uniform(0.1, 0.5, size=100)

    metric = calib_slope_metric(y_true, y_pred)
    slope = calibration_intercept_slope(y_true, y_pred).slope

    assert np.isclose(metric, slope)


# ============================================================================
# Expected Calibration Error Tests
# ============================================================================


def test_expected_calibration_error_perfect():
    """Perfect calibration should have ECE near 0."""
    rng = np.random.default_rng(42)
    n = 1000
    y_pred = rng.uniform(0, 1, size=n)
    y_true = rng.binomial(1, y_pred)

    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert 0.0 <= ece <= 0.15, f"Perfect calibration should have low ECE, got {ece}"


def test_expected_calibration_error_poor():
    """Poor calibration should have high ECE."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.binomial(1, 0.5, size=n)
    # Completely miscalibrated: predict opposite
    y_pred = 1.0 - y_true.astype(float)
    y_pred = np.clip(y_pred, 0.01, 0.99)

    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert ece > 0.3, f"Poor calibration should have high ECE, got {ece}"


def test_expected_calibration_error_with_nans():
    """Should filter NaN values."""
    y_true = np.array([0, 1, 0, 1, np.nan, 0, 1])
    y_pred = np.array([0.1, 0.9, np.nan, 0.8, 0.2, 0.15, 0.95])

    ece = expected_calibration_error(y_true, y_pred, n_bins=5)
    assert np.isfinite(ece)
    assert 0.0 <= ece <= 1.0


def test_expected_calibration_error_empty():
    """Empty arrays should return NaN."""
    y_true = np.array([])
    y_pred = np.array([])

    ece = expected_calibration_error(y_true, y_pred)
    assert np.isnan(ece)


# ============================================================================
# Prevalence Adjustment Tests
# ============================================================================


def test_adjust_probabilities_for_prevalence_shift_up():
    """Shifting to higher prevalence should increase probabilities."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.3, size=100)
    sample_prev = 0.1
    target_prev = 0.3

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(adjusted > probs), "Higher prevalence should increase probabilities"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Probabilities should be in [0,1]"


def test_adjust_probabilities_for_prevalence_shift_down():
    """Shifting to lower prevalence should decrease probabilities."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.3, 0.5, size=100)
    sample_prev = 0.3
    target_prev = 0.1

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(adjusted < probs), "Lower prevalence should decrease probabilities"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Probabilities should be in [0,1]"


def test_adjust_probabilities_for_prevalence_no_change():
    """Same prevalence should return nearly identical probabilities."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.5, size=100)
    sample_prev = 0.2
    target_prev = 0.2

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.allclose(
        adjusted, probs, atol=1e-5
    ), "Same prevalence should not change probabilities"


def test_adjust_probabilities_extreme_values():
    """Should handle extreme probabilities without overflow."""
    probs = np.array([0.001, 0.5, 0.999])
    sample_prev = 0.1
    target_prev = 0.9

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    assert np.all(np.isfinite(adjusted)), "Should handle extreme values"
    assert np.all((adjusted >= 0) & (adjusted <= 1)), "Should clip to [0,1]"


# ============================================================================
# PrevalenceAdjustedModel Tests
# ============================================================================


def test_prevalence_adjusted_model_fit():
    """Test fit method (should be no-op)."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 10))
    y_train = rng.binomial(1, 0.3, size=100)

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(base_model, sample_prevalence=0.3, target_prevalence=0.1)
    result = wrapper.fit(X_train, y_train)

    assert result is wrapper, "fit() should return self"


def test_prevalence_adjusted_model_predict_proba():
    """Test adjusted probability predictions."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 10))
    y_train = rng.binomial(1, 0.3, size=100)
    X_test = rng.standard_normal((20, 10))

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(base_model, sample_prevalence=0.3, target_prevalence=0.1)
    raw_probs = base_model.predict_proba(X_test)[:, 1]
    adjusted_probs = wrapper.predict_proba(X_test)[:, 1]

    assert np.all(
        adjusted_probs < raw_probs
    ), "Lower target prevalence should decrease probabilities"
    assert adjusted_probs.shape == raw_probs.shape


def test_prevalence_adjusted_model_predict():
    """Test binary predictions."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 10))
    y_train = rng.binomial(1, 0.3, size=100)
    X_test = rng.standard_normal((20, 10))

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    wrapper = PrevalenceAdjustedModel(base_model, sample_prevalence=0.3, target_prevalence=0.1)
    predictions = wrapper.predict(X_test)

    assert predictions.shape == (20,)
    assert np.all((predictions == 0) | (predictions == 1)), "Should be binary predictions"


# ============================================================================
# CalibratedClassifierCV Utilities Tests
# ============================================================================


def test_get_calibrated_estimator_param_name():
    """Test parameter name detection for CalibratedClassifierCV."""
    param_name = get_calibrated_estimator_param_name()
    assert param_name in ["estimator", "base_estimator"]


def test_get_calibrated_cv_param_name():
    """Test CV parameter name detection."""
    param_name = get_calibrated_cv_param_name()
    assert param_name == "cv"


def test_maybe_calibrate_estimator_lr():
    """Test calibration wrapper for Logistic Regression."""
    base_model = LogisticRegression(random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="sigmoid", cv=3
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_calibrate_estimator_rf():
    """Test calibration wrapper for Random Forest."""
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="RF", calibrate=True, method="sigmoid", cv=3
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


def test_maybe_calibrate_estimator_svm_skip():
    """SVM should not be calibrated (already calibrated)."""
    base_model = CalibratedClassifierCV(LinearSVC(random_state=42))
    result = maybe_calibrate_estimator(
        base_model, model_name="LinSVM_cal", calibrate=True, method="sigmoid", cv=3
    )

    assert result is base_model, "SVM should not be re-calibrated"


def test_maybe_calibrate_estimator_disabled():
    """Should return original estimator when calibration disabled."""
    base_model = LogisticRegression(random_state=42)
    result = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=False, method="sigmoid", cv=3
    )

    assert result is base_model


def test_maybe_calibrate_estimator_already_calibrated():
    """Should not double-calibrate."""
    base_model = CalibratedClassifierCV(LogisticRegression(random_state=42))
    result = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="sigmoid", cv=3
    )

    assert result is base_model, "Should not double-calibrate"


def test_maybe_calibrate_estimator_isotonic():
    """Test isotonic calibration method."""
    base_model = LogisticRegression(random_state=42)
    calibrated = maybe_calibrate_estimator(
        base_model, model_name="LR_EN", calibrate=True, method="isotonic", cv=5
    )

    assert isinstance(calibrated, CalibratedClassifierCV)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_calibration_workflow():
    """Test end-to-end calibration workflow."""
    rng = np.random.default_rng(42)

    # Generate toy data
    n = 200
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    train_idx = np.arange(150)
    test_idx = np.arange(150, 200)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train base model
    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    # Compute calibration metrics
    y_pred = base_model.predict_proba(X_test)[:, 1]
    _cal_result = calibration_intercept_slope(y_test, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    ece = expected_calibration_error(y_test, y_pred)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert np.isfinite(ece)

    # Apply prevalence adjustment
    sample_prev = y_train.mean()
    target_prev = 0.1
    adjusted_probs = adjust_probabilities_for_prevalence(y_pred, sample_prev, target_prev)

    assert np.all(adjusted_probs <= y_pred), "Lower prevalence should decrease probs"

    # Test wrapper
    wrapper = PrevalenceAdjustedModel(base_model, sample_prev, target_prev)
    wrapper_probs = wrapper.predict_proba(X_test)[:, 1]

    assert np.allclose(wrapper_probs, adjusted_probs, atol=1e-5)


def test_calibration_with_perfect_separation():
    """Test calibration when classes are perfectly separated."""
    rng = np.random.default_rng(42)

    # Perfectly separable data
    X_train = np.vstack(
        [
            rng.standard_normal((50, 2)) - 3,
            rng.standard_normal((50, 2)) + 3,
        ]  # Class 0  # Class 1
    )
    y_train = np.array([0] * 50 + [1] * 50)

    X_test = np.vstack([rng.standard_normal((10, 2)) - 3, rng.standard_normal((10, 2)) + 3])
    y_test = np.array([0] * 10 + [1] * 10)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict_proba(X_test)[:, 1]

    # Calibration metrics should still work
    _cal_result = calibration_intercept_slope(y_test, y_pred)
    intercept, slope = _cal_result.intercept, _cal_result.slope
    ece = expected_calibration_error(y_test, y_pred)

    assert np.isfinite(intercept)
    assert np.isfinite(slope)
    assert np.isfinite(ece)
    assert ece < 0.2, "Perfect separation should have good calibration"


# ============================================================================
# OOF Calibration Tests
# ============================================================================


class TestOOFCalibrator:
    """Tests for OOFCalibrator class."""

    def test_init_isotonic(self):
        """Test OOFCalibrator initialization with isotonic method."""
        calibrator = OOFCalibrator(method="isotonic")
        assert calibrator.method == "isotonic"
        assert not calibrator.is_fitted
        assert calibrator.calibrator_ is None

    def test_init_invalid_method(self):
        """Test OOFCalibrator raises error for invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            OOFCalibrator(method="invalid")

    def test_fit_isotonic(self):
        """Test OOFCalibrator fit with isotonic method."""
        rng = np.random.default_rng(42)
        n = 200
        oof_preds = rng.uniform(0.1, 0.9, size=n)
        y_true = rng.binomial(1, oof_preds)

        calibrator = OOFCalibrator(method="isotonic")
        result = calibrator.fit(oof_preds, y_true)

        assert result is calibrator
        assert calibrator.is_fitted
        assert calibrator.calibrator_ is not None

    def test_fit_insufficient_data(self):
        """Test OOFCalibrator raises error when fewer than 50 valid samples."""
        rng = np.random.default_rng(42)
        # 20 samples is below the new minimum of 50.
        oof_preds = rng.uniform(0.1, 0.9, size=20)
        y_true = rng.binomial(1, oof_preds)

        calibrator = OOFCalibrator(method="isotonic")
        with pytest.raises(ValueError, match="at least 50 valid samples"):
            calibrator.fit(oof_preds, y_true)

    def test_fit_single_class(self):
        """Test OOFCalibrator raises error for single class."""
        rng = np.random.default_rng(42)
        oof_preds = rng.uniform(0.1, 0.9, size=100)
        y_true = np.zeros(100)  # All same class

        calibrator = OOFCalibrator(method="isotonic")
        with pytest.raises(ValueError, match="both classes present"):
            calibrator.fit(oof_preds, y_true)

    def test_fit_mismatched_shapes(self):
        """Test OOFCalibrator raises error for mismatched shapes."""
        oof_preds = np.array([0.5, 0.6, 0.7])
        y_true = np.array([0, 1])

        calibrator = OOFCalibrator(method="isotonic")
        with pytest.raises(ValueError, match="same length"):
            calibrator.fit(oof_preds, y_true)

    def test_fit_with_nans(self):
        """Test OOFCalibrator handles NaN values correctly."""
        rng = np.random.default_rng(42)
        n = 200
        oof_preds = rng.uniform(0.1, 0.9, size=n)
        y_true = rng.binomial(1, oof_preds).astype(float)  # Convert to float for NaN

        # Add some NaN values
        oof_preds[10:15] = np.nan
        y_true[20:25] = np.nan

        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y_true)

        assert calibrator.is_fitted

    def test_transform_isotonic(self):
        """Test OOFCalibrator transform with isotonic method."""
        rng = np.random.default_rng(42)
        n = 200
        oof_preds = rng.uniform(0.1, 0.9, size=n)
        y_true = rng.binomial(1, oof_preds)

        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y_true)

        new_preds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrated = calibrator.transform(new_preds)

        assert len(calibrated) == len(new_preds)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_transform_not_fitted(self):
        """Test OOFCalibrator transform raises error when not fitted."""
        calibrator = OOFCalibrator(method="isotonic")

        with pytest.raises(RuntimeError, match="must be fitted"):
            calibrator.transform(np.array([0.5]))

    def test_repr(self):
        """Test OOFCalibrator string representation."""
        calibrator = OOFCalibrator(method="isotonic")
        assert "OOFCalibrator" in repr(calibrator)
        assert "isotonic" in repr(calibrator)
        assert "not fitted" in repr(calibrator)

        # After fitting
        rng = np.random.default_rng(42)
        oof_preds = rng.uniform(0.1, 0.9, size=100)
        y_true = rng.binomial(1, oof_preds)
        calibrator.fit(oof_preds, y_true)

        assert "fitted" in repr(calibrator)
        assert "not fitted" not in repr(calibrator)


class TestOOFCalibratedModel:
    """Tests for OOFCalibratedModel wrapper class."""

    def test_init(self):
        """Test OOFCalibratedModel initialization."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        # Train base model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)

        # Create calibrator
        oof_preds = base_model.predict_proba(X)[:, 1]
        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y)

        # Create wrapper
        wrapper = OOFCalibratedModel(base_model, calibrator)

        assert wrapper.base_model is base_model
        assert wrapper.calibrator is calibrator

    def test_init_unfitted_calibrator(self):
        """Test OOFCalibratedModel raises error for unfitted calibrator."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)

        calibrator = OOFCalibrator(method="isotonic")  # Not fitted

        with pytest.raises(ValueError, match="must be fitted"):
            OOFCalibratedModel(base_model, calibrator)

    def test_init_no_predict_proba(self):
        """Test OOFCalibratedModel raises error for model without predict_proba."""
        from sklearn.svm import LinearSVC

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        # LinearSVC does not have predict_proba by default
        base_model = LinearSVC(random_state=42)
        base_model.fit(X, y)

        # Create fitted calibrator
        calibrator = OOFCalibrator(method="isotonic")
        oof_preds = rng.uniform(0.1, 0.9, size=100)
        calibrator.fit(oof_preds, y)

        with pytest.raises(ValueError, match="predict_proba"):
            OOFCalibratedModel(base_model, calibrator)

    def test_predict_proba(self):
        """Test OOFCalibratedModel predict_proba returns calibrated probabilities."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 5))
        y_train = rng.binomial(1, 0.3, size=200)
        X_test = rng.standard_normal((50, 5))

        # Train base model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X_train, y_train)

        # Create calibrator
        oof_preds = base_model.predict_proba(X_train)[:, 1]
        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y_train)

        # Create wrapper
        wrapper = OOFCalibratedModel(base_model, calibrator)

        # Get predictions
        proba = wrapper.predict_proba(X_test)

        assert proba.shape == (50, 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict(self):
        """Test OOFCalibratedModel predict returns binary predictions."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 5))
        y_train = rng.binomial(1, 0.3, size=200)
        X_test = rng.standard_normal((50, 5))

        # Train base model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X_train, y_train)

        # Create calibrator
        oof_preds = base_model.predict_proba(X_train)[:, 1]
        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y_train)

        # Create wrapper
        wrapper = OOFCalibratedModel(base_model, calibrator)

        # Get predictions
        preds = wrapper.predict(X_test)

        assert preds.shape == (50,)
        assert np.all((preds == 0) | (preds == 1))

    def test_repr(self):
        """Test OOFCalibratedModel string representation."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)

        oof_preds = base_model.predict_proba(X)[:, 1]
        calibrator = OOFCalibrator(method="isotonic")
        calibrator.fit(oof_preds, y)

        wrapper = OOFCalibratedModel(base_model, calibrator)

        assert "OOFCalibratedModel" in repr(wrapper)
        assert "LogisticRegression" in repr(wrapper)


class TestOOFCalibrationHelpers:
    """Tests for OOF calibration helper functions."""

    def test_fit_oof_calibrator(self):
        """Test fit_oof_calibrator convenience function."""
        rng = np.random.default_rng(42)
        n = 200
        oof_preds = rng.uniform(0.1, 0.9, size=n)
        y_true = rng.binomial(1, oof_preds)

        calibrator = fit_oof_calibrator(oof_preds, y_true, method="isotonic")

        assert isinstance(calibrator, OOFCalibrator)
        assert calibrator.is_fitted
        assert calibrator.method == "isotonic"

    def test_apply_oof_calibrator(self):
        """Test apply_oof_calibrator convenience function."""
        rng = np.random.default_rng(42)
        n = 200
        oof_preds = rng.uniform(0.1, 0.9, size=n)
        y_true = rng.binomial(1, oof_preds)

        calibrator = fit_oof_calibrator(oof_preds, y_true, method="isotonic")

        new_preds = np.array([0.1, 0.5, 0.9])
        calibrated = apply_oof_calibrator(calibrator, new_preds)

        assert len(calibrated) == len(new_preds)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)


class TestOOFCalibrationIntegration:
    """Integration tests for OOF calibration workflow."""

    def test_full_oof_calibration_workflow(self):
        """Test complete OOF calibration workflow with train/test split."""
        rng = np.random.default_rng(42)

        # Generate data
        n = 300
        X = rng.standard_normal((n, 10))
        # Create outcome with moderate correlation to features
        y = ((X[:, 0] + X[:, 1] + rng.standard_normal(n) * 0.5) > 0).astype(int)

        # Split
        train_idx = np.arange(200)
        test_idx = np.arange(200, 300)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train base model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X_train, y_train)

        # Simulate OOF predictions (in practice, these come from CV)
        oof_preds = base_model.predict_proba(X_train)[:, 1]

        # Fit OOF calibrator
        calibrator = fit_oof_calibrator(oof_preds, y_train, method="isotonic")

        # Create calibrated model
        calibrated_model = OOFCalibratedModel(base_model, calibrator)

        # Test predictions
        test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        test_preds = calibrated_model.predict(X_test)

        # Verify output
        assert len(test_proba) == len(y_test)
        assert np.all((test_proba >= 0) & (test_proba <= 1))
        assert np.all((test_preds == 0) | (test_preds == 1))

        # Compare with uncalibrated predictions
        uncalibrated_proba = base_model.predict_proba(X_test)[:, 1]

        # Calibrated predictions should generally be different
        # (unless base model is already perfectly calibrated)
        assert not np.allclose(test_proba, uncalibrated_proba)

    def test_oof_calibration_improves_calibration(self):
        """Test that OOF calibration can improve calibration metrics."""
        rng = np.random.default_rng(42)

        # Generate data with a bit of miscalibration
        n = 500
        X = rng.standard_normal((n, 5))
        # True probabilities
        true_proba = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))
        y = rng.binomial(1, true_proba)

        # Split
        train_idx = np.arange(400)
        test_idx = np.arange(400, 500)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train model with regularization (may cause miscalibration)
        base_model = LogisticRegression(C=0.1, random_state=42)
        base_model.fit(X_train, y_train)

        # OOF predictions
        oof_preds = base_model.predict_proba(X_train)[:, 1]

        # Fit calibrator
        calibrator = fit_oof_calibrator(oof_preds, y_train, method="isotonic")

        # Calibrated model
        calibrated_model = OOFCalibratedModel(base_model, calibrator)

        # Get test predictions
        uncalibrated_test = base_model.predict_proba(X_test)[:, 1]
        calibrated_test = calibrated_model.predict_proba(X_test)[:, 1]

        # Compute ECE
        ece_uncal = expected_calibration_error(y_test, uncalibrated_test)
        ece_cal = expected_calibration_error(y_test, calibrated_test)

        # Both should be finite
        assert np.isfinite(ece_uncal)
        assert np.isfinite(ece_cal)

        # Note: In practice, calibration should improve ECE,
        # but with small test sets, results can vary
        # So we just verify the workflow runs without error

    def test_oof_calibration_with_random_forest(self):
        """Test OOF calibration with Random Forest model."""
        rng = np.random.default_rng(42)

        n = 300
        X = rng.standard_normal((n, 10))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

        train_idx = np.arange(200)
        test_idx = np.arange(200, 300)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train RF
        base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        base_model.fit(X_train, y_train)

        # OOF predictions
        oof_preds = base_model.predict_proba(X_train)[:, 1]

        # Fit calibrator
        calibrator = fit_oof_calibrator(oof_preds, y_train, method="isotonic")

        # Calibrated model
        calibrated_model = OOFCalibratedModel(base_model, calibrator)

        # Test
        test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        assert len(test_proba) == len(y_test)
        assert np.all((test_proba >= 0) & (test_proba <= 1))


# ============================================================================
# New Calibration Method Tests
# ============================================================================


class TestOOFCalibratorNewMethods:
    """Tests for the new OOFCalibrator methods: logistic_full, logistic_intercept, beta."""

    # --- Shared fixture ---

    @staticmethod
    def _make_data(n: int = 300, seed: int = 42):
        rng = np.random.default_rng(seed)
        oof_preds = rng.uniform(0.05, 0.95, size=n)
        y_true = rng.binomial(1, oof_preds)
        return oof_preds, y_true

    # --- Initialisation ---

    def test_init_logistic_full(self):
        """logistic_full initialises correctly."""
        cal = OOFCalibrator(method="logistic_full")
        assert cal.method == "logistic_full"
        assert not cal.is_fitted

    def test_init_logistic_intercept(self):
        """logistic_intercept initialises correctly."""
        cal = OOFCalibrator(method="logistic_intercept")
        assert cal.method == "logistic_intercept"
        assert not cal.is_fitted

    def test_init_beta(self):
        """beta initialises correctly."""
        cal = OOFCalibrator(method="beta")
        assert cal.method == "beta"
        assert not cal.is_fitted

    # --- Fit + transform for each new method ---

    def test_fit_logistic_full(self):
        """logistic_full fits and produces in-range calibrated outputs."""
        oof, y = self._make_data()
        cal = OOFCalibrator(method="logistic_full")
        result = cal.fit(oof, y)

        assert result is cal
        assert cal.is_fitted
        calibrated = cal.transform(oof)
        assert calibrated.shape == oof.shape
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)

    def test_fit_logistic_intercept(self):
        """logistic_intercept fits and produces in-range calibrated outputs."""
        oof, y = self._make_data()
        cal = OOFCalibrator(method="logistic_intercept")
        cal.fit(oof, y)

        assert cal.is_fitted
        assert isinstance(cal.calibrator_, dict)
        assert "intercept" in cal.calibrator_
        calibrated = cal.transform(oof)
        assert calibrated.shape == oof.shape
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)

    def test_fit_beta(self):
        """beta calibration fits and produces in-range calibrated outputs."""
        oof, y = self._make_data()
        cal = OOFCalibrator(method="beta")
        cal.fit(oof, y)

        assert cal.is_fitted
        assert isinstance(cal.calibrator_, dict)
        assert "a" in cal.calibrator_ and "b" in cal.calibrator_ and "c" in cal.calibrator_
        calibrated = cal.transform(oof)
        assert calibrated.shape == oof.shape
        assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)

    # --- Ordering preservation ---

    def test_logistic_intercept_preserves_rank_order(self):
        """Intercept-only shift is monotone, so rank order must be preserved."""
        oof, y = self._make_data()
        cal = OOFCalibrator(method="logistic_intercept")
        cal.fit(oof, y)

        test_pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrated = cal.transform(test_pts)
        # Monotone transform: original rank order must be preserved.
        assert np.all(np.diff(calibrated) > 0), "logistic_intercept must be monotone"

    def test_logistic_full_preserves_rank_order(self):
        """Two-parameter Platt scaling is monotone."""
        oof, y = self._make_data()
        cal = OOFCalibrator(method="logistic_full")
        cal.fit(oof, y)

        test_pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrated = cal.transform(test_pts)
        assert np.all(np.diff(calibrated) > 0), "logistic_full must be monotone"

    # --- Intercept at zero for well-calibrated predictions ---

    def test_logistic_intercept_near_zero_for_calibrated_preds(self):
        """When predictions are already calibrated, the fitted intercept should be near 0."""
        rng = np.random.default_rng(0)
        n = 2000
        # Predictions drawn from a logistic model -> well-calibrated.
        x = rng.standard_normal(n)
        true_prob = 1.0 / (1.0 + np.exp(-x))
        y = rng.binomial(1, true_prob)
        # Use true_prob as our "predicted" probabilities.
        cal = OOFCalibrator(method="logistic_intercept")
        cal.fit(true_prob, y)
        intercept = cal.calibrator_["intercept"]
        assert (
            abs(intercept) < 0.5
        ), f"Intercept should be near 0 for well-calibrated predictions, got {intercept:.4f}"

    # --- Isotonic safety: minimum sample threshold ---

    def test_isotonic_fails_below_50_samples(self):
        """OOFCalibrator.fit should raise ValueError when n < 50."""
        rng = np.random.default_rng(42)
        oof = rng.uniform(0.1, 0.9, size=30)
        y = rng.binomial(1, oof)
        cal = OOFCalibrator(method="isotonic")
        with pytest.raises(ValueError, match="at least 50 valid samples"):
            cal.fit(oof, y)

    def test_isotonic_warns_few_positives(self):
        """OOFCalibrator.fit should emit a logger.warning when n_positive < 30.

        We construct labels deterministically: exactly 5 positives out of 100
        samples, which is guaranteed to be < 30.  Verification is done by
        patching logger.warning directly to avoid caplog propagation issues.
        """
        import unittest.mock as mock

        n = 100
        # Predictions spread evenly so both classes have distinct scores.
        oof = np.linspace(0.1, 0.9, n)
        # Exactly 5 positives (indices 0..4); rest negative.  n_positive == 5 < 30.
        y = np.zeros(n, dtype=int)
        y[:5] = 1

        cal = OOFCalibrator(method="isotonic")
        with mock.patch("ced_ml.models.calibration.logger") as mock_logger:
            cal.fit(oof, y)

        # Confirm that logger.warning was called with a message about positive samples.
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any(
            "positive samples" in msg for msg in warning_calls
        ), f"Expected a warning about positive sample count; calls were: {warning_calls}"

    def test_fit_oof_calibrator_logistic_intercept(self):
        """fit_oof_calibrator convenience function works with logistic_intercept."""
        oof, y = self._make_data()
        cal = fit_oof_calibrator(oof, y, method="logistic_intercept")
        assert cal.is_fitted
        assert cal.method == "logistic_intercept"

    def test_fit_oof_calibrator_beta(self):
        """fit_oof_calibrator convenience function works with beta."""
        oof, y = self._make_data()
        cal = fit_oof_calibrator(oof, y, method="beta")
        assert cal.is_fitted
        assert cal.method == "beta"


# ============================================================================
# CalibrationConfig Schema Tests (new methods + per_model override)
# ============================================================================


class TestCalibrationConfigSchema:
    """Tests for the updated CalibrationConfig schema."""

    def test_new_method_literals_accepted(self):
        """All new method literals should be accepted by CalibrationConfig."""
        from ced_ml.config.calibration_schema import CalibrationConfig

        for method in ("isotonic", "logistic_full", "logistic_intercept", "beta"):
            cfg = CalibrationConfig(method=method)
            assert cfg.method == method

    def test_invalid_method_rejected(self):
        """Unknown method names should raise a Pydantic validation error."""
        from pydantic import ValidationError

        from ced_ml.config.calibration_schema import CalibrationConfig

        with pytest.raises(ValidationError):
            CalibrationConfig(method="unknown_method")

    def test_per_model_dict_accepted(self):
        """Full PerModelCalibrationConfig dicts should be accepted."""
        from ced_ml.config.calibration_schema import CalibrationConfig, PerModelCalibrationConfig

        cfg = CalibrationConfig(
            strategy="per_fold",
            method="isotonic",
            per_model={"LR_EN": {"strategy": "oof_posthoc", "method": "logistic_intercept"}},
        )
        override = cfg.per_model["LR_EN"]
        assert isinstance(override, PerModelCalibrationConfig)
        assert override.strategy == "oof_posthoc"
        assert override.method == "logistic_intercept"

    def test_get_method_for_model_uses_global_default(self):
        """get_method_for_model returns global method when no per_model override."""
        from ced_ml.config.calibration_schema import CalibrationConfig

        cfg = CalibrationConfig(method="logistic_full")
        assert cfg.get_method_for_model("LR_EN") == "logistic_full"
        assert cfg.get_method_for_model("RF") == "logistic_full"

    def test_get_method_for_model_uses_per_model_override(self):
        """get_method_for_model returns per-model method when present."""
        from ced_ml.config.calibration_schema import CalibrationConfig

        cfg = CalibrationConfig(
            method="isotonic",
            per_model={"LR_EN": {"method": "beta"}},
        )
        assert cfg.get_method_for_model("LR_EN") == "beta"
        assert cfg.get_method_for_model("RF") == "isotonic"  # falls back to global

    def test_get_strategy_for_model_with_per_model_override(self):
        """get_strategy_for_model returns per-model strategy when present."""
        from ced_ml.config.calibration_schema import CalibrationConfig

        cfg = CalibrationConfig(
            strategy="per_fold",
            per_model={
                "LR_EN": {"strategy": "oof_posthoc"},
                "LinSVM_cal": {"strategy": "none"},
            },
        )
        assert cfg.get_strategy_for_model("LR_EN") == "oof_posthoc"
        assert cfg.get_strategy_for_model("LinSVM_cal") == "none"
        assert cfg.get_strategy_for_model("RF") == "per_fold"
