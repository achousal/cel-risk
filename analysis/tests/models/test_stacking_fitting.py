"""Tests for StackingEnsemble fitting and training.

Tests cover:
- Meta-learner training on OOF predictions
- Calibrated meta-learner training
- Shape mismatch error handling
- Reproducibility with same random state
- sklearn-compatible fit methods
"""

import numpy as np
import pytest
from ced_ml.models.stacking import StackingEnsemble


def test_fit_from_oof(toy_oof_predictions):
    """Test fitting meta-learner on OOF predictions."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        meta_penalty="l2",
        meta_C=1.0,
        calibrate_meta=False,  # Skip calibration for simpler test
        random_state=42,
    )

    ensemble.fit_from_oof(oof_dict, y_train)

    assert ensemble.is_fitted_
    assert ensemble.meta_model is not None
    assert ensemble.scaler is not None  # scale_meta_features=True by default


def test_fit_from_oof_with_calibration(toy_oof_predictions):
    """Test fitting with calibrated meta-learner."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=True,
        calibration_cv=3,
        random_state=42,
    )

    ensemble.fit_from_oof(oof_dict, y_train)

    assert ensemble.is_fitted_
    # Meta-model should be CalibratedClassifierCV
    assert hasattr(ensemble.meta_model, "calibrated_classifiers_")


def test_fit_from_oof_shape_mismatch(toy_oof_predictions):
    """Test error on shape mismatch between OOF and labels."""
    oof_dict, y_train = toy_oof_predictions

    # Truncate y_train to cause mismatch
    y_short = y_train[:50]

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        ensemble.fit_from_oof(oof_dict, y_short)


def test_fit_from_oof_reproducibility(toy_oof_predictions):
    """Test that fit_from_oof() method produces reproducible results with same random_state."""
    oof_dict, y_train = toy_oof_predictions

    # First fit with random_state=42
    ensemble1 = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble1.fit_from_oof(oof_dict, y_train)
    coef1 = ensemble1.get_meta_model_coef()

    # Second fit with same random_state=42
    ensemble2 = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble2.fit_from_oof(oof_dict, y_train)
    coef2 = ensemble2.get_meta_model_coef()

    # Same seed should produce identical coefficients
    for key in coef1:
        np.testing.assert_almost_equal(coef1[key], coef2[key])


def test_sklearn_fit_predict(toy_oof_predictions):
    """Test sklearn-compatible fit/predict methods."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )

    # Build meta-features manually
    X_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)

    # Use sklearn-style fit
    ensemble.fit(X_meta, y_train)

    assert ensemble.is_fitted_

    # Use sklearn-style predict
    proba = ensemble.predict_proba(X_meta)
    assert proba.shape == (len(y_train), 2)

    labels = ensemble.predict(X_meta)
    assert len(labels) == len(y_train)


def test_sklearn_fit_reproducibility(toy_oof_predictions):
    """Test that fit() method produces reproducible results with same random_state."""
    oof_dict, y_train = toy_oof_predictions

    # Build meta-features once (to avoid fixture differences)
    ensemble_builder = StackingEnsemble(base_model_names=["LR_EN", "RF"])
    X_meta = ensemble_builder._build_meta_features(oof_dict, aggregate_repeats=True)

    # First fit with random_state=42
    ensemble1 = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble1.fit(X_meta.copy(), y_train)
    proba1 = ensemble1.predict_proba(X_meta)

    # Second fit with same random_state=42
    ensemble2 = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble2.fit(X_meta.copy(), y_train)
    proba2 = ensemble2.predict_proba(X_meta)

    # Third fit with different random_state=123
    ensemble3 = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=123,
    )
    ensemble3.fit(X_meta.copy(), y_train)
    _proba3 = ensemble3.predict_proba(X_meta)

    # Same seed should produce identical results
    np.testing.assert_array_almost_equal(proba1, proba2)

    # Different seed should produce different results (with high probability)
    # Note: LogisticRegression with lbfgs solver is deterministic given same data,
    # but the solver may produce slightly different results with different seeds
    # depending on initialization. If results are identical, that's also acceptable
    # since the key requirement is reproducibility with same seed.
