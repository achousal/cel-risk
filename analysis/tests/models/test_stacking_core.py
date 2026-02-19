"""Tests for StackingEnsemble core functionality.

Tests cover:
- Ensemble instantiation
- Default parameters
- Single base model handling
- Different penalty types
- meta_calibration_method parameter
- Warning emitted when calibrate_meta=True
"""

import logging

import numpy as np
import pytest

from ced_ml.models.stacking import StackingEnsemble


def test_stacking_ensemble_instantiation():
    """Test basic instantiation of StackingEnsemble."""
    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        meta_penalty="l2",
        meta_C=1.0,
        random_state=42,
    )

    assert ensemble.base_model_names == ["LR_EN", "RF", "XGBoost"]
    assert ensemble.meta_penalty == "l2"
    assert ensemble.meta_C == 1.0
    assert ensemble.random_state == 42
    assert not ensemble.is_fitted_


def test_stacking_ensemble_default_params():
    """Test default parameter values."""
    ensemble = StackingEnsemble()

    assert ensemble.base_model_names == []
    assert ensemble.meta_penalty == "l2"
    assert ensemble.meta_C == 1.0
    assert ensemble.meta_max_iter == 1000
    assert ensemble.use_probabilities is True
    assert ensemble.scale_meta_features is True
    # Default is False: base models are already OOF-calibrated and LR calibrates
    # naturally via the logistic link function.
    assert ensemble.calibrate_meta is False
    assert ensemble.meta_calibration_method == "isotonic"


def test_single_base_model():
    """Test ensemble with single base model (degenerate case)."""
    y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    oof_dict = {"LR_EN": np.array([[0.3, 0.2, 0.4, 0.8, 0.7, 0.3, 0.2, 0.9, 0.1, 0.6]])}

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN"],
        calibrate_meta=False,
        random_state=42,
    )

    # Should work but is effectively just recalibrating the base model
    ensemble.fit_from_oof(oof_dict, y_train)
    assert ensemble.is_fitted_


def test_different_penalty_types():
    """Test ensemble with different regularization penalties."""
    rng = np.random.default_rng(42)
    y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1] * 10)  # More samples
    oof_dict = {
        "LR_EN": rng.uniform(0.2, 0.8, (2, 100)),
        "RF": rng.uniform(0.2, 0.8, (2, 100)),
    }

    for penalty in ["l2", "l1", "none"]:
        solver = "saga" if penalty == "l1" else "lbfgs"
        ensemble = StackingEnsemble(
            base_model_names=["LR_EN", "RF"],
            meta_penalty=penalty,
            meta_solver=solver,
            calibrate_meta=False,
            random_state=42,
        )
        ensemble.fit_from_oof(oof_dict, y_train)
        assert ensemble.is_fitted_


def test_meta_calibration_method_stored():
    """meta_calibration_method is stored and flows into CalibratedClassifierCV."""
    rng = np.random.default_rng(0)
    y_train = np.array([0] * 80 + [1] * 20)
    oof_dict = {
        "LR_EN": rng.uniform(0.1, 0.9, (1, 100)),
        "RF": rng.uniform(0.1, 0.9, (1, 100)),
    }

    for method in ("isotonic", "sigmoid"):
        ensemble = StackingEnsemble(
            base_model_names=["LR_EN", "RF"],
            calibrate_meta=True,
            calibration_cv=3,
            meta_calibration_method=method,
            random_state=42,
        )
        ensemble.fit_from_oof(oof_dict, y_train)
        assert ensemble.meta_calibration_method == method
        # The fitted wrapper must be a CalibratedClassifierCV
        from sklearn.calibration import CalibratedClassifierCV

        assert isinstance(ensemble.meta_model, CalibratedClassifierCV)


def test_calibrate_meta_true_emits_warning():
    """A warning must be logged when calibrate_meta=True.

    Uses mock.patch instead of caplog to avoid propagation issues in full suite.
    """
    import unittest.mock as mock

    rng = np.random.default_rng(1)
    y_train = np.array([0] * 80 + [1] * 20)
    oof_dict = {
        "LR_EN": rng.uniform(0.1, 0.9, (1, 100)),
        "RF": rng.uniform(0.1, 0.9, (1, 100)),
    }

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=True,
        calibration_cv=3,
        random_state=42,
    )

    with mock.patch("ced_ml.models.stacking.logger") as mock_logger:
        ensemble.fit_from_oof(oof_dict, y_train)

    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any(
        "calibrate_meta=True" in msg for msg in warning_calls
    ), f"Expected a warning about double calibration; calls were: {warning_calls}"


def test_calibrate_meta_false_no_double_calibration_warning(caplog):
    """No double-calibration warning should be emitted when calibrate_meta=False."""
    rng = np.random.default_rng(2)
    y_train = np.array([0] * 80 + [1] * 20)
    oof_dict = {
        "LR_EN": rng.uniform(0.1, 0.9, (1, 100)),
        "RF": rng.uniform(0.1, 0.9, (1, 100)),
    }

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )

    with caplog.at_level(logging.WARNING, logger="ced_ml.models.stacking"):
        ensemble.fit_from_oof(oof_dict, y_train)

    double_calib_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "calibrate_meta=True" in r.message
    ]
    assert len(double_calib_warnings) == 0


@pytest.mark.parametrize("calibrate_meta", [True, False])
def test_ensemble_config_calibrate_meta_parameter(calibrate_meta):
    """EnsembleConfig.calibrate_meta is correctly read and stored in StackingEnsemble."""
    from ced_ml.config.ensemble_schema import EnsembleConfig

    cfg = EnsembleConfig(calibrate_meta=calibrate_meta)
    assert cfg.calibrate_meta == calibrate_meta

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN"],
        calibrate_meta=cfg.calibrate_meta,
        meta_calibration_method=cfg.meta_calibration_method,
    )
    assert ensemble.calibrate_meta == calibrate_meta
    assert ensemble.meta_calibration_method == cfg.meta_calibration_method
