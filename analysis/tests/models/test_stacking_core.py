"""Tests for StackingEnsemble core functionality.

Tests cover:
- Ensemble instantiation
- Default parameters
- Single base model handling
- Different penalty types
"""

import numpy as np

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
    assert ensemble.calibrate_meta is True


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
