"""Tests for StackingEnsemble prediction methods.

Tests cover:
- Prediction using base model predictions
- Class prediction
- Error handling for unfitted model
- Error handling for missing base model predictions
"""

import numpy as np
import pytest

from ced_ml.models.stacking import StackingEnsemble


def test_predict_proba_from_base_preds(toy_oof_predictions, toy_test_predictions):
    """Test prediction using base model predictions."""
    oof_dict, y_train = toy_oof_predictions
    preds_dict, y_test, _ = toy_test_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    proba = ensemble.predict_proba_from_base_preds(preds_dict)

    # Should be (n_test, 2) probability matrix
    assert proba.shape == (len(y_test), 2)

    # Probabilities should sum to 1
    np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(y_test)))

    # Probabilities should be in [0, 1]
    assert proba.min() >= 0
    assert proba.max() <= 1


def test_predict_from_base_preds(toy_oof_predictions, toy_test_predictions):
    """Test class prediction using base model predictions."""
    oof_dict, y_train = toy_oof_predictions
    preds_dict, y_test, _ = toy_test_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    labels = ensemble.predict_from_base_preds(preds_dict)

    # Should be binary labels
    assert set(labels).issubset({0, 1})
    assert len(labels) == len(y_test)


def test_predict_proba_not_fitted():
    """Test error when predicting before fitting."""
    ensemble = StackingEnsemble(base_model_names=["LR_EN", "RF"])

    with pytest.raises(RuntimeError, match="not fitted"):
        ensemble.predict_proba_from_base_preds({"LR_EN": [0.5], "RF": [0.5]})


def test_predict_proba_missing_model(toy_oof_predictions):
    """Test error when base model prediction is missing."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    # Missing XGBoost predictions
    incomplete_preds = {"LR_EN": [0.5, 0.6], "RF": [0.4, 0.7]}

    with pytest.raises(ValueError, match="Missing predictions"):
        ensemble.predict_proba_from_base_preds(incomplete_preds)
