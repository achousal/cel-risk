"""Tests for StackingEnsemble model inspection methods.

Tests cover:
- Extracting meta-model coefficients
- Error handling for unfitted model
"""

import pytest

from ced_ml.models.stacking import StackingEnsemble


def test_get_meta_model_coef(toy_oof_predictions):
    """Test extracting meta-model coefficients."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    coef = ensemble.get_meta_model_coef()

    # Should have coefficient for each base model
    assert set(coef.keys()) == {"oof_LR_EN", "oof_RF", "oof_XGBoost"}

    # Coefficients should be numeric
    for v in coef.values():
        assert isinstance(v, int | float)


def test_get_meta_model_coef_not_fitted():
    """Test error when getting coefficients before fitting."""
    ensemble = StackingEnsemble(base_model_names=["LR_EN", "RF"])

    with pytest.raises(RuntimeError, match="not fitted"):
        ensemble.get_meta_model_coef()
