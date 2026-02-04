"""Tests for StackingEnsemble meta-feature building.

Tests cover:
- Building meta-features from OOF predictions
- Logit transformation
- Missing model error handling
"""

import pytest
from ced_ml.models.stacking import StackingEnsemble


def test_build_meta_features_from_oof(toy_oof_predictions):
    """Test building meta-feature matrix from OOF predictions."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "XGBoost"],
    )

    X_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)

    # Should have n_samples x 3 features (one per base model)
    assert X_meta.shape == (len(y_train), 3)

    # Feature names should be set
    assert ensemble._feature_names == ["oof_LR_EN", "oof_RF", "oof_XGBoost"]


def test_build_meta_features_logits(toy_oof_predictions):
    """Test building meta-features with logit transformation."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        use_probabilities=False,  # Use logits
    )

    X_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)

    # Values should be logits (can be any real number)
    # Check that some values are outside [0, 1]
    assert X_meta.min() < 0 or X_meta.max() > 1


def test_build_meta_features_missing_model(toy_oof_predictions):
    """Test error when OOF predictions are missing for a model."""
    oof_dict, _ = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF", "MISSING_MODEL"],
    )

    with pytest.raises(ValueError, match="Missing OOF predictions"):
        ensemble._build_meta_features(oof_dict)
