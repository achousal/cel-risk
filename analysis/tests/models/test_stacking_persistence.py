"""Tests for StackingEnsemble persistence (save/load).

Tests cover:
- Saving and loading ensemble
- Loaded ensemble can make predictions
- Predictions match before and after save/load
"""

import tempfile
from pathlib import Path

import numpy as np

from ced_ml.models.stacking import StackingEnsemble


def test_save_and_load(toy_oof_predictions):
    """Test saving and loading ensemble."""
    oof_dict, y_train = toy_oof_predictions

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        meta_penalty="l2",
        meta_C=0.5,
        calibrate_meta=False,
        random_state=42,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ensemble.joblib"
        ensemble.save(save_path)

        # Verify file exists
        assert save_path.exists()

        # Load and verify
        loaded = StackingEnsemble.load(save_path)

        assert loaded.base_model_names == ensemble.base_model_names
        assert loaded.meta_penalty == ensemble.meta_penalty
        assert loaded.meta_C == ensemble.meta_C
        assert loaded.is_fitted_


def test_loaded_ensemble_predicts(toy_oof_predictions, toy_test_predictions):
    """Test that loaded ensemble can make predictions."""
    oof_dict, y_train = toy_oof_predictions
    preds_dict, y_test, _ = toy_test_predictions

    # Only use 2 models for this test
    oof_dict_2 = {k: oof_dict[k] for k in ["LR_EN", "RF"]}
    preds_dict_2 = {k: preds_dict[k] for k in ["LR_EN", "RF"]}

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )
    ensemble.fit_from_oof(oof_dict_2, y_train)

    # Get predictions before saving
    original_proba = ensemble.predict_proba_from_base_preds(preds_dict_2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "ensemble.joblib"
        ensemble.save(save_path)

        loaded = StackingEnsemble.load(save_path)
        loaded_proba = loaded.predict_proba_from_base_preds(preds_dict_2)

        # Predictions should match
        np.testing.assert_array_almost_equal(original_proba, loaded_proba)
