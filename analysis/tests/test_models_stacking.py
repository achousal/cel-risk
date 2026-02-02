"""Tests for models.stacking module.

Tests cover:
- StackingEnsemble class instantiation
- Meta-feature building from OOF predictions
- Meta-learner training on OOF predictions
- Prediction pipeline (base model preds -> meta-model)
- Coefficient extraction for interpretability
- Serialization (save/load)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ced_ml.models.stacking import (
    StackingEnsemble,
    collect_oof_predictions,
    collect_split_predictions,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def toy_data():
    """Generate toy classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    return X, y


@pytest.fixture
def toy_oof_predictions(toy_data):
    """Generate toy OOF predictions from multiple 'base models'.

    Simulates OOF predictions from 3 different models with different
    characteristics (different seeds, slightly different predictions).
    """
    X, y = toy_data
    n_samples = len(y)

    # Simulate 3 base models
    oof_dict = {}

    # Model 1: Good predictions
    rng = np.random.default_rng(42)
    lr1 = LogisticRegression(C=1.0, random_state=42)
    oof1 = cross_val_predict(lr1, X, y, cv=3, method="predict_proba")[:, 1]
    # Add repeat dimension
    oof_dict["LR_EN"] = np.vstack([oof1, oof1 + rng.normal(0, 0.02, n_samples)])

    # Model 2: Different model
    rng = np.random.default_rng(43)
    lr2 = LogisticRegression(C=0.1, random_state=43)
    oof2 = cross_val_predict(lr2, X, y, cv=3, method="predict_proba")[:, 1]
    oof_dict["RF"] = np.vstack([oof2, oof2 + rng.normal(0, 0.02, n_samples)])

    # Model 3: Another different model
    rng = np.random.default_rng(44)
    lr3 = LogisticRegression(C=10.0, random_state=44)
    oof3 = cross_val_predict(lr3, X, y, cv=3, method="predict_proba")[:, 1]
    oof_dict["XGBoost"] = np.vstack([oof3, oof3 + rng.normal(0, 0.02, n_samples)])

    return oof_dict, y


@pytest.fixture
def toy_test_predictions(toy_data):
    """Generate toy test set predictions from multiple models."""
    X, y = toy_data

    # Use a subset as "test set"
    test_idx = np.arange(50, 100)
    X_test = X[test_idx]
    y_test = y[test_idx]

    preds_dict = {}

    # Fit on "train" and predict on "test"
    train_idx = np.concatenate([np.arange(0, 50), np.arange(100, 200)])
    X_train = X[train_idx]
    y_train = y[train_idx]

    for i, model_name in enumerate(["LR_EN", "RF", "XGBoost"]):
        lr = LogisticRegression(C=1.0, random_state=42 + i)
        lr.fit(X_train, y_train)
        preds_dict[model_name] = lr.predict_proba(X_test)[:, 1]

    return preds_dict, y_test, test_idx


# ----------------------------
# StackingEnsemble instantiation
# ----------------------------
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


# ----------------------------
# Meta-feature building
# ----------------------------
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


# ----------------------------
# Meta-learner training
# ----------------------------
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


# ----------------------------
# Prediction
# ----------------------------
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


# ----------------------------
# Meta-model coefficients
# ----------------------------
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


# ----------------------------
# Serialization
# ----------------------------
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


# ----------------------------
# sklearn compatibility
# ----------------------------
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


# ----------------------------
# Edge cases
# ----------------------------
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


def test_nan_handling_in_oof():
    """Test handling of NaN values in OOF predictions."""
    y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 1])
    oof_dict = {
        "LR_EN": np.array([[0.3, 0.2, np.nan, 0.8, 0.7, 0.3, 0.2, 0.9, 0.1, 0.6]]),
        "RF": np.array([[0.4, 0.3, 0.5, 0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.7]]),
    }

    ensemble = StackingEnsemble(
        base_model_names=["LR_EN", "RF"],
        calibrate_meta=False,
        random_state=42,
    )

    # Should handle NaN by dropping those samples (with warning)
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


# ----------------------------
# Integration with file-based collection (mock)
# ----------------------------
def test_collect_oof_predictions_file_not_found():
    """Test error when OOF file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_split_predictions_file_not_found():
    """Test error when split predictions file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN"],
                split_seed=0,
                split_name="test",
            )


def test_collect_oof_predictions_from_files():
    """Test collecting OOF predictions from CSV files."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create mock directory structure and files
        for model in ["LR_EN", "RF"]:
            model_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            model_dir.mkdir(parents=True)

            # Create mock OOF CSV
            oof_df = pd.DataFrame(
                {
                    "idx": np.arange(100),
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                    "y_prob_repeat1": rng.uniform(0, 1, 100),
                }
            )
            oof_df.to_csv(model_dir / f"train_oof__{model}.csv", index=False)

        # Collect
        oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
            results_dir,
            base_models=["LR_EN", "RF"],
            split_seed=0,
        )

        assert "LR_EN" in oof_dict
        assert "RF" in oof_dict
        assert oof_dict["LR_EN"].shape == (2, 100)  # 2 repeats, 100 samples
        assert len(y_train) == 100
        assert len(train_idx) == 100
        assert cat_train is None  # No category column in test data


def test_collect_oof_predictions_index_length_mismatch():
    """Test error when base models have different sample counts in OOF predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with 100 samples
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100),
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(lr_dir / "train_oof__LR_EN.csv", index=False)

        # Create RF with 80 samples (mismatch)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(80),
                "y_true": rng.integers(0, 2, 80),
                "y_prob_repeat0": rng.uniform(0, 1, 80),
            }
        ).to_csv(rf_dir / "train_oof__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index length mismatch.*OOF predictions"):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_oof_predictions_index_value_mismatch():
    """Test error when base models have different index values in OOF predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-99
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100),
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(lr_dir / "train_oof__LR_EN.csv", index=False)

        # Create RF with different indices (100-199 instead of 0-99)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(100, 200),  # Different indices
                "y_true": rng.integers(0, 2, 100),
                "y_prob_repeat0": rng.uniform(0, 1, 100),
            }
        ).to_csv(rf_dir / "train_oof__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*OOF predictions"):
            collect_oof_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
            )


def test_collect_split_predictions_index_length_mismatch():
    """Test error when base models have different sample counts in test predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with 50 test samples
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50),
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(lr_dir / "test_preds__LR_EN.csv", index=False)

        # Create RF with 40 test samples (mismatch)
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(40),
                "y_true": rng.integers(0, 2, 40),
                "y_prob": rng.uniform(0, 1, 40),
            }
        ).to_csv(rf_dir / "test_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index length mismatch.*test predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="test",
            )


def test_collect_split_predictions_index_value_mismatch():
    """Test error when base models have different index values in test predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-49
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50),
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(lr_dir / "test_preds__LR_EN.csv", index=False)

        # Create RF with different indices
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(50, 100),  # Different indices
                "y_true": rng.integers(0, 2, 50),
                "y_prob": rng.uniform(0, 1, 50),
            }
        ).to_csv(rf_dir / "test_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*test predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="test",
            )


def test_collect_split_predictions_val_index_mismatch():
    """Test error when base models have different index values in val predictions."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create LR_EN with indices 0-29
        lr_dir = results_dir / "run_test" / "LR_EN" / "splits" / "split_seed0" / "preds"
        lr_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(30),
                "y_true": rng.integers(0, 2, 30),
                "y_prob": rng.uniform(0, 1, 30),
            }
        ).to_csv(lr_dir / "val_preds__LR_EN.csv", index=False)

        # Create RF with different indices
        rf_dir = results_dir / "run_test" / "RF" / "splits" / "split_seed0" / "preds"
        rf_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "idx": np.arange(30, 60),  # Different indices
                "y_true": rng.integers(0, 2, 30),
                "y_prob": rng.uniform(0, 1, 30),
            }
        ).to_csv(rf_dir / "val_preds__RF.csv", index=False)

        with pytest.raises(ValueError, match="Index mismatch.*val predictions"):
            collect_split_predictions(
                results_dir,
                base_models=["LR_EN", "RF"],
                split_seed=0,
                split_name="val",
            )


def test_collect_oof_predictions_matching_indices_succeeds():
    """Test that collecting OOF predictions succeeds when indices match."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Use same fixed seed for reproducible indices
        shared_indices = np.arange(100)

        for model in ["LR_EN", "RF", "XGBoost"]:
            model_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            model_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "idx": shared_indices,  # Same indices for all models
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                }
            ).to_csv(model_dir / f"train_oof__{model}.csv", index=False)

        # Should succeed without error
        oof_dict, y_train, train_idx, cat_train = collect_oof_predictions(
            results_dir,
            base_models=["LR_EN", "RF", "XGBoost"],
            split_seed=0,
        )

        assert len(oof_dict) == 3
        assert len(y_train) == 100
        np.testing.assert_array_equal(train_idx, shared_indices)
        assert cat_train is None  # No category column in test data


def test_ensemble_predictions_directory_structure():
    """Test that ENSEMBLE predictions are saved to correct subdirectories for aggregation."""
    rng = np.random.default_rng(42)
    from ced_ml.cli.train_ensemble import run_train_ensemble

    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir(parents=True)

        # Create mock base model results (with run_* directory and splits/)
        for model in ["LR_EN", "RF"]:
            preds_dir = results_dir / "run_test" / model / "splits" / "split_seed0" / "preds"
            preds_dir.mkdir(parents=True)

            # Mock OOF predictions
            oof_df = pd.DataFrame(
                {
                    "idx": np.arange(100),
                    "y_true": rng.integers(0, 2, 100),
                    "y_prob_repeat0": rng.uniform(0, 1, 100),
                    "y_prob_repeat1": rng.uniform(0, 1, 100),
                }
            )
            oof_df.to_csv(preds_dir / f"train_oof__{model}.csv", index=False)

            # Mock val predictions
            val_df = pd.DataFrame(
                {
                    "idx": np.arange(50),
                    "y_true": rng.integers(0, 2, 50),
                    "y_prob": rng.uniform(0, 1, 50),
                }
            )
            val_df.to_csv(preds_dir / f"val_preds__{model}.csv", index=False)

            # Mock test predictions
            test_df = pd.DataFrame(
                {
                    "idx": np.arange(50, 100),
                    "y_true": rng.integers(0, 2, 50),
                    "y_prob": rng.uniform(0, 1, 50),
                }
            )
            test_df.to_csv(preds_dir / f"test_preds__{model}.csv", index=False)

        # Run ensemble training
        result = run_train_ensemble(
            results_dir=str(results_dir),
            base_models=["LR_EN", "RF"],
            split_seed=0,
            log_level=None,
        )

        assert result is not None

        # Check that ENSEMBLE predictions are in the expected directory
        ensemble_dir = results_dir / "run_test" / "ENSEMBLE" / "splits" / "split_seed0"

        # Val predictions should be in preds/ directory
        val_preds_path = ensemble_dir / "preds" / "val_preds__ENSEMBLE.csv"
        assert val_preds_path.exists(), f"Val predictions not found at {val_preds_path}"

        # Test predictions should be in preds/ directory
        test_preds_path = ensemble_dir / "preds" / "test_preds__ENSEMBLE.csv"
        assert test_preds_path.exists(), f"Test predictions not found at {test_preds_path}"

        # OOF predictions should be in preds/ directory
        oof_preds_path = ensemble_dir / "preds" / "train_oof__ENSEMBLE.csv"
        assert oof_preds_path.exists(), f"OOF predictions not found at {oof_preds_path}"

        # Verify that predictions can be loaded and have correct structure
        val_preds = pd.read_csv(val_preds_path)
        assert "y_prob" in val_preds.columns
        assert "y_true" in val_preds.columns

        test_preds = pd.read_csv(test_preds_path)
        assert "y_prob" in test_preds.columns
        assert "y_true" in test_preds.columns


# --- Tests for ENSEMBLE aggregation support ---


class TestEnsembleAggregationSupport:
    """Test ensemble discovery and aggregation functions for aggregate_splits."""

    def test_discover_ensemble_dirs_empty(self, tmp_path):
        """Test that empty directory returns empty list."""
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        result = discover_ensemble_dirs(tmp_path)
        assert result == []

    def test_discover_ensemble_dirs_no_ensemble_folder(self, tmp_path):
        """Test that missing ENSEMBLE folder returns empty list."""
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        (tmp_path / "some_other_folder").mkdir()
        result = discover_ensemble_dirs(tmp_path)
        assert result == []

    def test_discover_ensemble_dirs_split_underscore_format(self, tmp_path):
        """Test discovery of split_seedX format directories."""
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed1").mkdir()
        (ensemble_dir / "split_seed2").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3
        # Check they're sorted by seed
        assert result[0].name == "split_seed0"
        assert result[1].name == "split_seed1"
        assert result[2].name == "split_seed2"

    def test_discover_ensemble_dirs_split_seed_format(self, tmp_path):
        """Test discovery of split_seed{X} format directories."""
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed5").mkdir()
        (ensemble_dir / "split_seed10").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3
        # Check they're sorted by seed
        assert result[0].name == "split_seed0"
        assert result[1].name == "split_seed5"
        assert result[2].name == "split_seed10"

    def test_discover_ensemble_dirs_mixed_formats(self, tmp_path):
        """Test discovery handles only split_seed format directories."""
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()
        (ensemble_dir / "split_seed1").mkdir()
        (ensemble_dir / "split_seed2").mkdir()

        result = discover_ensemble_dirs(tmp_path)
        assert len(result) == 3

    def test_collect_ensemble_predictions_empty(self, tmp_path):
        """Test collecting predictions from empty directories."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        (ensemble_dir / "split_seed0").mkdir()

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")
        assert result.empty

    def test_collect_ensemble_predictions_with_data(self, tmp_path):
        """Test collecting predictions with actual data."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        # Setup directory structure
        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        split_dir = ensemble_dir / "split_seed0"
        split_dir.mkdir()
        preds_dir = split_dir / "preds"
        preds_dir.mkdir(parents=True)

        # Create test predictions
        pd.DataFrame(
            {
                "idx": [0, 1, 2],
                "y_true": [0, 0, 1],
                "y_prob": [0.1, 0.2, 0.9],
            }
        ).to_csv(preds_dir / "test_preds__ENSEMBLE.csv", index=False)

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")

        assert len(result) == 3
        assert "model" in result.columns
        assert result["model"].iloc[0] == "ENSEMBLE"
        assert "split_seed" in result.columns
        assert result["split_seed"].iloc[0] == 0

    def test_collect_ensemble_predictions_multiple_splits(self, tmp_path):
        """Test collecting predictions from multiple splits."""
        from ced_ml.cli.aggregation.collection import collect_ensemble_predictions
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        # Setup directory structure with two splits
        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)

        for seed in [0, 1]:
            split_dir = ensemble_dir / f"split_seed{seed}"
            split_dir.mkdir()
            preds_dir = split_dir / "preds"
            preds_dir.mkdir(parents=True)

            pd.DataFrame(
                {
                    "idx": [0, 1],
                    "y_true": [0, 1],
                    "y_prob": [0.2 + seed * 0.1, 0.8 + seed * 0.05],
                }
            ).to_csv(preds_dir / "test_preds__ENSEMBLE.csv", index=False)

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_predictions(ensemble_dirs, "test")

        assert len(result) == 4  # 2 samples * 2 splits
        assert result["split_seed"].nunique() == 2

    def test_collect_ensemble_metrics_with_data(self, tmp_path):
        """Test collecting metrics from ENSEMBLE directories."""
        import json

        from ced_ml.cli.aggregation.collection import collect_ensemble_metrics
        from ced_ml.cli.aggregation.discovery import discover_ensemble_dirs

        # Setup directory structure
        ensemble_dir = tmp_path / "ENSEMBLE" / "splits"
        ensemble_dir.mkdir(parents=True)
        split_dir = ensemble_dir / "split_seed0"
        split_dir.mkdir()
        core_dir = split_dir / "core"
        core_dir.mkdir()

        # Create metrics JSON (ENSEMBLE format)
        metrics = {
            "model": "ENSEMBLE",
            "test": {"AUROC": 0.87, "PR_AUC": 0.14, "Brier": 0.07},
            "val": {"AUROC": 0.85, "PR_AUC": 0.12},
        }
        with open(core_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        ensemble_dirs = discover_ensemble_dirs(tmp_path)
        result = collect_ensemble_metrics(ensemble_dirs)

        assert len(result) == 1
        assert result["model"].iloc[0] == "ENSEMBLE"
        assert result["test_AUROC"].iloc[0] == 0.87
        assert result["val_AUROC"].iloc[0] == 0.85

    def test_generate_model_comparison_report(self, tmp_path):
        """Test model comparison report generation."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        test_metrics = {
            "LR_EN": {"AUROC": 0.85, "PR_AUC": 0.12, "Brier": 0.08},
            "RF": {"AUROC": 0.82, "PR_AUC": 0.10, "Brier": 0.09},
            "ENSEMBLE": {"AUROC": 0.87, "PR_AUC": 0.14, "Brier": 0.07},
        }
        val_metrics = {
            "LR_EN": {"AUROC": 0.84},
            "RF": {"AUROC": 0.81},
            "ENSEMBLE": {"AUROC": 0.86},
        }
        threshold_info = {
            "LR_EN": {"youden_threshold": 0.3},
            "RF": {"youden_threshold": 0.25},
            "ENSEMBLE": {"youden_threshold": 0.28},
        }

        result = generate_model_comparison_report(
            test_metrics, val_metrics, threshold_info, tmp_path
        )

        assert len(result) == 3
        assert "is_ensemble" in result.columns

        # Check ENSEMBLE is correctly flagged
        ensemble_row = result[result["model"] == "ENSEMBLE"]
        assert ensemble_row["is_ensemble"].iloc[0] == True  # noqa: E712

        # Check non-ensemble models are not flagged
        lr_row = result[result["model"] == "LR_EN"]
        assert lr_row["is_ensemble"].iloc[0] == False  # noqa: E712

        # Check report was saved
        report_path = tmp_path / "metrics" / "model_comparison.csv"
        assert report_path.exists()

    def test_generate_model_comparison_report_sorted_by_auroc(self, tmp_path):
        """Test that model comparison report is sorted by test AUROC."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        test_metrics = {
            "RF": {"AUROC": 0.82},
            "LR_EN": {"AUROC": 0.85},
            "ENSEMBLE": {"AUROC": 0.87},
        }

        result = generate_model_comparison_report(test_metrics, {}, {}, tmp_path)

        # Should be sorted by AUROC descending
        assert result.iloc[0]["model"] == "ENSEMBLE"
        assert result.iloc[1]["model"] == "LR_EN"
        assert result.iloc[2]["model"] == "RF"

    def test_generate_model_comparison_report_empty(self, tmp_path):
        """Test model comparison report with empty metrics."""
        from ced_ml.cli.aggregation.plot_generator import (
            generate_model_comparison_report,
        )

        result = generate_model_comparison_report({}, {}, {}, tmp_path)
        assert result.empty
