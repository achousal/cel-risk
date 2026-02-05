"""
Tests for models.calibration_strategy module.

Coverage areas:
- CalibrationStrategy protocol compliance
- NoCalibration strategy
- PerFoldCalibration strategy
- OOFPosthocCalibration strategy
- IsotonicCalibration and SigmoidCalibration convenience classes
- get_calibration_strategy factory function
- get_strategy_display_name utility
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ced_ml.config.calibration_schema import CalibrationConfig
from ced_ml.models.calibration_strategy import (
    CalibrationStrategy,
    IsotonicCalibration,
    NoCalibration,
    OOFPosthocCalibration,
    PerFoldCalibration,
    SigmoidCalibration,
    get_calibration_strategy,
    get_strategy_display_name,
)

# ============================================================================
# Protocol Compliance Tests
# ============================================================================


def test_no_calibration_implements_protocol():
    """NoCalibration should implement CalibrationStrategy protocol."""
    strategy = NoCalibration()
    assert isinstance(strategy, CalibrationStrategy)


def test_per_fold_calibration_implements_protocol():
    """PerFoldCalibration should implement CalibrationStrategy protocol."""
    strategy = PerFoldCalibration()
    assert isinstance(strategy, CalibrationStrategy)


def test_oof_posthoc_calibration_implements_protocol():
    """OOFPosthocCalibration should implement CalibrationStrategy protocol."""
    strategy = OOFPosthocCalibration()
    assert isinstance(strategy, CalibrationStrategy)


def test_isotonic_calibration_implements_protocol():
    """IsotonicCalibration should implement CalibrationStrategy protocol."""
    strategy = IsotonicCalibration()
    assert isinstance(strategy, CalibrationStrategy)


def test_sigmoid_calibration_implements_protocol():
    """SigmoidCalibration should implement CalibrationStrategy protocol."""
    strategy = SigmoidCalibration()
    assert isinstance(strategy, CalibrationStrategy)


# ============================================================================
# NoCalibration Tests
# ============================================================================


class TestNoCalibration:
    """Tests for NoCalibration strategy."""

    def test_name(self):
        """Name should be 'none'."""
        strategy = NoCalibration()
        assert strategy.name() == "none"

    def test_method(self):
        """Method should be None."""
        strategy = NoCalibration()
        assert strategy.method() is None

    def test_requires_oof_calibration(self):
        """Should not require OOF calibration."""
        strategy = NoCalibration()
        assert strategy.requires_oof_calibration() is False

    def test_requires_per_fold_calibration(self):
        """Should not require per-fold calibration."""
        strategy = NoCalibration()
        assert strategy.requires_per_fold_calibration() is False

    def test_should_skip_for_any_model(self):
        """Should skip for any model."""
        strategy = NoCalibration()
        assert strategy.should_skip_for_model("LR_EN") is True
        assert strategy.should_skip_for_model("RF") is True
        assert strategy.should_skip_for_model("LinSVM_cal") is True

    def test_get_cv_folds(self):
        """CV folds should be 0."""
        strategy = NoCalibration()
        assert strategy.get_cv_folds() == 0

    def test_repr(self):
        """Repr should be meaningful."""
        strategy = NoCalibration()
        assert "NoCalibration" in repr(strategy)


# ============================================================================
# PerFoldCalibration Tests
# ============================================================================


class TestPerFoldCalibration:
    """Tests for PerFoldCalibration strategy."""

    def test_init_default(self):
        """Default initialization."""
        strategy = PerFoldCalibration()
        assert strategy.method() == "isotonic"
        assert strategy.get_cv_folds() == 5

    def test_init_custom(self):
        """Custom initialization."""
        strategy = PerFoldCalibration(method="sigmoid", cv=3)
        assert strategy.method() == "sigmoid"
        assert strategy.get_cv_folds() == 3

    def test_init_invalid_method(self):
        """Should raise error for invalid method."""
        with pytest.raises(ValueError, match="must be 'isotonic' or 'sigmoid'"):
            PerFoldCalibration(method="invalid")

    def test_init_invalid_cv(self):
        """Should raise error for cv < 2."""
        with pytest.raises(ValueError, match="cv must be >= 2"):
            PerFoldCalibration(cv=1)

    def test_name(self):
        """Name should be 'per_fold'."""
        strategy = PerFoldCalibration()
        assert strategy.name() == "per_fold"

    def test_requires_oof_calibration(self):
        """Should not require OOF calibration."""
        strategy = PerFoldCalibration()
        assert strategy.requires_oof_calibration() is False

    def test_requires_per_fold_calibration(self):
        """Should require per-fold calibration."""
        strategy = PerFoldCalibration()
        assert strategy.requires_per_fold_calibration() is True

    def test_should_skip_for_svm(self):
        """Should skip for LinSVM_cal (already calibrated)."""
        strategy = PerFoldCalibration()
        assert strategy.should_skip_for_model("LinSVM_cal") is True

    def test_should_not_skip_for_other_models(self):
        """Should not skip for LR_EN, RF, etc."""
        strategy = PerFoldCalibration()
        assert strategy.should_skip_for_model("LR_EN") is False
        assert strategy.should_skip_for_model("RF") is False
        assert strategy.should_skip_for_model("XGBoost") is False

    def test_repr(self):
        """Repr should show method and cv."""
        strategy = PerFoldCalibration(method="sigmoid", cv=3)
        assert "PerFoldCalibration" in repr(strategy)
        assert "sigmoid" in repr(strategy)
        assert "3" in repr(strategy)

    def test_apply_basic(self):
        """Apply should calibrate a model."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)

        strategy = PerFoldCalibration(method="isotonic", cv=3)
        calibrated = strategy.apply(base_model, "LR_EN", X, y)

        assert isinstance(calibrated, CalibratedClassifierCV)

    def test_apply_skip_svm(self):
        """Apply should skip for LinSVM_cal."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)

        strategy = PerFoldCalibration()
        result = strategy.apply(base_model, "LinSVM_cal", X, y)

        # Should return original model
        assert result is base_model

    def test_apply_no_double_calibration(self):
        """Apply should not double-calibrate."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.binomial(1, 0.3, size=100)

        base_model = LogisticRegression(random_state=42)
        # Pre-calibrate
        calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=2)
        calibrated.fit(X, y)

        strategy = PerFoldCalibration()
        result = strategy.apply(calibrated, "LR_EN", X, y)

        # Should return same calibrated model
        assert result is calibrated


# ============================================================================
# OOFPosthocCalibration Tests
# ============================================================================


class TestOOFPosthocCalibration:
    """Tests for OOFPosthocCalibration strategy."""

    def test_init_default(self):
        """Default initialization."""
        strategy = OOFPosthocCalibration()
        assert strategy.method() == "isotonic"

    def test_init_custom(self):
        """Custom initialization."""
        strategy = OOFPosthocCalibration(method="sigmoid")
        assert strategy.method() == "sigmoid"

    def test_init_invalid_method(self):
        """Should raise error for invalid method."""
        with pytest.raises(ValueError, match="must be 'isotonic' or 'sigmoid'"):
            OOFPosthocCalibration(method="invalid")

    def test_name(self):
        """Name should be 'oof_posthoc'."""
        strategy = OOFPosthocCalibration()
        assert strategy.name() == "oof_posthoc"

    def test_requires_oof_calibration(self):
        """Should require OOF calibration."""
        strategy = OOFPosthocCalibration()
        assert strategy.requires_oof_calibration() is True

    def test_requires_per_fold_calibration(self):
        """Should not require per-fold calibration."""
        strategy = OOFPosthocCalibration()
        assert strategy.requires_per_fold_calibration() is False

    def test_should_skip_for_svm(self):
        """Should skip for LinSVM_cal (already calibrated)."""
        strategy = OOFPosthocCalibration()
        assert strategy.should_skip_for_model("LinSVM_cal") is True

    def test_should_not_skip_for_other_models(self):
        """Should not skip for LR_EN, RF, etc."""
        strategy = OOFPosthocCalibration()
        assert strategy.should_skip_for_model("LR_EN") is False
        assert strategy.should_skip_for_model("RF") is False

    def test_get_cv_folds(self):
        """CV folds should be 0 (not used)."""
        strategy = OOFPosthocCalibration()
        assert strategy.get_cv_folds() == 0

    def test_repr(self):
        """Repr should show method."""
        strategy = OOFPosthocCalibration(method="sigmoid")
        assert "OOFPosthocCalibration" in repr(strategy)
        assert "sigmoid" in repr(strategy)

    def test_create_calibrator(self):
        """create_calibrator should return an OOFCalibrator."""
        from ced_ml.models.calibration import OOFCalibrator

        strategy = OOFPosthocCalibration(method="isotonic")
        calibrator = strategy.create_calibrator()

        assert isinstance(calibrator, OOFCalibrator)
        assert calibrator.method == "isotonic"
        assert not calibrator.is_fitted


# ============================================================================
# Convenience Classes Tests
# ============================================================================


class TestIsotonicCalibration:
    """Tests for IsotonicCalibration convenience class."""

    def test_init(self):
        """Should initialize with isotonic method."""
        strategy = IsotonicCalibration(cv=3)
        assert strategy.method() == "isotonic"
        assert strategy.get_cv_folds() == 3

    def test_repr(self):
        """Repr should be IsotonicCalibration."""
        strategy = IsotonicCalibration()
        assert "IsotonicCalibration" in repr(strategy)


class TestSigmoidCalibration:
    """Tests for SigmoidCalibration convenience class."""

    def test_init(self):
        """Should initialize with sigmoid method."""
        strategy = SigmoidCalibration(cv=3)
        assert strategy.method() == "sigmoid"
        assert strategy.get_cv_folds() == 3

    def test_repr(self):
        """Repr should be SigmoidCalibration."""
        strategy = SigmoidCalibration()
        assert "SigmoidCalibration" in repr(strategy)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetCalibrationStrategy:
    """Tests for get_calibration_strategy factory function."""

    def _make_config(
        self,
        enabled: bool = True,
        strategy: str = "per_fold",
        method: str = "isotonic",
        cv: int = 5,
        per_model: dict | None = None,
    ):
        """Create a mock config object with calibration settings."""
        # Create a minimal config-like object
        calibration = CalibrationConfig(
            enabled=enabled,
            strategy=strategy,
            method=method,
            cv=cv,
            per_model=per_model,
        )

        class MockConfig:
            pass

        config = MockConfig()
        config.calibration = calibration
        return config

    def test_disabled_returns_no_calibration(self):
        """Disabled calibration should return NoCalibration."""
        config = self._make_config(enabled=False)
        strategy = get_calibration_strategy(config)

        assert isinstance(strategy, NoCalibration)

    def test_none_strategy(self):
        """Strategy 'none' should return NoCalibration."""
        config = self._make_config(enabled=True, strategy="none")
        strategy = get_calibration_strategy(config)

        assert isinstance(strategy, NoCalibration)

    def test_per_fold_strategy(self):
        """Strategy 'per_fold' should return PerFoldCalibration."""
        config = self._make_config(enabled=True, strategy="per_fold", method="sigmoid", cv=3)
        strategy = get_calibration_strategy(config)

        assert isinstance(strategy, PerFoldCalibration)
        assert strategy.method() == "sigmoid"
        assert strategy.get_cv_folds() == 3

    def test_oof_posthoc_strategy(self):
        """Strategy 'oof_posthoc' should return OOFPosthocCalibration."""
        config = self._make_config(enabled=True, strategy="oof_posthoc", method="isotonic")
        strategy = get_calibration_strategy(config)

        assert isinstance(strategy, OOFPosthocCalibration)
        assert strategy.method() == "isotonic"

    def test_per_model_override(self):
        """Per-model override should be respected."""
        config = self._make_config(
            enabled=True,
            strategy="per_fold",
            method="isotonic",
            per_model={"LR_EN": "oof_posthoc"},
        )

        # Without model name, should use global strategy
        strategy = get_calibration_strategy(config)
        assert isinstance(strategy, PerFoldCalibration)

        # With model name that has override
        strategy = get_calibration_strategy(config, model_name="LR_EN")
        assert isinstance(strategy, OOFPosthocCalibration)

        # With model name that doesn't have override
        strategy = get_calibration_strategy(config, model_name="RF")
        assert isinstance(strategy, PerFoldCalibration)

    def test_per_model_none_override(self):
        """Per-model override to 'none' should work."""
        config = self._make_config(
            enabled=True,
            strategy="per_fold",
            per_model={"LinSVM_cal": "none"},
        )

        strategy = get_calibration_strategy(config, model_name="LinSVM_cal")
        assert isinstance(strategy, NoCalibration)


# ============================================================================
# Display Name Tests
# ============================================================================


class TestGetStrategyDisplayName:
    """Tests for get_strategy_display_name utility."""

    def test_no_calibration(self):
        """NoCalibration should display as 'none'."""
        strategy = NoCalibration()
        assert get_strategy_display_name(strategy) == "none"

    def test_per_fold_calibration(self):
        """PerFoldCalibration should include method and strategy."""
        strategy = PerFoldCalibration(method="isotonic")
        display = get_strategy_display_name(strategy)
        assert "isotonic" in display
        assert "per_fold" in display

    def test_oof_posthoc_calibration(self):
        """OOFPosthocCalibration should include method and strategy."""
        strategy = OOFPosthocCalibration(method="sigmoid")
        display = get_strategy_display_name(strategy)
        assert "sigmoid" in display
        assert "oof_posthoc" in display


# ============================================================================
# Integration Tests
# ============================================================================


class TestCalibrationStrategyIntegration:
    """Integration tests for calibration strategy workflow."""

    def test_full_workflow_per_fold(self):
        """Test complete per-fold calibration workflow."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = rng.binomial(1, 0.3, size=200)

        # Create strategy
        strategy = PerFoldCalibration(method="isotonic", cv=3)

        # Verify strategy properties
        assert strategy.name() == "per_fold"
        assert strategy.requires_per_fold_calibration()
        assert not strategy.requires_oof_calibration()

        # Train and calibrate model
        base_model = LogisticRegression(random_state=42)
        base_model.fit(X, y)
        calibrated = strategy.apply(base_model, "LR_EN", X, y)

        # Verify calibrated model works
        proba = calibrated.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_full_workflow_oof_posthoc(self):
        """Test complete OOF posthoc calibration workflow."""
        rng = np.random.default_rng(42)

        # Create strategy
        strategy = OOFPosthocCalibration(method="isotonic")

        # Verify strategy properties
        assert strategy.name() == "oof_posthoc"
        assert strategy.requires_oof_calibration()
        assert not strategy.requires_per_fold_calibration()

        # Create calibrator
        calibrator = strategy.create_calibrator()
        assert not calibrator.is_fitted

        # Fit calibrator on OOF predictions
        oof_preds = rng.uniform(0.1, 0.9, size=200)
        y_true = rng.binomial(1, oof_preds)
        calibrator.fit(oof_preds, y_true)

        assert calibrator.is_fitted

        # Transform new predictions
        new_preds = np.array([0.1, 0.5, 0.9])
        calibrated_preds = calibrator.transform(new_preds)
        assert len(calibrated_preds) == 3
        assert np.all(calibrated_preds >= 0)
        assert np.all(calibrated_preds <= 1)
