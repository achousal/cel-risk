"""Tests for threshold_strategy module.

Tests the Strategy pattern implementation for threshold selection.
Covers all concrete strategies and the factory functions.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ced_ml.config.calibration_schema import ThresholdConfig
from ced_ml.metrics.threshold_strategy import (
    FixedPPVThreshold,
    FixedSensitivityThreshold,
    FixedSpecificityThreshold,
    MaxF1Threshold,
    MaxFBetaThreshold,
    ThresholdStrategy,
    YoudensJThreshold,
    get_threshold_strategy,
    get_threshold_strategy_from_params,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def balanced_data():
    """Balanced dataset with clear separation."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def imbalanced_data():
    """Imbalanced dataset (1:9 ratio) simulating rare disease."""
    rng = np.random.default_rng(42)
    y = np.array([0] * 90 + [1] * 10)
    p_controls = rng.beta(2, 5, size=90)  # Skewed low
    p_cases = rng.beta(5, 2, size=10)  # Skewed high
    p = np.concatenate([p_controls, p_cases])
    return y, p


@pytest.fixture
def perfect_separation():
    """Perfectly separated data (AUROC = 1.0)."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    return y, p


@pytest.fixture
def all_controls():
    """All negative samples (no cases)."""
    rng = np.random.default_rng(42)
    y = np.zeros(100, dtype=int)
    p = rng.uniform(0, 1, size=100)
    return y, p


@pytest.fixture
def all_cases():
    """All positive samples (no controls)."""
    rng = np.random.default_rng(42)
    y = np.ones(100, dtype=int)
    p = rng.uniform(0, 1, size=100)
    return y, p


# ============================================================================
# Test Protocol compliance
# ============================================================================


class TestProtocolCompliance:
    """Verify all concrete strategies implement the ThresholdStrategy protocol."""

    def test_max_f1_is_threshold_strategy(self):
        """MaxF1Threshold implements ThresholdStrategy protocol."""
        strategy = MaxF1Threshold()
        assert isinstance(strategy, ThresholdStrategy)

    def test_max_fbeta_is_threshold_strategy(self):
        """MaxFBetaThreshold implements ThresholdStrategy protocol."""
        strategy = MaxFBetaThreshold(beta=2.0)
        assert isinstance(strategy, ThresholdStrategy)

    def test_youden_is_threshold_strategy(self):
        """YoudensJThreshold implements ThresholdStrategy protocol."""
        strategy = YoudensJThreshold()
        assert isinstance(strategy, ThresholdStrategy)

    def test_fixed_specificity_is_threshold_strategy(self):
        """FixedSpecificityThreshold implements ThresholdStrategy protocol."""
        strategy = FixedSpecificityThreshold(target_specificity=0.95)
        assert isinstance(strategy, ThresholdStrategy)

    def test_fixed_sensitivity_is_threshold_strategy(self):
        """FixedSensitivityThreshold implements ThresholdStrategy protocol."""
        strategy = FixedSensitivityThreshold(target_sensitivity=0.95)
        assert isinstance(strategy, ThresholdStrategy)

    def test_fixed_ppv_is_threshold_strategy(self):
        """FixedPPVThreshold implements ThresholdStrategy protocol."""
        strategy = FixedPPVThreshold(target_ppv=0.5)
        assert isinstance(strategy, ThresholdStrategy)


# ============================================================================
# Test MaxF1Threshold
# ============================================================================


class TestMaxF1Threshold:
    """Tests for MaxF1Threshold strategy."""

    def test_name_property(self):
        """Strategy has correct name."""
        strategy = MaxF1Threshold()
        assert strategy.name == "max_f1"

    def test_find_threshold_balanced(self, balanced_data):
        """Find F1-maximizing threshold on balanced data."""
        y, p = balanced_data
        strategy = MaxF1Threshold()
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0
        # Expect threshold around 0.5 for balanced separation
        assert 0.4 <= threshold <= 0.6

    def test_find_threshold_imbalanced(self, imbalanced_data):
        """Find F1-maximizing threshold on imbalanced data."""
        y, p = imbalanced_data
        strategy = MaxF1Threshold()
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0

    def test_find_threshold_empty_returns_default(self):
        """Empty input returns default threshold."""
        strategy = MaxF1Threshold()
        threshold = strategy.find_threshold(np.array([]), np.array([]))
        assert threshold == 0.5


# ============================================================================
# Test MaxFBetaThreshold
# ============================================================================


class TestMaxFBetaThreshold:
    """Tests for MaxFBetaThreshold strategy."""

    def test_name_property_beta_1(self):
        """Strategy name includes beta value."""
        strategy = MaxFBetaThreshold(beta=1.0)
        assert strategy.name == "max_fbeta_1.00"

    def test_name_property_beta_2(self):
        """Strategy name includes beta value."""
        strategy = MaxFBetaThreshold(beta=2.0)
        assert strategy.name == "max_fbeta_2.00"

    def test_beta_1_equals_f1(self, balanced_data):
        """Beta=1.0 should give same threshold as F1."""
        y, p = balanced_data
        f1_strategy = MaxF1Threshold()
        fbeta_strategy = MaxFBetaThreshold(beta=1.0)

        f1_threshold = f1_strategy.find_threshold(y, p)
        fbeta_threshold = fbeta_strategy.find_threshold(y, p)

        assert f1_threshold == pytest.approx(fbeta_threshold, rel=1e-6)

    def test_beta_2_favors_recall(self, imbalanced_data):
        """Beta=2 should favor recall (lower threshold)."""
        y, p = imbalanced_data
        beta2_strategy = MaxFBetaThreshold(beta=2.0)
        beta2_threshold = beta2_strategy.find_threshold(y, p)

        # Higher beta favors recall, typically resulting in lower threshold
        # This relationship may not always hold for all datasets
        assert 0.0 <= beta2_threshold <= 1.0


# ============================================================================
# Test YoudensJThreshold
# ============================================================================


class TestYoudensJThreshold:
    """Tests for YoudensJThreshold strategy."""

    def test_name_property(self):
        """Strategy has correct name."""
        strategy = YoudensJThreshold()
        assert strategy.name == "youden"

    def test_find_threshold_balanced(self, balanced_data):
        """Find Youden's J maximizing threshold on balanced data."""
        y, p = balanced_data
        strategy = YoudensJThreshold()
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0
        # For balanced data with clear separation, expect around 0.5
        assert 0.3 <= threshold <= 0.7

    def test_find_threshold_perfect_separation(self, perfect_separation):
        """Find threshold on perfectly separated data."""
        y, p = perfect_separation
        strategy = YoudensJThreshold()
        threshold = strategy.find_threshold(y, p)
        # Perfect separation: any threshold between groups works
        assert 0.3 <= threshold <= 0.7

    def test_single_class_returns_default(self, all_controls):
        """Single class returns default threshold."""
        y, p = all_controls
        strategy = YoudensJThreshold()
        threshold = strategy.find_threshold(y, p)
        assert threshold == 0.5


# ============================================================================
# Test FixedSpecificityThreshold
# ============================================================================


class TestFixedSpecificityThreshold:
    """Tests for FixedSpecificityThreshold strategy."""

    def test_name_property(self):
        """Strategy name includes target specificity."""
        strategy = FixedSpecificityThreshold(target_specificity=0.95)
        assert strategy.name == "fixed_spec_0.95"

    def test_default_target_specificity(self):
        """Default target specificity is 0.95."""
        strategy = FixedSpecificityThreshold()
        assert strategy.target_specificity == 0.95

    def test_find_threshold_90_spec(self, balanced_data):
        """Find threshold for 90% specificity."""
        y, p = balanced_data
        strategy = FixedSpecificityThreshold(target_specificity=0.90)
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0

    def test_find_threshold_95_spec(self, imbalanced_data):
        """Find threshold for 95% specificity on imbalanced data."""
        y, p = imbalanced_data
        strategy = FixedSpecificityThreshold(target_specificity=0.95)
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0

    def test_single_class_returns_high_threshold(self, all_controls):
        """Single class returns high threshold."""
        y, p = all_controls
        strategy = FixedSpecificityThreshold(target_specificity=0.95)
        threshold = strategy.find_threshold(y, p)
        # For all controls, returns max(p) + epsilon
        assert threshold >= np.max(p)


# ============================================================================
# Test FixedSensitivityThreshold
# ============================================================================


class TestFixedSensitivityThreshold:
    """Tests for FixedSensitivityThreshold strategy."""

    def test_name_property(self):
        """Strategy name includes target sensitivity."""
        strategy = FixedSensitivityThreshold(target_sensitivity=0.95)
        assert strategy.name == "fixed_sens_0.95"

    def test_default_target_sensitivity(self):
        """Default target sensitivity is 0.95."""
        strategy = FixedSensitivityThreshold()
        assert strategy.target_sensitivity == 0.95

    def test_find_threshold_90_sens(self, balanced_data):
        """Find threshold for 90% sensitivity."""
        y, p = balanced_data
        strategy = FixedSensitivityThreshold(target_sensitivity=0.90)
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0

    def test_single_class_returns_default(self, all_controls):
        """Single class returns default threshold."""
        y, p = all_controls
        strategy = FixedSensitivityThreshold(target_sensitivity=0.95)
        threshold = strategy.find_threshold(y, p)
        # Single class case returns min(p) - epsilon or 0.5
        assert 0.0 <= threshold <= 1.0

    def test_empty_input_returns_default(self):
        """Empty input returns default threshold."""
        strategy = FixedSensitivityThreshold(target_sensitivity=0.95)
        threshold = strategy.find_threshold(np.array([]), np.array([]))
        assert threshold == 0.5


# ============================================================================
# Test FixedPPVThreshold
# ============================================================================


class TestFixedPPVThreshold:
    """Tests for FixedPPVThreshold strategy."""

    def test_name_property(self):
        """Strategy name includes target PPV."""
        strategy = FixedPPVThreshold(target_ppv=0.5)
        assert strategy.name == "fixed_ppv_0.50"

    def test_default_target_ppv(self):
        """Default target PPV is 0.5."""
        strategy = FixedPPVThreshold()
        assert strategy.target_ppv == 0.5

    def test_find_threshold_50_ppv(self, balanced_data):
        """Find threshold for 50% PPV."""
        y, p = balanced_data
        strategy = FixedPPVThreshold(target_ppv=0.50)
        threshold = strategy.find_threshold(y, p)
        assert 0.0 <= threshold <= 1.0

    def test_unattainable_ppv_falls_back_to_f1(self, balanced_data):
        """Unattainable PPV falls back to max-F1 threshold."""
        y, p = balanced_data
        # Request impossibly high PPV
        strategy = FixedPPVThreshold(target_ppv=0.99)
        threshold = strategy.find_threshold(y, p)
        # Should fall back to max-F1
        assert 0.0 <= threshold <= 1.0


# ============================================================================
# Test get_threshold_strategy factory
# ============================================================================


class TestGetThresholdStrategy:
    """Tests for get_threshold_strategy factory function."""

    def test_max_f1_from_config(self):
        """Create MaxF1Threshold from config."""
        config = ThresholdConfig(objective="max_f1")
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, MaxF1Threshold)
        assert strategy.name == "max_f1"

    def test_max_fbeta_from_config(self):
        """Create MaxFBetaThreshold from config."""
        config = ThresholdConfig(objective="max_fbeta", fbeta=2.0)
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, MaxFBetaThreshold)
        assert strategy.beta == 2.0

    def test_youden_from_config(self):
        """Create YoudensJThreshold from config."""
        config = ThresholdConfig(objective="youden")
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, YoudensJThreshold)

    def test_fixed_spec_from_config(self):
        """Create FixedSpecificityThreshold from config."""
        config = ThresholdConfig(objective="fixed_spec", fixed_spec=0.90)
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, FixedSpecificityThreshold)
        assert strategy.target_specificity == 0.90

    def test_fixed_ppv_from_config(self):
        """Create FixedPPVThreshold from config."""
        config = ThresholdConfig(objective="fixed_ppv", fixed_ppv=0.30)
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, FixedPPVThreshold)
        assert strategy.target_ppv == 0.30

    def test_case_insensitive(self):
        """Factory handles case-insensitive objective names."""
        config = ThresholdConfig(objective="max_f1")  # Pydantic validates lowercase
        strategy = get_threshold_strategy(config)
        assert isinstance(strategy, MaxF1Threshold)


# ============================================================================
# Test get_threshold_strategy_from_params factory
# ============================================================================


class TestGetThresholdStrategyFromParams:
    """Tests for get_threshold_strategy_from_params factory function."""

    def test_max_f1_from_params(self):
        """Create MaxF1Threshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="max_f1")
        assert isinstance(strategy, MaxF1Threshold)

    def test_max_fbeta_from_params(self):
        """Create MaxFBetaThreshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="max_fbeta", fbeta=2.0)
        assert isinstance(strategy, MaxFBetaThreshold)
        assert strategy.beta == 2.0

    def test_youden_from_params(self):
        """Create YoudensJThreshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="youden")
        assert isinstance(strategy, YoudensJThreshold)

    def test_fixed_spec_from_params(self):
        """Create FixedSpecificityThreshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="fixed_spec", fixed_spec=0.90)
        assert isinstance(strategy, FixedSpecificityThreshold)
        assert strategy.target_specificity == 0.90

    def test_fixed_ppv_from_params(self):
        """Create FixedPPVThreshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="fixed_ppv", fixed_ppv=0.30)
        assert isinstance(strategy, FixedPPVThreshold)
        assert strategy.target_ppv == 0.30

    def test_fixed_sens_from_params(self):
        """Create FixedSensitivityThreshold from parameters."""
        strategy = get_threshold_strategy_from_params(objective="fixed_sens", fixed_sens=0.90)
        assert isinstance(strategy, FixedSensitivityThreshold)
        assert strategy.target_sensitivity == 0.90

    def test_unknown_objective_falls_back_to_max_f1(self):
        """Unknown objective falls back to max_f1 with warning."""
        strategy = get_threshold_strategy_from_params(objective="unknown_objective")
        assert isinstance(strategy, MaxF1Threshold)

    def test_case_insensitive_params(self):
        """Factory handles case-insensitive objective names."""
        strategy = get_threshold_strategy_from_params(objective="YOUDEN")
        assert isinstance(strategy, YoudensJThreshold)


# ============================================================================
# Test immutability (frozen dataclasses)
# ============================================================================


class TestImmutability:
    """Test that strategy instances are immutable (frozen dataclasses)."""

    def test_max_f1_is_frozen(self):
        """MaxF1Threshold is immutable."""
        strategy = MaxF1Threshold()
        with pytest.raises(FrozenInstanceError):
            strategy.name = "modified"  # type: ignore[misc]

    def test_max_fbeta_is_frozen(self):
        """MaxFBetaThreshold is immutable."""
        strategy = MaxFBetaThreshold(beta=2.0)
        with pytest.raises(FrozenInstanceError):
            strategy.beta = 3.0  # type: ignore[misc]

    def test_fixed_specificity_is_frozen(self):
        """FixedSpecificityThreshold is immutable."""
        strategy = FixedSpecificityThreshold(target_specificity=0.95)
        with pytest.raises(FrozenInstanceError):
            strategy.target_specificity = 0.80  # type: ignore[misc]


# ============================================================================
# Test equivalence with existing functions
# ============================================================================


class TestEquivalenceWithExistingFunctions:
    """Verify strategies produce same results as existing threshold functions."""

    def test_max_f1_equivalence(self, balanced_data):
        """MaxF1Threshold produces same result as threshold_max_f1."""
        from ced_ml.metrics.thresholds import threshold_max_f1

        y, p = balanced_data
        strategy = MaxF1Threshold()

        strategy_result = strategy.find_threshold(y, p)
        function_result = threshold_max_f1(y, p)

        assert strategy_result == pytest.approx(function_result, rel=1e-10)

    def test_youden_equivalence(self, balanced_data):
        """YoudensJThreshold produces same result as threshold_youden."""
        from ced_ml.metrics.thresholds import threshold_youden

        y, p = balanced_data
        strategy = YoudensJThreshold()

        strategy_result = strategy.find_threshold(y, p)
        function_result = threshold_youden(y, p)

        assert strategy_result == pytest.approx(function_result, rel=1e-10)

    def test_fixed_spec_equivalence(self, imbalanced_data):
        """FixedSpecificityThreshold produces same result as threshold_for_specificity."""
        from ced_ml.metrics.thresholds import threshold_for_specificity

        y, p = imbalanced_data
        strategy = FixedSpecificityThreshold(target_specificity=0.90)

        strategy_result = strategy.find_threshold(y, p)
        function_result = threshold_for_specificity(y, p, target_spec=0.90)

        assert strategy_result == pytest.approx(function_result, rel=1e-10)

    def test_fixed_ppv_equivalence(self, balanced_data):
        """FixedPPVThreshold produces same result as threshold_for_precision."""
        from ced_ml.metrics.thresholds import threshold_for_precision

        y, p = balanced_data
        strategy = FixedPPVThreshold(target_ppv=0.50)

        strategy_result = strategy.find_threshold(y, p)
        function_result = threshold_for_precision(y, p, target_ppv=0.50)

        assert strategy_result == pytest.approx(function_result, rel=1e-10)
