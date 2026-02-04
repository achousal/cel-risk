"""
Tests for models.prevalence module.

Tests prevalence adjustment logic and PrevalenceAdjustedModel wrapper.
"""

import numpy as np
import pytest
from ced_ml.models.prevalence import (
    PrevalenceAdjustedModel,
    _inv_logit,
    _logit,
    adjust_probabilities_for_prevalence,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# =============================================================================
# Logit/Inverse Logit Tests
# =============================================================================


def test_logit_basic():
    """Test logit function on basic inputs."""
    p = np.array([0.1, 0.5, 0.9])
    logits = _logit(p)

    # Check shape
    assert logits.shape == p.shape

    # Check specific values
    assert np.isclose(logits[1], 0.0)  # logit(0.5) = 0
    assert logits[0] < 0  # logit(p < 0.5) is negative
    assert logits[2] > 0  # logit(p > 0.5) is positive


def test_logit_edge_cases():
    """Test logit function handles edge cases safely."""
    # Very small probabilities
    p_small = np.array([1e-10, 1e-5])
    logits_small = _logit(p_small)
    assert np.all(np.isfinite(logits_small))
    assert np.all(logits_small < 0)

    # Very large probabilities
    p_large = np.array([0.99999, 1.0 - 1e-10])
    logits_large = _logit(p_large)
    assert np.all(np.isfinite(logits_large))
    assert np.all(logits_large > 0)


def test_inv_logit_basic():
    """Test inverse logit function on basic inputs."""
    z = np.array([-2.0, 0.0, 2.0])
    probs = _inv_logit(z)

    # Check shape
    assert probs.shape == z.shape

    # Check specific values
    assert np.isclose(probs[1], 0.5)  # inv_logit(0) = 0.5
    assert probs[0] < 0.5  # inv_logit(z < 0) < 0.5
    assert probs[2] > 0.5  # inv_logit(z > 0) > 0.5

    # Check range
    assert np.all((probs >= 0) & (probs <= 1))


def test_logit_inv_logit_inverse():
    """Test that logit and inv_logit are inverses."""
    p_original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Round trip: p -> logit -> inv_logit -> p
    logits = _logit(p_original)
    p_recovered = _inv_logit(logits)

    np.testing.assert_allclose(p_recovered, p_original, rtol=1e-6)


# =============================================================================
# Prevalence Adjustment Tests
# =============================================================================


def test_adjust_probabilities_basic():
    """Test prevalence adjustment with typical inputs."""
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sample_prev = 0.5  # Balanced training data
    target_prev = 0.1  # Imbalanced target population

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    # Check shape preserved
    assert adjusted.shape == probs.shape

    # Adjusted probabilities should be lower (target prev < sample prev)
    assert np.all(adjusted < probs)

    # Check valid probability range
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_adjust_probabilities_increase():
    """Test prevalence adjustment when target > sample."""
    probs = np.array([0.1, 0.3, 0.5])
    sample_prev = 0.1  # Low training prevalence
    target_prev = 0.5  # Higher target prevalence

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    # Adjusted probabilities should be higher
    assert np.all(adjusted > probs)


def test_adjust_probabilities_no_change():
    """Test prevalence adjustment when prevalences match."""
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sample_prev = 0.3
    target_prev = 0.3

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    # Should be identical when prevalences match
    np.testing.assert_allclose(adjusted, probs, rtol=1e-6)


def test_adjust_probabilities_edge_cases():
    """Test prevalence adjustment handles edge cases."""
    probs = np.array([0.01, 0.5, 0.99])

    # Very low target prevalence
    adjusted_low = adjust_probabilities_for_prevalence(probs, 0.5, 0.001)
    assert np.all(adjusted_low < probs)
    assert np.all(adjusted_low > 0)  # Still positive

    # Very high target prevalence
    adjusted_high = adjust_probabilities_for_prevalence(probs, 0.5, 0.999)
    assert np.all(adjusted_high > probs)
    assert np.all(adjusted_high < 1)  # Still less than 1


def test_adjust_probabilities_invalid_inputs():
    """Test prevalence adjustment returns unadjusted when inputs invalid."""
    probs = np.array([0.1, 0.3, 0.5])

    # Invalid sample prevalence (negative)
    result = adjust_probabilities_for_prevalence(probs, -0.1, 0.3)
    np.testing.assert_array_equal(result, probs)

    # Invalid target prevalence (> 1)
    result = adjust_probabilities_for_prevalence(probs, 0.5, 1.5)
    np.testing.assert_array_equal(result, probs)

    # Invalid sample prevalence (0)
    result = adjust_probabilities_for_prevalence(probs, 0.0, 0.3)
    np.testing.assert_array_equal(result, probs)

    # Invalid target prevalence (1)
    result = adjust_probabilities_for_prevalence(probs, 0.5, 1.0)
    np.testing.assert_array_equal(result, probs)

    # NaN prevalence
    result = adjust_probabilities_for_prevalence(probs, np.nan, 0.3)
    np.testing.assert_array_equal(result, probs)


def test_adjust_probabilities_celiac_scenario():
    """Test prevalence adjustment with realistic CeD scenario."""
    # Training: 1:5 case:control ratio (prevalence ~0.167)
    # Target: 0.34% prevalence (0.0034)
    probs = np.array([0.05, 0.1, 0.2, 0.5, 0.8])
    sample_prev = 0.167
    target_prev = 0.0034

    adjusted = adjust_probabilities_for_prevalence(probs, sample_prev, target_prev)

    # Adjusted probabilities should be much lower
    assert np.all(adjusted < probs * 0.1)

    # High-risk patients still retain relative ordering
    assert adjusted[-1] > adjusted[-2] > adjusted[-3]


# =============================================================================
# PrevalenceAdjustedModel Tests
# =============================================================================


@pytest.fixture
def simple_rf_model():
    """Create simple fitted RF model for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, 100)

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture
def simple_lr_model():
    """Create simple fitted LR model for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = rng.integers(0, 2, 100)

    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    return lr


def test_prevalence_model_initialization(simple_rf_model):
    """Test PrevalenceAdjustedModel initialization."""
    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.3, target_prevalence=0.01
    )

    assert wrapper.base_model is simple_rf_model
    assert wrapper.sample_prevalence == 0.3
    assert wrapper.target_prevalence == 0.01
    assert wrapper.classes_ is not None


def test_prevalence_model_predict_proba(simple_rf_model):
    """Test predict_proba with prevalence adjustment."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((20, 5))

    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    # Get predictions
    base_probs = simple_rf_model.predict_proba(X_test)
    adjusted_probs = wrapper.predict_proba(X_test)

    # Check shape
    assert adjusted_probs.shape == base_probs.shape
    assert adjusted_probs.shape[1] == 2

    # Check valid probabilities
    assert np.all(adjusted_probs >= 0)
    assert np.all(adjusted_probs <= 1)
    np.testing.assert_allclose(adjusted_probs.sum(axis=1), 1.0, rtol=1e-6)

    # Positive class probabilities should be lower (target < sample)
    # Exclude boundary predictions (exactly 0 or 1) where adjustment is clamped
    interior = (base_probs[:, 1] > 1e-6) & (base_probs[:, 1] < 1 - 1e-6)
    assert np.all(adjusted_probs[interior, 1] < base_probs[interior, 1])


def test_prevalence_model_predict(simple_rf_model):
    """Test predict method uses adjusted probabilities."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((20, 5))

    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    # Get predictions
    base_preds = simple_rf_model.predict(X_test)
    adjusted_preds = wrapper.predict(X_test)

    # Check shape and type
    assert adjusted_preds.shape == base_preds.shape
    assert adjusted_preds.dtype == base_preds.dtype

    # Predictions may differ due to threshold effects
    # But should be valid class labels
    assert set(adjusted_preds).issubset({0, 1})


def test_prevalence_model_no_adjustment_when_invalid(simple_rf_model):
    """Test no adjustment applied when prevalences invalid."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((20, 5))

    # Invalid target prevalence
    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=-0.1
    )

    base_probs = simple_rf_model.predict_proba(X_test)
    adjusted_probs = wrapper.predict_proba(X_test)

    # Should be identical (no adjustment)
    np.testing.assert_array_equal(adjusted_probs, base_probs)


def test_prevalence_model_get_base_model(simple_rf_model):
    """Test get_base_model returns base model."""
    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    retrieved = wrapper.get_base_model()
    assert retrieved is simple_rf_model


def test_prevalence_model_attribute_delegation(simple_rf_model):
    """Test wrapper delegates attributes to base model."""
    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.1
    )

    # Check delegation works
    assert wrapper.n_estimators == simple_rf_model.n_estimators
    assert wrapper.n_features_in_ == simple_rf_model.n_features_in_
    assert hasattr(wrapper, "feature_importances_")


def test_prevalence_model_with_lr(simple_lr_model):
    """Test wrapper works with LogisticRegression."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((20, 5))

    wrapper = PrevalenceAdjustedModel(
        base_model=simple_lr_model, sample_prevalence=0.5, target_prevalence=0.05
    )

    # Get predictions
    base_probs = simple_lr_model.predict_proba(X_test)
    adjusted_probs = wrapper.predict_proba(X_test)

    # Check valid output
    assert adjusted_probs.shape == base_probs.shape
    assert np.all((adjusted_probs >= 0) & (adjusted_probs <= 1))

    # Positive class should be lower
    assert np.all(adjusted_probs[:, 1] < base_probs[:, 1])


def test_prevalence_model_extreme_adjustment(simple_rf_model):
    """Test wrapper handles extreme prevalence adjustments."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((20, 5))

    # Extreme downward adjustment (training 50%, target 0.1%)
    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.001
    )

    base_probs = simple_rf_model.predict_proba(X_test)
    adjusted_probs = wrapper.predict_proba(X_test)

    # Should still be valid probabilities
    assert np.all((adjusted_probs >= 0) & (adjusted_probs <= 1))
    np.testing.assert_allclose(adjusted_probs.sum(axis=1), 1.0, rtol=1e-6)

    # Adjusted probabilities should be lower than base (downward adjustment)
    # Exclude boundary predictions (exactly 0 or 1) where adjustment is clamped
    interior = (base_probs[:, 1] > 1e-6) & (base_probs[:, 1] < 1 - 1e-6)
    assert np.all(adjusted_probs[interior, 1] <= base_probs[interior, 1])

    # Median probability should be much lower (more robust than mean to outliers)
    assert np.median(adjusted_probs[:, 1]) < np.median(base_probs[:, 1]) * 0.05


def test_prevalence_model_serialization(simple_rf_model, tmp_path):
    """Test wrapper can be serialized and deserialized."""
    import joblib

    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.3, target_prevalence=0.01
    )

    # Save
    model_path = tmp_path / "model.joblib"
    joblib.dump(wrapper, model_path)

    # Load
    loaded = joblib.load(model_path)

    # Check attributes preserved
    assert loaded.sample_prevalence == 0.3
    assert loaded.target_prevalence == 0.01

    # Check predictions match
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((10, 5))

    probs_original = wrapper.predict_proba(X_test)
    probs_loaded = loaded.predict_proba(X_test)

    np.testing.assert_array_equal(probs_loaded, probs_original)


def test_prevalence_model_preserves_ordering(simple_rf_model):
    """Test prevalence adjustment preserves probability ordering."""
    rng = np.random.default_rng(42)
    X_test = rng.standard_normal((50, 5))

    wrapper = PrevalenceAdjustedModel(
        base_model=simple_rf_model, sample_prevalence=0.5, target_prevalence=0.05
    )

    base_probs = simple_rf_model.predict_proba(X_test)[:, 1]
    adjusted_probs = wrapper.predict_proba(X_test)[:, 1]

    # Check ordering preserved (monotonic transformation)
    base_order = np.argsort(base_probs)
    adjusted_order = np.argsort(adjusted_probs)

    np.testing.assert_array_equal(base_order, adjusted_order)
