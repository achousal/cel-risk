"""Tests for utils.math_utils module.

Coverage areas:
- logit / inv_logit roundtrip consistency
- Numerical stability at boundaries (0, 1, near-zero)
- Epsilon constants are positive and ordered
- jeffreys_smooth correctness and edge cases
"""

import numpy as np
import pytest
from ced_ml.utils.math_utils import (
    EPSILON_BOUNDS,
    EPSILON_LOGIT,
    EPSILON_LOGLOSS,
    EPSILON_PREVALENCE,
    JEFFREYS_ALPHA,
    inv_logit,
    jeffreys_smooth,
    logit,
)

# ============================================================================
# Epsilon Constants
# ============================================================================


class TestEpsilonConstants:
    def test_all_positive(self):
        for eps in (EPSILON_LOGIT, EPSILON_LOGLOSS, EPSILON_PREVALENCE, EPSILON_BOUNDS):
            assert eps > 0.0

    def test_all_less_than_one(self):
        for eps in (EPSILON_LOGIT, EPSILON_LOGLOSS, EPSILON_PREVALENCE, EPSILON_BOUNDS):
            assert eps < 1.0

    def test_jeffreys_alpha_positive(self):
        assert JEFFREYS_ALPHA > 0.0


# ============================================================================
# logit()
# ============================================================================


class TestLogit:
    def test_midpoint_is_zero(self):
        assert logit(np.array([0.5]))[0] == pytest.approx(0.0)

    def test_monotonically_increasing(self):
        p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = logit(p)
        assert np.all(np.diff(result) > 0)

    def test_boundary_zero_no_nan(self):
        result = logit(np.array([0.0]))
        assert np.isfinite(result).all()

    def test_boundary_one_no_nan(self):
        result = logit(np.array([1.0]))
        assert np.isfinite(result).all()

    def test_near_zero_finite(self):
        result = logit(np.array([1e-15]))
        assert np.isfinite(result).all()

    def test_near_one_finite(self):
        result = logit(np.array([1.0 - 1e-15]))
        assert np.isfinite(result).all()

    def test_scalar_input(self):
        result = logit(0.5)
        assert result == pytest.approx(0.0)

    def test_custom_eps(self):
        eps = 0.01
        result = logit(np.array([0.0]), eps=eps)
        expected = np.log(eps / (1.0 - eps))
        assert result[0] == pytest.approx(expected)


# ============================================================================
# inv_logit()
# ============================================================================


class TestInvLogit:
    def test_zero_gives_half(self):
        assert inv_logit(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive_near_one(self):
        result = inv_logit(np.array([100.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_near_zero(self):
        result = inv_logit(np.array([-100.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_output_in_unit_interval(self):
        z = np.linspace(-10, 10, 100)
        result = inv_logit(z)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_scalar_input(self):
        result = inv_logit(0.0)
        assert result == pytest.approx(0.5)


# ============================================================================
# Roundtrip: logit <-> inv_logit
# ============================================================================


class TestRoundtrip:
    def test_logit_then_inv_logit(self):
        p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        np.testing.assert_allclose(inv_logit(logit(p)), p, atol=1e-6)

    def test_inv_logit_then_logit(self):
        z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(logit(inv_logit(z)), z, atol=1e-6)


# ============================================================================
# jeffreys_smooth()
# ============================================================================


class TestJeffreysSmooth:
    def test_zero_successes(self):
        result = jeffreys_smooth(0, 10)
        expected = (0 + 0.5) / (10 + 1.0)
        assert result == pytest.approx(expected)

    def test_all_successes(self):
        result = jeffreys_smooth(10, 10)
        expected = (10 + 0.5) / (10 + 1.0)
        assert result == pytest.approx(expected)

    def test_custom_alpha(self):
        result = jeffreys_smooth(5, 20, alpha=1.0)
        expected = (5 + 1.0) / (20 + 2.0)
        assert result == pytest.approx(expected)

    def test_output_in_unit_interval(self):
        for k in range(11):
            result = jeffreys_smooth(k, 10)
            assert 0.0 < result < 1.0

    def test_zero_trials(self):
        result = jeffreys_smooth(0, 0)
        expected = 0.5 / 1.0
        assert result == pytest.approx(expected)
