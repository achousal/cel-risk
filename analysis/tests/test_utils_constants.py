"""Tests for utils.constants module.

Coverage areas:
- Value range consistency (MIN < MAX bounds)
- CI parameter consistency (lower + upper = 100, alpha matches)
- Bootstrap minimum is positive
- All constants have expected types
"""

from ced_ml.utils.constants import (
    CI_ALPHA,
    CI_LOWER_PCT,
    CI_UPPER_PCT,
    DEFAULT_TARGET_SPEC,
    MAX_SAFE_PREVALENCE,
    MIN_BOOTSTRAP_SAMPLES,
    MIN_SAFE_PREVALENCE,
    Z_CRITICAL_005,
)


class TestCIConstants:
    def test_alpha_in_unit_interval(self):
        assert 0.0 < CI_ALPHA < 1.0

    def test_lower_less_than_upper(self):
        assert CI_LOWER_PCT < CI_UPPER_PCT

    def test_percentiles_sum_to_100(self):
        assert CI_LOWER_PCT + CI_UPPER_PCT == 100.0

    def test_percentiles_consistent_with_alpha(self):
        expected_lower = 100.0 * CI_ALPHA / 2.0
        assert CI_LOWER_PCT == expected_lower
        assert CI_UPPER_PCT == 100.0 - expected_lower


class TestPrevalenceBounds:
    def test_min_less_than_max(self):
        assert MIN_SAFE_PREVALENCE < MAX_SAFE_PREVALENCE

    def test_min_positive(self):
        assert MIN_SAFE_PREVALENCE > 0.0

    def test_max_at_most_one(self):
        assert MAX_SAFE_PREVALENCE <= 1.0


class TestMiscConstants:
    def test_bootstrap_samples_positive_integer(self):
        assert isinstance(MIN_BOOTSTRAP_SAMPLES, int)
        assert MIN_BOOTSTRAP_SAMPLES > 0

    def test_z_critical_positive(self):
        assert Z_CRITICAL_005 > 0.0

    def test_default_target_spec_in_unit_interval(self):
        assert 0.0 < DEFAULT_TARGET_SPEC < 1.0
