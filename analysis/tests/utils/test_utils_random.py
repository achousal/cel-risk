"""Tests for utils.random module (seed management)."""

import os

import numpy as np
from ced_ml.utils.random import apply_seed_global, get_cv_seed


class TestApplySeedGlobal:
    """Tests for apply_seed_global (SEED_GLOBAL env var)."""

    def setup_method(self):
        """Remove SEED_GLOBAL from environment before each test."""
        os.environ.pop("SEED_GLOBAL", None)

    def teardown_method(self):
        """Clean up SEED_GLOBAL after each test."""
        os.environ.pop("SEED_GLOBAL", None)

    def test_returns_none_when_unset(self):
        """Test returns None when SEED_GLOBAL not in environment."""
        result = apply_seed_global()
        assert result is None

    def test_returns_none_when_empty(self):
        """Test returns None when SEED_GLOBAL is empty string."""
        os.environ["SEED_GLOBAL"] = ""
        result = apply_seed_global()
        assert result is None

    def test_returns_none_when_whitespace(self):
        """Test returns None when SEED_GLOBAL is whitespace."""
        os.environ["SEED_GLOBAL"] = "   "
        result = apply_seed_global()
        assert result is None

    def test_returns_none_when_non_integer(self):
        """Test returns None when SEED_GLOBAL is not a valid integer."""
        os.environ["SEED_GLOBAL"] = "abc"
        result = apply_seed_global()
        assert result is None

    def test_returns_seed_when_valid(self):
        """Test returns parsed seed when SEED_GLOBAL is valid integer."""
        os.environ["SEED_GLOBAL"] = "42"
        result = apply_seed_global()
        assert result == 42

    def test_applies_global_seed(self):
        """Test that SEED_GLOBAL actually seeds the RNG deterministically."""
        os.environ["SEED_GLOBAL"] = "123"
        apply_seed_global()
        a = np.random.random(5)

        os.environ["SEED_GLOBAL"] = "123"
        apply_seed_global()
        b = np.random.random(5)

        np.testing.assert_array_equal(a, b)

    def test_negative_seed_rejected(self):
        """Test that negative seeds are rejected (numpy range: [0, 2^32-1])."""
        os.environ["SEED_GLOBAL"] = "-1"
        result = apply_seed_global()
        assert result is None

    def test_too_large_seed_rejected(self):
        """Test that seeds > 2^32-1 are rejected."""
        os.environ["SEED_GLOBAL"] = str(2**32)
        result = apply_seed_global()
        assert result is None

    def test_zero_seed(self):
        """Test that seed 0 is valid."""
        os.environ["SEED_GLOBAL"] = "0"
        result = apply_seed_global()
        assert result == 0

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        os.environ["SEED_GLOBAL"] = "  42  "
        result = apply_seed_global()
        assert result == 42


class TestGetCvSeed:
    """Tests for get_cv_seed."""

    def test_base_case(self):
        """Test seed generation with fold 0, repeat 0."""
        assert get_cv_seed(100, 0, 0) == 100

    def test_fold_offset(self):
        """Test fold index offsets seed."""
        assert get_cv_seed(100, 3, 0) == 103

    def test_repeat_offset(self):
        """Test repeat index offsets seed by 1000."""
        assert get_cv_seed(100, 0, 2) == 2100

    def test_combined(self):
        """Test fold + repeat combined offset."""
        assert get_cv_seed(100, 5, 3) == 3105

    def test_deterministic(self):
        """Test same inputs produce same output."""
        assert get_cv_seed(42, 2, 1) == get_cv_seed(42, 2, 1)

    def test_different_folds_differ(self):
        """Test different folds produce different seeds."""
        assert get_cv_seed(42, 0) != get_cv_seed(42, 1)
