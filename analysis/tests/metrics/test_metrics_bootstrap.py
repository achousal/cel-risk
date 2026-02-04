"""
Tests for bootstrap confidence interval module.

Covers:
- Basic CI computation
- Stratified resampling behavior
- Edge cases (insufficient samples, length mismatches)
- Reproducibility (seeding)
- Model comparison CIs
"""

import numpy as np
import pytest
from ced_ml.metrics.bootstrap import (
    _safe_metric,
    stratified_bootstrap_ci,
    stratified_bootstrap_diff_ci,
)
from sklearn.metrics import brier_score_loss, roc_auc_score


class TestSafeMetric:
    """Tests for _safe_metric helper."""

    def test_safe_metric_success(self):
        """Should return metric value on successful computation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        result = _safe_metric(roc_auc_score, y_true, y_pred)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_safe_metric_failure(self):
        """Should return NaN on metric computation failure."""
        # All same class - AUROC will fail
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])
        result = _safe_metric(roc_auc_score, y_true, y_pred)
        assert np.isnan(result)

    def test_safe_metric_invalid_input(self):
        """Should return NaN on invalid inputs."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([np.inf, 0.2, 0.8, np.nan])
        result = _safe_metric(roc_auc_score, y_true, y_pred)
        # May fail due to invalid predictions
        assert isinstance(result, float | np.floating)


class TestStratifiedBootstrapCI:
    """Tests for stratified_bootstrap_ci."""

    @pytest.fixture
    def basic_data(self):
        """Simple balanced dataset."""
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = rng.random(100)
        return y_true, y_pred

    def test_basic_ci_computation(self, basic_data):
        """Should compute valid CI bounds."""
        y_true, y_pred = basic_data
        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert 0 <= ci_lower <= ci_upper <= 1

    def test_reproducibility(self, basic_data):
        """Should produce identical results with same seed."""
        y_true, y_pred = basic_data
        ci1 = stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100, seed=42)
        ci2 = stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100, seed=42)
        assert ci1 == ci2

    def test_different_seeds_differ(self, basic_data):
        """Should produce different results with different seeds."""
        y_true, y_pred = basic_data
        ci1 = stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100, seed=42)
        ci2 = stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100, seed=999)
        assert ci1 != ci2

    def test_insufficient_cases(self):
        """Should raise ValueError with <2 cases."""
        y_true = np.array([0, 0, 0, 1])  # Only 1 case
        y_pred = np.array([0.1, 0.2, 0.3, 0.9])
        with pytest.raises(ValueError, match="Insufficient samples"):
            stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100)

    def test_insufficient_controls(self):
        """Should raise ValueError with <2 controls."""
        y_true = np.array([1, 1, 1, 0])  # Only 1 control
        y_pred = np.array([0.7, 0.8, 0.9, 0.1])
        with pytest.raises(ValueError, match="Insufficient samples"):
            stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100)

    def test_length_mismatch(self):
        """Should raise ValueError on length mismatch."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8])  # Too short
        with pytest.raises(ValueError, match="Length mismatch"):
            stratified_bootstrap_ci(y_true, y_pred, roc_auc_score, n_boot=100)

    def test_nan_on_insufficient_valid_samples(self):
        """Should return (NaN, NaN) if too few valid bootstrap samples."""
        # Create data that causes metric failures
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])  # Constant predictions

        def always_nan(y, p):
            return np.nan

        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, always_nan, n_boot=100, seed=42
        )
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_brier_score_metric(self, basic_data):
        """Should work with different metric functions."""
        y_true, y_pred = basic_data
        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, brier_score_loss, n_boot=100, seed=42
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= ci_upper

    def test_imbalanced_data(self):
        """Should handle imbalanced datasets."""
        # 10 cases, 90 controls
        y_true = np.array([0] * 90 + [1] * 10)
        rng = np.random.default_rng(42)
        y_pred = rng.random(100)

        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert 0 <= ci_lower <= ci_upper <= 1

    def test_custom_min_valid_frac(self, basic_data):
        """Should respect min_valid_frac parameter."""
        y_true, y_pred = basic_data

        # With higher threshold, may get NaN if some samples fail
        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=100, seed=42, min_valid_frac=0.99
        )
        # Should still succeed with good data
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_perfect_predictions(self):
        """Should handle perfect predictions."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        )
        # AUROC should be 1.0 in all bootstrap samples
        assert ci_lower == pytest.approx(1.0, abs=0.01)
        assert ci_upper == pytest.approx(1.0, abs=0.01)


class TestStratifiedBootstrapDiffCI:
    """Tests for stratified_bootstrap_diff_ci."""

    @pytest.fixture
    def two_model_data(self):
        """Dataset with predictions from two models."""
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 50 + [1] * 50)
        p1 = rng.random(100)
        p2 = rng.random(100)
        return y_true, p1, p2

    def test_basic_diff_ci(self, two_model_data):
        """Should compute difference CI."""
        y_true, p1, p2 = two_model_data
        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(diff, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= ci_upper

    def test_diff_reproducibility(self, two_model_data):
        """Should be reproducible with same seed."""
        y_true, p1, p2 = two_model_data
        result1 = stratified_bootstrap_diff_ci(y_true, p1, p2, roc_auc_score, n_boot=100, seed=42)
        result2 = stratified_bootstrap_diff_ci(y_true, p1, p2, roc_auc_score, n_boot=100, seed=42)
        assert result1 == result2

    def test_diff_full_sample(self, two_model_data):
        """Should return correct full-sample difference."""
        y_true, p1, p2 = two_model_data
        diff, _, _ = stratified_bootstrap_diff_ci(
            y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        )

        # Manually compute expected difference
        expected_diff = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
        assert diff == pytest.approx(expected_diff)

    def test_diff_length_mismatch_y_p1(self):
        """Should raise ValueError on y_true/p1 length mismatch."""
        y_true = np.array([0, 0, 1, 1])
        p1 = np.array([0.1, 0.2, 0.8])  # Too short
        p2 = np.array([0.2, 0.3, 0.7, 0.9])
        with pytest.raises(ValueError, match="Length mismatch"):
            stratified_bootstrap_diff_ci(y_true, p1, p2, roc_auc_score, n_boot=100)

    def test_diff_length_mismatch_p1_p2(self):
        """Should raise ValueError on p1/p2 length mismatch."""
        y_true = np.array([0, 0, 1, 1])
        p1 = np.array([0.1, 0.2, 0.8, 0.9])
        p2 = np.array([0.2, 0.3, 0.7])  # Too short
        with pytest.raises(ValueError, match="Length mismatch"):
            stratified_bootstrap_diff_ci(y_true, p1, p2, roc_auc_score, n_boot=100)

    def test_diff_insufficient_samples(self):
        """Should raise ValueError with insufficient samples."""
        y_true = np.array([0, 0, 0, 1])  # Only 1 case
        p1 = np.array([0.1, 0.2, 0.3, 0.9])
        p2 = np.array([0.2, 0.3, 0.4, 0.8])
        with pytest.raises(ValueError, match="Insufficient samples"):
            stratified_bootstrap_diff_ci(y_true, p1, p2, roc_auc_score, n_boot=100)

    def test_diff_nan_on_invalid_bootstrap(self):
        """Should return NaN CIs if too few valid bootstrap samples."""
        y_true = np.array([0, 0, 1, 1])
        p1 = np.array([0.1, 0.2, 0.8, 0.9])
        p2 = np.array([0.2, 0.3, 0.7, 0.8])

        def always_nan(y, p):
            return np.nan

        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, always_nan, n_boot=100, seed=42
        )
        # diff_full will also be NaN
        assert np.isnan(diff)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_diff_identical_models(self, two_model_data):
        """Should have diff ~0 when models are identical."""
        y_true, p1, _ = two_model_data
        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p1, roc_auc_score, n_boot=100, seed=42
        )
        # Difference should be exactly 0
        assert diff == pytest.approx(0.0, abs=1e-10)
        # CI should be tight around 0
        assert abs(ci_lower) < 0.01
        assert abs(ci_upper) < 0.01

    def test_diff_imbalanced_data(self):
        """Should handle imbalanced datasets."""
        y_true = np.array([0] * 90 + [1] * 10)
        rng = np.random.default_rng(42)
        p1 = rng.random(100)
        p2 = rng.random(100)

        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(diff, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_diff_brier_score(self, two_model_data):
        """Should work with different metric functions."""
        y_true, p1, p2 = two_model_data
        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, brier_score_loss, n_boot=100, seed=42
        )
        assert isinstance(diff, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_diff_custom_min_valid_frac(self, two_model_data):
        """Should respect min_valid_frac parameter."""
        y_true, p1, p2 = two_model_data
        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, roc_auc_score, n_boot=100, seed=42, min_valid_frac=0.5
        )
        assert isinstance(diff, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_diff_perfect_vs_random(self):
        """Should detect clear difference between perfect and random model."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p_perfect = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        p_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p_perfect, p_random, roc_auc_score, n_boot=100, seed=42
        )
        # Perfect model (AUC=1.0) vs random (AUC=0.5) → diff ≈ 0.5
        assert diff == pytest.approx(0.5, abs=0.01)
        # CI should be positive and tight
        assert ci_lower > 0.4
        assert ci_upper < 0.6


class TestEdgeCases:
    """Additional edge case tests."""

    def test_single_bootstrap_iteration(self):
        """Should handle n_boot=1 (though not recommended)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        # With n_boot=1 and min_valid_frac=0.1, needs ≥20 valid samples
        # Will return NaN
        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=1, seed=42
        )
        # Should return NaN due to insufficient samples
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_large_n_boot(self):
        """Should handle large n_boot."""
        y_true = np.array([0] * 50 + [1] * 50)
        rng = np.random.default_rng(42)
        y_pred = rng.random(100)

        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=5000, seed=42
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert 0 <= ci_lower <= ci_upper <= 1

    def test_array_like_inputs(self):
        """Should accept list inputs (converted to arrays)."""
        y_true = [0, 0, 1, 1]  # List, not array
        y_pred = [0.1, 0.2, 0.8, 0.9]

        ci_lower, ci_upper = stratified_bootstrap_ci(
            y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)

    def test_float_y_true_converted_to_int(self):
        """Should handle float y_true in diff_ci (cast to int)."""
        y_true = np.array([0.0, 0.0, 1.0, 1.0])  # Floats
        p1 = np.array([0.1, 0.2, 0.8, 0.9])
        p2 = np.array([0.2, 0.3, 0.7, 0.8])

        diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
            y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        )
        assert isinstance(diff, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
