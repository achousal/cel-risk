"""
Tests for discrimination metrics module.

Tests cover:
- AUROC computation and edge cases
- PR-AUC computation and imbalanced data behavior
- Youden's J statistic
- Alpha (sensitivity at specificity)
- Brier score and log loss
- Aggregate metric computation
- Numerical stability and error handling
"""

import numpy as np
import pytest
from ced_ml.metrics.discrimination import (
    alpha_sensitivity_at_specificity,
    auroc,
    compute_brier_score,
    compute_discrimination_metrics,
    compute_log_loss,
    prauc,
    youden_j,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


class TestAUROC:
    """Test AUROC computation."""

    def test_perfect_separation(self):
        """AUROC = 1.0 for perfect ranking."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert auroc(y_true, y_pred) == 1.0

    def test_random_classifier(self):
        """AUROC ≈ 0.5 for random predictions."""
        np.random.seed(42)
        y_true = np.array([0, 1] * 50)
        y_pred = np.random.random(100)
        auc = auroc(y_true, y_pred)
        # Random classifier should be close to 0.5 (allow some variance)
        assert 0.4 <= auc <= 0.6

    def test_inverted_predictions(self):
        """AUROC = 0.0 for completely inverted predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])
        assert auroc(y_true, y_pred) == 0.0

    def test_imbalanced_data(self):
        """AUROC handles imbalanced datasets correctly."""
        rng = np.random.default_rng(42)
        # 95 negatives, 5 positives
        y_true = np.array([0] * 95 + [1] * 5)
        y_pred = np.concatenate([rng.uniform(0, 0.3, 95), rng.uniform(0.7, 1.0, 5)])
        auc = auroc(y_true, y_pred)
        assert 0.9 <= auc <= 1.0

    def test_tied_predictions(self):
        """AUROC handles tied predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        auc = auroc(y_true, y_pred)
        # All tied predictions give AUROC = 0.5
        assert auc == 0.5

    def test_single_class_returns_nan(self):
        """AUROC returns NaN for single-class inputs (sklearn behavior)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        # sklearn returns NaN with warning for single-class
        result = auroc(y_true, y_pred)
        assert np.isnan(result)

    def test_type_conversion(self):
        """AUROC handles list inputs and converts to numpy arrays."""
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.8, 0.9]
        assert auroc(y_true, y_pred) == 1.0

    def test_matches_sklearn(self):
        """AUROC matches sklearn implementation."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = np.random.random(100)
        expected = roc_auc_score(y_true, y_pred)
        assert auroc(y_true, y_pred) == pytest.approx(expected)


class TestPRAUC:
    """Test PR-AUC computation."""

    def test_perfect_separation(self):
        """PR-AUC = 1.0 for perfect ranking."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert prauc(y_true, y_pred) == 1.0

    def test_imbalanced_baseline(self):
        """PR-AUC baseline equals prevalence for random classifier."""
        # For imbalanced data, random classifier PR-AUC ~ prevalence
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 95 + [1] * 5)
        y_pred = rng.random(100)
        pr = prauc(y_true, y_pred)
        prevalence = 5 / 100
        # Random predictions give PR-AUC near prevalence, allow generous range
        assert prevalence * 0.3 <= pr <= prevalence * 4.0

    def test_all_positives_ranked_last(self):
        """PR-AUC is low when positives ranked below negatives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.8, 0.9, 0.1, 0.2])
        pr = prauc(y_true, y_pred)
        # Should be below random (0.5 for balanced data)
        assert pr < 0.5

    def test_single_class_all_positives(self):
        """PR-AUC returns NaN for all positives (single-class guard)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        # With single-class guard, returns NaN and warns
        with pytest.warns(UserWarning, match="PR-AUC requires both classes"):
            result = prauc(y_true, y_pred)
        assert np.isnan(result)

    def test_matches_sklearn(self):
        """PR-AUC matches sklearn implementation."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = np.random.random(100)
        expected = average_precision_score(y_true, y_pred)
        assert prauc(y_true, y_pred) == pytest.approx(expected)


class TestYoudenJ:
    """Test Youden's J statistic."""

    def test_perfect_separation(self):
        """Youden J = 1.0 for perfect separation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert youden_j(y_true, y_pred) == 1.0

    def test_no_discrimination(self):
        """Youden J = 0.0 for no discrimination."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        j = youden_j(y_true, y_pred)
        assert j == pytest.approx(0.0, abs=1e-10)

    def test_partial_separation(self):
        """Youden J between 0 and 1 for partial separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.6, 0.4, 0.7, 0.8])
        j = youden_j(y_true, y_pred)
        assert 0.0 < j < 1.0

    def test_imbalanced_data(self):
        """Youden J handles imbalanced data correctly."""
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 95 + [1] * 5)
        y_pred = np.concatenate([rng.uniform(0, 0.3, 95), rng.uniform(0.7, 1.0, 5)])
        j = youden_j(y_true, y_pred)
        assert 0.0 <= j <= 1.0

    def test_single_class_returns_nan(self):
        """Youden J returns NaN for single-class inputs."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        # roc_curve returns empty arrays for single class, nanmax returns NaN
        result = youden_j(y_true, y_pred)
        assert np.isnan(result)


class TestAlphaSensitivityAtSpecificity:
    """Test Alpha (sensitivity at target specificity) metric."""

    def test_perfect_separation_high_specificity(self):
        """Alpha = 1.0 for perfect separation at any specificity."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        alpha = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
        assert alpha == 1.0

    def test_achievable_target(self):
        """Alpha returns max sensitivity when target specificity is achievable."""
        # Create data where 95% specificity is achievable
        y_true = np.array([0] * 100 + [1] * 10)
        # Negatives: scores 0.0-0.5, positives: scores 0.6-1.0
        y_pred = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.6, 1.0, 10)])
        alpha = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
        # At 95% specificity (5% FPR), should catch most positives
        assert alpha > 0.5

    def test_unachievable_target(self):
        """Alpha returns sensitivity at closest specificity when target unachievable."""
        # All predictions are identical, cannot achieve any specific specificity
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        alpha = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.99)
        # Should return sensitivity at closest achievable specificity
        assert 0.0 <= alpha <= 1.0

    def test_different_targets(self):
        """Alpha varies with target specificity."""
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 100 + [1] * 10)
        y_pred = np.concatenate([rng.uniform(0, 0.5, 100), rng.uniform(0.5, 1.0, 10)])
        alpha_95 = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
        alpha_99 = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.99)
        # Higher specificity requirement typically yields lower sensitivity
        assert alpha_99 <= alpha_95

    def test_invalid_target_raises(self):
        """Alpha raises error for invalid target specificity."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        with pytest.raises(ValueError, match="target_specificity must be in"):
            alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.0)

        with pytest.raises(ValueError, match="target_specificity must be in"):
            alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=1.0)

        with pytest.raises(ValueError, match="target_specificity must be in"):
            alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=1.5)

    def test_default_target(self):
        """Alpha uses 0.95 as default target specificity."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        alpha_default = alpha_sensitivity_at_specificity(y_true, y_pred)
        alpha_explicit = alpha_sensitivity_at_specificity(y_true, y_pred, target_specificity=0.95)
        assert alpha_default == alpha_explicit


class TestComputeDiscriminationMetrics:
    """Test aggregate discrimination metrics computation."""

    def test_all_metrics_included(self):
        """All metrics computed when all flags enabled."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_discrimination_metrics(
            y_true, y_pred, include_youden=True, include_alpha=True
        )
        assert set(metrics.keys()) == {"auroc", "prauc", "Youden", "Alpha"}
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_exclude_youden(self):
        """Youden excluded when flag disabled."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_discrimination_metrics(
            y_true, y_pred, include_youden=False, include_alpha=True
        )
        assert "Youden" not in metrics
        assert "Alpha" in metrics

    def test_exclude_alpha(self):
        """Alpha excluded when flag disabled."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_discrimination_metrics(
            y_true, y_pred, include_youden=True, include_alpha=False
        )
        assert "Youden" in metrics
        assert "Alpha" not in metrics

    def test_minimal_metrics(self):
        """Only AUROC and PR-AUC when both flags disabled."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = compute_discrimination_metrics(
            y_true, y_pred, include_youden=False, include_alpha=False
        )
        assert set(metrics.keys()) == {"auroc", "prauc"}

    def test_custom_alpha_target(self):
        """Custom alpha target specificity is respected."""
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 100 + [1] * 10)
        y_pred = np.concatenate([rng.uniform(0, 0.5, 100), rng.uniform(0.5, 1.0, 10)])
        metrics_95 = compute_discrimination_metrics(
            y_true, y_pred, include_alpha=True, alpha_target_specificity=0.95
        )
        metrics_99 = compute_discrimination_metrics(
            y_true, y_pred, include_alpha=True, alpha_target_specificity=0.99
        )
        # Higher specificity requirement should yield lower or equal sensitivity
        assert metrics_99["Alpha"] <= metrics_95["Alpha"]

    def test_perfect_predictions(self):
        """All metrics = 1.0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        metrics = compute_discrimination_metrics(
            y_true, y_pred, include_youden=True, include_alpha=True
        )
        assert metrics["auroc"] == 1.0
        assert metrics["prauc"] == 1.0
        assert metrics["Youden"] == 1.0
        assert metrics["Alpha"] == 1.0


class TestBrierScore:
    """Test Brier score computation."""

    def test_perfect_predictions(self):
        """Brier = 0.0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        assert compute_brier_score(y_true, y_pred) == 0.0

    def test_constant_prediction(self):
        """Brier = 0.25 for constant 0.5 prediction on balanced data."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        assert compute_brier_score(y_true, y_pred) == 0.25

    def test_worst_predictions(self):
        """Brier = 1.0 for completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])
        assert compute_brier_score(y_true, y_pred) == 1.0

    def test_matches_sklearn(self):
        """Brier score matches sklearn implementation."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = np.random.random(100)
        expected = brier_score_loss(y_true, y_pred)
        assert compute_brier_score(y_true, y_pred) == pytest.approx(expected)


class TestLogLoss:
    """Test log loss computation."""

    def test_perfect_predictions(self):
        """Log loss ≈ 0.0 for perfect predictions (with clipping)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        ll = compute_log_loss(y_true, y_pred)
        # With clipping, cannot reach exactly 0.0
        assert ll < 1e-10

    def test_constant_prediction(self):
        """Log loss ≈ log(2) for constant 0.5 prediction."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        ll = compute_log_loss(y_true, y_pred)
        assert ll == pytest.approx(np.log(2), rel=0.01)

    def test_clipping_prevents_inf(self):
        """Clipping prevents log(0) = inf."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])  # Would cause log(0) without clipping
        ll = compute_log_loss(y_true, y_pred)
        assert np.isfinite(ll)

    def test_custom_eps(self):
        """Custom epsilon for clipping is respected."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        ll_small_eps = compute_log_loss(y_true, y_pred, eps=1e-15)
        ll_large_eps = compute_log_loss(y_true, y_pred, eps=1e-3)
        # Larger epsilon means more aggressive clipping, higher loss
        assert ll_large_eps > ll_small_eps

    def test_matches_sklearn_with_clipping(self):
        """Log loss matches sklearn when probabilities are clipped."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = np.random.random(100)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        expected = log_loss(y_true, y_pred_clipped)
        assert compute_log_loss(y_true, y_pred) == pytest.approx(expected)


class TestNumericalStability:
    """Test numerical stability across edge cases."""

    def test_extreme_probabilities(self):
        """Metrics handle extreme probabilities (0.0, 1.0) without errors."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])

        assert auroc(y_true, y_pred) == 1.0
        assert prauc(y_true, y_pred) == 1.0
        assert youden_j(y_true, y_pred) == 1.0
        assert alpha_sensitivity_at_specificity(y_true, y_pred) == 1.0
        assert compute_brier_score(y_true, y_pred) == 0.0
        assert np.isfinite(compute_log_loss(y_true, y_pred))

    def test_very_small_probabilities(self):
        """Metrics handle very small but non-zero probabilities."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1e-10, 1e-9, 0.999999, 1.0 - 1e-10])

        metrics = compute_discrimination_metrics(y_true, y_pred)
        assert all(np.isfinite(v) for v in metrics.values())

    def test_large_sample_size(self):
        """Metrics handle large sample sizes efficiently."""
        rng = np.random.default_rng(42)
        n = 100_000
        y_true = rng.integers(0, 2, n)
        y_pred = np.random.random(n)

        metrics = compute_discrimination_metrics(y_true, y_pred)
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestSingleClassGuards:
    """Test handling of single-class edge cases."""

    def test_auroc_all_negatives(self):
        """AUROC returns NaN when only negative class present."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(UserWarning, match="AUROC requires both classes"):
            result = auroc(y_true, y_pred)
        assert np.isnan(result)

    def test_auroc_all_positives(self):
        """AUROC returns NaN when only positive class present."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.warns(UserWarning, match="AUROC requires both classes"):
            result = auroc(y_true, y_pred)
        assert np.isnan(result)

    def test_prauc_all_negatives(self):
        """PR-AUC returns NaN when only negative class present."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(UserWarning, match="PR-AUC requires both classes"):
            result = prauc(y_true, y_pred)
        assert np.isnan(result)

    def test_prauc_all_positives(self):
        """PR-AUC returns NaN when only positive class present."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.warns(UserWarning, match="PR-AUC requires both classes"):
            result = prauc(y_true, y_pred)
        assert np.isnan(result)

    def test_youden_all_negatives(self):
        """Youden's J returns NaN when only negative class present."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(UserWarning, match="Youden's J requires both classes"):
            result = youden_j(y_true, y_pred)
        assert np.isnan(result)

    def test_compute_discrimination_metrics_single_class(self):
        """Aggregate function returns all NaN for single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(
            UserWarning, match="compute_discrimination_metrics requires both classes"
        ):
            metrics = compute_discrimination_metrics(y_true, y_pred)

        assert np.isnan(metrics["auroc"])
        assert np.isnan(metrics["prauc"])
        assert np.isnan(metrics["Youden"])
        assert np.isnan(metrics["Alpha"])

    def test_alpha_all_negatives(self):
        """Alpha returns NaN when only negative class present."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(
            UserWarning, match="Alpha \\(sensitivity at specificity\\) requires both classes"
        ):
            result = alpha_sensitivity_at_specificity(y_true, y_pred)
        assert np.isnan(result)

    def test_alpha_all_positives(self):
        """Alpha returns NaN when only positive class present."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.warns(
            UserWarning, match="Alpha \\(sensitivity at specificity\\) requires both classes"
        ):
            result = alpha_sensitivity_at_specificity(y_true, y_pred)
        assert np.isnan(result)

    def test_brier_score_single_class_still_works(self):
        """Brier score doesn't require both classes (it's calibration)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # Should not warn or fail
        result = compute_brier_score(y_true, y_pred)
        assert np.isfinite(result)
        assert result >= 0.0


class TestStrictMode:
    """Test strict mode for hard failure on invalid inputs."""

    def test_auroc_strict_raises_on_single_class(self):
        """AUROC raises ValueError when strict=True and single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        with pytest.raises(ValueError, match="AUROC requires both classes"):
            auroc(y_true, y_pred, strict=True)

    def test_auroc_strict_false_returns_nan(self):
        """AUROC returns NaN with warning when strict=False (default)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        with pytest.warns(UserWarning, match="AUROC requires both classes"):
            result = auroc(y_true, y_pred, strict=False)
        assert np.isnan(result)

    def test_auroc_strict_normal_case_works(self):
        """AUROC works normally with both classes regardless of strict setting."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        result_default = auroc(y_true, y_pred)
        result_strict = auroc(y_true, y_pred, strict=True)

        assert result_default == 1.0
        assert result_strict == 1.0

    def test_prauc_strict_raises_on_single_class(self):
        """PR-AUC raises ValueError when strict=True and single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="PR-AUC requires both classes"):
            prauc(y_true, y_pred, strict=True)

    def test_prauc_strict_false_returns_nan(self):
        """PR-AUC returns NaN with warning when strict=False (default)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(UserWarning, match="PR-AUC requires both classes"):
            result = prauc(y_true, y_pred, strict=False)
        assert np.isnan(result)

    def test_youden_j_strict_raises_on_single_class(self):
        """Youden's J raises ValueError when strict=True and single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.raises(ValueError, match="Youden's J requires both classes"):
            youden_j(y_true, y_pred, strict=True)

    def test_youden_j_strict_false_returns_nan(self):
        """Youden's J returns NaN with warning when strict=False (default)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.6, 0.7, 0.8, 0.9])

        with pytest.warns(UserWarning, match="Youden's J requires both classes"):
            result = youden_j(y_true, y_pred, strict=False)
        assert np.isnan(result)

    def test_alpha_strict_raises_on_single_class(self):
        """Alpha raises ValueError when strict=True and single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(
            ValueError, match="Alpha \\(sensitivity at specificity\\) requires both classes"
        ):
            alpha_sensitivity_at_specificity(y_true, y_pred, strict=True)

    def test_alpha_strict_false_returns_nan(self):
        """Alpha returns NaN with warning when strict=False (default)."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(
            UserWarning, match="Alpha \\(sensitivity at specificity\\) requires both classes"
        ):
            result = alpha_sensitivity_at_specificity(y_true, y_pred, strict=False)
        assert np.isnan(result)

    def test_compute_discrimination_metrics_strict_raises(self):
        """Aggregate function raises ValueError when strict=True and single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(
            ValueError, match="compute_discrimination_metrics requires both classes"
        ):
            compute_discrimination_metrics(y_true, y_pred, strict=True)

    def test_compute_discrimination_metrics_strict_false_returns_nan(self):
        """Aggregate function returns NaN values when strict=False and single class."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        with pytest.warns(
            UserWarning, match="compute_discrimination_metrics requires both classes"
        ):
            metrics = compute_discrimination_metrics(y_true, y_pred, strict=False)

        assert np.isnan(metrics["auroc"])
        assert np.isnan(metrics["prauc"])
        assert np.isnan(metrics["Youden"])
        assert np.isnan(metrics["Alpha"])

    def test_compute_discrimination_metrics_strict_normal_works(self):
        """Aggregate function works normally with both classes regardless of strict."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])

        metrics_default = compute_discrimination_metrics(y_true, y_pred)
        metrics_strict = compute_discrimination_metrics(y_true, y_pred, strict=True)

        assert metrics_default["auroc"] == 1.0
        assert metrics_strict["auroc"] == 1.0
        assert metrics_default["prauc"] == 1.0
        assert metrics_strict["prauc"] == 1.0
