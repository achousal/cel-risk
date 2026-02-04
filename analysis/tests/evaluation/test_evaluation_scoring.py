"""
Tests for evaluation scoring module.

Tests the compute_selection_score function and related utilities
for model selection based on composite metrics.
"""

import numpy as np
import pytest
from ced_ml.evaluation.scoring import (
    DEFAULT_WEIGHTS,
    _is_valid_metric,
    compute_selection_score,
    compute_selection_scores_for_models,
    rank_models_by_selection_score,
)


class TestComputeSelectionScore:
    """Tests for compute_selection_score function."""

    def test_perfect_model_scores_one(self):
        """Perfect model (AUROC=1.0, Brier=0, slope=1.0) should score 1.0."""
        metrics = {"auroc": 1.0, "brier_score": 0.0, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        assert score == pytest.approx(1.0)

    def test_random_model_score(self):
        """Random model (AUROC=0.5, Brier=0.25, slope=1.0) should score ~0.675."""
        metrics = {"auroc": 0.5, "brier_score": 0.25, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # 0.50 * 0.5 + 0.30 * (1 - 0.25) + 0.20 * 1.0 = 0.25 + 0.225 + 0.2 = 0.675
        assert score == pytest.approx(0.675, rel=0.01)

    def test_custom_weights(self):
        """Test with custom weights emphasizing AUROC."""
        metrics = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0}
        weights = {"auroc": 0.7, "brier": 0.2, "slope": 0.1}
        score = compute_selection_score(metrics, weights)
        # 0.7 * 0.9 + 0.2 * 0.9 + 0.1 * 1.0 = 0.63 + 0.18 + 0.1 = 0.91
        assert score == pytest.approx(0.91, rel=0.01)

    def test_missing_auroc_uses_default(self):
        """Missing AUROC should use default value of 0.5."""
        metrics = {"brier_score": 0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # With AUROC=0.5 (default): 0.5*0.5 + 0.3*0.9 + 0.2*1.0 = 0.25 + 0.27 + 0.2 = 0.72
        assert score == pytest.approx(0.72, rel=0.01)

    def test_missing_brier_uses_default(self):
        """Missing Brier should use default value of 0.25."""
        metrics = {"auroc": 0.8, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # With Brier=0.25 (default): 0.5*0.8 + 0.3*0.75 + 0.2*1.0 = 0.4 + 0.225 + 0.2 = 0.825
        assert score == pytest.approx(0.825, rel=0.01)

    def test_missing_slope_uses_default(self):
        """Missing slope should use default value of 1.0."""
        metrics = {"auroc": 0.85, "brier_score": 0.12}
        score = compute_selection_score(metrics)
        # With slope=1.0 (default): 0.5*0.85 + 0.3*0.88 + 0.2*1.0 = 0.425 + 0.264 + 0.2 = 0.889
        assert score == pytest.approx(0.889, rel=0.01)

    def test_all_missing_metrics_uses_defaults(self):
        """Empty metrics dict should use all defaults."""
        metrics = {}
        score = compute_selection_score(metrics)
        # All defaults: AUROC=0.5, Brier=0.25, slope=1.0
        # 0.5*0.5 + 0.3*0.75 + 0.2*1.0 = 0.25 + 0.225 + 0.2 = 0.675
        assert score == pytest.approx(0.675, rel=0.01)

    def test_canonical_keys_required(self):
        """Test that canonical metric keys are required."""
        # Canonical keys should work
        metrics = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        assert score == pytest.approx(0.92, rel=0.01)

    def test_slope_deviation_penalty(self):
        """Test that slope deviation from 1.0 is penalized."""
        # Perfect slope
        metrics_perfect = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0}
        score_perfect = compute_selection_score(metrics_perfect)

        # Slope of 0.8 (deviation of 0.2)
        metrics_low = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 0.8}
        score_low = compute_selection_score(metrics_low)

        # Slope of 1.2 (deviation of 0.2)
        metrics_high = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.2}
        score_high = compute_selection_score(metrics_high)

        assert score_perfect > score_low
        assert score_perfect > score_high
        # Both deviations of 0.2 should give same penalty
        assert score_low == pytest.approx(score_high, rel=0.01)

    def test_slope_large_deviation_clamped(self):
        """Test that slope deviation is clamped to [0, 1]."""
        # Slope of 2.5 (deviation of 1.5, clamped to 1.0)
        metrics = {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 2.5}
        score = compute_selection_score(metrics)
        # slope_component = 1 - min(1.5, 1.0) = 0.0
        # 0.5*0.9 + 0.3*0.9 + 0.2*0.0 = 0.45 + 0.27 + 0 = 0.72
        assert score == pytest.approx(0.72, rel=0.01)

    def test_nan_metric_uses_default(self):
        """NaN metric values should use defaults."""
        metrics = {"auroc": np.nan, "brier_score": 0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # AUROC defaults to 0.5
        # 0.5*0.5 + 0.3*0.9 + 0.2*1.0 = 0.25 + 0.27 + 0.2 = 0.72
        assert score == pytest.approx(0.72, rel=0.01)

    def test_inf_metric_uses_default(self):
        """Infinite metric values should use defaults."""
        metrics = {"auroc": 0.9, "brier_score": np.inf, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # Brier defaults to 0.25
        # 0.5*0.9 + 0.3*0.75 + 0.2*1.0 = 0.45 + 0.225 + 0.2 = 0.875
        assert score == pytest.approx(0.875, rel=0.01)

    def test_none_metric_uses_default(self):
        """None metric values should use defaults."""
        metrics = {"auroc": 0.9, "brier_score": None, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        # Brier defaults to 0.25
        assert score == pytest.approx(0.875, rel=0.01)

    def test_auroc_clamped_to_valid_range(self):
        """AUROC values outside [0, 1] should be clamped."""
        # AUROC > 1 (clamped to 1.0)
        metrics = {"auroc": 1.5, "brier_score": 0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        expected_metrics = {"auroc": 1.0, "brier_score": 0.1, "calibration_slope": 1.0}
        expected_score = compute_selection_score(expected_metrics)
        assert score == pytest.approx(expected_score, rel=0.01)

    def test_brier_clamped_to_valid_range(self):
        """Brier values outside [0, 1] should be clamped."""
        # Brier < 0 (clamped to 0.0)
        metrics = {"auroc": 0.9, "brier_score": -0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics)
        expected_metrics = {"auroc": 0.9, "brier_score": 0.0, "calibration_slope": 1.0}
        expected_score = compute_selection_score(expected_metrics)
        assert score == pytest.approx(expected_score, rel=0.01)

    def test_normalized_weights(self):
        """Weights are normalized to sum to 1.0."""
        metrics = {"auroc": 1.0, "brier_score": 0.0, "calibration_slope": 1.0}
        # Non-unit sum weights
        weights = {"auroc": 1.0, "brier": 0.6, "slope": 0.4}  # Sum = 2.0
        score = compute_selection_score(metrics, weights)
        # With perfect metrics, score should still be 1.0
        assert score == pytest.approx(1.0)

    def test_empty_weights_uses_default(self):
        """Empty weights dict should use defaults."""
        metrics = {"auroc": 0.8, "brier_score": 0.1, "calibration_slope": 1.0}
        score = compute_selection_score(metrics, weights={})
        expected = compute_selection_score(metrics, weights=None)
        assert score == pytest.approx(expected, rel=0.01)

    def test_score_in_zero_one_range(self):
        """Score should always be in [0, 1] range."""
        test_cases = [
            {"auroc": 0.0, "brier_score": 1.0, "calibration_slope": 0.0},  # Worst case
            {"auroc": 1.0, "brier_score": 0.0, "calibration_slope": 1.0},  # Best case
            {"auroc": 0.5, "brier_score": 0.5, "calibration_slope": 0.5},  # Middle case
            {"auroc": 0.7, "brier_score": 0.3, "calibration_slope": 1.5},  # Mixed
        ]
        for metrics in test_cases:
            score = compute_selection_score(metrics)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {metrics}"


class TestComputeSelectionScoresForModels:
    """Tests for compute_selection_scores_for_models function."""

    def test_multiple_models(self):
        """Test scoring multiple models."""
        model_metrics = {
            "LR_EN": {"auroc": 0.85, "brier_score": 0.10, "calibration_slope": 1.05},
            "RF": {"auroc": 0.82, "brier_score": 0.12, "calibration_slope": 0.90},
            "XGBoost": {"auroc": 0.88, "brier_score": 0.08, "calibration_slope": 1.10},
        }
        scores = compute_selection_scores_for_models(model_metrics)

        assert len(scores) == 3
        assert "LR_EN" in scores
        assert "RF" in scores
        assert "XGBoost" in scores

        # All scores should be valid
        for model_name, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Invalid score for {model_name}"

    def test_empty_model_metrics(self):
        """Empty input should return empty dict."""
        scores = compute_selection_scores_for_models({})
        assert scores == {}

    def test_single_model(self):
        """Single model should work."""
        model_metrics = {"LR_EN": {"auroc": 0.85, "brier_score": 0.10, "calibration_slope": 1.0}}
        scores = compute_selection_scores_for_models(model_metrics)
        assert len(scores) == 1
        assert "LR_EN" in scores

    def test_custom_weights_applied(self):
        """Custom weights should be applied to all models."""
        model_metrics = {
            "model_a": {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0},
            "model_b": {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0},
        }
        weights = {"auroc": 0.7, "brier": 0.2, "slope": 0.1}
        scores = compute_selection_scores_for_models(model_metrics, weights)

        # Both models have same metrics so should have same score
        assert scores["model_a"] == pytest.approx(scores["model_b"])


class TestRankModelsBySelectionScore:
    """Tests for rank_models_by_selection_score function."""

    def test_ranking_order(self):
        """Models should be ranked by score descending."""
        model_metrics = {
            "best": {"auroc": 0.95, "brier_score": 0.05, "calibration_slope": 1.0},
            "worst": {"auroc": 0.60, "brier_score": 0.30, "calibration_slope": 0.5},
            "middle": {"auroc": 0.80, "brier_score": 0.15, "calibration_slope": 1.1},
        }
        ranking = rank_models_by_selection_score(model_metrics)

        assert len(ranking) == 3
        assert ranking[0][0] == "best"
        assert ranking[-1][0] == "worst"

        # Scores should be decreasing
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_empty_input(self):
        """Empty input should return empty list."""
        ranking = rank_models_by_selection_score({})
        assert ranking == []

    def test_single_model(self):
        """Single model should return single-element list."""
        model_metrics = {
            "only_model": {"auroc": 0.85, "brier_score": 0.10, "calibration_slope": 1.0}
        }
        ranking = rank_models_by_selection_score(model_metrics)
        assert len(ranking) == 1
        assert ranking[0][0] == "only_model"

    def test_tie_handling(self):
        """Models with same score should be in ranking."""
        model_metrics = {
            "model_a": {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0},
            "model_b": {"auroc": 0.9, "brier_score": 0.1, "calibration_slope": 1.0},
        }
        ranking = rank_models_by_selection_score(model_metrics)

        assert len(ranking) == 2
        assert ranking[0][1] == pytest.approx(ranking[1][1])


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_valid_metric_finite_number(self):
        """Finite numbers are valid."""
        assert _is_valid_metric(0.5)
        assert _is_valid_metric(0)
        assert _is_valid_metric(1.0)
        assert _is_valid_metric(-0.5)

    def test_is_valid_metric_nan(self):
        """NaN is not valid."""
        assert not _is_valid_metric(np.nan)
        assert not _is_valid_metric(float("nan"))

    def test_is_valid_metric_inf(self):
        """Infinity is not valid."""
        assert not _is_valid_metric(np.inf)
        assert not _is_valid_metric(-np.inf)
        assert not _is_valid_metric(float("inf"))

    def test_is_valid_metric_none(self):
        """None is not valid."""
        assert not _is_valid_metric(None)

    def test_is_valid_metric_string(self):
        """Strings are not valid."""
        assert _is_valid_metric("0.5")  # Can be converted to float
        assert not _is_valid_metric("hello")


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS constant."""

    def test_default_weights_values(self):
        """Verify default weight values."""
        assert DEFAULT_WEIGHTS["auroc"] == 0.50
        assert DEFAULT_WEIGHTS["brier"] == 0.30
        assert DEFAULT_WEIGHTS["slope"] == 0.20

    def test_default_weights_sum(self):
        """Default weights should sum to 1.0."""
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_default_weights_keys(self):
        """Default weights should have expected keys."""
        assert set(DEFAULT_WEIGHTS.keys()) == {"auroc", "brier", "slope"}
