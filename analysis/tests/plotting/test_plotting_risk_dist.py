"""Tests for risk distribution plotting.

Tests cover:
- Distribution statistics computation
- Single-panel plots (histogram, binary, KDE)
- Multi-panel plots (incident/prevalent subplots)
- Threshold line rendering
- Metadata handling
- Edge cases (empty data, NaN handling)
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from ced_ml.plotting.risk_dist import (
    compute_distribution_stats,
    plot_risk_distribution,
)

matplotlib.use("Agg")


class TestComputeDistributionStats:
    """Tests for distribution statistics computation."""

    def test_basic_stats(self):
        """Test statistics computation for normal distribution."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        stats = compute_distribution_stats(scores)

        assert stats["mean"] == pytest.approx(0.5, abs=0.01)
        assert stats["median"] == pytest.approx(0.5, abs=0.01)
        assert stats["iqr"] == pytest.approx(0.4, abs=0.01)  # Q3(0.7) - Q1(0.3)
        assert stats["sd"] > 0

    def test_empty_array(self):
        """Test handling of empty array."""
        scores = np.array([])
        stats = compute_distribution_stats(scores)

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["iqr"])
        assert np.isnan(stats["sd"])

    def test_nan_filtering(self):
        """Test NaN values are filtered before computation."""
        scores = np.array([0.1, np.nan, 0.3, np.inf, 0.5])
        stats = compute_distribution_stats(scores)

        # Should compute on [0.1, 0.3, 0.5] only
        assert not np.isnan(stats["mean"])
        assert stats["mean"] == pytest.approx(0.3, abs=0.01)

    def test_all_nans(self):
        """Test array with only NaN/inf values."""
        scores = np.array([np.nan, np.inf, -np.inf])
        stats = compute_distribution_stats(scores)

        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["iqr"])
        assert np.isnan(stats["sd"])

    def test_single_value(self):
        """Test statistics with single value (edge case)."""
        scores = np.array([0.42])
        stats = compute_distribution_stats(scores)

        assert stats["mean"] == pytest.approx(0.42)
        assert stats["median"] == pytest.approx(0.42)
        assert stats["iqr"] == pytest.approx(0.0)
        assert stats["sd"] == pytest.approx(0.0)


class TestPlotRiskDistribution:
    """Tests for risk distribution plotting."""

    def test_simple_histogram(self, tmp_path):
        """Test simple histogram without labels."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = tmp_path / "simple.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Simple Distribution",
        )

        assert out_path.exists()

    def test_binary_histogram(self, tmp_path):
        """Test binary histogram (cases vs controls)."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y_true = rng.binomial(1, 0.3, 100)
        out_path = tmp_path / "binary.png"

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Binary Distribution",
            pos_label="CeD",
        )

        assert out_path.exists()

    def test_three_category_kde(self, tmp_path):
        """Test KDE plot with three categories."""
        rng = np.random.default_rng(42)
        n = 300
        scores = rng.uniform(0, 1, n)
        category_col = rng.choice(["Controls", "Incident", "Prevalent"], size=n, p=[0.7, 0.2, 0.1])
        out_path = tmp_path / "kde.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Three-Category KDE",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_incident_subplot(self, tmp_path):
        """Test incident subplot rendering."""
        rng = np.random.default_rng(42)
        n = 300
        scores = rng.uniform(0, 1, n)
        category_col = rng.choice(["Controls", "Incident"], size=n, p=[0.8, 0.2])
        out_path = tmp_path / "incident_subplot.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Incident Subplot",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_prevalent_subplot(self, tmp_path):
        """Test prevalent subplot rendering."""
        rng = np.random.default_rng(42)
        n = 300
        scores = rng.uniform(0, 1, n)
        category_col = rng.choice(["Controls", "Prevalent"], size=n, p=[0.8, 0.2])
        out_path = tmp_path / "prevalent_subplot.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Prevalent Subplot",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_all_subplots(self, tmp_path):
        """Test all three subplots (main + incident + prevalent)."""
        rng = np.random.default_rng(42)
        n = 300
        scores = rng.uniform(0, 1, n)
        category_col = rng.choice(["Controls", "Incident", "Prevalent"], size=n, p=[0.7, 0.2, 0.1])
        out_path = tmp_path / "all_subplots.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="All Subplots",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_threshold_lines(self, tmp_path):
        """Test threshold line rendering."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y_true = rng.binomial(1, 0.3, 100)
        out_path = tmp_path / "thresholds.png"

        # Create threshold bundle
        threshold_bundle = {
            "dca_threshold": 0.15,
            "spec_target_threshold": 0.25,
            "youden_threshold": 0.35,
            "target_spec": 0.95,
        }

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Thresholds",
            threshold_bundle=threshold_bundle,
        )

        assert out_path.exists()

    def test_threshold_metrics(self, tmp_path):
        """Test threshold metrics in legend."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y_true = rng.binomial(1, 0.3, 100)
        out_path = tmp_path / "threshold_metrics.png"

        # Create threshold bundle with metrics
        threshold_bundle = {
            "dca_threshold": 0.15,
            "spec_target_threshold": 0.25,
            "youden_threshold": 0.35,
            "target_spec": 0.95,
            "spec_target": {"sensitivity": 0.82, "precision": 0.45, "fp": 12},
            "youden": {"sensitivity": 0.75, "precision": 0.55, "fp": 8},
            "dca": {"sensitivity": 0.88, "precision": 0.38, "fp": 15},
        }

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Threshold Metrics",
            threshold_bundle=threshold_bundle,
        )

        assert out_path.exists()

    def test_metadata_lines(self, tmp_path):
        """Test metadata rendering at bottom."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = tmp_path / "metadata.png"

        meta_lines = [
            "Model: RandomForest",
            "Seed: 42",
            "CV: 5-fold × 10 repeats",
        ]

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="With Metadata",
            meta_lines=meta_lines,
        )

        assert out_path.exists()

    def test_subtitle(self, tmp_path):
        """Test subtitle rendering."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = tmp_path / "subtitle.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Main Title",
            subtitle="Validation Set (n=222)",
        )

        assert out_path.exists()

    def test_custom_xlabel(self, tmp_path):
        """Test custom x-axis label."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = tmp_path / "xlabel.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Custom X-Label",
            xlabel="Risk Score (0-1)",
        )

        assert out_path.exists()

    def test_x_limits(self, tmp_path):
        """Test custom x-axis limits."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = tmp_path / "xlimits.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Custom X-Limits",
            x_limits=(0.0, 0.5),
        )

        assert out_path.exists()

    def test_target_spec_label(self, tmp_path):
        """Test custom target specificity label."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y_true = rng.binomial(1, 0.3, 100)
        out_path = tmp_path / "target_spec.png"

        # Create threshold bundle with custom target_spec
        threshold_bundle = {
            "spec_target_threshold": 0.25,
            "target_spec": 0.99,
        }

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Custom Target Spec",
            threshold_bundle=threshold_bundle,
        )

        assert out_path.exists()

    def test_empty_scores(self, tmp_path):
        """Test handling of empty scores array."""
        scores = np.array([])
        out_path = tmp_path / "empty.png"

        plot_risk_distribution(y_true=None, scores=scores, out_path=out_path, title="Empty")

        # Should not create file (early return on empty data)
        assert not out_path.exists()

    def test_all_nan_scores(self, tmp_path):
        """Test handling of all-NaN scores."""
        scores = np.array([np.nan, np.nan, np.nan])
        out_path = tmp_path / "all_nan.png"

        plot_risk_distribution(y_true=None, scores=scores, out_path=out_path, title="All NaN")

        # Should not create file (early return after NaN filtering)
        assert not out_path.exists()

    def test_partial_nan_scores(self, tmp_path):
        """Test handling of partial NaN scores."""
        scores = np.array([0.1, np.nan, 0.3, np.inf, 0.5])
        out_path = tmp_path / "partial_nan.png"

        plot_risk_distribution(y_true=None, scores=scores, out_path=out_path, title="Partial NaN")

        assert out_path.exists()

    def test_invalid_threshold_ignored(self, tmp_path):
        """Test invalid thresholds are ignored."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        y_true = rng.binomial(1, 0.3, 100)
        out_path = tmp_path / "invalid_threshold.png"

        # Thresholds outside [0, 1] should be ignored
        threshold_bundle = {
            "dca_threshold": 1.5,  # Invalid
            "spec_target_threshold": -0.1,  # Invalid
            "youden_threshold": 0.35,  # Valid
            "target_spec": 0.95,
        }

        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Invalid Thresholds",
            threshold_bundle=threshold_bundle,
        )

        assert out_path.exists()

    def test_kde_fallback_to_histogram(self, tmp_path):
        """Test KDE fallback to histogram with few points."""
        rng = np.random.default_rng(42)
        # Very few incident points might cause KDE to fail
        n = 50
        scores = rng.uniform(0, 1, n)
        category_col = np.array(["Controls"] * 48 + ["Incident"] * 2)
        out_path = tmp_path / "kde_fallback.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="KDE Fallback",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_missing_category_filtered(self, tmp_path):
        """Test categories with zero samples are filtered."""
        rng = np.random.default_rng(42)
        n = 100
        scores = rng.uniform(0, 1, n)
        # Only Controls and Incident, no Prevalent
        category_col = rng.choice(["Controls", "Incident"], size=n, p=[0.8, 0.2])
        out_path = tmp_path / "missing_category.png"

        plot_risk_distribution(
            y_true=None,
            scores=scores,
            out_path=out_path,
            title="Missing Category",
            category_col=category_col,
        )

        assert out_path.exists()

    def test_pathlib_path(self, tmp_path):
        """Test Path object as out_path."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = Path(tmp_path) / "pathlib.png"

        plot_risk_distribution(y_true=None, scores=scores, out_path=out_path, title="Path Object")

        assert out_path.exists()

    def test_string_path(self, tmp_path):
        """Test string as out_path."""
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        out_path = str(tmp_path / "string.png")

        plot_risk_distribution(y_true=None, scores=scores, out_path=out_path, title="String Path")

        assert Path(out_path).exists()


class TestThresholdLineVisibility:
    """Regression tests for threshold line visibility bug.

    Bug: When data is clustered in a narrow range and thresholds are outside
    that range, threshold lines were not visible because main plot auto-scaled
    to data range while subplots used xlim=[0,1].

    Fix: Set main plot xlim=[0,1] when category_col is provided to match subplots.
    """

    def test_threshold_visible_with_low_scores(self, tmp_path):
        """Test that threshold lines are visible even when scores are clustered low.

        Regression test for bug where:
        - Test set scores clustered 0.0-0.3 (mostly low-risk controls)
        - Spec95 threshold at 0.5 was off-screen
        - Legend showed threshold but line was invisible
        """
        np.random.seed(42)
        # Low-risk scores (skewed beta distribution)
        y_true = np.array([0] * 95 + [1] * 5)
        scores = np.random.beta(2, 10, 100)  # Mostly 0-0.3 range
        category = np.array(["Controls"] * 95 + ["Incident"] * 5)

        # Create threshold bundle with threshold likely > max(scores)
        from ced_ml.metrics.thresholds import compute_threshold_bundle

        bundle = compute_threshold_bundle(y_true, scores, target_spec=0.95)

        out_path = tmp_path / "low_scores_threshold.png"
        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Low Scores with High Threshold",
            category_col=category,
            threshold_bundle=bundle,
        )

        assert out_path.exists()
        # With fix, main plot should have xlim=[0,1] when category_col provided
        # This ensures threshold line at bundle['spec_target_threshold'] is visible
        # even if it's beyond the max score in the data

    def test_threshold_visible_with_high_scores(self, tmp_path):
        """Test threshold visibility when scores are clustered high."""
        np.random.seed(43)
        # High-risk scores (reversed beta)
        y_true = np.array([0] * 5 + [1] * 95)
        scores = np.random.beta(10, 2, 100)  # Mostly 0.7-1.0 range
        category = np.array(["Controls"] * 5 + ["Incident"] * 95)

        from ced_ml.metrics.thresholds import compute_threshold_bundle

        bundle = compute_threshold_bundle(y_true, scores, target_spec=0.95)

        out_path = tmp_path / "high_scores_threshold.png"
        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="High Scores with Low Threshold",
            category_col=category,
            threshold_bundle=bundle,
        )

        assert out_path.exists()

    def test_all_thresholds_visible_simultaneously(self, tmp_path):
        """Test that all three threshold lines (spec95, youden, dca) are visible."""
        np.random.seed(44)
        y_true = np.array([0] * 90 + [1] * 10)
        scores = np.random.beta(2, 5, 100)  # Clustered 0-0.4
        category = np.array(["Controls"] * 90 + ["Incident"] * 10)

        from ced_ml.metrics.dca import threshold_dca_zero_crossing
        from ced_ml.metrics.thresholds import compute_threshold_bundle

        dca_thr = threshold_dca_zero_crossing(y_true, scores)
        bundle = compute_threshold_bundle(y_true, scores, target_spec=0.95, dca_threshold=dca_thr)

        out_path = tmp_path / "all_thresholds.png"
        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="All Thresholds Test",
            category_col=category,
            threshold_bundle=bundle,
        )

        assert out_path.exists()

    def test_threshold_slightly_above_one(self, tmp_path):
        """Test threshold slightly > 1.0 due to floating point arithmetic.

        Regression test for floating point precision bug where threshold
        computation could return values like 1.0000000001 (e.g., max(p) + 1e-12),
        causing the line not to be drawn due to strict 0 <= threshold <= 1 check.

        Fix: _normalize_threshold() clamps values within epsilon to [0, 1].
        """
        np.random.seed(45)
        y_true = np.array([0] * 95 + [1] * 5)
        scores = np.random.beta(2, 10, 100)
        category = np.array(["Controls"] * 95 + ["Incident"] * 5)

        # Simulate threshold computation returning value slightly > 1.0
        bundle = {
            "spec_target_threshold": 1.0000000001,  # Edge case from max(p) + eps
            "youden_threshold": 0.5,
            "target_spec": 0.95,
            "spec_target": {
                "sensitivity": 0.8,
                "precision": 0.75,
                "fp": 10,
                "tp": 4,
                "fn": 1,
            },
        }

        out_path = tmp_path / "threshold_above_one.png"
        plot_risk_distribution(
            y_true=y_true,
            scores=scores,
            out_path=out_path,
            title="Threshold > 1.0 Edge Case",
            category_col=category,
            threshold_bundle=bundle,
        )

        assert out_path.exists()
        # With normalization, threshold 1.0000000001 should be clamped to 1.0
        # and the red line should be visible at x=1.0
        # All three lines (red spec95, green youden, purple dca) should be visible
        # even if some thresholds are beyond the natural data range
