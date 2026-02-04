"""
Tests for calibration plotting utilities.

Tests cover:
- Probability-space calibration plots
- Logit-space calibration plots
- Multi-split aggregation
- Binned logits computation
- Edge cases (all one class, NaN handling, small samples)
"""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ced_ml.plotting.calibration import (
    _binned_logits,
    _plot_logit_calibration_panel,
    _plot_prob_calibration_panel,
    plot_calibration_curve,
)


@pytest.fixture
def simple_predictions():
    """Create simple well-calibrated predictions for testing."""
    rng = np.random.default_rng(42)
    n = 500
    # Well-calibrated: probabilities match true outcomes
    probs = np.random.beta(2, 5, n)
    y_true = (rng.random(n) < probs).astype(int)
    return y_true, probs


@pytest.fixture
def miscalibrated_predictions():
    """Create miscalibrated predictions (overconfident)."""
    rng = np.random.default_rng(42)
    n = 500
    true_probs = np.random.beta(2, 5, n)
    # Overconfident: predicted probs more extreme than true
    pred_probs = np.clip(true_probs * 1.5, 0, 1)
    y_true = (rng.random(n) < true_probs).astype(int)
    return y_true, pred_probs


@pytest.fixture
def multi_split_predictions():
    """Create multi-split predictions for aggregation testing."""
    rng = np.random.default_rng(42)
    n_per_split = 200
    n_splits = 5

    y_list, p_list, split_ids = [], [], []
    for i in range(n_splits):
        probs = np.random.beta(2, 5, n_per_split)
        y = (rng.random(n_per_split) < probs).astype(int)
        y_list.append(y)
        p_list.append(probs)
        split_ids.extend([i] * n_per_split)

    y_true = np.concatenate(y_list)
    probs = np.concatenate(p_list)
    split_ids = np.array(split_ids)

    return y_true, probs, split_ids


class TestBinnedLogits:
    """Tests for _binned_logits helper function."""

    def test_basic_binning(self, simple_predictions):
        """Test basic binned logits computation."""
        y_true, probs = simple_predictions
        result = _binned_logits(y_true, probs, n_bins=10, bin_strategy="uniform")
        xs, ys, ys_lo, ys_hi, ys_sd, sizes = result

        # Should return valid arrays
        assert xs is not None
        assert ys is not None
        assert ys_lo is not None
        assert ys_hi is not None
        assert ys_sd is not None
        assert sizes is not None

        # All should be finite
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(ys))
        assert np.all(np.isfinite(ys_lo))
        assert np.all(np.isfinite(ys_hi))
        assert np.all(np.isfinite(ys_sd))

        # CI bounds should be ordered correctly
        assert np.all(ys_lo <= ys)
        assert np.all(ys <= ys_hi)

        # Sizes should be positive
        assert np.all(sizes > 0)

    def test_quantile_binning(self, simple_predictions):
        """Test quantile-based binning."""
        y_true, probs = simple_predictions
        result = _binned_logits(y_true, probs, n_bins=10, bin_strategy="quantile")
        xs, ys, ys_lo, ys_hi, ys_sd, sizes = result

        assert xs is not None
        assert len(xs) > 0
        # Quantile bins should have roughly equal sizes
        assert sizes.std() < sizes.mean() * 0.5

    def test_empty_input(self):
        """Test handling of empty input."""
        result = _binned_logits(np.array([]), np.array([]))
        assert all(x is None for x in result)

    def test_all_one_class(self):
        """Test handling when all predictions are one class."""
        rng = np.random.default_rng(42)
        y = np.ones(100)
        p = rng.random(100) * 0.5 + 0.5  # All high probabilities
        result = _binned_logits(y, p, n_bins=10)

        # Should still return valid result (with CIs)
        xs, ys, ys_lo, ys_hi, ys_sd, sizes = result
        if xs is not None:  # May merge bins
            assert np.all(np.isfinite(xs))
            assert np.all(np.isfinite(ys))

    def test_nan_handling(self):
        """Test that NaN values are filtered out."""
        y = np.array([0, 1, 0, 1, np.nan, 1])
        p = np.array([0.2, 0.8, np.nan, 0.7, 0.5, 0.9])

        result = _binned_logits(y, p, n_bins=2)
        xs, ys, ys_lo, ys_hi, ys_sd, sizes = result

        # Should filter NaNs and return valid data
        if xs is not None:
            assert np.all(np.isfinite(xs))
            assert np.all(np.isfinite(ys))


class TestPlotProbCalibrationPanel:
    """Tests for _plot_prob_calibration_panel."""

    def test_single_split_plot(self, simple_predictions):
        """Test probability calibration panel for single split."""
        import matplotlib.pyplot as plt

        y_true, probs = simple_predictions
        fig, ax = plt.subplots()

        # Create bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        _plot_prob_calibration_panel(
            ax,
            y_true,
            probs,
            bins,
            bin_centers,
            10,
            "uniform",
            panel_title="Test Calibration",
        )

        # Should have created plot elements
        assert len(ax.lines) > 0  # At least the diagonal line
        assert ax.get_title() == "Test Calibration"
        assert ax.get_xlabel() == "Predicted probability"
        assert ax.get_ylabel() == "Expected frequency"

        plt.close(fig)

    def test_multi_split_plot(self, multi_split_predictions):
        """Test probability calibration with multi-split aggregation."""
        import matplotlib.pyplot as plt

        y_true, probs, split_ids = multi_split_predictions
        unique_splits = np.unique(split_ids)
        fig, ax = plt.subplots()

        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        _plot_prob_calibration_panel(
            ax,
            y_true,
            probs,
            bins,
            bin_centers,
            10,
            "uniform",
            split_ids=split_ids,
            unique_splits=unique_splits,
            panel_title="Multi-Split Calibration",
        )

        # Should show confidence bands
        assert len(ax.collections) > 0  # Scatter or fill_between

        plt.close(fig)

    def test_quantile_vs_uniform(self, simple_predictions):
        """Test that quantile and uniform binning produce different plots."""
        import matplotlib.pyplot as plt

        y_true, probs = simple_predictions

        # Quantile binning
        fig1, ax1 = plt.subplots()
        quantiles = np.linspace(0, 100, 11)
        bins_q = np.percentile(probs, quantiles)
        bins_q = np.unique(bins_q)
        bin_centers_q = (bins_q[:-1] + bins_q[1:]) / 2

        _plot_prob_calibration_panel(
            ax1, y_true, probs, bins_q, bin_centers_q, len(bins_q) - 1, "quantile"
        )

        # Uniform binning
        fig2, ax2 = plt.subplots()
        bins_u = np.linspace(0, 1, 11)
        bin_centers_u = (bins_u[:-1] + bins_u[1:]) / 2

        _plot_prob_calibration_panel(ax2, y_true, probs, bins_u, bin_centers_u, 10, "uniform")

        # Both should create valid plots
        assert len(ax1.lines) > 0
        assert len(ax2.lines) > 0

        plt.close(fig1)
        plt.close(fig2)


class TestPlotLogitCalibrationPanel:
    """Tests for _plot_logit_calibration_panel."""

    def test_basic_logit_plot(self, simple_predictions):
        """Test basic logit calibration panel."""
        import matplotlib.pyplot as plt

        y_true, probs = simple_predictions
        fig, ax = plt.subplots()

        _plot_logit_calibration_panel(
            ax,
            y_true,
            probs,
            n_bins=10,
            bin_strategy="uniform",
            split_ids=None,
            unique_splits=None,
            panel_title="Logit Calibration",
            calib_intercept=0.0,
            calib_slope=1.0,
        )

        # Should have diagonal line and possibly recalibration line
        assert len(ax.lines) >= 1
        assert ax.get_xlabel() == "Predicted logit: logit(p̂)"

        plt.close(fig)

    def test_multi_split_logit(self, multi_split_predictions):
        """Test logit calibration with multi-split aggregation."""
        import matplotlib.pyplot as plt

        y_true, probs, split_ids = multi_split_predictions
        unique_splits = np.unique(split_ids)
        fig, ax = plt.subplots()

        _plot_logit_calibration_panel(
            ax,
            y_true,
            probs,
            n_bins=10,
            bin_strategy="quantile",
            split_ids=split_ids,
            unique_splits=unique_splits,
            panel_title="Multi-Split Logit",
            calib_intercept=0.1,
            calib_slope=0.9,
        )

        # Should create plot with aggregated bands
        assert len(ax.collections) > 0 or len(ax.lines) > 0

        plt.close(fig)

    def test_with_recalibration_line(self, miscalibrated_predictions):
        """Test that recalibration line is plotted."""
        import matplotlib.pyplot as plt

        y_true, probs = miscalibrated_predictions
        fig, ax = plt.subplots()

        _plot_logit_calibration_panel(
            ax,
            y_true,
            probs,
            n_bins=10,
            bin_strategy="uniform",
            split_ids=None,
            unique_splits=None,
            panel_title="With Recalibration",
            calib_intercept=0.5,  # Significant miscalibration
            calib_slope=0.7,
        )

        # Should have ideal line + recalibration line
        assert len(ax.lines) >= 2

        plt.close(fig)


class TestPlotCalibrationCurve:
    """Tests for main plot_calibration_curve function."""

    def test_basic_calibration_plot(self, simple_predictions):
        """Test basic 4-panel calibration plot generation."""
        y_true, probs = simple_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration.png"

            plot_calibration_curve(y_true, probs, out_path, title="Test Calibration", n_bins=10)

            # File should be created
            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_with_metadata(self, simple_predictions):
        """Test calibration plot with metadata footer."""
        y_true, probs = simple_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_meta.png"

            meta_lines = ["Model: Random Forest", "Dataset: Test", "N=500"]

            plot_calibration_curve(
                y_true,
                probs,
                out_path,
                title="Calibration with Metadata",
                meta_lines=meta_lines,
            )

            assert out_path.exists()

    def test_with_split_ids(self, multi_split_predictions):
        """Test calibration plot with multi-split aggregation."""
        y_true, probs, split_ids = multi_split_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_splits.png"

            plot_calibration_curve(
                y_true,
                probs,
                out_path,
                title="Multi-Split Calibration",
                split_ids=split_ids,
                n_bins=10,
            )

            assert out_path.exists()

    def test_with_recalibration_params(self, miscalibrated_predictions):
        """Test calibration plot with recalibration parameters."""
        y_true, probs = miscalibrated_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_recal.png"

            plot_calibration_curve(
                y_true,
                probs,
                out_path,
                title="Miscalibrated Model",
                calib_intercept=0.3,
                calib_slope=0.8,
            )

            assert out_path.exists()

    def test_edge_case_all_same_class(self):
        """Test handling when all predictions are same class."""
        rng = np.random.default_rng(42)
        y_true = np.ones(100)
        probs = rng.random(100) * 0.5 + 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_one_class.png"

            # Should not crash
            plot_calibration_curve(y_true, probs, out_path, title="All One Class")

            assert out_path.exists()

    def test_edge_case_perfect_predictions(self):
        """Test with perfect predictions (all 0 or 1)."""
        y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
        probs = y_true.astype(float)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_perfect.png"

            plot_calibration_curve(y_true, probs, out_path, title="Perfect Predictions")

            assert out_path.exists()

    def test_with_nan_values(self):
        """Test that NaN values are filtered correctly."""
        y_true = np.array([0, 1, 0, 1, np.nan, 1, 0, 1] * 20)
        probs = np.array([0.2, 0.8, np.nan, 0.7, 0.5, 0.9, 0.3, 0.6] * 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_nan.png"

            plot_calibration_curve(y_true, probs, out_path, title="With NaN Values")

            assert out_path.exists()

    def test_small_sample(self):
        """Test with very small sample size."""
        y_true = np.array([0, 1, 0, 1, 1])
        probs = np.array([0.2, 0.8, 0.3, 0.7, 0.9])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "calibration_small.png"

            plot_calibration_curve(
                y_true,
                probs,
                out_path,
                title="Small Sample",
                n_bins=2,  # Fewer bins for small sample
            )

            assert out_path.exists()

    def test_output_directory_creation(self, simple_predictions):
        """Test that output directory is created if it doesn't exist."""
        y_true, probs = simple_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "nested" / "calibration.png"

            plot_calibration_curve(y_true, probs, out_path, title="Nested Output")

            assert out_path.exists()
            assert out_path.parent.exists()


class TestEdgeCases:
    """Additional edge case tests."""

    def test_extreme_probabilities(self):
        """Test with probabilities very close to 0 and 1."""
        y_true = np.array([0, 0, 1, 1] * 25)
        probs = np.array([1e-7, 1e-6, 1 - 1e-6, 1 - 1e-7] * 25)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "extreme_probs.png"

            plot_calibration_curve(y_true, probs, out_path, title="Extreme Probabilities")

            assert out_path.exists()

    def test_imbalanced_data(self):
        """Test with highly imbalanced data (like CeD)."""
        np.random.seed(42)
        n_controls = 1000
        n_cases = 10

        y_true = np.concatenate([np.zeros(n_controls), np.ones(n_cases)])
        probs = np.concatenate(
            [
                np.random.beta(1, 10, n_controls),  # Low probabilities for controls
                np.random.beta(3, 2, n_cases),  # Higher probabilities for cases
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "imbalanced.png"

            plot_calibration_curve(y_true, probs, out_path, title="Imbalanced Data (1:100)")

            assert out_path.exists()

    def test_subtitle_formatting(self, simple_predictions):
        """Test that subtitle is incorporated correctly."""
        y_true, probs = simple_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "with_subtitle.png"

            plot_calibration_curve(
                y_true, probs, out_path, title="Main Title", subtitle="Validation Set"
            )

            assert out_path.exists()
