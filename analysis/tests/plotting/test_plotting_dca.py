"""
Tests for DCA plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ced_ml.plotting.dca import apply_plot_metadata, plot_dca, plot_dca_curve


@pytest.fixture
def sample_dca_df():
    """Create sample DCA DataFrame for testing."""
    rng = np.random.default_rng(42)
    thresholds = np.linspace(0, 20, 50)  # 0-20% as percentages
    nb_model = 0.02 - 0.003 * thresholds + rng.standard_normal(50) * 0.001
    nb_all = 0.02 - 0.004 * thresholds
    nb_none = np.zeros(50)

    return pd.DataFrame(
        {
            "threshold_pct": thresholds,
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": nb_none,
        }
    )


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    y_true = np.array([0] * 800 + [1] * 200)
    y_pred = np.random.beta(2, 5, size=800).tolist() + np.random.beta(5, 2, size=200).tolist()
    return np.array(y_true), np.array(y_pred)


class TestApplyPlotMetadata:
    """Test apply_plot_metadata function."""

    def test_no_metadata_returns_default_margin(self):
        """Empty metadata should return default margin."""
        fig = plt.figure()
        margin = apply_plot_metadata(fig, None)
        assert margin == 0.12
        plt.close()

    def test_metadata_increases_margin(self):
        """Metadata lines should increase bottom margin."""
        fig = plt.figure()
        meta = ["Line 1", "Line 2", "Line 3"]
        margin = apply_plot_metadata(fig, meta)
        assert margin > 0.10
        assert margin <= 0.30  # Capped
        plt.close()

    def test_margin_calculation_formula(self):
        """Verify margin calculation: 0.12 + (0.022 * n_lines)."""
        fig = plt.figure()
        meta = ["Line 1", "Line 2"]
        margin = apply_plot_metadata(fig, meta)
        expected = 0.12 + (0.022 * 2)
        assert margin == pytest.approx(expected)
        plt.close()

    def test_margin_capped_at_30_percent(self):
        """Margin should not exceed 30%."""
        fig = plt.figure()
        meta = ["Line"] * 20  # Many lines
        margin = apply_plot_metadata(fig, meta)
        assert margin <= 0.30
        plt.close()

    def test_empty_list_returns_default_margin(self):
        """Empty list should return default margin."""
        fig = plt.figure()
        margin = apply_plot_metadata(fig, [])
        assert margin == 0.12
        plt.close()

    def test_filters_none_values(self):
        """None values in metadata should be filtered."""
        fig = plt.figure()
        meta = ["Line 1", None, "Line 2", None]
        margin = apply_plot_metadata(fig, meta)
        expected = 0.12 + (0.022 * 2)  # Only 2 valid lines
        assert margin == pytest.approx(expected)
        plt.close()


class TestPlotDCA:
    """Test plot_dca function."""

    def test_creates_plot_file(self, sample_dca_df, tmp_path):
        """Test that plot_dca creates output file."""
        out_path = tmp_path / "dca.png"
        plot_dca(sample_dca_df, str(out_path))
        assert out_path.exists()

    def test_plot_with_metadata(self, sample_dca_df, tmp_path):
        """Test plot_dca with metadata lines."""
        out_path = tmp_path / "dca_meta.png"
        meta = ["Model: LR", "Dataset: Test"]
        plot_dca(sample_dca_df, str(out_path), meta_lines=meta)
        assert out_path.exists()

    def test_handles_negative_net_benefit(self, tmp_path):
        """Test plot_dca handles negative net benefits."""
        dca_df = pd.DataFrame(
            {
                "threshold_pct": [0, 5, 10, 15, 20],
                "net_benefit_model": [0.02, 0.01, -0.005, -0.01, -0.015],
                "net_benefit_all": [0.02, 0.01, 0.0, -0.01, -0.02],
                "net_benefit_none": [0, 0, 0, 0, 0],
            }
        )
        out_path = tmp_path / "dca_negative.png"
        plot_dca(dca_df, str(out_path))
        assert out_path.exists()

    def test_plot_with_single_threshold(self, tmp_path):
        """Test plot_dca with single threshold point."""
        dca_df = pd.DataFrame(
            {
                "threshold_pct": [5.0],
                "net_benefit_model": [0.01],
                "net_benefit_all": [0.005],
                "net_benefit_none": [0.0],
            }
        )
        out_path = tmp_path / "dca_single.png"
        plot_dca(dca_df, str(out_path))
        assert out_path.exists()


class TestPlotDCACurve:
    """Test plot_dca_curve function."""

    def test_creates_plot_from_predictions(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve creates output from raw predictions."""
        y_true, y_pred = binary_classification_data
        out_path = tmp_path / "dca_curve.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Test DCA")
        assert out_path.exists()

    def test_plot_with_subtitle(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve with subtitle."""
        y_true, y_pred = binary_classification_data
        out_path = tmp_path / "dca_subtitle.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Test DCA", subtitle="Validation Set")
        assert out_path.exists()

    def test_plot_with_metadata(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve with metadata."""
        y_true, y_pred = binary_classification_data
        out_path = tmp_path / "dca_meta.png"
        meta = ["N=1000", "Prevalence=20%"]
        plot_dca_curve(y_true, y_pred, str(out_path), title="Test DCA", meta_lines=meta)
        assert out_path.exists()

    def test_custom_threshold_range(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve with custom threshold range."""
        y_true, y_pred = binary_classification_data
        out_path = tmp_path / "dca_custom_range.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Test DCA", max_pt=0.10, step=0.01)
        assert out_path.exists()

    def test_handles_empty_predictions(self, tmp_path):
        """Test plot_dca_curve with empty arrays."""
        out_path = tmp_path / "dca_empty.png"
        plot_dca_curve(np.array([]), np.array([]), str(out_path), title="Empty")
        # Should return early without creating file (no data to plot)
        # Function returns None but doesn't raise error

    def test_handles_nan_values(self, tmp_path):
        """Test plot_dca_curve filters NaN values."""
        y_true = np.array([0, 1, 0, 1, np.nan, 0, 1])
        y_pred = np.array([0.2, 0.8, np.nan, 0.7, 0.5, 0.3, 0.9])
        out_path = tmp_path / "dca_nan.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="NaN Test")
        # Should filter NaN and create plot with valid data
        assert out_path.exists()

    def test_single_split(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve with single split ID."""
        y_true, y_pred = binary_classification_data
        split_ids = np.ones(len(y_true))
        out_path = tmp_path / "dca_single_split.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Single Split", split_ids=split_ids)
        assert out_path.exists()

    def test_multi_split_averaging(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve with multiple splits shows confidence bands."""
        y_true, y_pred = binary_classification_data
        # Create 3 splits
        split_ids = np.repeat([0, 1, 2], len(y_true) // 3 + 1)[: len(y_true)]
        out_path = tmp_path / "dca_multi_split.png"
        plot_dca_curve(
            y_true,
            y_pred,
            str(out_path),
            title="Multi-Split DCA",
            split_ids=split_ids,
        )
        assert out_path.exists()

    def test_split_with_insufficient_data(self, tmp_path):
        """Test plot_dca_curve skips splits with insufficient data."""
        # Create data where one split has only one class
        y_true = np.array([0, 0, 1, 1, 0, 0])
        y_pred = np.array([0.2, 0.3, 0.7, 0.8, 0.1, 0.2])
        split_ids = np.array([0, 0, 1, 1, 2, 2])  # Split 2 has only class 0
        out_path = tmp_path / "dca_insufficient.png"
        plot_dca_curve(
            y_true,
            y_pred,
            str(out_path),
            title="Insufficient Data",
            split_ids=split_ids,
        )
        # Should handle gracefully and create plot with valid splits
        assert out_path.exists()

    def test_all_splits_fail(self, tmp_path):
        """Test plot_dca_curve fallback when all splits fail."""
        # Create data where each split has only one sample
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.2, 0.8, 0.3])
        split_ids = np.array([0, 1, 2])
        out_path = tmp_path / "dca_all_fail.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="All Splits Fail", split_ids=split_ids)
        # Should fall back to single curve using all data
        assert out_path.exists()

    def test_plot_sets_reasonable_ylim(self, binary_classification_data, tmp_path):
        """Test plot_dca_curve sets y-axis limits with padding."""
        y_true, y_pred = binary_classification_data
        out_path = tmp_path / "dca_ylim.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Y-Limit Test")
        # Just verify it doesn't crash; actual limits tested visually
        assert out_path.exists()

    def test_zero_range_ylim(self, tmp_path):
        """Test plot_dca_curve handles zero range in y-values."""
        # Contrived case where all net benefits are identical
        y_true = np.ones(100)
        y_pred = np.ones(100) * 0.5
        out_path = tmp_path / "dca_zero_range.png"
        plot_dca_curve(y_true, y_pred, str(out_path), title="Zero Range")
        # Should set default ylim
        assert out_path.exists()


class TestDCAPlottingIntegration:
    """Integration tests for DCA plotting workflow."""

    def test_plot_dca_then_curve_same_data(self, binary_classification_data, tmp_path):
        """Test both plotting functions on same data produce valid outputs."""
        from ced_ml.metrics.dca import decision_curve_analysis

        y_true, y_pred = binary_classification_data

        # Compute DCA
        thresholds = np.arange(0.001, 0.20, 0.005)
        dca_df = decision_curve_analysis(y_true, y_pred, thresholds=thresholds)

        # Plot using both functions
        out1 = tmp_path / "dca_from_df.png"
        out2 = tmp_path / "dca_from_raw.png"

        # Convert thresholds to percentages for plot_dca
        dca_df_pct = dca_df.copy()
        dca_df_pct["threshold_pct"] = dca_df_pct["threshold"] * 100

        plot_dca(dca_df_pct, str(out1))
        plot_dca_curve(y_true, y_pred, str(out2), title="DCA Curve", max_pt=0.20, step=0.005)

        assert out1.exists()
        assert out2.exists()

    def test_multi_split_workflow(self, tmp_path):
        """Test complete workflow with multi-split DCA."""
        np.random.seed(42)

        # Create 5 splits
        y_true_list = []
        y_pred_list = []
        split_ids_list = []

        for i in range(5):
            y = np.array([0] * 160 + [1] * 40)
            p = np.random.beta(2, 5, size=160).tolist() + np.random.beta(5, 2, size=40).tolist()
            y_true_list.append(y)
            y_pred_list.append(p)
            split_ids_list.append(np.ones(len(y)) * i)

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        split_ids = np.concatenate(split_ids_list)

        out_path = tmp_path / "dca_multi_workflow.png"
        plot_dca_curve(
            y_true,
            y_pred,
            str(out_path),
            title="5-Split DCA",
            subtitle="With confidence bands",
            split_ids=split_ids,
            meta_lines=["N=1000", "Splits=5"],
        )

        assert out_path.exists()
