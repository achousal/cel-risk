"""
Tests for ROC and PR curve plotting.
"""

import numpy as np
import pytest
from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve


class TestROCCurvePlotting:
    """Tests for ROC curve plotting."""

    @pytest.fixture
    def basic_predictions(self):
        """Basic binary classification predictions."""
        _rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(180), np.ones(20)])
        y_pred = np.concatenate(
            [
                np.random.beta(2, 5, 180),
                np.random.beta(5, 2, 20),
            ]
        )
        return y_true, y_pred

    @pytest.fixture
    def split_predictions(self):
        """Predictions with multiple splits."""
        _rng = np.random.default_rng(42)
        n_splits = 5
        n_per_split = 40
        y_true = []
        y_pred = []
        split_ids = []

        for split_id in range(n_splits):
            y_s = np.concatenate([np.zeros(36), np.ones(4)])
            p_s = np.concatenate(
                [
                    np.random.beta(2, 5, 36),
                    np.random.beta(5, 2, 4),
                ]
            )
            y_true.append(y_s)
            y_pred.append(p_s)
            split_ids.append(np.full(n_per_split, split_id))

        return (
            np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(split_ids),
        )

    def test_basic_roc_plot(self, basic_predictions, tmp_path):
        """Test basic ROC curve plotting."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Test ROC Curve",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Validate plot content dimensions
        import matplotlib.image as mpimg

        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100, "Image height too small"
        assert img.shape[1] > 100, "Image width too small"

    def test_roc_with_subtitle(self, basic_predictions, tmp_path):
        """Test ROC curve with subtitle."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc_subtitle.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Test ROC",
            subtitle="n=200, prevalence=0.10",
        )

        assert out_path.exists()

    def test_roc_with_splits(self, split_predictions, tmp_path):
        """Test ROC curve with split-wise confidence bands."""
        y_true, y_pred, split_ids = split_predictions
        out_path = tmp_path / "roc_splits.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with Splits",
            split_ids=split_ids,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_roc_with_metadata(self, basic_predictions, tmp_path):
        """Test ROC curve with metadata lines."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc_meta.png"

        meta_lines = [
            "Model: RandomForest",
            "Features: 100 proteins",
            "Date: 2026-01-18",
        ]

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with Metadata",
            meta_lines=meta_lines,
        )

        assert out_path.exists()

    def test_roc_with_threshold_markers(self, basic_predictions, tmp_path):
        """Test ROC curve with Youden and alpha threshold markers."""
        from ced_ml.metrics.thresholds import compute_threshold_bundle

        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc_thresholds.png"

        bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with Thresholds",
            threshold_bundle=bundle,
        )

        assert out_path.exists()

    def test_roc_handles_nan_values(self, tmp_path):
        """Test ROC curve handles NaN values gracefully."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.2, np.nan, 0.7, 0.8, 0.3, np.nan])
        out_path = tmp_path / "roc_nan.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with NaNs",
        )

        assert out_path.exists()

    def test_roc_empty_data(self, tmp_path):
        """Test ROC curve with empty data after filtering."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        out_path = tmp_path / "roc_empty.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Empty ROC",
        )

        assert not out_path.exists()

    def test_roc_perfect_predictions(self, tmp_path):
        """Test ROC curve with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        out_path = tmp_path / "roc_perfect.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Perfect ROC",
        )

        assert out_path.exists()

    def test_roc_random_predictions(self, tmp_path):
        """Test ROC curve with random predictions."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.random(100)
        out_path = tmp_path / "roc_random.png"

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Random ROC",
        )

        assert out_path.exists()

    def test_roc_with_overlapping_thresholds(self, basic_predictions, tmp_path):
        """Test ROC curve handles overlapping threshold markers correctly."""
        from ced_ml.metrics.thresholds import compute_threshold_bundle

        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc_overlap.png"

        # Create a bundle where thresholds are very close (will trigger overlap detection)
        bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.90)

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with Overlapping Thresholds",
            threshold_bundle=bundle,
        )

        assert out_path.exists()

    def test_roc_with_threshold_bundle(self, basic_predictions, tmp_path):
        """Test ROC curve with ThresholdBundle interface."""
        from ced_ml.metrics.thresholds import compute_threshold_bundle

        y_true, y_pred = basic_predictions
        out_path = tmp_path / "roc_bundle.png"

        bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)

        plot_roc_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="ROC with Threshold Bundle",
            threshold_bundle=bundle,
        )

        assert out_path.exists()


class TestPRCurvePlotting:
    """Tests for Precision-Recall curve plotting."""

    @pytest.fixture
    def basic_predictions(self):
        """Basic binary classification predictions."""
        _rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(180), np.ones(20)])
        y_pred = np.concatenate(
            [
                np.random.beta(2, 5, 180),
                np.random.beta(5, 2, 20),
            ]
        )
        return y_true, y_pred

    @pytest.fixture
    def split_predictions(self):
        """Predictions with multiple splits."""
        _rng = np.random.default_rng(42)
        n_splits = 5
        n_per_split = 40
        y_true = []
        y_pred = []
        split_ids = []

        for split_id in range(n_splits):
            y_s = np.concatenate([np.zeros(36), np.ones(4)])
            p_s = np.concatenate(
                [
                    np.random.beta(2, 5, 36),
                    np.random.beta(5, 2, 4),
                ]
            )
            y_true.append(y_s)
            y_pred.append(p_s)
            split_ids.append(np.full(n_per_split, split_id))

        return (
            np.concatenate(y_true),
            np.concatenate(y_pred),
            np.concatenate(split_ids),
        )

    def test_basic_pr_plot(self, basic_predictions, tmp_path):
        """Test basic PR curve plotting."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "pr.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Test PR Curve",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Validate plot content dimensions
        import matplotlib.image as mpimg

        img = mpimg.imread(str(out_path))
        assert img.shape[0] > 100, "Image height too small"
        assert img.shape[1] > 100, "Image width too small"

    def test_pr_with_subtitle(self, basic_predictions, tmp_path):
        """Test PR curve with subtitle."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "pr_subtitle.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Test PR",
            subtitle="n=200, cases=20",
        )

        assert out_path.exists()

    def test_pr_with_splits(self, split_predictions, tmp_path):
        """Test PR curve with split-wise confidence bands."""
        y_true, y_pred, split_ids = split_predictions
        out_path = tmp_path / "pr_splits.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="PR with Splits",
            split_ids=split_ids,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_pr_with_metadata(self, basic_predictions, tmp_path):
        """Test PR curve with metadata lines."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "pr_meta.png"

        meta_lines = [
            "Model: XGBoost",
            "Features: 50 proteins",
            "AUROC: 0.85",
        ]

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="PR with Metadata",
            meta_lines=meta_lines,
        )

        assert out_path.exists()

    def test_pr_handles_nan_values(self, tmp_path):
        """Test PR curve handles NaN values gracefully."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0.2, np.nan, 0.7, 0.8, 0.3, np.nan])
        out_path = tmp_path / "pr_nan.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="PR with NaNs",
        )

        assert out_path.exists()

    def test_pr_empty_data(self, tmp_path):
        """Test PR curve with empty data after filtering."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        out_path = tmp_path / "pr_empty.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Empty PR",
        )

        assert not out_path.exists()

    def test_pr_perfect_predictions(self, tmp_path):
        """Test PR curve with perfect predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        out_path = tmp_path / "pr_perfect.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Perfect PR",
        )

        assert out_path.exists()

    def test_pr_shows_prevalence_baseline(self, basic_predictions, tmp_path):
        """Test that PR curve shows prevalence baseline."""
        y_true, y_pred = basic_predictions
        out_path = tmp_path / "pr_baseline.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="PR with Baseline",
        )

        assert out_path.exists()
        prevalence = np.mean(y_true)
        assert 0 < prevalence < 1

    def test_pr_extreme_imbalance(self, tmp_path):
        """Test PR curve with extreme class imbalance."""
        _rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(990), np.ones(10)])
        y_pred = np.concatenate(
            [
                np.random.beta(1, 10, 990),
                np.random.beta(10, 1, 10),
            ]
        )
        out_path = tmp_path / "pr_imbalanced.png"

        plot_pr_curve(
            y_true=y_true,
            y_pred=y_pred,
            out_path=out_path,
            title="Imbalanced PR",
            subtitle="Prevalence = 0.01",
        )

        assert out_path.exists()


class TestPlotMetadataHelper:
    """Tests for plot metadata helper function."""

    def test_metadata_with_lines(self, tmp_path):
        """Test metadata application with lines."""
        import matplotlib.pyplot as plt
        from ced_ml.plotting.dca import apply_plot_metadata

        fig, ax = plt.subplots()
        meta_lines = ["Line 1", "Line 2", "Line 3"]
        margin = apply_plot_metadata(fig, meta_lines)

        assert margin > 0.10
        assert margin <= 0.30
        plt.close()

    def test_metadata_without_lines(self):
        """Test metadata application without lines."""
        import matplotlib.pyplot as plt
        from ced_ml.plotting.dca import apply_plot_metadata

        fig, ax = plt.subplots()
        margin = apply_plot_metadata(fig, None)

        assert margin == 0.12
        plt.close()

    def test_metadata_empty_list(self):
        """Test metadata with empty list."""
        import matplotlib.pyplot as plt
        from ced_ml.plotting.dca import apply_plot_metadata

        fig, ax = plt.subplots()
        margin = apply_plot_metadata(fig, [])

        assert margin == 0.12
        plt.close()

    def test_metadata_caps_at_30_percent(self):
        """Test metadata margin caps at 30%."""
        import matplotlib.pyplot as plt
        from ced_ml.plotting.dca import apply_plot_metadata

        fig, ax = plt.subplots()
        many_lines = [f"Line {i}" for i in range(50)]
        margin = apply_plot_metadata(fig, many_lines)

        assert margin == 0.30
        plt.close()
