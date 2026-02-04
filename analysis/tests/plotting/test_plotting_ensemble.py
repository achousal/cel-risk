"""Tests for ensemble-specific plotting functions."""

import numpy as np
from ced_ml.plotting.ensemble import (
    plot_aggregated_weights,
    plot_meta_learner_weights,
    plot_model_comparison,
)


class TestPlotMetaLearnerWeights:
    """Tests for plot_meta_learner_weights."""

    def test_basic_plot(self, tmp_path):
        """Generates a horizontal bar chart with valid coefficients."""
        coef = {"LR_EN": 0.45, "RF": 0.35, "XGBoost": 0.20}
        out_path = tmp_path / "weights.png"

        plot_meta_learner_weights(coef, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_negative_coefficients(self, tmp_path):
        """Handles negative coefficients with correct coloring."""
        coef = {"LR_EN": 0.60, "RF": -0.15, "XGBoost": 0.30}
        out_path = tmp_path / "weights_neg.png"

        plot_meta_learner_weights(coef, out_path)

        assert out_path.exists()

    def test_single_model(self, tmp_path):
        """Works with a single base model coefficient."""
        coef = {"LR_EN": 0.90}
        out_path = tmp_path / "weights_single.png"

        plot_meta_learner_weights(coef, out_path)

        assert out_path.exists()

    def test_empty_coef_skips(self, tmp_path):
        """Empty coefficient dict does not create a file."""
        out_path = tmp_path / "weights_empty.png"

        plot_meta_learner_weights({}, out_path)

        assert not out_path.exists()

    def test_with_meta_lines(self, tmp_path):
        """Metadata lines are included without error."""
        coef = {"LR_EN": 0.5, "RF": 0.3}
        out_path = tmp_path / "weights_meta.png"

        plot_meta_learner_weights(
            coef,
            out_path,
            meta_lines=["Model: ENSEMBLE", "Split seed: 0"],
        )

        assert out_path.exists()

    def test_custom_title_and_subtitle(self, tmp_path):
        """Custom title and subtitle render correctly."""
        coef = {"LR_EN": 0.4, "RF": 0.35, "XGBoost": 0.25}
        out_path = tmp_path / "weights_title.png"

        plot_meta_learner_weights(
            coef,
            out_path,
            title="Custom Title",
            subtitle="split_seed=5",
            meta_penalty="l1",
            meta_c=0.5,
        )

        assert out_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Creates parent directories if they do not exist."""
        coef = {"LR_EN": 0.5}
        out_path = tmp_path / "subdir" / "deep" / "weights.png"

        plot_meta_learner_weights(coef, out_path)

        assert out_path.exists()

    def test_many_models(self, tmp_path):
        """Handles many base models gracefully."""
        coef = {f"Model_{i}": np.random.uniform(-1, 1) for i in range(10)}
        out_path = tmp_path / "weights_many.png"

        plot_meta_learner_weights(coef, out_path)

        assert out_path.exists()


class TestPlotModelComparison:
    """Tests for plot_model_comparison."""

    def test_basic_comparison(self, tmp_path):
        """Grouped bar chart with multiple models."""
        metrics = {
            "LR_EN": {"auroc": 0.89, "prauc": 0.30, "brier": 0.012},
            "RF": {"auroc": 0.87, "prauc": 0.28, "brier": 0.013},
            "ENSEMBLE": {"auroc": 0.91, "prauc": 0.33, "brier": 0.011},
        }
        out_path = tmp_path / "comparison.png"

        plot_model_comparison(metrics, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_highlight_ensemble(self, tmp_path):
        """ENSEMBLE bar has distinct visual treatment."""
        metrics = {
            "LR_EN": {"auroc": 0.85, "prauc": 0.25, "brier": 0.015},
            "ENSEMBLE": {"auroc": 0.88, "prauc": 0.28, "brier": 0.013},
        }
        out_path = tmp_path / "comparison_highlight.png"

        plot_model_comparison(metrics, out_path, highlight_model="ENSEMBLE")

        assert out_path.exists()

    def test_custom_metric_names(self, tmp_path):
        """Can specify a subset of metrics to display."""
        metrics = {
            "LR_EN": {"auroc": 0.89, "prauc": 0.30, "brier": 0.012},
            "ENSEMBLE": {"auroc": 0.91, "prauc": 0.33, "brier": 0.011},
        }
        out_path = tmp_path / "comparison_custom.png"

        plot_model_comparison(metrics, out_path, metric_names=["auroc", "brier"])

        assert out_path.exists()

    def test_single_model_skips(self, tmp_path):
        """Does not create a plot with fewer than 2 models."""
        metrics = {"LR_EN": {"auroc": 0.89}}
        out_path = tmp_path / "comparison_single.png"

        plot_model_comparison(metrics, out_path)

        assert not out_path.exists()

    def test_empty_metrics_skips(self, tmp_path):
        """Empty metrics dict does not create a file."""
        out_path = tmp_path / "comparison_empty.png"

        plot_model_comparison({}, out_path)

        assert not out_path.exists()

    def test_partial_metrics(self, tmp_path):
        """Handles models with missing metric keys."""
        metrics = {
            "LR_EN": {"auroc": 0.89, "prauc": 0.30},
            "ENSEMBLE": {"auroc": 0.91, "brier": 0.011},
        }
        out_path = tmp_path / "comparison_partial.png"

        plot_model_comparison(metrics, out_path)

        assert out_path.exists()

    def test_with_meta_lines(self, tmp_path):
        """Metadata lines are applied without error."""
        metrics = {
            "LR_EN": {"auroc": 0.89, "prauc": 0.30, "brier": 0.012},
            "RF": {"auroc": 0.87, "prauc": 0.28, "brier": 0.013},
        }
        out_path = tmp_path / "comparison_meta.png"

        plot_model_comparison(metrics, out_path, meta_lines=["n_models=2", "pooled test"])

        assert out_path.exists()

    def test_many_models(self, tmp_path):
        """Handles many models without layout issues."""
        metrics = {
            f"Model_{i}": {
                "auroc": 0.80 + i * 0.01,
                "prauc": 0.20 + i * 0.01,
                "brier": 0.015 - i * 0.001,
            }
            for i in range(6)
        }
        out_path = tmp_path / "comparison_many.png"

        plot_model_comparison(metrics, out_path)

        assert out_path.exists()


class TestPlotAggregatedWeights:
    """Tests for plot_aggregated_weights."""

    def test_multiple_splits(self, tmp_path):
        """Mean and SD error bars shown across splits."""
        coefs = {
            0: {"LR_EN": 0.45, "RF": 0.35, "XGBoost": 0.20},
            1: {"LR_EN": 0.50, "RF": 0.30, "XGBoost": 0.22},
            2: {"LR_EN": 0.42, "RF": 0.38, "XGBoost": 0.18},
        }
        out_path = tmp_path / "agg_weights.png"

        plot_aggregated_weights(coefs, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_single_split(self, tmp_path):
        """Degrades gracefully with a single split (zero SD, no error bars)."""
        coefs = {0: {"LR_EN": 0.6, "RF": 0.4}}
        out_path = tmp_path / "agg_weights_single.png"

        plot_aggregated_weights(coefs, out_path)

        assert out_path.exists()

    def test_empty_coefs_skips(self, tmp_path):
        """Empty dict does not create a file."""
        out_path = tmp_path / "agg_weights_empty.png"

        plot_aggregated_weights({}, out_path)

        assert not out_path.exists()

    def test_inconsistent_models_across_splits(self, tmp_path):
        """Handles splits with different base model sets."""
        coefs = {
            0: {"LR_EN": 0.5, "RF": 0.3, "XGBoost": 0.2},
            1: {"LR_EN": 0.55, "RF": 0.35},  # Missing XGBoost
        }
        out_path = tmp_path / "agg_weights_inconsistent.png"

        plot_aggregated_weights(coefs, out_path)

        assert out_path.exists()

    def test_negative_coefficients(self, tmp_path):
        """Negative mean coefficients colored correctly."""
        coefs = {
            0: {"LR_EN": 0.6, "RF": -0.2},
            1: {"LR_EN": 0.5, "RF": -0.3},
        }
        out_path = tmp_path / "agg_weights_neg.png"

        plot_aggregated_weights(coefs, out_path)

        assert out_path.exists()

    def test_with_subtitle(self, tmp_path):
        """Subtitle renders correctly."""
        coefs = {
            0: {"LR_EN": 0.5, "RF": 0.3},
            1: {"LR_EN": 0.55, "RF": 0.35},
        }
        out_path = tmp_path / "agg_weights_subtitle.png"

        plot_aggregated_weights(
            coefs,
            out_path,
            subtitle="Custom subtitle",
        )

        assert out_path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if needed."""
        coefs = {0: {"LR_EN": 0.5}}
        out_path = tmp_path / "nested" / "dir" / "agg.png"

        plot_aggregated_weights(coefs, out_path)

        assert out_path.exists()
