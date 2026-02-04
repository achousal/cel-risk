"""Tests for plotting.learning_curve module."""

import numpy as np
import pandas as pd
import pytest
from ced_ml.plotting.learning_curve import (
    _normalize_metric_scores,
    aggregate_learning_curve_runs,
    compute_learning_curve,
    plot_learning_curve,
    plot_learning_curve_summary,
    save_learning_curve_csv,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification data."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 5

    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    return X, y


@pytest.fixture
def simple_pipeline():
    """Create simple logistic regression pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])


class TestComputeLearningCurve:
    """Tests for compute_learning_curve."""

    def test_basic_computation(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        sizes, train_scores, val_scores = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=3, n_points=4, seed=42
        )

        assert len(sizes) == 4
        assert train_scores.shape == (4, 3)
        assert val_scores.shape == (4, 3)

    def test_different_cv_folds(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        sizes, train_scores, val_scores = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=5, n_points=3, seed=42
        )

        assert train_scores.shape == (3, 5)
        assert val_scores.shape == (3, 5)

    def test_different_n_points(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        sizes, train_scores, val_scores = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=3, n_points=6, seed=42
        )

        assert len(sizes) == 6
        assert train_scores.shape == (6, 3)

    def test_min_frac_parameter(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        sizes, _, _ = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=3, min_frac=0.5, n_points=3
        )

        # First size should be approximately 50% of CV training set (which is smaller than full data)
        # With cv=3, each fold uses ~67% of data for training
        # min_frac=0.5 means 50% of that training set
        assert sizes[0] >= 0.30 * len(y)  # Conservative lower bound

    def test_reproducibility_with_seed(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        sizes1, train1, val1 = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=3, seed=42
        )
        sizes2, train2, val2 = compute_learning_curve(
            simple_pipeline, X, y, scoring="roc_auc", cv=3, seed=42
        )

        np.testing.assert_array_equal(sizes1, sizes2)
        np.testing.assert_array_almost_equal(train1, train2)
        np.testing.assert_array_almost_equal(val1, val2)

    def test_different_scoring_metrics(self, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data

        # Test with different scoring metrics
        for scoring in ["roc_auc", "neg_brier_score", "accuracy"]:
            sizes, train_scores, val_scores = compute_learning_curve(
                simple_pipeline, X, y, scoring=scoring, cv=3, n_points=3
            )
            assert train_scores.shape == (3, 3)
            assert val_scores.shape == (3, 3)


class TestNormalizeMetricScores:
    """Tests for _normalize_metric_scores."""

    def test_negative_metric_normalization(self):
        train_scores = np.array([[0.1, 0.2], [0.3, 0.4]])
        val_scores = np.array([[0.15, 0.25], [0.35, 0.45]])

        label, is_error, train_norm, val_norm = _normalize_metric_scores(
            "neg_brier_score", train_scores, val_scores
        )

        assert label == "brier_score"
        assert is_error is True
        np.testing.assert_array_equal(train_norm, -train_scores)
        np.testing.assert_array_equal(val_norm, -val_scores)

    def test_positive_metric_unchanged(self):
        train_scores = np.array([[0.8, 0.9], [0.85, 0.95]])
        val_scores = np.array([[0.75, 0.85], [0.80, 0.90]])

        label, is_error, train_norm, val_norm = _normalize_metric_scores(
            "roc_auc", train_scores, val_scores
        )

        assert label == "roc_auc"
        assert is_error is False
        np.testing.assert_array_equal(train_norm, train_scores)
        np.testing.assert_array_equal(val_norm, val_scores)

    def test_neg_log_loss(self):
        train_scores = np.array([[-0.5, -0.6]])
        val_scores = np.array([[-0.55, -0.65]])

        label, is_error, train_norm, val_norm = _normalize_metric_scores(
            "neg_log_loss", train_scores, val_scores
        )

        assert label == "log_loss"
        assert is_error is True
        assert (train_norm > 0).all()
        assert (val_norm > 0).all()


class TestSaveLearningCurveCsv:
    """Tests for save_learning_curve_csv."""

    def test_csv_creation(self, tmp_path, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data
        out_csv = tmp_path / "learning_curve.csv"

        save_learning_curve_csv(simple_pipeline, X, y, out_csv, scoring="roc_auc", cv=3, n_points=3)

        assert out_csv.exists()

        df = pd.read_csv(out_csv)
        assert "train_size" in df.columns
        assert "cv_split" in df.columns
        assert "train_score" in df.columns
        assert "val_score" in df.columns
        assert "scoring" in df.columns

    def test_csv_content_structure(self, tmp_path, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data
        out_csv = tmp_path / "learning_curve.csv"

        save_learning_curve_csv(simple_pipeline, X, y, out_csv, scoring="roc_auc", cv=3, n_points=4)

        df = pd.read_csv(out_csv)

        # Should have 4 points × 3 folds = 12 rows
        assert len(df) == 12

        # Check unique train sizes
        assert df["train_size"].nunique() == 4

        # Check CV splits
        assert set(df["cv_split"].unique()) == {0, 1, 2}

    def test_metric_metadata(self, tmp_path, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data
        out_csv = tmp_path / "learning_curve_neg.csv"

        save_learning_curve_csv(
            simple_pipeline, X, y, out_csv, scoring="neg_brier_score", cv=3, n_points=3
        )

        df = pd.read_csv(out_csv)

        assert (df["scoring"] == "neg_brier_score").all()
        assert (df["error_metric"] == "brier_score").all()
        assert (df["metric_direction"] == "lower_is_better").all()

    def test_with_plot_generation(self, tmp_path, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data
        out_csv = tmp_path / "learning_curve.csv"
        out_plot = tmp_path / "learning_curve.png"

        save_learning_curve_csv(
            simple_pipeline, X, y, out_csv, scoring="roc_auc", cv=3, out_plot=out_plot
        )

        assert out_csv.exists()
        assert out_plot.exists()

    def test_summary_statistics(self, tmp_path, simple_pipeline, simple_classification_data):
        X, y = simple_classification_data
        out_csv = tmp_path / "learning_curve.csv"

        save_learning_curve_csv(simple_pipeline, X, y, out_csv, scoring="roc_auc", cv=3, n_points=3)

        df = pd.read_csv(out_csv)

        # Check that mean and SD columns exist
        assert "train_score_mean" in df.columns
        assert "train_score_sd" in df.columns
        assert "val_score_mean" in df.columns
        assert "val_score_sd" in df.columns

        # Check that means are reasonable
        assert df["train_score_mean"].notna().all()
        assert df["val_score_mean"].notna().all()


class TestPlotLearningCurve:
    """Tests for plot_learning_curve."""

    def test_plot_creation(self, tmp_path):
        train_sizes = np.array([50, 100, 150])
        train_scores = np.array([[0.8, 0.82, 0.81], [0.85, 0.87, 0.86], [0.88, 0.90, 0.89]])
        val_scores = np.array([[0.75, 0.77, 0.76], [0.78, 0.80, 0.79], [0.80, 0.82, 0.81]])

        out_plot = tmp_path / "lc_plot.png"

        plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            out_plot,
            metric_label="roc_auc",
            metric_is_error=False,
        )

        assert out_plot.exists()

    def test_plot_with_error_metric(self, tmp_path):
        train_sizes = np.array([50, 100, 150])
        train_scores = np.array([[0.2, 0.18, 0.19], [0.15, 0.13, 0.14], [0.12, 0.10, 0.11]])
        val_scores = np.array([[0.25, 0.23, 0.24], [0.20, 0.18, 0.19], [0.18, 0.16, 0.17]])

        out_plot = tmp_path / "lc_error.png"

        plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            out_plot,
            metric_label="brier_score",
            metric_is_error=True,
        )

        assert out_plot.exists()

    def test_plot_with_metadata(self, tmp_path):
        train_sizes = np.array([50, 100])
        train_scores = np.array([[0.8, 0.82], [0.85, 0.87]])
        val_scores = np.array([[0.75, 0.77], [0.78, 0.80]])

        out_plot = tmp_path / "lc_meta.png"
        meta_lines = ["Model: LogisticRegression", "CV: 5-fold", "Seed: 42"]

        plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            out_plot,
            metric_label="roc_auc",
            metric_is_error=False,
            meta_lines=meta_lines,
        )

        assert out_plot.exists()

    def test_handles_invalid_input(self, tmp_path):
        # Test with 1D arrays (should return early)
        train_sizes = np.array([50, 100])
        train_scores = np.array([0.8, 0.85])  # 1D instead of 2D
        val_scores = np.array([0.75, 0.78])  # 1D instead of 2D

        out_plot = tmp_path / "lc_invalid.png"

        # Should not raise, just return early without creating plot
        plot_learning_curve(
            train_sizes,
            train_scores,
            val_scores,
            out_plot,
            metric_label="roc_auc",
            metric_is_error=False,
        )

        assert not out_plot.exists()


class TestAggregateLearningCurveRuns:
    """Tests for aggregate_learning_curve_runs."""

    def test_empty_list(self):
        result = aggregate_learning_curve_runs([])
        assert result.empty

    def test_single_run_aggregation(self):
        df1 = pd.DataFrame(
            {
                "train_size": [50, 100, 150, 50, 100, 150],
                "cv_split": [0, 0, 0, 1, 1, 1],
                "train_score": [0.8, 0.85, 0.88, 0.82, 0.87, 0.90],
                "val_score": [0.75, 0.78, 0.80, 0.77, 0.80, 0.82],
                "error_metric": ["roc_auc"] * 6,
                "metric_direction": ["higher_is_better"] * 6,
                "scoring": ["roc_auc"] * 6,
            }
        )

        result = aggregate_learning_curve_runs([df1])

        assert "train_size" in result.columns
        assert "train_mean" in result.columns
        assert "val_mean" in result.columns
        assert "train_sd" in result.columns
        assert "val_sd" in result.columns
        assert len(result) == 3  # 3 unique train sizes

    def test_multiple_runs_aggregation(self):
        df1 = pd.DataFrame(
            {
                "train_size": [50, 100, 50, 100],
                "train_score": [0.8, 0.85, 0.82, 0.87],
                "val_score": [0.75, 0.78, 0.77, 0.80],
                "error_metric": ["roc_auc"] * 4,
                "metric_direction": ["higher_is_better"] * 4,
                "scoring": ["roc_auc"] * 4,
            }
        )
        df1["run_dir"] = "run1"

        df2 = pd.DataFrame(
            {
                "train_size": [50, 100, 50, 100],
                "train_score": [0.78, 0.83, 0.80, 0.85],
                "val_score": [0.73, 0.76, 0.75, 0.78],
                "error_metric": ["roc_auc"] * 4,
                "metric_direction": ["higher_is_better"] * 4,
                "scoring": ["roc_auc"] * 4,
            }
        )
        df2["run_dir"] = "run2"

        result = aggregate_learning_curve_runs([df1, df2])

        assert len(result) == 2  # 2 unique train sizes
        assert result["n_runs"].iloc[0] == 2

    def test_confidence_intervals(self):
        # Create multiple runs to get CIs
        runs = []
        for i in range(10):
            df = pd.DataFrame(
                {
                    "train_size": [100, 200],
                    "train_score": [0.8 + i * 0.01, 0.85 + i * 0.01],
                    "val_score": [0.75 + i * 0.01, 0.80 + i * 0.01],
                    "error_metric": ["roc_auc"] * 2,
                    "metric_direction": ["higher_is_better"] * 2,
                    "scoring": ["roc_auc"] * 2,
                }
            )
            df["run_dir"] = f"run{i}"
            runs.append(df)

        result = aggregate_learning_curve_runs(runs)

        assert "train_ci_lo" in result.columns
        assert "train_ci_hi" in result.columns
        assert "val_ci_lo" in result.columns
        assert "val_ci_hi" in result.columns

        # CI bounds should be finite
        assert result["train_ci_lo"].notna().all()
        assert result["train_ci_hi"].notna().all()

    def test_metric_metadata_preserved(self):
        df1 = pd.DataFrame(
            {
                "train_size": [50, 100],
                "train_score": [0.8, 0.85],
                "val_score": [0.75, 0.78],
                "error_metric": ["brier_score", "brier_score"],
                "metric_direction": ["lower_is_better", "lower_is_better"],
                "scoring": ["neg_brier_score", "neg_brier_score"],
            }
        )

        result = aggregate_learning_curve_runs([df1])

        assert result["metric_label"].iloc[0] == "brier_score"
        assert result["metric_direction"].iloc[0] == "lower_is_better"
        assert result["scoring"].iloc[0] == "neg_brier_score"

    def test_handles_none_and_empty(self):
        df1 = pd.DataFrame({"train_size": [50], "train_score": [0.8], "val_score": [0.75]})
        df_empty = pd.DataFrame()

        result = aggregate_learning_curve_runs([None, df_empty, df1])

        # Should process only valid df1
        assert not result.empty


class TestPlotLearningCurveSummary:
    """Tests for plot_learning_curve_summary."""

    def test_summary_plot_creation(self, tmp_path):
        df = pd.DataFrame(
            {
                "train_size": [50, 100, 150],
                "train_mean": [0.8, 0.85, 0.88],
                "train_sd": [0.02, 0.015, 0.01],
                "train_ci_lo": [0.78, 0.83, 0.86],
                "train_ci_hi": [0.82, 0.87, 0.90],
                "val_mean": [0.75, 0.78, 0.80],
                "val_sd": [0.03, 0.025, 0.02],
                "val_ci_lo": [0.72, 0.75, 0.77],
                "val_ci_hi": [0.78, 0.81, 0.83],
                "metric_label": ["roc_auc"] * 3,
                "metric_direction": ["higher_is_better"] * 3,
            }
        )

        out_plot = tmp_path / "summary.png"

        plot_learning_curve_summary(df, out_plot, title="Learning Curve Summary")

        assert out_plot.exists()

    def test_empty_dataframe(self, tmp_path):
        out_plot = tmp_path / "empty.png"

        # Should not raise, just return early
        plot_learning_curve_summary(pd.DataFrame(), out_plot, title="Empty")

        assert not out_plot.exists()

    def test_with_metadata(self, tmp_path):
        df = pd.DataFrame(
            {
                "train_size": [50, 100],
                "train_mean": [0.8, 0.85],
                "train_sd": [0.02, 0.015],
                "train_ci_lo": [0.78, 0.83],
                "train_ci_hi": [0.82, 0.87],
                "val_mean": [0.75, 0.78],
                "val_sd": [0.03, 0.025],
                "val_ci_lo": [0.72, 0.75],
                "val_ci_hi": [0.78, 0.81],
                "metric_label": ["brier_score"] * 2,
                "metric_direction": ["lower_is_better"] * 2,
            }
        )

        out_plot = tmp_path / "meta_summary.png"
        meta_lines = ["Aggregated across 10 runs", "CV: 5-fold"]

        plot_learning_curve_summary(df, out_plot, title="Test", meta_lines=meta_lines)

        assert out_plot.exists()
