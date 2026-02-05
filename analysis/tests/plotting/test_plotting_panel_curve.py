"""Tests for panel_curve plotting module."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ced_ml.plotting.panel_curve import (
    plot_feature_ranking,
    plot_pareto_curve,
    plot_rfecv_selection_curve,
)

# Check matplotlib availability
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

pytestmark = pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not available")


class TestPlotParetoCurve:
    """Tests for plot_pareto_curve function."""

    def test_basic_plot(self):
        """Basic Pareto curve plot generation."""
        curve = [
            {
                "size": 100,
                "auroc_val": 0.89,
                "auroc_cv": 0.88,
                "auroc_cv_std": 0.02,
                "auroc_val_std": 0.025,
                "auroc_val_ci_low": 0.841,
                "auroc_val_ci_high": 0.939,
            },
            {
                "size": 50,
                "auroc_val": 0.87,
                "auroc_cv": 0.86,
                "auroc_cv_std": 0.03,
                "auroc_val_std": 0.035,
                "auroc_val_ci_low": 0.801,
                "auroc_val_ci_high": 0.939,
            },
            {
                "size": 25,
                "auroc_val": 0.82,
                "auroc_cv": 0.81,
                "auroc_cv_std": 0.04,
                "auroc_val_std": 0.045,
                "auroc_val_ci_low": 0.732,
                "auroc_val_ci_high": 0.908,
            },
            {
                "size": 10,
                "auroc_val": 0.75,
                "auroc_cv": 0.74,
                "auroc_cv_std": 0.05,
                "auroc_val_std": 0.055,
                "auroc_val_ci_low": 0.642,
                "auroc_val_ci_high": 0.858,
            },
        ]
        recommended = {
            "min_size_95pct": 50,
            "min_size_90pct": 25,
            "knee_point": 25,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pareto.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                title="Test Pareto Curve",
                model_name="LR_EN",
            )

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_empty_curve(self):
        """Empty curve returns without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty.png"
            plot_pareto_curve(
                curve=[],
                recommended={},
                out_path=out_path,
            )
            # Should not create file for empty curve
            assert not out_path.exists()

    def test_custom_thresholds(self):
        """Custom threshold annotations work."""
        curve = [
            {"size": 50, "auroc_val": 0.90, "auroc_cv": 0.89, "auroc_cv_std": 0.01},
            {
                "size": 25,
                "auroc_val": 0.85,
                "auroc_cv": 0.84,
                "auroc_cv_std": 0.02,
                "auroc_val_std": 0.025,
            },
        ]
        recommended = {"min_size_95pct": 50, "min_size_85pct": 25}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "custom_thresh.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                thresholds_to_show=[0.95, 0.85],
            )

            assert out_path.exists()

    def test_confidence_intervals_shown(self):
        """Confidence intervals are displayed when show_ci=True."""
        curve = [
            {"size": 100, "auroc_val": 0.92, "auroc_cv": 0.91, "auroc_cv_std": 0.020},
            {"size": 50, "auroc_val": 0.90, "auroc_cv": 0.89, "auroc_cv_std": 0.025},
            {"size": 25, "auroc_val": 0.85, "auroc_cv": 0.84, "auroc_cv_std": 0.030},
        ]
        recommended = {"min_size_95pct": 50, "knee_point": 50}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "with_ci.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                show_ci=True,
                ci_alpha=0.2,
            )

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_confidence_intervals_hidden(self):
        """Confidence intervals can be disabled with show_ci=False."""
        curve = [
            {
                "size": 50,
                "auroc_val": 0.90,
                "auroc_cv": 0.89,
                "auroc_cv_std": 0.02,
                "auroc_val_std": 0.025,
            },
            {
                "size": 25,
                "auroc_val": 0.85,
                "auroc_cv": 0.84,
                "auroc_cv_std": 0.03,
                "auroc_val_std": 0.035,
            },
        ]
        recommended = {"min_size_95pct": 50}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "no_ci.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                show_ci=False,
            )

            assert out_path.exists()

    def test_zero_std_handling(self):
        """Handles zero standard deviation (no CI) gracefully."""
        curve = [
            {"size": 50, "auroc_val": 0.90, "auroc_cv": 0.90, "auroc_cv_std": 0.0},
            {"size": 25, "auroc_val": 0.85, "auroc_cv": 0.85, "auroc_cv_std": 0.0},
        ]
        recommended = {"min_size_95pct": 50}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "zero_std.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                show_ci=True,
            )

            assert out_path.exists()

    def test_comparison_annotations(self):
        """Statistical comparison annotations work correctly."""
        # Create curve with significant and non-significant differences
        curve = [
            {"size": 100, "auroc_val": 0.92, "auroc_cv": 0.91, "auroc_cv_std": 0.010},
            {"size": 50, "auroc_val": 0.90, "auroc_cv": 0.89, "auroc_cv_std": 0.015},
            {"size": 25, "auroc_val": 0.88, "auroc_cv": 0.87, "auroc_cv_std": 0.012},
            {"size": 10, "auroc_val": 0.75, "auroc_cv": 0.74, "auroc_cv_std": 0.025},
        ]
        recommended = {
            "min_size_95pct": 50,  # Should be compared with 90pct
            "min_size_90pct": 25,  # Should be compared with 95pct
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "comparisons.png"
            plot_pareto_curve(
                curve=curve,
                recommended=recommended,
                out_path=out_path,
                thresholds_to_show=[0.95, 0.90],
            )

            assert out_path.exists()


class TestPlotFeatureRanking:
    """Tests for plot_feature_ranking function."""

    def test_basic_ranking_plot(self):
        """Basic feature ranking plot generation."""
        feature_ranking = {
            "PROT_A": 10,
            "PROT_B": 9,
            "PROT_C": 8,
            "PROT_D": 5,
            "PROT_E": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ranking.png"
            plot_feature_ranking(
                feature_ranking=feature_ranking,
                out_path=out_path,
                top_n=5,
                title="Test Feature Ranking",
            )

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_empty_ranking(self):
        """Empty ranking returns without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty_rank.png"
            plot_feature_ranking(
                feature_ranking={},
                out_path=out_path,
            )
            # Should not create file for empty ranking
            assert not out_path.exists()

    def test_top_n_limit(self):
        """Respects top_n parameter."""
        feature_ranking = {f"PROT_{i}": i for i in range(50)}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "top10.png"
            plot_feature_ranking(
                feature_ranking=feature_ranking,
                out_path=out_path,
                top_n=10,
            )

            assert out_path.exists()


class TestPlotRFECVSelectionCurve:
    """Tests for plot_rfecv_selection_curve function."""

    def test_basic_selection_curve(self):
        """Basic RFECV selection curve plot generation."""
        # Create mock cv_scores_curve.csv
        cv_data = []
        for fold in [0, 1, 2]:
            for n_feat in range(1, 11):
                # Simulate increasing then plateauing scores
                score = 0.60 + 0.03 * n_feat - 0.001 * (n_feat**2)
                score += np.random.normal(0, 0.01)  # Slight noise
                cv_data.append({"fold": fold, "n_features": n_feat, "cv_score": score})

        df = pd.DataFrame(cv_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            cv_path = Path(tmpdir) / "cv_scores_curve.csv"
            df.to_csv(cv_path, index=False)

            out_path = Path(tmpdir) / "selection_curve.png"
            plot_rfecv_selection_curve(
                cv_scores_curve_path=cv_path,
                out_path=out_path,
                title="Test RFECV Curve",
                model_name="RF",
            )

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_empty_cv_data(self):
        """Empty CV data returns without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cv_path = Path(tmpdir) / "empty.csv"
            pd.DataFrame(columns=["fold", "n_features", "cv_score"]).to_csv(cv_path, index=False)

            out_path = Path(tmpdir) / "empty_curve.png"
            plot_rfecv_selection_curve(
                cv_scores_curve_path=cv_path,
                out_path=out_path,
            )
            # Should not create file for empty data
            assert not out_path.exists()

    def test_nonexistent_cv_file(self):
        """Nonexistent CV file returns without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cv_path = Path(tmpdir) / "nonexistent.csv"
            out_path = Path(tmpdir) / "curve.png"

            plot_rfecv_selection_curve(
                cv_scores_curve_path=cv_path,
                out_path=out_path,
            )

            assert not out_path.exists()

    def test_single_fold(self):
        """Works with single fold data."""
        cv_data = [{"fold": 0, "n_features": i, "cv_score": 0.70 + 0.02 * i} for i in range(1, 6)]
        df = pd.DataFrame(cv_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            cv_path = Path(tmpdir) / "single_fold.csv"
            df.to_csv(cv_path, index=False)

            out_path = Path(tmpdir) / "single_fold_curve.png"
            plot_rfecv_selection_curve(
                cv_scores_curve_path=cv_path,
                out_path=out_path,
            )

            assert out_path.exists()

    def test_multiple_folds_different_lengths(self):
        """Handles folds with different numbers of features."""
        cv_data = []
        # Fold 0: 10 features
        for n in range(1, 11):
            cv_data.append({"fold": 0, "n_features": n, "cv_score": 0.75 + 0.01 * n})
        # Fold 1: 8 features
        for n in range(1, 9):
            cv_data.append({"fold": 1, "n_features": n, "cv_score": 0.73 + 0.01 * n})

        df = pd.DataFrame(cv_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            cv_path = Path(tmpdir) / "diff_lengths.csv"
            df.to_csv(cv_path, index=False)

            out_path = Path(tmpdir) / "diff_lengths_curve.png"
            plot_rfecv_selection_curve(
                cv_scores_curve_path=cv_path,
                out_path=out_path,
            )

            assert out_path.exists()


class TestPlottingIntegration:
    """Integration tests for plotting functions."""

    def test_all_plots_together(self):
        """Generate all plot types in one test."""
        curve = [
            {
                "size": 20,
                "auroc_val": 0.88,
                "auroc_cv": 0.87,
                "auroc_cv_std": 0.02,
                "auroc_val_std": 0.025,
            },
            {
                "size": 10,
                "auroc_val": 0.82,
                "auroc_cv": 0.81,
                "auroc_cv_std": 0.03,
                "auroc_val_std": 0.035,
            },
        ]
        recommended = {"min_size_95pct": 20, "knee_point": 15}
        feature_ranking = {f"PROT_{i}": i for i in range(20)}

        cv_data = [
            {"fold": f, "n_features": n, "cv_score": 0.70 + 0.02 * n + 0.01 * f}
            for f in range(3)
            for n in range(1, 11)
        ]
        cv_df = pd.DataFrame(cv_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Pareto curve
            pareto_path = tmpdir / "pareto.png"
            plot_pareto_curve(curve, recommended, pareto_path)

            # Feature ranking
            ranking_path = tmpdir / "ranking.png"
            plot_feature_ranking(feature_ranking, ranking_path, top_n=10)

            # RFECV selection curve
            cv_path = tmpdir / "cv_scores.csv"
            cv_df.to_csv(cv_path, index=False)
            selection_path = tmpdir / "selection.png"
            plot_rfecv_selection_curve(cv_path, selection_path)

            # All should exist
            assert pareto_path.exists()
            assert ranking_path.exists()
            assert selection_path.exists()
