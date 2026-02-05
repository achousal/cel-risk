"""Tests for Nested RFECV module."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ced_ml.features.nested_rfe import (
    NestedRFECVResult,
    RFECVFoldResult,
    aggregate_rfecv_results,
    compute_consensus_panel,
    run_rfecv_within_fold,
    save_nested_rfecv_results,
)


@pytest.fixture
def sample_data(sample_data_nested_rfe):
    """Alias for sample_data_nested_rfe from conftest for backward compatibility."""
    return sample_data_nested_rfe


@pytest.fixture
def train_val_split(sample_data):
    """Split sample data into train and validation."""
    X_df, y, feature_names = sample_data
    n_train = 150
    X_train = X_df.iloc[:n_train]
    y_train = y[:n_train]
    X_val = X_df.iloc[n_train:]
    y_val = y[n_train:]
    return X_train, y_train, X_val, y_val, feature_names


class TestRunRFECVWithinFold:
    """Tests for run_rfecv_within_fold function."""

    def test_basic_rfecv(self, train_val_split):
        """RFECV runs and returns valid results."""
        X_train, y_train, X_val, y_val, feature_names = train_val_split

        estimator = LogisticRegression(
            solver="saga", penalty="l1", C=0.1, max_iter=1000, random_state=42
        )

        result = run_rfecv_within_fold(
            X_train_fold=X_train,
            y_train_fold=y_train,
            X_val_fold=X_val,
            y_val_fold=y_val,
            estimator=estimator,
            feature_names=feature_names,
            fold_idx=0,
            min_features=5,
            step=5,  # Faster for testing
            cv_folds=3,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(result, RFECVFoldResult)
        assert result.optimal_n_features >= 5
        assert result.optimal_n_features <= len(feature_names)
        assert len(result.selected_features) == result.optimal_n_features
        assert 0.0 <= result.val_auroc <= 1.0
        assert len(result.cv_scores) > 0
        assert all(f in feature_names for f in result.selected_features)

    def test_rfecv_with_rf(self, train_val_split):
        """RFECV works with RandomForest (feature_importances_)."""
        X_train, y_train, X_val, y_val, feature_names = train_val_split

        estimator = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=1)

        result = run_rfecv_within_fold(
            X_train_fold=X_train,
            y_train_fold=y_train,
            X_val_fold=X_val,
            y_val_fold=y_val,
            estimator=estimator,
            feature_names=feature_names,
            fold_idx=0,
            min_features=5,
            step=10,
            cv_folds=2,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(result, RFECVFoldResult)
        assert result.optimal_n_features >= 5
        assert len(result.selected_features) > 0

    def test_feature_ranking_valid(self, train_val_split):
        """Feature ranking contains all features with valid ranks."""
        X_train, y_train, X_val, y_val, feature_names = train_val_split

        estimator = LogisticRegression(
            solver="saga", penalty="l1", C=0.1, max_iter=1000, random_state=42
        )

        result = run_rfecv_within_fold(
            X_train_fold=X_train,
            y_train_fold=y_train,
            X_val_fold=X_val,
            y_val_fold=y_val,
            estimator=estimator,
            feature_names=feature_names,
            fold_idx=0,
            min_features=5,
            step=10,
            cv_folds=2,
            random_state=42,
        )

        # All features should have a ranking
        assert set(result.feature_ranking.keys()) == set(feature_names)
        # Selected features should have rank 1
        for f in result.selected_features:
            assert result.feature_ranking[f] == 1

    def test_rfecv_with_linearsvc(self, train_val_split):
        """RFECV works with LinearSVC (uses decision_function instead of predict_proba)."""
        X_train, y_train, X_val, y_val, feature_names = train_val_split

        estimator = LinearSVC(C=0.1, max_iter=1000, random_state=42, dual="auto")

        result = run_rfecv_within_fold(
            X_train_fold=X_train,
            y_train_fold=y_train,
            X_val_fold=X_val,
            y_val_fold=y_val,
            estimator=estimator,
            feature_names=feature_names,
            fold_idx=0,
            min_features=5,
            step=10,
            cv_folds=2,
            scoring="roc_auc",
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(result, RFECVFoldResult)
        assert result.optimal_n_features >= 5
        assert len(result.selected_features) > 0
        assert 0.0 <= result.val_auroc <= 1.0


class TestComputeConsensusPanel:
    """Tests for compute_consensus_panel function."""

    def test_basic_consensus(self):
        """Basic consensus panel computation."""
        fold_selections = [
            ["A", "B", "C", "D"],
            ["A", "B", "C", "E"],
            ["A", "B", "D", "E"],
            ["A", "B", "C", "D"],
            ["A", "B", "C", "E"],
        ]
        # A: 5/5 = 1.0, B: 5/5 = 1.0, C: 4/5 = 0.8, D: 3/5 = 0.6, E: 3/5 = 0.6
        consensus, stability = compute_consensus_panel(fold_selections, threshold=0.80)

        assert "A" in consensus
        assert "B" in consensus
        assert "C" in consensus
        assert "D" not in consensus  # 0.6 < 0.8
        assert "E" not in consensus  # 0.6 < 0.8
        assert stability["A"] == 1.0
        assert stability["B"] == 1.0
        assert stability["C"] == 0.8
        assert stability["D"] == 0.6
        assert stability["E"] == 0.6

    def test_high_threshold(self):
        """High threshold filters more features."""
        fold_selections = [
            ["A", "B", "C"],
            ["A", "B", "D"],
            ["A", "C", "D"],
        ]
        consensus, _ = compute_consensus_panel(fold_selections, threshold=1.0)
        # Only A appears in all folds
        assert consensus == ["A"]

    def test_low_threshold(self):
        """Low threshold includes more features."""
        fold_selections = [
            ["A", "B"],
            ["C", "D"],
        ]
        consensus, _ = compute_consensus_panel(fold_selections, threshold=0.5)
        # All features appear in 50% of folds
        assert set(consensus) == {"A", "B", "C", "D"}

    def test_empty_input(self):
        """Empty input returns empty results."""
        consensus, stability = compute_consensus_panel([])
        assert consensus == []
        assert stability == {}

    def test_single_fold(self):
        """Single fold returns all features."""
        fold_selections = [["A", "B", "C"]]
        consensus, stability = compute_consensus_panel(fold_selections, threshold=1.0)
        assert set(consensus) == {"A", "B", "C"}
        assert all(s == 1.0 for s in stability.values())


class TestAggregateRFECVResults:
    """Tests for aggregate_rfecv_results function."""

    def test_basic_aggregation(self):
        """Basic aggregation of fold results."""
        fold_results = [
            RFECVFoldResult(
                fold_idx=0,
                optimal_n_features=10,
                selected_features=["A", "B", "C"],
                cv_scores=[0.8, 0.82, 0.85],
                feature_ranking={"A": 1, "B": 1, "C": 1, "D": 2},
                val_auroc=0.85,
            ),
            RFECVFoldResult(
                fold_idx=1,
                optimal_n_features=12,
                selected_features=["A", "B", "D"],
                cv_scores=[0.78, 0.80, 0.83],
                feature_ranking={"A": 1, "B": 1, "C": 2, "D": 1},
                val_auroc=0.82,
            ),
        ]

        result = aggregate_rfecv_results(fold_results, consensus_threshold=0.80)

        assert isinstance(result, NestedRFECVResult)
        assert len(result.fold_results) == 2
        assert result.optimal_sizes == [10, 12]
        assert result.mean_optimal_size == 11.0
        assert result.fold_val_aurocs == [0.85, 0.82]
        # A and B in both folds (100%), C and D in one fold (50%)
        assert "A" in result.consensus_panel
        assert "B" in result.consensus_panel
        assert result.feature_stability["A"] == 1.0
        assert result.feature_stability["B"] == 1.0


class TestSaveNestedRFECVResults:
    """Tests for save_nested_rfecv_results function."""

    def test_saves_all_artifacts(self):
        """All expected artifacts are saved."""
        result = NestedRFECVResult(
            fold_results=[
                RFECVFoldResult(
                    fold_idx=0,
                    optimal_n_features=10,
                    selected_features=["A", "B"],
                    cv_scores=[0.8, 0.85],
                    feature_ranking={"A": 1, "B": 1, "C": 2},
                    val_auroc=0.85,
                ),
            ],
            consensus_panel=["A", "B"],
            feature_stability={"A": 1.0, "B": 1.0, "C": 0.5},
            optimal_sizes=[10],
            mean_optimal_size=10.0,
            fold_val_aurocs=[0.85],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_nested_rfecv_results(
                result=result,
                output_dir=tmpdir,
                model_name="LR_EN",
                split_seed=0,
            )

            # Check all expected files exist
            assert Path(paths["consensus_panel"]).exists()
            assert Path(paths["feature_stability"]).exists()
            assert Path(paths["fold_results"]).exists()
            assert Path(paths["summary"]).exists()

            # Verify consensus panel content
            consensus_df = pd.read_csv(paths["consensus_panel"])
            assert set(consensus_df["protein"]) == {"A", "B"}

            # Verify summary JSON
            with open(paths["summary"]) as f:
                summary = json.load(f)
            assert summary["model"] == "LR_EN"
            assert summary["split_seed"] == 0
            assert summary["consensus_panel_size"] == 2
            assert summary["mean_optimal_size"] == 10.0

    def test_cv_scores_curve_saved(self):
        """CV scores curve is saved when present."""
        result = NestedRFECVResult(
            fold_results=[
                RFECVFoldResult(
                    fold_idx=0,
                    optimal_n_features=3,
                    selected_features=["A"],
                    cv_scores=[0.7, 0.75, 0.8],  # 3 points
                    feature_ranking={"A": 1},
                    val_auroc=0.8,
                ),
            ],
            consensus_panel=["A"],
            feature_stability={"A": 1.0},
            optimal_sizes=[3],
            mean_optimal_size=3.0,
            fold_val_aurocs=[0.8],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_nested_rfecv_results(
                result=result, output_dir=tmpdir, model_name="RF", split_seed=1
            )

            assert "cv_scores_curve" in paths
            cv_df = pd.read_csv(paths["cv_scores_curve"])
            assert len(cv_df) == 3  # 3 CV score points
            assert "fold" in cv_df.columns
            assert "n_features" in cv_df.columns
            assert "cv_score" in cv_df.columns

    def test_rfecv_selection_curve_plot_generated(self):
        """RFECV selection curve plot is generated when matplotlib available."""
        result = NestedRFECVResult(
            fold_results=[
                RFECVFoldResult(
                    fold_idx=0,
                    optimal_n_features=5,
                    selected_features=["A", "B"],
                    cv_scores=[0.70, 0.75, 0.80, 0.82, 0.85],
                    feature_ranking={"A": 1, "B": 1},
                    val_auroc=0.85,
                ),
                RFECVFoldResult(
                    fold_idx=1,
                    optimal_n_features=4,
                    selected_features=["A", "C"],
                    cv_scores=[0.72, 0.77, 0.81, 0.83],
                    feature_ranking={"A": 1, "C": 1},
                    val_auroc=0.83,
                ),
            ],
            consensus_panel=["A"],
            feature_stability={"A": 1.0, "B": 0.5, "C": 0.5},
            optimal_sizes=[5, 4],
            mean_optimal_size=4.5,
            fold_val_aurocs=[0.85, 0.83],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_nested_rfecv_results(
                result=result, output_dir=tmpdir, model_name="XGBoost", split_seed=2
            )

            # Check that cv_scores_curve exists
            assert "cv_scores_curve" in paths
            assert Path(paths["cv_scores_curve"]).exists()

            # Check if plot was generated (conditional on matplotlib)
            try:
                import matplotlib  # noqa: F401

                assert "selection_curve_plot" in paths
                assert Path(paths["selection_curve_plot"]).exists()
                assert paths["selection_curve_plot"].endswith(".png")
            except ImportError:
                # matplotlib not available, plot should not be generated
                assert "selection_curve_plot" not in paths
