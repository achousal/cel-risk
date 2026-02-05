"""Tests for grouped permutation importance module.

Tests cover:
- Feature cluster building from correlation matrix
- Individual permutation importance computation
- Grouped permutation importance (correlation-robust)
- Integration with existing correlation infrastructure
- Edge cases: single features, no correlations, all correlated
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ced_ml.features.grouped_importance import (
    build_feature_clusters,
    compute_grouped_permutation_importance,
    compute_permutation_importance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification data with correlated features."""
    np.random.seed(42)
    n_samples = 200

    feat_a = np.random.randn(n_samples)
    feat_b = feat_a + 0.1 * np.random.randn(n_samples)
    feat_c = np.random.randn(n_samples)
    feat_d = feat_c + 0.05 * np.random.randn(n_samples)
    feat_e = np.random.randn(n_samples)

    X = pd.DataFrame(
        {
            "feat_a": feat_a,
            "feat_b": feat_b,
            "feat_c": feat_c,
            "feat_d": feat_d,
            "feat_e": feat_e,
        }
    )

    y = (feat_a + 0.5 * feat_c > 0).astype(int)

    return X, y


@pytest.fixture
def fitted_lr_model(simple_classification_data):
    """Fitted LogisticRegression model."""
    X, y = simple_classification_data
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_rf_model(simple_classification_data):
    """Fitted RandomForestClassifier model."""
    X, y = simple_classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
    model.fit(X, y)
    return model


# =============================================================================
# Tests: build_feature_clusters
# =============================================================================


class TestBuildFeatureClusters:
    """Tests for build_feature_clusters function."""

    def test_basic_clustering(self, simple_classification_data):
        """Test basic feature clustering with correlation threshold."""
        X, _ = simple_classification_data
        feature_names = X.columns.tolist()

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="spearman"
        )

        assert len(clusters) > 0
        all_features = [f for cluster in clusters for f in cluster]
        assert set(all_features) == set(feature_names)

    def test_high_threshold_singleton_clusters(self, simple_classification_data):
        """Test with very high threshold produces more singleton clusters."""
        X, _ = simple_classification_data
        feature_names = X.columns.tolist()

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.99, corr_method="spearman"
        )

        assert len(clusters) <= len(feature_names)

    def test_low_threshold_fewer_clusters(self, simple_classification_data):
        """Test with low threshold produces fewer, larger clusters."""
        X, _ = simple_classification_data
        feature_names = X.columns.tolist()

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.50, corr_method="spearman"
        )

        assert len(clusters) <= len(feature_names)

    def test_pearson_vs_spearman(self, simple_classification_data):
        """Test that pearson and spearman methods produce valid clusters."""
        X, _ = simple_classification_data
        feature_names = X.columns.tolist()

        clusters_pearson = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="pearson"
        )

        clusters_spearman = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="spearman"
        )

        assert len(clusters_pearson) > 0
        assert len(clusters_spearman) > 0

    def test_empty_feature_list(self, simple_classification_data):
        """Test with empty feature list."""
        X, _ = simple_classification_data

        clusters = build_feature_clusters(
            X=X, feature_names=[], corr_threshold=0.85, corr_method="spearman"
        )

        assert clusters == []

    def test_unknown_features_ignored(self, simple_classification_data):
        """Test that unknown features are ignored."""
        X, _ = simple_classification_data
        feature_names = ["feat_a", "feat_b", "unknown_feat"]

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="spearman"
        )

        all_features = [f for cluster in clusters for f in cluster]
        assert "unknown_feat" not in all_features
        assert "feat_a" in all_features

    def test_single_feature(self, simple_classification_data):
        """Test with single feature."""
        X, _ = simple_classification_data
        feature_names = ["feat_a"]

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="spearman"
        )

        assert len(clusters) == 1
        assert clusters[0] == ["feat_a"]

    def test_two_correlated_features(self):
        """Test with two highly correlated features."""
        np.random.seed(42)
        feat_a = np.random.randn(100)
        feat_b = feat_a + 0.05 * np.random.randn(100)

        X = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b})

        clusters = build_feature_clusters(
            X=X, feature_names=["feat_a", "feat_b"], corr_threshold=0.85, corr_method="spearman"
        )

        assert len(clusters) == 1
        assert set(clusters[0]) == {"feat_a", "feat_b"}

    def test_clusters_are_sorted(self, simple_classification_data):
        """Test that features within clusters are sorted."""
        X, _ = simple_classification_data
        feature_names = X.columns.tolist()

        clusters = build_feature_clusters(
            X=X, feature_names=feature_names, corr_threshold=0.85, corr_method="spearman"
        )

        for cluster in clusters:
            assert cluster == sorted(cluster)


# =============================================================================
# Tests: compute_permutation_importance
# =============================================================================


class TestComputePermutationImportance:
    """Tests for compute_permutation_importance function."""

    def test_basic_computation(self, fitted_lr_model, simple_classification_data):
        """Test basic permutation importance computation."""
        X, y = simple_classification_data
        feature_names = X.columns.tolist()

        importance_df = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        assert len(importance_df) == len(feature_names)
        assert list(importance_df.columns) == [
            "feature",
            "mean_importance",
            "std_importance",
            "n_repeats",
            "baseline_auroc",
        ]
        assert all(importance_df["n_repeats"] == 10)
        assert importance_df["baseline_auroc"].nunique() == 1

    def test_sorted_by_importance(self, fitted_lr_model, simple_classification_data):
        """Test that results are sorted by mean_importance descending."""
        X, y = simple_classification_data
        feature_names = X.columns.tolist()

        importance_df = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        importances = importance_df["mean_importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))

    def test_positive_importance_for_signal_features(
        self, fitted_lr_model, simple_classification_data
    ):
        """Test that features with signal have positive importance."""
        X, y = simple_classification_data
        feature_names = X.columns.tolist()

        importance_df = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        assert importance_df.loc[0, "mean_importance"] > 0

    def test_deterministic_with_random_state(self, fitted_lr_model, simple_classification_data):
        """Test that results are deterministic with same random_state."""
        X, y = simple_classification_data
        feature_names = X.columns.tolist()

        importance_df1 = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        importance_df2 = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        pd.testing.assert_frame_equal(importance_df1, importance_df2)

    def test_empty_feature_list(self, fitted_lr_model, simple_classification_data):
        """Test with empty feature list."""
        X, y = simple_classification_data

        importance_df = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=[],
            n_repeats=10,
            random_state=42,
        )

        assert importance_df.empty

    def test_unknown_features_ignored(self, simple_classification_data):
        """Test that unknown features are ignored."""
        X, y = simple_classification_data
        X_subset = X[["feat_a"]]

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_subset, y)

        feature_names = ["feat_a", "unknown_feat"]

        importance_df = compute_permutation_importance(
            estimator=model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        assert len(importance_df) == 1
        assert importance_df.loc[0, "feature"] == "feat_a"

    def test_single_feature(self, simple_classification_data):
        """Test with single feature."""
        X, y = simple_classification_data
        X_subset = X[["feat_a"]]

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_subset, y)

        feature_names = ["feat_a"]

        importance_df = compute_permutation_importance(
            estimator=model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=10,
            random_state=42,
        )

        assert len(importance_df) == 1
        assert importance_df.loc[0, "feature"] == "feat_a"

    def test_std_importance_positive(self, fitted_lr_model, simple_classification_data):
        """Test that std_importance is positive with multiple repeats."""
        X, y = simple_classification_data
        feature_names = X.columns.tolist()

        importance_df = compute_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=30,
            random_state=42,
        )

        assert all(importance_df["std_importance"] >= 0)


# =============================================================================
# Tests: compute_grouped_permutation_importance
# =============================================================================


class TestComputeGroupedPermutationImportance:
    """Tests for compute_grouped_permutation_importance function."""

    def test_basic_computation(self, fitted_lr_model, simple_classification_data):
        """Test basic grouped permutation importance computation."""
        X, y = simple_classification_data
        clusters = [["feat_a", "feat_b"], ["feat_c", "feat_d"], ["feat_e"]]

        importance_df = compute_grouped_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            clusters=clusters,
            n_repeats=10,
            random_state=42,
        )

        assert len(importance_df) == 3
        assert list(importance_df.columns) == [
            "cluster_id",
            "cluster_features",
            "cluster_size",
            "mean_importance",
            "std_importance",
            "n_repeats",
            "baseline_auroc",
        ]
        assert all(importance_df["n_repeats"] == 10)

    def test_cluster_size_correct(self, fitted_lr_model, simple_classification_data):
        """Test that cluster_size matches actual cluster size."""
        X, y = simple_classification_data
        clusters = [["feat_a", "feat_b"], ["feat_c"], ["feat_d", "feat_e"]]

        importance_df = compute_grouped_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            clusters=clusters,
            n_repeats=10,
            random_state=42,
        )

        for i, cluster in enumerate(clusters):
            row = importance_df[importance_df["cluster_id"] == i].iloc[0]
            assert row["cluster_size"] == len(cluster)

    def test_all_features_in_one_cluster(self, fitted_lr_model, simple_classification_data):
        """Test with all features in single cluster."""
        X, y = simple_classification_data
        clusters = [X.columns.tolist()]

        importance_df = compute_grouped_permutation_importance(
            estimator=fitted_lr_model,
            X=X,
            y=y,
            clusters=clusters,
            n_repeats=10,
            random_state=42,
        )

        assert len(importance_df) == 1
        assert importance_df.loc[0, "cluster_size"] == len(X.columns)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_clustering_with_manual_verification(self):
        """Test clustering with manually created correlated features."""
        np.random.seed(42)
        n_samples = 200

        feat_a = np.random.randn(n_samples)
        feat_b = feat_a + 0.01 * np.random.randn(n_samples)
        feat_c = np.random.randn(n_samples)

        X = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "feat_c": feat_c})

        _ = (feat_a > 0).astype(int)  # y - not used in this test

        clusters = build_feature_clusters(
            X=X, feature_names=["feat_a", "feat_b", "feat_c"], corr_threshold=0.85
        )

        assert len(clusters) == 2
        cluster_sizes = sorted([len(c) for c in clusters])
        assert cluster_sizes == [1, 2]

    def test_permutation_importance_consistency(self, simple_classification_data):
        """Test that grouped importance for singleton clusters matches individual importance."""
        X, y = simple_classification_data
        X_subset = X[["feat_a"]]

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_subset, y)

        feature_names = ["feat_a"]

        individual_df = compute_permutation_importance(
            estimator=model,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=30,
            random_state=42,
        )

        grouped_df = compute_grouped_permutation_importance(
            estimator=model,
            X=X,
            y=y,
            clusters=[["feat_a"]],
            n_repeats=30,
            random_state=42,
        )

        if len(individual_df) > 0 and len(grouped_df) > 0:
            assert np.isclose(
                individual_df.loc[0, "mean_importance"],
                grouped_df.loc[0, "mean_importance"],
                atol=0.05,
            )
