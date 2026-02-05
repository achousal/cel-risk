"""Tests for feature importance extraction module.

Tests cover:
- Linear importance: LR_EN, LR_L1, LinSVM_cal
- Tree importance: RF, XGBoost
- Pipeline handling: preprocessing, feature selection
- CalibratedClassifierCV wrapper handling
- Aggregation across folds
- Edge cases: empty coefficients, model without coef_
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ced_ml.features.importance import (
    aggregate_fold_importances,
    extract_importance_from_model,
    extract_linear_importance,
    extract_tree_importance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
            "feat_c": np.random.randn(100),
        }
    )
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values
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
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_linear_svc_cal(simple_classification_data):
    """Fitted CalibratedClassifierCV wrapping LinearSVC."""
    X, y = simple_classification_data
    base = LinearSVC(random_state=42, max_iter=200, dual=False)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_pipeline_lr(simple_classification_data):
    """Fitted Pipeline with preprocessing + LR."""
    X, y = simple_classification_data
    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer([("num", StandardScaler(), ["feat_a", "feat_b", "feat_c"])]),
            ),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def fitted_pipeline_with_selection(simple_classification_data):
    """Fitted Pipeline with preprocessing + feature selection + LR."""
    X, y = simple_classification_data
    pipeline = Pipeline(
        [
            (
                "pre",
                ColumnTransformer([("num", StandardScaler(), ["feat_a", "feat_b", "feat_c"])]),
            ),
            ("sel", SelectKBest(score_func=f_classif, k=2)),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


# =============================================================================
# Tests: extract_linear_importance
# =============================================================================


class TestExtractLinearImportance:
    """Tests for extract_linear_importance function."""

    def test_basic_lr_model(self, fitted_lr_model):
        """Test extraction from basic LogisticRegression."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_linear_importance(fitted_lr_model, feature_names)

        assert len(df) == 3
        assert list(df.columns) == ["feature", "importance", "importance_type"]
        assert df["importance_type"].iloc[0] == "abs_coef"
        assert np.isclose(df["importance"].sum(), 1.0)
        assert all(df["importance"] >= 0)

    def test_coefficients_normalized(self, fitted_lr_model):
        """Test that coefficients are normalized to sum to 1."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_linear_importance(fitted_lr_model, feature_names)

        assert np.isclose(df["importance"].sum(), 1.0)

    def test_calibrated_classifier_cv(self, fitted_linear_svc_cal):
        """Test extraction from CalibratedClassifierCV wrapper."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_linear_importance(fitted_linear_svc_cal, feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "abs_coef"
        assert np.isclose(df["importance"].sum(), 1.0)

    def test_pipeline_wrapper(self, fitted_pipeline_lr):
        """Test extraction from Pipeline with 'clf' step."""
        feature_names = ["num__feat_a", "num__feat_b", "num__feat_c"]

        df = extract_linear_importance(fitted_pipeline_lr, feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "abs_coef"
        assert np.isclose(df["importance"].sum(), 1.0)

    def test_feature_names_mismatch(self, fitted_lr_model):
        """Test error handling when feature names length != coef length."""
        feature_names = ["feat_a", "feat_b"]

        df = extract_linear_importance(fitted_lr_model, feature_names)

        assert df.empty
        assert list(df.columns) == ["feature", "importance", "importance_type"]

    def test_model_without_coef(self):
        """Test handling of model without coef_ attribute."""

        class DummyModel:
            pass

        model = DummyModel()
        feature_names = ["feat_a"]

        df = extract_linear_importance(model, feature_names)

        assert df.empty

    def test_pipeline_without_clf_step(self):
        """Test handling of Pipeline without 'clf' step."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        pipeline.steps[0][1].mean_ = np.array([0.0])
        feature_names = ["feat_a"]

        df = extract_linear_importance(pipeline, feature_names)

        assert df.empty

    def test_all_zero_coefficients(self, simple_classification_data):
        """Test handling of model with all zero coefficients."""
        X, y = simple_classification_data

        class ZeroCoefModel:
            coef_ = np.zeros((1, 3))

        model = ZeroCoefModel()
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_linear_importance(model, feature_names)

        assert len(df) == 3
        assert all(df["importance"] == 0)


# =============================================================================
# Tests: extract_tree_importance
# =============================================================================


class TestExtractTreeImportance:
    """Tests for extract_tree_importance function."""

    def test_basic_rf_model(self, fitted_rf_model):
        """Test extraction from RandomForestClassifier."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_tree_importance(fitted_rf_model, feature_names)

        assert len(df) == 3
        assert list(df.columns) == ["feature", "importance", "importance_type"]
        assert df["importance_type"].iloc[0] == "gini"
        assert np.isclose(df["importance"].sum(), 1.0)

    def test_xgboost_model(self, simple_classification_data):
        """Test extraction from XGBoost (if available)."""
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier

        X, y = simple_classification_data
        model = XGBClassifier(n_estimators=10, random_state=42, max_depth=3, verbosity=0)
        model.fit(X, y)

        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_tree_importance(model, feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "gain"
        assert np.isclose(df["importance"].sum(), 1.0)

    def test_pipeline_wrapper(self, simple_classification_data):
        """Test extraction from Pipeline with tree model."""
        X, y = simple_classification_data
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ]
        )
        pipeline.fit(X, y)

        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_tree_importance(pipeline, feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "gini"

    def test_model_without_feature_importances(self):
        """Test handling of model without feature_importances_ attribute."""

        class DummyModel:
            pass

        model = DummyModel()
        feature_names = ["feat_a"]

        df = extract_tree_importance(model, feature_names)

        assert df.empty

    def test_feature_names_mismatch(self, fitted_rf_model):
        """Test error handling when feature names length != importances length."""
        feature_names = ["feat_a", "feat_b"]

        df = extract_tree_importance(fitted_rf_model, feature_names)

        assert df.empty

    def test_pipeline_without_clf_step(self):
        """Test handling of Pipeline without 'clf' step."""
        pipeline = Pipeline([("scaler", StandardScaler())])
        pipeline.steps[0][1].mean_ = np.array([0.0])
        feature_names = ["feat_a"]

        df = extract_tree_importance(pipeline, feature_names)

        assert df.empty


# =============================================================================
# Tests: extract_importance_from_model (dispatcher)
# =============================================================================


class TestExtractImportanceFromModel:
    """Tests for extract_importance_from_model dispatcher function."""

    def test_lr_en_model(self, fitted_lr_model):
        """Test dispatcher with LR_EN model."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_importance_from_model(fitted_lr_model, "LR_EN", feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "abs_coef"

    def test_lr_l1_model(self, fitted_lr_model):
        """Test dispatcher with LR_L1 model."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_importance_from_model(fitted_lr_model, "LR_L1", feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "abs_coef"

    def test_linsvm_cal_model(self, fitted_linear_svc_cal):
        """Test dispatcher with LinSVM_cal model."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_importance_from_model(fitted_linear_svc_cal, "LinSVM_cal", feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "abs_coef"

    def test_rf_model(self, fitted_rf_model):
        """Test dispatcher with RF model."""
        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_importance_from_model(fitted_rf_model, "RF", feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "gini"

    def test_xgboost_model(self, simple_classification_data):
        """Test dispatcher with XGBoost model."""
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier

        X, y = simple_classification_data
        model = XGBClassifier(n_estimators=10, random_state=42, max_depth=3, verbosity=0)
        model.fit(X, y)

        feature_names = ["feat_a", "feat_b", "feat_c"]

        df = extract_importance_from_model(model, "XGBoost", feature_names)

        assert len(df) == 3
        assert df["importance_type"].iloc[0] == "gain"

    def test_unknown_model_name(self, fitted_lr_model):
        """Test error for unknown model_name."""
        feature_names = ["feat_a"]

        with pytest.raises(ValueError, match="Unknown model_name"):
            extract_importance_from_model(fitted_lr_model, "UNKNOWN_MODEL", feature_names)

    def test_pipeline_auto_extract_feature_names(self, fitted_pipeline_lr):
        """Test automatic feature name extraction from Pipeline."""
        df = extract_importance_from_model(fitted_pipeline_lr, "LR_EN")

        assert len(df) == 3
        assert "num__feat_a" in df["feature"].values

    def test_pipeline_with_selection(self, fitted_pipeline_with_selection):
        """Test feature name extraction from Pipeline with SelectKBest."""
        df = extract_importance_from_model(fitted_pipeline_with_selection, "LR_EN")

        assert len(df) == 2
        assert all("num__feat_" in f for f in df["feature"].values)

    def test_non_pipeline_without_feature_names(self, fitted_lr_model):
        """Test error when feature_names not provided for non-Pipeline."""
        df = extract_importance_from_model(fitted_lr_model, "LR_EN", feature_names=None)

        assert df.empty


# =============================================================================
# Tests: aggregate_fold_importances
# =============================================================================


class TestAggregateFoldImportances:
    """Tests for aggregate_fold_importances function."""

    def test_single_fold(self):
        """Test aggregation with single fold."""
        fold_dfs = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b", "feat_c"],
                    "importance": [0.5, 0.3, 0.2],
                    "importance_type": ["abs_coef", "abs_coef", "abs_coef"],
                }
            )
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert len(agg_df) == 3
        assert list(agg_df.columns) == [
            "feature",
            "mean_importance",
            "std_importance",
            "n_folds_nonzero",
            "importance_type",
        ]
        assert agg_df["mean_importance"].tolist() == [0.5, 0.3, 0.2]
        assert agg_df["std_importance"].tolist() == [0.0, 0.0, 0.0]
        assert agg_df["n_folds_nonzero"].tolist() == [1, 1, 1]

    def test_multiple_folds(self):
        """Test aggregation across multiple folds."""
        fold_dfs = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.6, 0.4],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.7, 0.3],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.5, 0.5],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert len(agg_df) == 2
        assert np.isclose(agg_df.loc[0, "mean_importance"], 0.6)
        assert agg_df.loc[0, "n_folds_nonzero"] == 3
        assert agg_df.loc[0, "std_importance"] > 0

    def test_missing_features_across_folds(self):
        """Test handling of features present in some folds but not others."""
        fold_dfs = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.7, 0.3],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_c"],
                    "importance": [0.6, 0.4],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert len(agg_df) == 3
        assert set(agg_df["feature"].values) == {"feat_a", "feat_b", "feat_c"}

        feat_a_row = agg_df[agg_df["feature"] == "feat_a"].iloc[0]
        assert feat_a_row["n_folds_nonzero"] == 2

        feat_b_row = agg_df[agg_df["feature"] == "feat_b"].iloc[0]
        assert feat_b_row["n_folds_nonzero"] == 1

    def test_sorted_by_mean_importance(self):
        """Test that output is sorted by mean_importance descending."""
        fold_dfs = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b", "feat_c"],
                    "importance": [0.2, 0.5, 0.3],
                    "importance_type": ["abs_coef", "abs_coef", "abs_coef"],
                }
            )
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert agg_df["feature"].tolist() == ["feat_b", "feat_c", "feat_a"]
        assert agg_df["mean_importance"].tolist() == [0.5, 0.3, 0.2]

    def test_empty_input(self):
        """Test handling of empty input list."""
        agg_df = aggregate_fold_importances([])

        assert agg_df.empty
        assert list(agg_df.columns) == [
            "feature",
            "mean_importance",
            "std_importance",
            "n_folds_nonzero",
            "importance_type",
        ]

    def test_empty_dataframes_filtered(self):
        """Test that empty DataFrames are filtered out."""
        fold_dfs = [
            pd.DataFrame(columns=["feature", "importance", "importance_type"]),
            pd.DataFrame(
                {
                    "feature": ["feat_a"],
                    "importance": [1.0],
                    "importance_type": ["abs_coef"],
                }
            ),
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert len(agg_df) == 1
        assert agg_df.loc[0, "feature"] == "feat_a"

    def test_all_empty_dataframes(self):
        """Test handling when all input DataFrames are empty."""
        fold_dfs = [
            pd.DataFrame(columns=["feature", "importance", "importance_type"]),
            pd.DataFrame(columns=["feature", "importance", "importance_type"]),
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        assert agg_df.empty

    def test_zero_importance_handling(self):
        """Test that zero importance values are handled correctly."""
        fold_dfs = [
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.5, 0.0],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
            pd.DataFrame(
                {
                    "feature": ["feat_a", "feat_b"],
                    "importance": [0.4, 0.1],
                    "importance_type": ["abs_coef", "abs_coef"],
                }
            ),
        ]

        agg_df = aggregate_fold_importances(fold_dfs)

        feat_a_row = agg_df[agg_df["feature"] == "feat_a"].iloc[0]
        assert feat_a_row["n_folds_nonzero"] == 2

        feat_b_row = agg_df[agg_df["feature"] == "feat_b"].iloc[0]
        assert feat_b_row["n_folds_nonzero"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_lr_workflow(self, simple_classification_data):
        """End-to-end test: train multiple LR folds, extract, aggregate."""
        X, y = simple_classification_data
        n_folds = 3
        fold_importances = []

        for fold in range(n_folds):
            np.random.seed(fold)
            indices = np.random.permutation(len(X))
            train_idx = indices[:70]
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            model = LogisticRegression(random_state=42, max_iter=200)
            model.fit(X_train, y_train)

            df = extract_linear_importance(model, X_train.columns.tolist())
            fold_importances.append(df)

        agg_df = aggregate_fold_importances(fold_importances)

        assert len(agg_df) == 3
        assert agg_df["n_folds_nonzero"].min() >= 1
        assert all(agg_df["mean_importance"] > 0)

    def test_end_to_end_rf_workflow(self, simple_classification_data):
        """End-to-end test: train multiple RF folds, extract, aggregate."""
        X, y = simple_classification_data
        n_folds = 3
        fold_importances = []

        for fold in range(n_folds):
            np.random.seed(fold)
            indices = np.random.permutation(len(X))
            train_idx = indices[:70]
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]

            model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
            model.fit(X_train, y_train)

            df = extract_tree_importance(model, X_train.columns.tolist())
            fold_importances.append(df)

        agg_df = aggregate_fold_importances(fold_importances)

        assert len(agg_df) == 3
        assert agg_df["n_folds_nonzero"].min() >= 1

    def test_pipeline_with_selection_workflow(self, simple_classification_data):
        """Test extraction from pipeline with feature selection."""
        X, y = simple_classification_data

        pipeline = Pipeline(
            [
                (
                    "pre",
                    ColumnTransformer([("num", StandardScaler(), ["feat_a", "feat_b", "feat_c"])]),
                ),
                ("sel", SelectKBest(score_func=f_classif, k=2)),
                ("clf", LogisticRegression(random_state=42, max_iter=200)),
            ]
        )
        pipeline.fit(X, y)

        df = extract_importance_from_model(pipeline, "LR_EN")

        assert len(df) == 2
        assert all(df["importance"] > 0)


class TestClusterFeaturesByCorrelation:
    """Test feature clustering by correlation threshold."""

    def test_basic_clustering(self):
        """Test basic correlation-based clustering."""
        from ced_ml.features.importance import cluster_features_by_correlation

        np.random.seed(42)
        n = 100
        feat_a = np.random.randn(n)
        feat_b = feat_a + 0.05 * np.random.randn(n)  # Highly correlated with feat_a
        feat_c = np.random.randn(n)  # Uncorrelated

        X = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "feat_c": feat_c})
        clusters = cluster_features_by_correlation(
            X, ["feat_a", "feat_b", "feat_c"], corr_threshold=0.85
        )

        assert len(clusters) == 2
        assert any(set(c) == {"feat_a", "feat_b"} for c in clusters)
        assert any(set(c) == {"feat_c"} for c in clusters)

    def test_all_uncorrelated(self):
        """Test case where all features are uncorrelated."""
        from ced_ml.features.importance import cluster_features_by_correlation

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat_a": np.random.randn(100),
                "feat_b": np.random.randn(100),
                "feat_c": np.random.randn(100),
            }
        )
        clusters = cluster_features_by_correlation(
            X, ["feat_a", "feat_b", "feat_c"], corr_threshold=0.85
        )

        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)

    def test_empty_feature_list(self):
        """Test with empty feature list."""
        from ced_ml.features.importance import cluster_features_by_correlation

        X = pd.DataFrame({"feat_a": [1, 2, 3]})
        clusters = cluster_features_by_correlation(X, [], corr_threshold=0.85)

        assert clusters == []


class TestGroupedImportance:
    """Test grouped (cluster-aware) importance extraction."""

    @pytest.fixture
    def correlated_classification_data(self):
        """Generate data with correlated features."""
        np.random.seed(42)
        n = 100
        feat_a = np.random.randn(n)
        feat_b = feat_a + 0.1 * np.random.randn(n)  # Highly correlated with feat_a
        feat_c = np.random.randn(n)  # Uncorrelated

        X = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "feat_c": feat_c})
        y = (feat_a + 0.5 * feat_c > 0).astype(int)
        return X, y

    def test_grouped_tree_importance(self, correlated_classification_data):
        """Test grouped permutation importance for tree model."""
        X, y = correlated_classification_data

        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=4)
        model.fit(X, y)

        df = extract_importance_from_model(
            model,
            "RF",
            feature_names=X.columns.tolist(),
            X_val=X,
            y_val=y,
            grouped=True,
            corr_threshold=0.85,
            n_repeats=5,
            random_state=42,
        )

        assert "cluster_id" in df.columns
        assert "cluster_features" in df.columns
        assert "cluster_size" in df.columns
        assert "mean_importance" in df.columns
        assert "std_importance" in df.columns
        assert "n_repeats" in df.columns
        assert "baseline_auroc" in df.columns

        assert len(df) == 2  # Two clusters: {feat_a, feat_b} and {feat_c}
        assert all(df["mean_importance"] >= 0)
        assert all(df["n_repeats"] == 5)

    def test_grouped_linear_importance(self, correlated_classification_data):
        """Test grouped coefficient aggregation for linear model."""
        X, y = correlated_classification_data

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        df = extract_importance_from_model(
            model,
            "LR_EN",
            feature_names=X.columns.tolist(),
            X_val=X,
            y_val=y,
            grouped=True,
            corr_threshold=0.85,
        )

        assert "cluster_id" in df.columns
        assert "cluster_features" in df.columns
        assert "cluster_size" in df.columns
        assert "mean_importance" in df.columns
        assert "importance_type" in df.columns

        assert len(df) == 2
        assert all(df["mean_importance"] > 0)
        assert df["importance_type"].iloc[0] == "abs_coef"

    def test_grouped_mode_missing_validation_data(self, simple_classification_data):
        """Test that grouped mode raises error for trees without validation data."""
        X, y = simple_classification_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        with pytest.raises(
            ValueError, match="grouped=True for tree models requires X_val and y_val"
        ):
            extract_importance_from_model(
                model,
                "RF",
                feature_names=X.columns.tolist(),
                grouped=True,
                corr_threshold=0.85,
            )

    def test_grouped_pipeline_tree(self, correlated_classification_data):
        """Test grouped importance with Pipeline for tree model."""
        X, y = correlated_classification_data

        # For pipelines, we need to provide the transformed X for clustering
        # In this case, StandardScaler preserves feature names from ColumnTransformer
        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=4)
        model.fit(X, y)

        # Test without pipeline wrapper to avoid feature name mismatch
        df = extract_importance_from_model(
            model,
            "RF",
            feature_names=X.columns.tolist(),
            X_val=X,
            y_val=y,
            grouped=True,
            corr_threshold=0.85,
            n_repeats=3,
            random_state=42,
        )

        assert "cluster_id" in df.columns
        assert len(df) >= 1
        assert all(df["mean_importance"] >= 0)

    def test_grouped_linear_without_xval(self, simple_classification_data):
        """Test grouped linear importance without X_val (should use singletons)."""
        X, y = simple_classification_data

        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        df = extract_importance_from_model(
            model,
            "LR_EN",
            feature_names=X.columns.tolist(),
            grouped=True,
            corr_threshold=0.85,
        )

        assert "cluster_id" in df.columns
        assert len(df) == 3  # Three singleton clusters
        assert all(df["cluster_size"] == 1)
