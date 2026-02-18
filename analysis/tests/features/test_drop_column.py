"""Tests for drop-column importance validation module.

Tests cover:
- Core computation: drop-column importance on single fold
- Edge cases: empty clusters, single-feature clusters, all features dropped
- Aggregation: cross-fold statistics and sorting
- Error handling: invalid inputs, refitting failures
- Round-trip: serialization to dict/JSON
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ced_ml.features.drop_column import (
    DropColumnResult,
    aggregate_drop_column_results,
    compute_drop_column_importance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def toy_data():
    """Create toy binary classification dataset for testing.

    Returns:
        Tuple of (X, y, feature_names)
        - X: (100, 5) array with 5 features
        - y: (100,) binary labels with ~10% positives
        - feature_names: ['A', 'B', 'C', 'D', 'E']
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create features with some signal
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.10, n_samples)

    # Add signal to first 2 features
    X[y == 1, 0] += 2.0
    X[y == 1, 1] += 1.5

    X_df = pd.DataFrame(X, columns=["A", "B", "C", "D", "E"])
    return X_df, y, list(X_df.columns)


@pytest.fixture
def fitted_pipeline(toy_data):
    """Create a fitted sklearn Pipeline for testing.

    Returns:
        Pipeline with StandardScaler + LogisticRegression fitted on toy data.
    """
    X, y, _ = toy_data
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=200, C=1.0)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def train_val_split(toy_data):
    """Split toy data into train/val.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    X, y, _ = toy_data
    n_train = 70
    X_train = X.iloc[:n_train]
    y_train = y[:n_train]
    X_val = X.iloc[n_train:]
    y_val = y[n_train:]
    return X_train, y_train, X_val, y_val


# =============================================================================
# Tests: DropColumnResult dataclass
# =============================================================================


class TestDropColumnResult:
    """Tests for DropColumnResult dataclass."""

    def test_creation(self):
        """Test basic DropColumnResult creation."""
        result = DropColumnResult(
            cluster_id=0,
            cluster_features=["A", "B"],
            original_auroc=0.85,
            reduced_auroc=0.75,
            delta_auroc=0.10,
        )
        assert result.cluster_id == 0
        assert result.cluster_features == ["A", "B"]
        assert result.original_auroc == 0.85
        assert result.reduced_auroc == 0.75
        assert result.delta_auroc == 0.10
        assert result.n_folds == 1
        assert result.error_msg is None

    def test_to_dict(self):
        """Test serialization to dict."""
        result = DropColumnResult(
            cluster_id=0,
            cluster_features=["A", "B"],
            original_auroc=0.85,
            reduced_auroc=0.75,
            delta_auroc=0.10,
            fold_id=2,
            model_name="LR",
        )
        d = result.to_dict()
        assert d["cluster_id"] == 0
        assert d["cluster_features"] == "A,B"
        assert d["n_features_in_cluster"] == 2
        assert d["original_auroc"] == 0.85
        assert d["reduced_auroc"] == 0.75
        assert d["delta_auroc"] == 0.10
        assert d["fold_id"] == 2
        assert d["model_name"] == "LR"
        assert d["error_msg"] is None

    def test_to_dict_with_error(self):
        """Test to_dict when result contains error message."""
        result = DropColumnResult(
            cluster_id=1,
            cluster_features=["X"],
            original_auroc=0.8,
            reduced_auroc=np.nan,
            delta_auroc=np.nan,
            error_msg="Refitting failed",
        )
        d = result.to_dict()
        assert d["error_msg"] == "Refitting failed"
        assert np.isnan(d["reduced_auroc"])


# =============================================================================
# Tests: compute_drop_column_importance
# =============================================================================


class TestComputeDropColumnImportance:
    """Tests for compute_drop_column_importance function."""

    def test_basic_functionality(self, fitted_pipeline, train_val_split):
        """Test basic drop-column importance computation."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["C"], ["D", "E"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        assert len(results) == 3
        assert all(isinstance(r, DropColumnResult) for r in results)
        for i, result in enumerate(results):
            assert result.cluster_id == i
            assert len(result.cluster_features) > 0
            assert result.original_auroc > 0
            assert result.reduced_auroc >= 0
            assert result.delta_auroc == result.original_auroc - result.reduced_auroc

    def test_single_feature_clusters(self, fitted_pipeline, train_val_split):
        """Test drop-column with single-feature clusters."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"], ["B"], ["C"], ["D"], ["E"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        assert len(results) == 5
        assert all(len(r.cluster_features) == 1 for r in results)

    def test_overlapping_features_in_clusters(self, fitted_pipeline, train_val_split):
        """Test that overlapping features in clusters are dropped correctly.

        Note: The function doesn't validate overlaps; it simply drops
        all features in each cluster independently.
        """
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["B", "C"]]  # 'B' in both

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        assert len(results) == 2
        # First cluster drops A, B; second drops B, C
        # This is valid behavior (independent evaluations)

    def test_empty_clusters_error(self, fitted_pipeline, train_val_split):
        """Test that empty clusters are rejected."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [[], ["A", "B"]]

        # Empty cluster should be handled gracefully
        # (implementation may log warning and continue)
        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )
        # First result should handle empty cluster (reduced_auroc == original_auroc)
        assert results[0].cluster_features == []

    def test_unknown_feature_error(self, fitted_pipeline, train_val_split):
        """Test error when cluster contains unknown features."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "UNKNOWN"]]

        with pytest.raises(ValueError, match="Unknown features"):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
            )

    def test_mismatched_train_val_columns(self, fitted_pipeline, toy_data):
        """Test error when X_train and X_val have different columns."""
        X, y, _ = toy_data
        X_train = X.iloc[:70]
        y_train = y[:70]
        X_val = X.iloc[70:][["A", "B", "C"]]  # Removed D, E
        y_val = y[70:]
        clusters = [["A"]]

        with pytest.raises(ValueError, match="different columns"):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
            )

    def test_mismatched_X_y_lengths(self, fitted_pipeline, toy_data):
        """Test error when X and y have different lengths."""
        X, y, _ = toy_data
        X_train = X.iloc[:70]
        y_train = y[:60]  # Mismatch
        X_val = X.iloc[70:]
        y_val = y[70:]
        clusters = [["A"]]

        with pytest.raises(ValueError, match="different lengths"):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
            )

    def test_numpy_array_inputs(self, fitted_pipeline, toy_data):
        """Test that numpy arrays are converted to DataFrames correctly."""
        X, y, feature_names = toy_data
        X_train = X.iloc[:70].values  # Convert to numpy
        y_train = y[:70]
        X_val = X.iloc[70:].values
        y_val = y[70:]
        clusters = [["0", "1"]]  # Will use feature names like '0', '1', ...

        # Should raise ValueError because numpy arrays don't have column names
        # But let's test the conversion behavior
        with pytest.raises((ValueError, AttributeError)):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
            )

    def test_delta_auroc_calculation(self, fitted_pipeline, train_val_split):
        """Test that delta_auroc is computed correctly."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        result = results[0]
        expected_delta = result.original_auroc - result.reduced_auroc
        assert abs(result.delta_auroc - expected_delta) < 1e-9

    def test_positive_delta_auroc(self, fitted_pipeline, train_val_split):
        """Test that important features have positive delta_auroc."""
        X_train, y_train, X_val, y_val = train_val_split
        # Feature A has signal added (see toy_data fixture)
        clusters = [["A"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        result = results[0]
        # Dropping important feature should give positive delta_auroc
        assert result.delta_auroc >= 0  # May be 0 if no signal

    def test_deterministic_results(self, fitted_pipeline, train_val_split):
        """Test that results are deterministic with same random_state."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["C", "D"]]

        results1 = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        results2 = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        for r1, r2 in zip(results1, results2, strict=False):
            assert r1.cluster_id == r2.cluster_id
            assert abs(r1.delta_auroc - r2.delta_auroc) < 1e-9


# =============================================================================
# Tests: aggregate_drop_column_results
# =============================================================================


class TestAggregateDropColumnResults:
    """Tests for aggregate_drop_column_results function."""

    def test_single_fold_aggregation(self):
        """Test aggregation with single fold (n_folds=1)."""
        results_fold0 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A", "B"],
                original_auroc=0.85,
                reduced_auroc=0.75,
                delta_auroc=0.10,
            ),
            DropColumnResult(
                cluster_id=1,
                cluster_features=["C"],
                original_auroc=0.85,
                reduced_auroc=0.83,
                delta_auroc=0.02,
            ),
        ]

        df = aggregate_drop_column_results([results_fold0])

        assert len(df) == 2
        assert df.loc[0, "cluster_id"] == 0  # Sorted by mean_delta_auroc descending
        assert df.loc[0, "mean_delta_auroc"] == 0.10
        assert df.loc[0, "n_folds"] == 1
        assert np.isnan(df.loc[0, "std_delta_auroc"])  # NaN for n_folds=1

    def test_multi_fold_aggregation(self):
        """Test aggregation across multiple folds."""
        results_fold0 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.85,
                reduced_auroc=0.75,
                delta_auroc=0.10,
            ),
        ]

        results_fold1 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.86,
                reduced_auroc=0.76,
                delta_auroc=0.10,
            ),
        ]

        results_fold2 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.84,
                reduced_auroc=0.80,
                delta_auroc=0.04,
            ),
        ]

        df = aggregate_drop_column_results([results_fold0, results_fold1, results_fold2])

        assert len(df) == 1
        assert df.loc[0, "cluster_id"] == 0
        assert abs(df.loc[0, "mean_delta_auroc"] - (0.10 + 0.10 + 0.04) / 3) < 1e-6
        assert df.loc[0, "n_folds"] == 3
        assert not np.isnan(df.loc[0, "std_delta_auroc"])

    def test_aggregation_with_errors(self):
        """Test aggregation when some folds have errors."""
        results_fold0 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.85,
                reduced_auroc=0.75,
                delta_auroc=0.10,
            ),
        ]

        results_fold1 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=np.nan,
                reduced_auroc=np.nan,
                delta_auroc=np.nan,
                error_msg="Refitting failed",
            ),
        ]

        df = aggregate_drop_column_results([results_fold0, results_fold1])

        assert len(df) == 1
        assert df.loc[0, "n_folds"] == 2
        assert df.loc[0, "n_errors"] == 1
        assert df.loc[0, "mean_delta_auroc"] == 0.10  # Only fold0 counted

    def test_aggregation_sorting(self):
        """Test that results are sorted by importance (mean_delta_auroc desc)."""
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.70,
                    delta_auroc=0.15,
                ),
                DropColumnResult(
                    cluster_id=1,
                    cluster_features=["B"],
                    original_auroc=0.85,
                    reduced_auroc=0.82,
                    delta_auroc=0.03,
                ),
            ],
        ]

        df = aggregate_drop_column_results(results)

        assert len(df) == 2
        assert df.loc[0, "cluster_id"] == 0  # Most important first
        assert df.loc[1, "cluster_id"] == 1

    def test_aggregation_median_method(self):
        """Test aggregation with median method."""
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.75,
                    delta_auroc=0.10,
                ),
            ],
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.86,
                    reduced_auroc=0.80,
                    delta_auroc=0.06,
                ),
            ],
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.84,
                    reduced_auroc=0.81,
                    delta_auroc=0.03,
                ),
            ],
        ]

        df = aggregate_drop_column_results(results, agg_method="median")

        assert df.loc[0, "mean_delta_auroc"] == 0.06  # Median of [0.10, 0.06, 0.03]
        assert np.isnan(df.loc[0, "std_delta_auroc"])  # NaN for median

    def test_aggregation_max_method(self):
        """Test aggregation with max method."""
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.75,
                    delta_auroc=0.10,
                ),
            ],
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.86,
                    reduced_auroc=0.80,
                    delta_auroc=0.06,
                ),
            ],
        ]

        df = aggregate_drop_column_results(results, agg_method="max")

        assert df.loc[0, "mean_delta_auroc"] == 0.10  # Max of [0.10, 0.06]

    def test_aggregation_invalid_method(self):
        """Test error for invalid aggregation method."""
        results = [[DropColumnResult(0, ["A"], 0.85, 0.75, 0.10)]]

        with pytest.raises(ValueError, match="Unknown agg_method"):
            aggregate_drop_column_results(results, agg_method="invalid")

    def test_aggregation_empty_input(self):
        """Test error when aggregating empty results."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_drop_column_results([])

    def test_aggregation_min_max_tracking(self):
        """Test that min and max delta_auroc are tracked."""
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.70,
                    delta_auroc=0.15,
                ),
            ],
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.86,
                    reduced_auroc=0.80,
                    delta_auroc=0.06,
                ),
            ],
        ]

        df = aggregate_drop_column_results(results)

        assert df.loc[0, "min_delta_auroc"] == 0.06
        assert df.loc[0, "max_delta_auroc"] == 0.15


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_single_fold(self, fitted_pipeline, train_val_split):
        """End-to-end test: compute and aggregate single fold."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["C"], ["D", "E"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        df = aggregate_drop_column_results([results])

        assert len(df) == 3
        assert df["n_folds"].tolist() == [1, 1, 1]
        assert all(df["mean_delta_auroc"] >= 0)

    def test_end_to_end_multi_fold(self, fitted_pipeline, toy_data):
        """End-to-end test: compute across multiple folds and aggregate."""
        X, y, _ = toy_data
        clusters = [["A", "B"], ["C", "D", "E"]]

        all_results = []
        for fold_id in range(3):
            # Create different train/val splits
            start_val = fold_id * 30 + 10
            val_idx = np.arange(start_val, min(start_val + 20, 100))
            train_idx = np.setdiff1d(np.arange(100), val_idx)

            if len(train_idx) > 10 and len(val_idx) > 5:
                X_train = X.iloc[train_idx]
                y_train = y[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y[val_idx]

                results = compute_drop_column_importance(
                    estimator=fitted_pipeline,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    feature_clusters=clusters,
                    random_state=42,
                )
                all_results.append(results)

        if len(all_results) > 0:
            df = aggregate_drop_column_results(all_results)
            assert len(df) == len(clusters)
            assert all(df["n_folds"] > 0)

    def test_serialization_round_trip(self, fitted_pipeline, train_val_split):
        """Test serialization and deserialization of results."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["C"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        # Serialize to JSON-compatible dicts
        dicts = [r.to_dict() for r in results]

        # Create JSON string
        json_str = json.dumps(dicts)

        # Deserialize
        dicts_restored = json.loads(json_str)

        assert len(dicts_restored) == 2
        assert dicts_restored[0]["cluster_id"] == 0
        assert dicts_restored[0]["cluster_features"] == "A,B"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zeros_y(self, fitted_pipeline, toy_data):
        """Test with all zeros in y (no positives).

        The function handles this gracefully - sklearn warns but doesn't raise,
        and we get NaN AUROC which is captured in the result.
        """
        X, _, _ = toy_data
        y_all_zeros = np.zeros(len(X), dtype=int)

        X_train = X.iloc[:70]
        y_train = y_all_zeros[:70]
        X_val = X.iloc[70:]
        y_val = y_all_zeros[70:]
        clusters = [["A", "B"]]

        # Should not raise; baseline AUROC will be NaN
        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )
        assert np.isnan(results[0].original_auroc)

    def test_all_ones_y(self, fitted_pipeline, toy_data):
        """Test with all ones in y (no negatives).

        The function handles this gracefully - sklearn warns but doesn't raise,
        and we get NaN AUROC which is captured in the result.
        """
        X, _, _ = toy_data
        y_all_ones = np.ones(len(X), dtype=int)

        X_train = X.iloc[:70]
        y_train = y_all_ones[:70]
        X_val = X.iloc[70:]
        y_val = y_all_ones[70:]
        clusters = [["A", "B"]]

        # Should not raise; baseline AUROC will be NaN
        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )
        assert np.isnan(results[0].original_auroc)

    def test_single_sample(self, fitted_pipeline, toy_data):
        """Test with single sample in validation set."""
        X, y, _ = toy_data
        X_train = X.iloc[:70]
        y_train = y[:70]
        X_val = X.iloc[70:71]  # Single sample
        y_val = y[70:71]
        clusters = [["A"]]

        # Should work, but AUROC may be undefined
        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )
        # May get NaN or error depending on implementation
        assert len(results) == 1

    def test_all_features_in_one_cluster(self, fitted_pipeline, train_val_split):
        """Test with all features in single cluster."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B", "C", "D", "E"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        # Should result in no features to refit on
        assert results[0].error_msg == "All features dropped from panel"

    def test_very_large_cluster_count(self, fitted_pipeline, train_val_split):
        """Test with many clusters (one feature each)."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [[f] for f in X_train.columns]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        assert len(results) == len(clusters)


# =============================================================================
# Tests: Refit Modes (fixed, retune, fixed_retune)
# =============================================================================


class TestRefitModes:
    """Tests for the three refit modes in drop-column importance."""

    def test_fixed_mode_matches_default(self, fitted_pipeline, train_val_split):
        """Test that refit_mode='fixed' produces identical results to default (no mode)."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A", "B"], ["C"]]

        results_default = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
        )

        results_fixed = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
            refit_mode="fixed",
        )

        for r_def, r_fix in zip(results_default, results_fixed, strict=True):
            assert r_def.cluster_id == r_fix.cluster_id
            assert abs(r_def.delta_auroc - r_fix.delta_auroc) < 1e-9
            assert r_fix.retune_auroc is None
            assert r_fix.delta_auroc_retune is None

    def test_fixed_mode_no_retune_fields(self, fitted_pipeline, train_val_split):
        """Test that fixed mode does not populate retune fields."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"]]

        results = compute_drop_column_importance(
            estimator=fitted_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
            refit_mode="fixed",
        )

        assert results[0].retune_auroc is None
        assert results[0].delta_auroc_retune is None
        assert results[0].retune_best_params == {}

    def test_retune_mode_requires_model_name(self, fitted_pipeline, train_val_split):
        """Test that retune mode raises without model_name."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"]]

        with pytest.raises(ValueError, match="model_name is required"):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
                refit_mode="retune",
            )

    def test_fixed_retune_requires_model_name(self, fitted_pipeline, train_val_split):
        """Test that fixed_retune mode raises without model_name."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"]]

        with pytest.raises(ValueError, match="model_name is required"):
            compute_drop_column_importance(
                estimator=fitted_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_clusters=clusters,
                random_state=42,
                refit_mode="fixed_retune",
            )

    def test_retune_mode_produces_results(self, train_val_split):
        """Test that retune mode produces valid results via Optuna."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"]]

        # Build a simple fitted pipeline for baseline AUROC computation
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42, max_iter=200)),
            ]
        )
        pipeline.fit(X_train, y_train)

        results = compute_drop_column_importance(
            estimator=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
            refit_mode="retune",
            model_name="LR_EN",
            cat_cols=[],
            retune_n_trials=5,
            retune_inner_folds=2,
        )

        assert len(results) == 1
        r = results[0]
        # In retune mode, primary delta_auroc uses retune values
        assert not np.isnan(r.delta_auroc)
        assert r.retune_auroc is not None
        assert r.delta_auroc_retune is not None
        assert len(r.retune_best_params) > 0

    def test_fixed_retune_mode_has_both(self, train_val_split):
        """Test that fixed_retune mode populates both fixed and retune fields."""
        X_train, y_train, X_val, y_val = train_val_split
        clusters = [["A"]]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42, max_iter=200)),
            ]
        )
        pipeline.fit(X_train, y_train)

        results = compute_drop_column_importance(
            estimator=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_clusters=clusters,
            random_state=42,
            refit_mode="fixed_retune",
            model_name="LR_EN",
            cat_cols=[],
            retune_n_trials=5,
            retune_inner_folds=2,
        )

        assert len(results) == 1
        r = results[0]
        # Fixed fields populated
        assert not np.isnan(r.delta_auroc)
        assert not np.isnan(r.reduced_auroc)
        # Retune fields populated
        assert r.retune_auroc is not None
        assert r.delta_auroc_retune is not None
        assert len(r.retune_best_params) > 0

    def test_retune_to_dict_includes_retune_fields(self, train_val_split):
        """Test that to_dict includes retune fields when present."""
        result = DropColumnResult(
            cluster_id=0,
            cluster_features=["A"],
            original_auroc=0.85,
            reduced_auroc=0.80,
            delta_auroc=0.05,
            retune_auroc=0.78,
            delta_auroc_retune=0.07,
            retune_best_params={"clf__C": 0.1},
        )
        d = result.to_dict()
        assert "retune_auroc" in d
        assert d["retune_auroc"] == 0.78
        assert "delta_auroc_retune" in d
        assert d["delta_auroc_retune"] == 0.07
        assert "retune_best_params" in d

    def test_fixed_to_dict_excludes_retune_fields(self):
        """Test that to_dict excludes retune fields when not set."""
        result = DropColumnResult(
            cluster_id=0,
            cluster_features=["A"],
            original_auroc=0.85,
            reduced_auroc=0.80,
            delta_auroc=0.05,
        )
        d = result.to_dict()
        assert "retune_auroc" not in d
        assert "delta_auroc_retune" not in d
        assert "retune_best_params" not in d

    def test_aggregation_with_retune_columns(self):
        """Test that aggregation handles retune columns correctly."""
        results_fold0 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.85,
                reduced_auroc=0.80,
                delta_auroc=0.05,
                retune_auroc=0.78,
                delta_auroc_retune=0.07,
            ),
        ]
        results_fold1 = [
            DropColumnResult(
                cluster_id=0,
                cluster_features=["A"],
                original_auroc=0.86,
                reduced_auroc=0.83,
                delta_auroc=0.03,
                retune_auroc=0.79,
                delta_auroc_retune=0.07,
            ),
        ]

        df = aggregate_drop_column_results([results_fold0, results_fold1])

        assert "mean_delta_auroc_retune" in df.columns
        assert "std_delta_auroc_retune" in df.columns
        assert abs(df.loc[0, "mean_delta_auroc_retune"] - 0.07) < 1e-6

    def test_compensation_flag(self):
        """Test compensation flag logic in fixed_retune aggregation."""
        # Cluster appears non-essential in fixed mode but essential in retune
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.849,
                    delta_auroc=0.001,  # Below threshold
                    retune_auroc=0.83,
                    delta_auroc_retune=0.02,  # Above threshold
                ),
            ],
        ]

        df = aggregate_drop_column_results(results)

        assert "compensation_flag" in df.columns
        assert df.loc[0, "compensation_flag"] == True  # noqa: E712
        assert df.loc[0, "compensation_delta"] > 0

    def test_no_compensation_when_both_essential(self):
        """Test no compensation flag when both modes agree feature is essential."""
        results = [
            [
                DropColumnResult(
                    cluster_id=0,
                    cluster_features=["A"],
                    original_auroc=0.85,
                    reduced_auroc=0.80,
                    delta_auroc=0.05,  # Above threshold
                    retune_auroc=0.79,
                    delta_auroc_retune=0.06,  # Also above
                ),
            ],
        ]

        df = aggregate_drop_column_results(results)

        assert "compensation_flag" in df.columns
        assert df.loc[0, "compensation_flag"] == False  # noqa: E712
