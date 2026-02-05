"""Tests for permutation testing module.

Tests cover:
- P-value computation with known null distributions
- Single permutation execution
- Full permutation test workflow
- Aggregation across folds
- Edge cases: all null >= observed, observed is best, failures
- Serialization of results
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ced_ml.significance.permutation_test import (
    PermutationTestResult,
    aggregate_permutation_results,
    compute_p_value,
    run_permutation_for_fold,
    run_permutation_test,
    save_null_distributions,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    return X_df.values, y


@pytest.fixture
def simple_pipeline():
    """Create simple sklearn Pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=200)),
        ]
    )


@pytest.fixture
def train_test_indices():
    """Generate train/test indices."""
    train_idx = np.arange(120)
    test_idx = np.arange(120, 200)
    return train_idx, test_idx


# =============================================================================
# Tests: compute_p_value
# =============================================================================


class TestComputePValue:
    """Tests for compute_p_value function."""

    def test_basic_p_value(self):
        """Test basic p-value computation."""
        observed = 0.75
        null_distribution = [0.48, 0.52, 0.55, 0.49, 0.51]

        p_value = compute_p_value(observed, null_distribution)

        expected_p = (1 + 0) / (1 + 5)
        assert p_value == pytest.approx(expected_p)

    def test_observed_equals_max_null(self):
        """Test when observed equals maximum null value."""
        observed = 0.60
        null_distribution = [0.50, 0.55, 0.60]

        p_value = compute_p_value(observed, null_distribution)

        expected_p = (1 + 1) / (1 + 3)
        assert p_value == pytest.approx(expected_p)

    def test_all_null_greater_than_observed(self):
        """Test when all null values >= observed."""
        observed = 0.50
        null_distribution = [0.60, 0.70, 0.80, 0.90]

        p_value = compute_p_value(observed, null_distribution)

        # expected_p = (1 + 4) / (1 + 4)
        assert p_value == pytest.approx(1.0)

    def test_observed_is_best(self):
        """Test when observed is better than all null values."""
        observed = 0.90
        null_distribution = [0.50, 0.52, 0.48, 0.51]

        p_value = compute_p_value(observed, null_distribution)

        # expected_p = (1 + 0) / (1 + 4)
        assert p_value == pytest.approx(0.2)

    def test_single_null_value(self):
        """Test with single null value."""
        observed = 0.75
        null_distribution = [0.50]

        p_value = compute_p_value(observed, null_distribution)

        # expected_p = (1 + 0) / (1 + 1)
        assert p_value == pytest.approx(0.5)

    def test_empty_null_distribution_raises(self):
        """Test that empty null distribution raises error."""
        observed = 0.75
        null_distribution = []

        with pytest.raises(ValueError, match="Null distribution is empty"):
            compute_p_value(observed, null_distribution)

    def test_p_value_never_zero(self):
        """Test that p-value is never exactly zero (Phipson & Smyth 2010)."""
        observed = 1.0
        null_distribution = [0.5] * 100

        p_value = compute_p_value(observed, null_distribution)

        assert p_value > 0
        expected_p = 1 / 101
        assert p_value == pytest.approx(expected_p)

    def test_p_value_never_one_when_ties(self):
        """Test p-value with ties."""
        observed = 0.60
        null_distribution = [0.60, 0.60, 0.60]

        p_value = compute_p_value(observed, null_distribution)

        # expected_p = (1 + 3) / (1 + 3)
        assert p_value == pytest.approx(1.0)


# =============================================================================
# Tests: run_permutation_for_fold
# =============================================================================


class TestRunPermutationForFold:
    """Tests for run_permutation_for_fold function."""

    def test_basic_permutation(self, simple_pipeline, simple_classification_data):
        """Test basic permutation execution."""
        X, y = simple_classification_data
        X_train = X[:120]
        y_train = y[:120]
        X_test = X[120:]
        y_test = y[120:]

        auroc = run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=0,
        )

        assert isinstance(auroc, float)
        assert 0.0 <= auroc <= 1.0

    def test_different_perm_idx_different_results(
        self, simple_pipeline, simple_classification_data
    ):
        """Test that different perm_idx gives different results."""
        X, y = simple_classification_data
        X_train = X[:120]
        y_train = y[:120]
        X_test = X[120:]
        y_test = y[120:]

        auroc1 = run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=0,
        )

        auroc2 = run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=1,
        )

        assert auroc1 != auroc2

    def test_deterministic_with_same_perm_idx(self, simple_pipeline, simple_classification_data):
        """Test that same perm_idx gives same result."""
        X, y = simple_classification_data
        X_train = X[:120]
        y_train = y[:120]
        X_test = X[120:]
        y_test = y[120:]

        auroc1 = run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=0,
        )

        auroc2 = run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=0,
        )

        assert auroc1 == auroc2

    def test_permutation_preserves_held_out(self, simple_pipeline, simple_classification_data):
        """Test that held-out data is never permuted."""
        X, y = simple_classification_data
        X_train = X[:120]
        y_train = y[:120]
        X_test = X[120:]
        y_test = y[120:]

        y_test_original = y_test.copy()

        run_permutation_for_fold(
            pipeline=simple_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42,
            perm_idx=0,
        )

        np.testing.assert_array_equal(y_test, y_test_original)


# =============================================================================
# Tests: run_permutation_test
# =============================================================================


class TestRunPermutationTest:
    """Tests for run_permutation_test function."""

    def test_basic_permutation_test(self, simple_pipeline, simple_classification_data):
        """Test basic permutation test execution."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=10,
            n_jobs=1,
            random_state=42,
        )

        assert isinstance(result, PermutationTestResult)
        assert result.model == "LR"
        assert result.split_seed == 0
        assert result.outer_fold == 0
        assert result.n_perms == 10
        assert result.random_state == 42
        assert 0.0 <= result.observed_auroc <= 1.0
        assert 0.0 <= result.p_value <= 1.0
        assert len(result.null_aurocs) == 10

    def test_deterministic_with_random_state(self, simple_pipeline, simple_classification_data):
        """Test that results are deterministic with same random_state."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result1 = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=10,
            n_jobs=1,
            random_state=42,
        )

        result2 = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=10,
            n_jobs=1,
            random_state=42,
        )

        assert result1.observed_auroc == result2.observed_auroc
        assert result1.p_value == result2.p_value
        assert result1.null_aurocs == result2.null_aurocs

    def test_observed_auroc_computed_correctly(self, simple_pipeline, simple_classification_data):
        """Test that observed AUROC is computed from unpermuted data."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        from sklearn.metrics import roc_auc_score

        pipeline_fitted = simple_pipeline.fit(X[train_idx], y[train_idx])
        expected_auroc = roc_auc_score(
            y[test_idx], pipeline_fitted.predict_proba(X[test_idx])[:, 1]
        )

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=5,
            n_jobs=1,
            random_state=42,
        )

        assert result.observed_auroc == pytest.approx(expected_auroc, abs=0.001)

    def test_null_aurocs_different_from_observed(self, simple_pipeline, simple_classification_data):
        """Test that null AUROCs are different from observed (on average)."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=20,
            n_jobs=1,
            random_state=42,
        )

        mean_null = np.mean(result.null_aurocs)
        assert abs(mean_null - result.observed_auroc) > 0.01

    def test_p_value_in_valid_range(self, simple_pipeline, simple_classification_data):
        """Test that p-value is in valid range."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=10,
            n_jobs=1,
            random_state=42,
        )

        assert 0.0 < result.p_value <= 1.0

    def test_parallel_execution(self, simple_pipeline, simple_classification_data):
        """Test that parallel execution works (n_jobs > 1)."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=10,
            n_jobs=2,
            random_state=42,
        )

        assert len(result.null_aurocs) == 10

    def test_small_n_perms(self, simple_pipeline, simple_classification_data):
        """Test that small n_perms works correctly."""
        X, y = simple_classification_data
        train_idx = np.arange(120)
        test_idx = np.arange(120, 200)

        result = run_permutation_test(
            pipeline=simple_pipeline,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            model_name="LR",
            split_seed=0,
            outer_fold=0,
            n_perms=3,
            n_jobs=1,
            random_state=42,
        )

        assert len(result.null_aurocs) == 3
        assert result.n_perms == 3


# =============================================================================
# Tests: PermutationTestResult dataclass
# =============================================================================


class TestPermutationTestResult:
    """Tests for PermutationTestResult dataclass."""

    def test_creation(self):
        """Test basic PermutationTestResult creation."""
        result = PermutationTestResult(
            model="LR",
            split_seed=0,
            outer_fold=1,
            observed_auroc=0.85,
            null_aurocs=[0.50, 0.52, 0.48],
            p_value=0.25,
            n_perms=3,
            random_state=42,
        )

        assert result.model == "LR"
        assert result.split_seed == 0
        assert result.outer_fold == 1
        assert result.observed_auroc == 0.85
        assert len(result.null_aurocs) == 3
        assert result.p_value == 0.25
        assert result.n_perms == 3

    def test_to_dict_excludes_null_aurocs(self):
        """Test that to_dict excludes null_aurocs."""
        result = PermutationTestResult(
            model="LR",
            split_seed=0,
            outer_fold=0,
            observed_auroc=0.85,
            null_aurocs=[0.50, 0.52, 0.48],
            p_value=0.25,
            n_perms=3,
            random_state=42,
        )

        result_dict = result.to_dict()

        assert "null_aurocs" not in result_dict
        assert "observed_auroc" in result_dict
        assert result_dict["observed_auroc"] == 0.85

    def test_summary_stats(self):
        """Test summary statistics computation."""
        result = PermutationTestResult(
            model="LR",
            split_seed=0,
            outer_fold=0,
            observed_auroc=0.85,
            null_aurocs=[0.50, 0.52, 0.48, 0.51],
            p_value=0.2,
            n_perms=4,
            random_state=42,
        )

        stats = result.summary_stats()

        assert "null_mean" in stats
        assert "null_std" in stats
        assert "null_min" in stats
        assert "null_max" in stats
        assert "null_median" in stats
        assert stats["null_min"] == 0.48
        assert stats["null_max"] == 0.52
        assert stats["null_mean"] == pytest.approx(0.5025)


# =============================================================================
# Tests: aggregate_permutation_results
# =============================================================================


class TestAggregatePermutationResults:
    """Tests for aggregate_permutation_results function."""

    def test_basic_aggregation(self):
        """Test basic aggregation of multiple results."""
        results = [
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.85,
                null_aurocs=[0.50, 0.52],
                p_value=0.33,
                n_perms=2,
                random_state=42,
            ),
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=1,
                observed_auroc=0.80,
                null_aurocs=[0.48, 0.51],
                p_value=0.33,
                n_perms=2,
                random_state=42,
            ),
        ]

        df = aggregate_permutation_results(results)

        assert len(df) == 2
        assert list(df["outer_fold"]) == [0, 1]
        assert "null_mean" in df.columns
        assert "null_std" in df.columns
        assert "p_value" in df.columns

    def test_sorted_by_model_split_fold(self):
        """Test that results are sorted correctly."""
        results = [
            PermutationTestResult(
                model="RF",
                split_seed=1,
                outer_fold=2,
                observed_auroc=0.85,
                null_aurocs=[0.50],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.80,
                null_aurocs=[0.48],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
        ]

        df = aggregate_permutation_results(results)

        assert df.iloc[0]["model"] == "LR"
        assert df.iloc[1]["model"] == "RF"

    def test_summary_stats_included(self):
        """Test that summary statistics are included."""
        results = [
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.85,
                null_aurocs=[0.50, 0.52, 0.48],
                p_value=0.25,
                n_perms=3,
                random_state=42,
            )
        ]

        df = aggregate_permutation_results(results)

        assert "null_mean" in df.columns
        assert "null_std" in df.columns
        assert "null_min" in df.columns
        assert "null_max" in df.columns
        assert "null_median" in df.columns

    def test_empty_results_raises(self):
        """Test that empty results list raises error."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_permutation_results([])


# =============================================================================
# Tests: save_null_distributions
# =============================================================================


class TestSaveNullDistributions:
    """Tests for save_null_distributions function."""

    def test_basic_save(self, tmp_path):
        """Test basic saving of null distributions."""
        results = [
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.85,
                null_aurocs=[0.50, 0.52],
                p_value=0.33,
                n_perms=2,
                random_state=42,
            )
        ]

        output_path = tmp_path / "null_distributions.csv"
        save_null_distributions(results, str(output_path))

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert list(df.columns) == ["model", "split_seed", "outer_fold", "perm_index", "null_auroc"]
        assert list(df["perm_index"]) == [0, 1]
        assert list(df["null_auroc"]) == [0.50, 0.52]

    def test_multiple_folds(self, tmp_path):
        """Test saving with multiple folds."""
        results = [
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.85,
                null_aurocs=[0.50],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=1,
                observed_auroc=0.80,
                null_aurocs=[0.48],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
        ]

        output_path = tmp_path / "null_distributions.csv"
        save_null_distributions(results, str(output_path))

        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert list(df["outer_fold"]) == [0, 1]

    def test_sorted_output(self, tmp_path):
        """Test that output is sorted."""
        results = [
            PermutationTestResult(
                model="RF",
                split_seed=1,
                outer_fold=0,
                observed_auroc=0.85,
                null_aurocs=[0.50],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
            PermutationTestResult(
                model="LR",
                split_seed=0,
                outer_fold=0,
                observed_auroc=0.80,
                null_aurocs=[0.48],
                p_value=0.5,
                n_perms=1,
                random_state=42,
            ),
        ]

        output_path = tmp_path / "null_distributions.csv"
        save_null_distributions(results, str(output_path))

        df = pd.read_csv(output_path)
        assert df.iloc[0]["model"] == "LR"
        assert df.iloc[1]["model"] == "RF"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self, simple_pipeline, simple_classification_data):
        """End-to-end test: run permutation test, aggregate, save."""
        X, y = simple_classification_data

        results = []
        for fold in range(2):
            train_idx = np.arange(fold * 60, fold * 60 + 100)
            test_idx = np.setdiff1d(np.arange(200), train_idx)[:40]

            result = run_permutation_test(
                pipeline=simple_pipeline,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                model_name="LR",
                split_seed=0,
                outer_fold=fold,
                n_perms=5,
                n_jobs=1,
                random_state=42,
            )
            results.append(result)

        df = aggregate_permutation_results(results)

        assert len(df) == 2
        assert all(df["n_perms"] == 5)
        assert all(df["p_value"] > 0)
