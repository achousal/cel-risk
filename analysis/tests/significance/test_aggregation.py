"""Tests for pooled-null permutation aggregation module.

Tests cover:
- PooledNullResult dataclass: creation, to_dict(), summary_stats()
- compute_pooled_p_value: p-value computation with known distributions
- pool_null_distribution: aggregation across seeds/folds
"""

import numpy as np
import pandas as pd
import pytest

from ced_ml.significance.aggregation import (
    PooledNullResult,
    compute_pooled_p_value,
    pool_null_distribution,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pooled_null() -> np.ndarray:
    """Generate sample pooled null distribution."""
    np.random.seed(42)
    return np.random.rand(100) * 0.2 + 0.45  # Range ~[0.45, 0.65]


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Generate sample results DataFrame for pool_null_distribution."""
    np.random.seed(42)
    n_seeds = 3
    n_folds = 4
    n_perms = 50

    records = []
    for seed in range(n_seeds):
        for fold in range(n_folds):
            for perm_idx in range(n_perms):
                records.append(
                    {
                        "model": "LR_EN",
                        "split_seed": seed,
                        "outer_fold": fold,
                        "perm_index": perm_idx,
                        "null_auroc": np.random.rand() * 0.2 + 0.45,
                        "observed_auroc": 0.75,
                    }
                )

    return pd.DataFrame.from_records(records)


@pytest.fixture
def multi_model_results_df() -> pd.DataFrame:
    """Generate results DataFrame with multiple models."""
    np.random.seed(42)
    n_perms = 50

    records = []
    for model in ["LR_EN", "RF", "XGBoost"]:
        for perm_idx in range(n_perms):
            records.append(
                {
                    "model": model,
                    "split_seed": 0,
                    "outer_fold": 0,
                    "perm_index": perm_idx,
                    "null_auroc": np.random.rand() * 0.2 + 0.45,
                    "observed_auroc": 0.75,
                }
            )

    return pd.DataFrame.from_records(records)


# =============================================================================
# Tests: PooledNullResult
# =============================================================================


class TestPooledNullResult:
    """Tests for PooledNullResult dataclass."""

    def test_creation(self, sample_pooled_null):
        """Test basic creation of PooledNullResult."""
        result = PooledNullResult(
            model="LR_EN",
            observed_auroc=0.75,
            pooled_null=sample_pooled_null,
            empirical_p_value=0.01,
            n_seeds=3,
            n_perms_total=100,
            significant=True,
            alpha=0.05,
        )

        assert result.model == "LR_EN"
        assert result.observed_auroc == 0.75
        assert len(result.pooled_null) == 100
        assert result.empirical_p_value == 0.01
        assert result.n_seeds == 3
        assert result.n_perms_total == 100
        assert result.significant is True
        assert result.alpha == 0.05

    def test_summary_stats(self, sample_pooled_null):
        """Test summary_stats() returns correct statistics."""
        result = PooledNullResult(
            model="LR_EN",
            observed_auroc=0.75,
            pooled_null=sample_pooled_null,
            empirical_p_value=0.01,
            n_seeds=3,
            n_perms_total=100,
            significant=True,
            alpha=0.05,
        )

        stats = result.summary_stats()

        assert "null_mean" in stats
        assert "null_std" in stats
        assert "null_min" in stats
        assert "null_max" in stats
        assert "null_median" in stats
        assert "null_q25" in stats
        assert "null_q75" in stats

        assert stats["null_mean"] == pytest.approx(np.mean(sample_pooled_null))
        assert stats["null_std"] == pytest.approx(np.std(sample_pooled_null))
        assert stats["null_min"] == pytest.approx(np.min(sample_pooled_null))
        assert stats["null_max"] == pytest.approx(np.max(sample_pooled_null))

    def test_summary_stats_empty_null(self):
        """Test summary_stats() with empty pooled_null."""
        result = PooledNullResult(
            model="LR_EN",
            observed_auroc=0.75,
            pooled_null=np.array([]),
            empirical_p_value=1.0,
            n_seeds=0,
            n_perms_total=0,
            significant=False,
            alpha=0.05,
        )

        stats = result.summary_stats()

        assert np.isnan(stats["null_mean"])
        assert np.isnan(stats["null_std"])
        assert np.isnan(stats["null_min"])
        assert np.isnan(stats["null_max"])

    def test_to_dict(self, sample_pooled_null):
        """Test to_dict() returns correct dictionary."""
        result = PooledNullResult(
            model="LR_EN",
            observed_auroc=0.75,
            pooled_null=sample_pooled_null,
            empirical_p_value=0.01,
            n_seeds=3,
            n_perms_total=100,
            significant=True,
            alpha=0.05,
        )

        d = result.to_dict()

        assert d["model"] == "LR_EN"
        assert d["observed_auroc"] == 0.75
        assert d["empirical_p_value"] == 0.01
        assert d["n_seeds"] == 3
        assert d["n_perms_total"] == 100
        assert d["significant"] is True
        assert d["alpha"] == 0.05
        # Should include summary stats
        assert "null_mean" in d
        assert "null_std" in d
        # Should NOT include the full pooled_null array
        assert "pooled_null" not in d

    def test_to_dict_serializable(self, sample_pooled_null):
        """Test that to_dict() output is JSON-serializable."""
        import json

        result = PooledNullResult(
            model="LR_EN",
            observed_auroc=0.75,
            pooled_null=sample_pooled_null,
            empirical_p_value=0.01,
            n_seeds=3,
            n_perms_total=100,
            significant=True,
            alpha=0.05,
        )

        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0


# =============================================================================
# Tests: compute_pooled_p_value
# =============================================================================


class TestComputePooledPValue:
    """Tests for compute_pooled_p_value function."""

    def test_basic_p_value(self):
        """Test basic p-value computation."""
        observed = 0.75
        null_pool = np.array([0.48, 0.52, 0.55, 0.49, 0.51])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 0) / (1 + 5) = 0.1667
        expected_p = (1 + 0) / (1 + 5)
        assert p_value == pytest.approx(expected_p)

    def test_observed_equals_max_null(self):
        """Test when observed equals maximum null value."""
        observed = 0.60
        null_pool = np.array([0.50, 0.55, 0.60])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 1) / (1 + 3) = 0.5
        expected_p = (1 + 1) / (1 + 3)
        assert p_value == pytest.approx(expected_p)

    def test_all_null_greater_than_observed(self):
        """Test when all null values >= observed."""
        observed = 0.50
        null_pool = np.array([0.60, 0.70, 0.80, 0.90])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 4) / (1 + 4) = 1.0
        assert p_value == pytest.approx(1.0)

    def test_observed_is_best(self):
        """Test when observed is better than all null values."""
        observed = 0.90
        null_pool = np.array([0.50, 0.52, 0.48, 0.51])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 0) / (1 + 4) = 0.2
        assert p_value == pytest.approx(0.2)

    def test_single_null_value(self):
        """Test with single null value."""
        observed = 0.75
        null_pool = np.array([0.50])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 0) / (1 + 1) = 0.5
        assert p_value == pytest.approx(0.5)

    def test_empty_null_pool_raises(self):
        """Test that empty null pool raises error."""
        observed = 0.75
        null_pool = np.array([])

        with pytest.raises(ValueError, match="Null pool is empty"):
            compute_pooled_p_value(observed, null_pool)

    def test_p_value_never_zero(self):
        """Test that p-value is never exactly zero (Phipson & Smyth 2010)."""
        observed = 1.0
        null_pool = np.array([0.5] * 100)

        p_value = compute_pooled_p_value(observed, null_pool)

        assert p_value > 0
        expected_p = 1 / 101
        assert p_value == pytest.approx(expected_p)

    def test_p_value_with_ties(self):
        """Test p-value with ties."""
        observed = 0.60
        null_pool = np.array([0.60, 0.60, 0.60])

        p_value = compute_pooled_p_value(observed, null_pool)

        # expected_p = (1 + 3) / (1 + 3) = 1.0
        assert p_value == pytest.approx(1.0)

    def test_accepts_list_input(self):
        """Test that function accepts list input."""
        observed = 0.75
        null_pool = [0.48, 0.52, 0.55]

        p_value = compute_pooled_p_value(observed, np.array(null_pool))

        assert isinstance(p_value, float)
        assert 0 < p_value <= 1


# =============================================================================
# Tests: pool_null_distribution
# =============================================================================


class TestPoolNullDistribution:
    """Tests for pool_null_distribution function."""

    def test_basic_pooling(self, sample_results_df):
        """Test basic pooling of null distribution."""
        result = pool_null_distribution(sample_results_df, model="LR_EN", alpha=0.05)

        assert result.model == "LR_EN"
        assert result.n_seeds == 3
        assert result.n_perms_total == 600  # 3 seeds * 4 folds * 50 perms
        assert len(result.pooled_null) == 600
        assert result.alpha == 0.05
        assert isinstance(result.empirical_p_value, float)
        assert isinstance(result.significant, bool)

    def test_uses_first_model_if_none(self, sample_results_df):
        """Test that first model is used if model is None."""
        result = pool_null_distribution(sample_results_df, model=None, alpha=0.05)

        assert result.model == "LR_EN"

    def test_multi_model_filtering(self, multi_model_results_df):
        """Test filtering to specific model in multi-model results."""
        result = pool_null_distribution(multi_model_results_df, model="RF", alpha=0.05)

        assert result.model == "RF"
        assert result.n_perms_total == 50  # Only RF perms

    def test_missing_columns_raises(self):
        """Test that missing required columns raises error."""
        df = pd.DataFrame({"foo": [1, 2, 3]})

        with pytest.raises(ValueError, match="missing required columns"):
            pool_null_distribution(df, model="LR_EN")

    def test_empty_df_raises(self):
        """Test that empty DataFrame raises error."""
        df = pd.DataFrame({"model": [], "null_auroc": []})

        with pytest.raises(ValueError, match="empty"):
            pool_null_distribution(df, model="LR_EN")

    def test_nonexistent_model_raises(self, sample_results_df):
        """Test that nonexistent model raises error."""
        with pytest.raises(ValueError, match="No results found for model"):
            pool_null_distribution(sample_results_df, model="NonExistent")

    def test_nan_values_removed(self):
        """Test that NaN values are removed from pooled null."""
        df = pd.DataFrame(
            {
                "model": ["LR_EN"] * 10,
                "split_seed": [0] * 10,
                "null_auroc": [0.5, 0.55, np.nan, 0.52, np.nan, 0.48, 0.51, 0.53, 0.49, 0.50],
                "observed_auroc": [0.75] * 10,
            }
        )

        result = pool_null_distribution(df, model="LR_EN")

        assert result.n_perms_total == 8  # 10 - 2 NaNs
        assert not np.any(np.isnan(result.pooled_null))

    def test_all_nan_returns_empty_result(self):
        """Test that all-NaN values return empty result with p=1.0."""
        df = pd.DataFrame(
            {
                "model": ["LR_EN"] * 5,
                "split_seed": [0] * 5,
                "null_auroc": [np.nan] * 5,
            }
        )

        result = pool_null_distribution(df, model="LR_EN")

        assert result.n_perms_total == 0
        assert result.empirical_p_value == 1.0
        assert result.significant is False

    def test_observed_auroc_from_column(self):
        """Test that observed AUROC is computed from column."""
        df = pd.DataFrame(
            {
                "model": ["LR_EN"] * 10,
                "split_seed": [0] * 10,
                "null_auroc": [0.5] * 10,
                "observed_auroc": [0.75, 0.76, 0.74, 0.75, 0.73, 0.77, 0.75, 0.74, 0.76, 0.75],
            }
        )

        result = pool_null_distribution(df, model="LR_EN")

        assert result.observed_auroc == pytest.approx(0.75)

    def test_custom_alpha(self, sample_results_df):
        """Test custom alpha level."""
        result = pool_null_distribution(sample_results_df, model="LR_EN", alpha=0.01)

        assert result.alpha == 0.01
