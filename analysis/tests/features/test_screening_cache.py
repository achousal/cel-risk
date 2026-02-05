"""Tests for screening results caching."""

import numpy as np
import pandas as pd
import pytest

from ced_ml.features.screening import screen_proteins
from ced_ml.features.screening_cache import get_screening_cache


@pytest.fixture
def sample_data(sample_data_screening):
    """Alias for sample_data_screening from conftest for backward compatibility."""
    return sample_data_screening


@pytest.fixture
def fresh_cache():
    """Get a fresh cache for testing."""
    cache = get_screening_cache()
    cache.clear()
    yield cache
    cache.clear()


def test_screening_cache_basic(sample_data, fresh_cache):
    """Test basic cache functionality."""
    X, y, protein_cols = sample_data

    # First call - should be a miss
    selected1, stats1, _ = screen_proteins(
        X_train=X,
        y_train=y,
        protein_cols=protein_cols,
        method="mannwhitney",
        top_n=20,
        use_cache=True,
    )

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 1
    assert cache_stats["hits"] == 0
    assert cache_stats["size"] == 1

    # Second call with same parameters - should be a hit
    selected2, stats2, _ = screen_proteins(
        X_train=X,
        y_train=y,
        protein_cols=protein_cols,
        method="mannwhitney",
        top_n=20,
        use_cache=True,
    )

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 1
    assert cache_stats["hits"] == 1
    assert cache_stats["size"] == 1

    # Results should be identical
    assert selected1 == selected2
    pd.testing.assert_frame_equal(stats1, stats2)


def test_screening_cache_different_params(sample_data, fresh_cache):
    """Test that cache distinguishes different parameters."""
    X, y, protein_cols = sample_data

    # Different top_n
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=10, use_cache=True)
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 2
    assert cache_stats["size"] == 2

    # Different method
    screen_proteins(X, y, protein_cols, method="f_classif", top_n=10, use_cache=True)

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 3
    assert cache_stats["size"] == 3


def test_screening_cache_different_data(sample_data, fresh_cache):
    """Test that cache distinguishes different data."""
    X, y, protein_cols = sample_data

    # Original data
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    # Modified X
    X_modified = X.copy()
    X_modified.iloc[0, 0] = 999.0
    screen_proteins(X_modified, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    # Modified y
    y_modified = y.copy()
    y_modified[0] = 1 - y_modified[0]
    screen_proteins(X, y_modified, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 3
    assert cache_stats["size"] == 3


def test_screening_cache_disabled(sample_data, fresh_cache):
    """Test that cache can be disabled."""
    X, y, protein_cols = sample_data

    # First call with cache disabled
    selected1, stats1, _ = screen_proteins(
        X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=False
    )

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 0
    assert cache_stats["hits"] == 0
    assert cache_stats["size"] == 0

    # Second call with cache disabled
    selected2, stats2, _ = screen_proteins(
        X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=False
    )

    cache_stats = fresh_cache.stats()
    assert cache_stats["misses"] == 0
    assert cache_stats["hits"] == 0
    assert cache_stats["size"] == 0

    # Results should still be identical (deterministic screening)
    assert selected1 == selected2
    pd.testing.assert_frame_equal(stats1, stats2)


def test_screening_cache_clear(sample_data, fresh_cache):
    """Test cache clearing."""
    X, y, protein_cols = sample_data

    # Add entries to cache
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)
    screen_proteins(X, y, protein_cols, method="f_classif", top_n=20, use_cache=True)

    cache_stats = fresh_cache.stats()
    assert cache_stats["size"] == 2

    # Clear cache
    fresh_cache.clear()

    cache_stats = fresh_cache.stats()
    assert cache_stats["size"] == 0
    assert cache_stats["hits"] == 0
    assert cache_stats["misses"] == 0


def test_screening_cache_hit_rate(sample_data, fresh_cache):
    """Test hit rate calculation."""
    X, y, protein_cols = sample_data

    # 1 miss
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    # 2 hits
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)
    screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True)

    cache_stats = fresh_cache.stats()
    assert cache_stats["hits"] == 2
    assert cache_stats["misses"] == 1
    assert cache_stats["hit_rate"] == pytest.approx(2 / 3)


def test_screening_cache_sequential_calls(sample_data, fresh_cache):
    """Test cache works correctly with sequential repeated calls."""
    X, y, protein_cols = sample_data

    # Make 5 sequential calls
    results = []
    for _ in range(5):
        selected, stats, _ = screen_proteins(
            X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True
        )
        results.append((selected, stats))

    # All results should be identical (deterministic screening)
    first_selected, first_stats = results[0]
    for selected, stats in results[1:]:
        assert selected == first_selected
        pd.testing.assert_frame_equal(stats, first_stats)

    # Should have 1 miss + 4 hits
    cache_stats = fresh_cache.stats()
    assert cache_stats["hits"] == 4
    assert cache_stats["misses"] == 1


def test_screening_cache_empty_data(fresh_cache):
    """Test cache with edge cases."""
    # Empty protein list
    X = pd.DataFrame()
    y = np.array([0, 1])
    protein_cols = []

    selected, stats, _ = screen_proteins(
        X, y, protein_cols, method="mannwhitney", top_n=20, use_cache=True
    )

    assert selected == []
    assert stats.empty

    # Cache should not store empty results
    cache_stats = fresh_cache.stats()
    assert cache_stats["size"] == 0


def test_screening_cache_top_n_zero(sample_data, fresh_cache):
    """Test that top_n=0 (all proteins) is cached correctly."""
    X, y, protein_cols = sample_data

    # First call with top_n=0
    selected1, stats1, _ = screen_proteins(
        X, y, protein_cols, method="mannwhitney", top_n=0, use_cache=True
    )

    # Second call should hit cache
    selected2, stats2, _ = screen_proteins(
        X, y, protein_cols, method="mannwhitney", top_n=0, use_cache=True
    )

    cache_stats = fresh_cache.stats()
    assert cache_stats["hits"] == 1
    assert cache_stats["misses"] == 1

    # Results should include all proteins (no filtering, though may be reordered by p-value)
    assert set(selected1) == set(protein_cols)
    assert set(selected2) == set(protein_cols)
    assert len(selected1) == len(protein_cols)
    assert len(selected2) == len(protein_cols)
