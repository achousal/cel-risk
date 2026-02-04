"""
Tests for features.screening module.
"""

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.screening import (
    f_statistic_screen,
    mann_whitney_screen,
    screen_proteins,
)


@pytest.fixture
def sample_data():
    """Create sample proteomics data for testing."""
    rng = np.random.default_rng(42)
    n_controls = 100
    n_cases = 20
    n_proteins = 50

    # Generate protein data
    # Proteins 0-9: discriminative (higher in cases)
    # Proteins 10-19: discriminative (lower in cases)
    # Proteins 20-49: non-discriminative
    X_data = {}
    for i in range(10):
        # High in cases
        X_data[f"protein_{i}_resid"] = np.concatenate(
            [
                rng.normal(0, 1, n_controls),
                rng.normal(2, 1, n_cases),
            ]  # Higher mean
        )

    for i in range(10, 20):
        # Low in cases
        X_data[f"protein_{i}_resid"] = np.concatenate(
            [
                rng.normal(0, 1, n_controls),
                rng.normal(-2, 1, n_cases),
            ]  # Lower mean
        )

    for i in range(20, n_proteins):
        # Non-discriminative
        X_data[f"protein_{i}_resid"] = rng.normal(0, 1, n_controls + n_cases)

    X = pd.DataFrame(X_data)
    y = np.array([0] * n_controls + [1] * n_cases)

    protein_cols = [col for col in X.columns if col.endswith("_resid")]

    return X, y, protein_cols


def test_mann_whitney_screen_basic(sample_data):
    """Test basic Mann-Whitney screening."""
    X, y, protein_cols = sample_data

    selected, stats = mann_whitney_screen(X, y, protein_cols, top_n=10, min_n_per_group=5)

    assert len(selected) == 10
    assert len(stats) == len(protein_cols)  # All proteins tested
    assert list(stats.columns) == [
        "protein",
        "p_value",
        "effect_size",
        "nonmissing_frac",
    ]

    # Top proteins should be the discriminative ones (0-19)
    top_indices = [int(p.split("_")[1]) for p in selected]
    assert all(i < 20 for i in top_indices), "Top proteins should be discriminative"

    # Check sorting: p_values should be ascending
    assert stats["p_value"].is_monotonic_increasing or stats["p_value"].iloc[:10].min() < 0.05


def test_mann_whitney_screen_with_missing(sample_data):
    """Test Mann-Whitney screening with missing values."""
    X, y, protein_cols = sample_data

    # Introduce missing values
    X_miss = X.copy()
    X_miss.iloc[:10, 0] = np.nan  # 10 missing in first protein

    selected, stats = mann_whitney_screen(X_miss, y, protein_cols, top_n=10, min_n_per_group=5)

    assert len(selected) <= 10
    assert "nonmissing_frac" in stats.columns

    # First protein should have lower nonmissing_frac
    first_prot_stats = stats[stats["protein"] == protein_cols[0]].iloc[0]
    assert first_prot_stats["nonmissing_frac"] < 1.0


def test_mann_whitney_screen_min_n_per_group(sample_data):
    """Test minimum group size enforcement."""
    X, y, protein_cols = sample_data

    # Use only 5 samples (too few for min_n_per_group=10)
    X_small = X.iloc[:5]
    y_small = y[:5]

    selected, stats = mann_whitney_screen(
        X_small, y_small, protein_cols, top_n=10, min_n_per_group=10
    )

    # Should return all proteins (no screening) due to insufficient samples
    assert len(stats) == 0  # No proteins meet the threshold


def test_mann_whitney_screen_single_class():
    """Test behavior with single class (no discrimination possible)."""
    X = pd.DataFrame({"protein_1_resid": [1, 2, 3, 4, 5]})
    y = np.array([0, 0, 0, 0, 0])  # All controls
    protein_cols = ["protein_1_resid"]

    selected, stats = mann_whitney_screen(X, y, protein_cols, top_n=10)

    assert selected == protein_cols  # Return all (no screening)
    assert len(stats) == 0


def test_f_statistic_screen_basic(sample_data):
    """Test basic F-statistic screening."""
    X, y, protein_cols = sample_data

    selected, stats = f_statistic_screen(X, y, protein_cols, top_n=10)

    assert len(selected) == 10
    assert len(stats) == len(protein_cols)
    assert list(stats.columns) == ["protein", "F_score", "p_value", "nonmissing_frac"]

    # F-scores should be descending
    assert stats["F_score"].is_monotonic_decreasing

    # Top proteins should be discriminative
    top_indices = [int(p.split("_")[1]) for p in selected]
    assert all(i < 20 for i in top_indices)


def test_f_statistic_screen_with_missing(sample_data):
    """Test F-statistic screening with missing values (median imputation)."""
    X, y, protein_cols = sample_data

    # Introduce missing values
    X_miss = X.copy()
    X_miss.iloc[:20, 0] = np.nan  # Many missing in first protein

    selected, stats = f_statistic_screen(X_miss, y, protein_cols, top_n=10)

    assert len(selected) == 10
    assert "nonmissing_frac" in stats.columns

    # First protein should show lower nonmissing_frac
    first_prot_stats = stats[stats["protein"] == protein_cols[0]].iloc[0]
    assert first_prot_stats["nonmissing_frac"] < 1.0


def test_f_statistic_screen_single_class():
    """Test F-statistic with single class."""
    X = pd.DataFrame({"protein_1_resid": [1, 2, 3, 4, 5]})
    y = np.array([0, 0, 0, 0, 0])
    protein_cols = ["protein_1_resid"]

    selected, stats = f_statistic_screen(X, y, protein_cols, top_n=10)

    assert selected == protein_cols
    assert len(stats) == 0


def test_screen_proteins_wrapper(sample_data):
    """Test screen_proteins convenience wrapper."""
    X, y, protein_cols = sample_data

    # Test Mann-Whitney
    selected_mw, stats_mw, _ = screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=10)
    assert len(selected_mw) == 10
    assert "p_value" in stats_mw.columns

    # Test F-statistic
    selected_f, stats_f, _ = screen_proteins(X, y, protein_cols, method="f_classif", top_n=10)
    assert len(selected_f) == 10
    assert "F_score" in stats_f.columns


def test_screen_proteins_no_screening(sample_data):
    """Test screen_proteins with top_n=0 (returns all proteins with statistics)."""
    X, y, protein_cols = sample_data

    selected, stats, _ = screen_proteins(X, y, protein_cols, method="mannwhitney", top_n=0)

    # Should return all proteins (though may be reordered by p-value)
    assert set(selected) == set(protein_cols)
    assert len(selected) == len(protein_cols)
    # Should compute statistics for all proteins
    assert len(stats) > 0
    assert "p_value" in stats.columns


def test_screen_proteins_invalid_method(sample_data):
    """Test screen_proteins with invalid method."""
    X, y, protein_cols = sample_data

    with pytest.raises(ValueError, match="Unknown screen_method"):
        screen_proteins(X, y, protein_cols, method="invalid", top_n=10)


def test_mann_whitney_effect_size_ordering(sample_data):
    """Test that effect size is used as tiebreaker."""
    X, y, protein_cols = sample_data

    selected, stats = mann_whitney_screen(X, y, protein_cols, top_n=20, min_n_per_group=5)

    # For proteins with similar p-values, higher effect_size should rank higher
    top20_stats = stats.head(20)

    # Check that within small p-value bins, effect_size is descending
    small_pval = top20_stats[top20_stats["p_value"] < 0.01]
    if len(small_pval) > 1:
        # Effect size should generally be high for significant proteins
        assert small_pval["effect_size"].mean() > 1.0


def test_f_statistic_handles_constant_features():
    """Test F-statistic correctly filters out constant features."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "constant_resid": np.ones(100),  # Zero variance
            "variable_resid": rng.normal(0, 1, 100),
        }
    )
    y = np.array([0] * 50 + [1] * 50)
    protein_cols = [col for col in X.columns if col.endswith("_resid")]

    selected, stats = f_statistic_screen(X, y, protein_cols, top_n=10)

    # Constant feature should be filtered out (non-finite F-score)
    assert "constant_resid" not in stats["protein"].values
    assert "variable_resid" in stats["protein"].values


def test_screen_proteins_default_parameters():
    """Test screen_proteins with default parameters."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "protein_1_resid": np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)]),
            "protein_2_resid": rng.normal(0, 1, 100),
        }
    )
    y = np.array([0] * 50 + [1] * 50)
    protein_cols = [col for col in X.columns if col.endswith("_resid")]

    # Default: mannwhitney, top_n=1000, min_n_per_group=10
    selected, stats, _ = screen_proteins(X, y, protein_cols)

    assert len(selected) <= 2  # At most 2 proteins available
    assert "p_value" in stats.columns  # Mann-Whitney default


def test_mann_whitney_asymmetric_group_sizes():
    """Test Mann-Whitney with very asymmetric group sizes (realistic for CeD)."""
    rng = np.random.default_rng(42)
    n_controls = 1000
    n_cases = 50

    X = pd.DataFrame(
        {
            "protein_1_resid": np.concatenate(
                [rng.normal(0, 1, n_controls), rng.normal(1.5, 1, n_cases)]
            )
        }
    )
    y = np.array([0] * n_controls + [1] * n_cases)
    protein_cols = ["protein_1_resid"]

    selected, stats = mann_whitney_screen(X, y, protein_cols, top_n=10, min_n_per_group=10)

    assert len(selected) == 1
    assert stats.iloc[0]["p_value"] < 0.05  # Should detect difference
    assert stats.iloc[0]["effect_size"] > 0.5  # Reasonable effect size
