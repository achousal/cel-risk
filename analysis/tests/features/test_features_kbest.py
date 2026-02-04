"""Tests for K-best feature selection module."""

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.kbest import (
    compute_f_classif_scores,
    rank_features_by_score,
)
from sklearn.feature_selection import f_classif


@pytest.fixture
def simple_protein_data():
    """Create simple protein dataset for testing."""
    rng = np.random.default_rng(42)

    # 3 proteins: P1 (strong signal), P2 (weak), P3 (noise)
    n_samples = 100
    n_case = 20

    X = pd.DataFrame(
        {
            "P1": np.concatenate(
                [
                    rng.normal(0, 1, n_samples - n_case),  # controls
                    rng.normal(2, 1, n_case),  # cases (shifted)
                ]
            ),
            "P2": np.concatenate(
                [
                    rng.normal(0, 1, n_samples - n_case),
                    rng.normal(0.5, 1, n_case),  # weak signal
                ]
            ),
            "P3": rng.normal(0, 1, n_samples),  # pure noise
        }
    )

    y = np.array([0] * (n_samples - n_case) + [1] * n_case)

    return X, y


@pytest.fixture
def protein_data_with_missing():
    """Protein data with missing values."""
    X = pd.DataFrame(
        {
            "P1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "P2": [10.0, np.nan, 30.0, 40.0, 50.0],
            "P3": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    y = np.array([0, 0, 1, 1, 1])

    return X, y


class TestComputeFClassifScores:
    """Tests for compute_f_classif_scores()."""

    def test_returns_correct_shape(self, simple_protein_data):
        X, y = simple_protein_data

        scores = compute_f_classif_scores(X.to_numpy(), y)

        assert scores.shape == (3,)  # 3 features

    def test_returns_positive_scores(self, simple_protein_data):
        X, y = simple_protein_data

        scores = compute_f_classif_scores(X.to_numpy(), y)

        assert np.all(scores >= 0)

    def test_higher_score_for_stronger_signal(self):
        # Feature 0: strong signal, Feature 1: no signal
        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [5, 0],
                [5, 1],
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1])

        scores = compute_f_classif_scores(X, y)

        assert scores[0] > scores[1]  # Feature 0 has stronger signal

    def test_raises_on_single_class(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 0])  # Single class

        with pytest.raises(ValueError, match="at least 2 classes"):
            compute_f_classif_scores(X, y)

    def test_matches_sklearn_f_classif(self, simple_protein_data):
        X, y = simple_protein_data

        our_scores = compute_f_classif_scores(X.to_numpy(), y)
        sklearn_scores, _ = f_classif(X.to_numpy(), y)

        np.testing.assert_array_almost_equal(our_scores, sklearn_scores)


class TestRankFeaturesByScore:
    """Tests for rank_features_by_score()."""

    def test_returns_descending_order(self):
        scores = np.array([0.5, 2.0, 1.0, 3.0])

        indices = rank_features_by_score(scores, k=4)

        # Expected order: [3, 1, 2, 0] (scores: 3.0, 2.0, 1.0, 0.5)
        expected = np.array([3, 1, 2, 0])
        np.testing.assert_array_equal(indices, expected)

    def test_selects_top_k(self):
        scores = np.array([0.5, 2.0, 1.0, 3.0])

        indices = rank_features_by_score(scores, k=2)

        assert len(indices) == 2
        assert indices[0] == 3  # Highest score
        assert indices[1] == 1  # Second highest

    def test_clips_k_to_array_length(self):
        scores = np.array([1.0, 2.0])

        indices = rank_features_by_score(scores, k=10)

        assert len(indices) == 2

    def test_handles_negative_scores(self):
        scores = np.array([-1.0, -3.0, -2.0])

        indices = rank_features_by_score(scores, k=2)

        # Should still rank descending: -1.0, -2.0, -3.0
        assert indices[0] == 0  # -1.0 (highest)
        assert indices[1] == 2  # -2.0


def test_screening_transformer_attributes():
    """Test ScreeningTransformer sets both selected_features_ and selected_proteins_."""
    from ced_ml.features.kbest import ScreeningTransformer

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "P1": rng.normal(0, 1, 100),
            "P2": rng.normal(1.5, 1, 100),
            "P3": rng.normal(0, 1, 100),
        }
    )
    y = np.concatenate([np.zeros(50), np.ones(50)])

    screener = ScreeningTransformer(method="mannwhitney", top_n=2, protein_cols=["P1", "P2", "P3"])
    screener.fit(X, y)

    # Both attributes should exist
    assert hasattr(screener, "selected_features_"), "Missing selected_features_ attribute"
    assert hasattr(screener, "selected_proteins_"), "Missing selected_proteins_ attribute"

    # They should be identical (selected_proteins_ is an alias)
    assert (
        screener.selected_features_ == screener.selected_proteins_
    ), "selected_features_ and selected_proteins_ should be identical"

    # Should have selected exactly 2 proteins
    assert (
        len(screener.selected_proteins_) == 2
    ), f"Expected 2 proteins, got {len(screener.selected_proteins_)}"


def test_screening_transformer_precomputed_features():
    """Test ScreeningTransformer skips screening when precomputed_features is set."""
    from ced_ml.features.kbest import ScreeningTransformer

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "P1": rng.normal(0, 1, 100),
            "P2": rng.normal(1.5, 1, 100),
            "P3": rng.normal(0, 1, 100),
            "meta_col": rng.choice(["A", "B"], 100),
        }
    )
    y = np.concatenate([np.zeros(50), np.ones(50)])

    precomputed = ["P1", "P3"]
    screener = ScreeningTransformer(
        method="mannwhitney",
        top_n=2,
        protein_cols=["P1", "P2", "P3"],
        precomputed_features=precomputed,
    )
    screener.fit(X, y)

    assert screener.selected_features_ == ["P1", "P3"]
    assert screener.selected_proteins_ == ["P1", "P3"]
    assert screener.screening_stats_ is None

    # Transform should keep precomputed proteins + non-protein columns
    result = screener.transform(X)
    assert "P1" in result.columns
    assert "P3" in result.columns
    assert "meta_col" in result.columns
    assert "P2" not in result.columns
