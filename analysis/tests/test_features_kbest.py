"""Tests for K-best feature selection module."""

import numpy as np
import pandas as pd
import pytest
from ced_ml.features.kbest import (
    compute_f_classif_scores,
    compute_protein_statistics,
    extract_selected_proteins_from_kbest,
    rank_features_by_score,
    select_kbest_features,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


class TestSelectKBestFeatures:
    """Tests for select_kbest_features()."""

    def test_selects_correct_number(self, simple_protein_data):
        X, y = simple_protein_data
        protein_cols = ["P1", "P2", "P3"]

        selected = select_kbest_features(X, y, k=2, protein_cols=protein_cols)

        assert len(selected) == 2

    def test_selects_highest_signal_features(self, simple_protein_data):
        X, y = simple_protein_data
        protein_cols = ["P1", "P2", "P3"]

        selected = select_kbest_features(X, y, k=1, protein_cols=protein_cols)

        # P1 has strongest signal (largest mean difference)
        assert selected == ["P1"]

    def test_clips_k_to_available_proteins(self, simple_protein_data):
        X, y = simple_protein_data
        protein_cols = ["P1", "P2"]

        # Request k=10 but only 2 proteins available
        selected = select_kbest_features(X, y, k=10, protein_cols=protein_cols)

        assert len(selected) == 2

    def test_handles_missing_values_median_imputation(self, protein_data_with_missing):
        X, y = protein_data_with_missing
        protein_cols = ["P1", "P2", "P3"]

        selected = select_kbest_features(
            X, y, k=2, protein_cols=protein_cols, missing_strategy="median"
        )

        assert len(selected) == 2
        # Should not raise despite NaN values

    def test_handles_missing_values_mean_imputation(self, protein_data_with_missing):
        X, y = protein_data_with_missing
        protein_cols = ["P1", "P2", "P3"]

        selected = select_kbest_features(
            X, y, k=2, protein_cols=protein_cols, missing_strategy="mean"
        )

        assert len(selected) == 2

    def test_raises_when_no_protein_columns_found(self, simple_protein_data):
        X, y = simple_protein_data

        with pytest.raises(ValueError, match="No valid protein columns"):
            select_kbest_features(X, y, k=2, protein_cols=["NonExistent"])

    def test_fallback_to_variance_on_error(self):
        # Create data where F-test might fail (single class)
        X = pd.DataFrame({"P1": [1, 2, 3], "P2": [4, 5, 6]})
        y = np.array([0, 0, 0])  # All same class

        with pytest.raises(ValueError):
            select_kbest_features(X, y, k=1, protein_cols=["P1", "P2"], fallback_to_variance=False)

    def test_returns_deterministic_results(self, simple_protein_data):
        X, y = simple_protein_data
        protein_cols = ["P1", "P2", "P3"]

        selected1 = select_kbest_features(X, y, k=2, protein_cols=protein_cols)
        selected2 = select_kbest_features(X, y, k=2, protein_cols=protein_cols)

        assert selected1 == selected2


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


class TestComputeProteinStatistics:
    """Tests for compute_protein_statistics()."""

    def test_returns_all_required_keys(self):
        X = pd.DataFrame({"P1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y = np.array([0, 0, 1, 1, 1])

        stats = compute_protein_statistics(X, y, "P1")

        required_keys = [
            "protein",
            "n_total",
            "n_case",
            "n_control",
            "mean_case",
            "mean_control",
            "sd_case",
            "sd_control",
            "cohens_d",
            "p_ttest",
        ]
        for key in required_keys:
            assert key in stats

    def test_computes_correct_sample_counts(self):
        X = pd.DataFrame({"P1": [1, 2, 3, 4, 5]})
        y = np.array([0, 0, 1, 1, 1])

        stats = compute_protein_statistics(X, y, "P1")

        assert stats["n_total"] == 5
        assert stats["n_case"] == 3
        assert stats["n_control"] == 2

    def test_computes_correct_means(self):
        X = pd.DataFrame({"P1": [1.0, 2.0, 10.0, 20.0]})
        y = np.array([0, 0, 1, 1])

        stats = compute_protein_statistics(X, y, "P1")

        assert stats["mean_control"] == pytest.approx(1.5)
        assert stats["mean_case"] == pytest.approx(15.0)

    def test_handles_missing_values(self):
        X = pd.DataFrame({"P1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
        y = np.array([0, 0, 1, 1, 1, 1])

        stats = compute_protein_statistics(X, y, "P1")

        assert stats["n_total"] == 5  # Excludes NaN
        assert stats["n_control"] == 2
        assert stats["n_case"] == 3

    def test_returns_none_for_insufficient_data(self):
        X = pd.DataFrame({"P1": [1.0, 2.0, 3.0]})
        y = np.array([0, 0, 0])

        stats = compute_protein_statistics(X, y, "P1")

        assert stats is None  # Single class

    def test_returns_none_for_missing_column(self):
        X = pd.DataFrame({"P1": [1, 2, 3]})
        y = np.array([0, 1, 1])

        stats = compute_protein_statistics(X, y, "NonExistent")

        assert stats is None

    def test_cohens_d_sign_correct(self):
        # Cases higher than controls (with some variance)
        X = pd.DataFrame({"P1": [1.0, 1.5, 5.0, 5.5]})
        y = np.array([0, 0, 1, 1])

        stats = compute_protein_statistics(X, y, "P1")

        assert stats["cohens_d"] > 0  # Positive effect (cases > controls)

    def test_p_value_significant_for_large_effect(self):
        # Large separation should yield small p-value
        X = pd.DataFrame({"P1": [0] * 20 + [10] * 20})
        y = np.array([0] * 20 + [1] * 20)

        stats = compute_protein_statistics(X, y, "P1")

        assert stats["p_ttest"] < 0.001  # Highly significant


class TestExtractSelectedProteinsFromKBest:
    """Tests for extract_selected_proteins_from_kbest()."""

    def test_extracts_from_simple_pipeline(self):
        # This is an integration test with sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        X = pd.DataFrame(
            {
                "P1": [1, 2, 3, 4],
                "P2": [10, 20, 30, 40],
                "P3": [0.1, 0.2, 0.3, 0.4],
            }
        )
        y = np.array([0, 0, 1, 1])

        # Build pipeline
        preprocessor = ColumnTransformer([("num", StandardScaler(), ["P1", "P2", "P3"])])

        selector = SelectKBest(score_func=f_classif, k=2)

        pipe = Pipeline(
            [
                ("pre", preprocessor),
                ("sel", selector),
            ]
        )

        pipe.fit(X, y)

        # Extract selected proteins
        selected = extract_selected_proteins_from_kbest(
            pipe, protein_cols=["P1", "P2", "P3"], step_name="sel"
        )

        assert len(selected) == 2
        assert all(p in ["P1", "P2", "P3"] for p in selected)

    def test_returns_empty_for_missing_step(self):
        pipe = Pipeline([("dummy", StandardScaler())])

        selected = extract_selected_proteins_from_kbest(pipe, protein_cols=["P1"], step_name="sel")

        assert selected == []

    def test_returns_empty_for_non_pipeline(self):
        scaler = StandardScaler()

        selected = extract_selected_proteins_from_kbest(
            scaler, protein_cols=["P1"], step_name="sel"
        )

        assert selected == []


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_selection_workflow(self, simple_protein_data):
        X, y = simple_protein_data
        protein_cols = ["P1", "P2", "P3"]

        # Select features
        selected = select_kbest_features(X, y, k=2, protein_cols=protein_cols)

        # Compute statistics for selected features
        stats_list = []
        for protein in selected:
            stats = compute_protein_statistics(X, y, protein)
            if stats:
                stats_list.append(stats)

        assert len(stats_list) == 2
        assert all(s["n_total"] > 0 for s in stats_list)

    def test_selection_reproducibility_across_seeds(self):
        # Same data should give same selection
        rng = np.random.default_rng(123)
        X1 = pd.DataFrame(
            {
                "P1": rng.normal(0, 1, 50),
                "P2": rng.normal(0, 1, 50),
            }
        )
        y1 = np.array([0] * 25 + [1] * 25)

        rng = np.random.default_rng(123)
        X2 = pd.DataFrame(
            {
                "P1": rng.normal(0, 1, 50),
                "P2": rng.normal(0, 1, 50),
            }
        )
        y2 = np.array([0] * 25 + [1] * 25)

        selected1 = select_kbest_features(X1, y1, k=1, protein_cols=["P1", "P2"])
        selected2 = select_kbest_features(X2, y2, k=1, protein_cols=["P1", "P2"])

        assert selected1 == selected2


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
