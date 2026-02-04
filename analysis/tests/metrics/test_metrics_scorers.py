"""
Tests for custom scorers (metrics/scorers.py).
"""

import numpy as np
from ced_ml.metrics.scorers import (
    get_scorer,
    make_tpr_at_fpr_scorer,
    tpr_at_fpr_score,
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class TestTPRAtFPRScore:
    """Test tpr_at_fpr_score function."""

    def test_perfect_separation(self):
        """Perfect classifier should achieve TPR=1.0 at any FPR."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])

        tpr = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert tpr == 1.0

    def test_random_classifier(self):
        """Random classifier should have TPR close to target_fpr."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 1000)
        y_score = rng.rand(1000)

        tpr = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert 0.0 <= tpr <= 0.2

    def test_impossible_constraint(self):
        """Should return 0.0 when no threshold achieves target_fpr."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])

        tpr = tpr_at_fpr_score(y_true, y_score, target_fpr=0.0)
        assert tpr == 0.0

    def test_different_fpr_targets(self):
        """Higher FPR threshold should allow higher TPR."""
        _rng = np.random.RandomState(42)  # noqa: F841
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]

        tpr_at_5 = tpr_at_fpr_score(y, y_score, target_fpr=0.05)
        tpr_at_10 = tpr_at_fpr_score(y, y_score, target_fpr=0.10)

        assert tpr_at_10 >= tpr_at_5


class TestMakeTPRAtFPRScorer:
    """Test make_tpr_at_fpr_scorer function."""

    def test_scorer_creation(self):
        """Scorer should be created successfully."""
        scorer = make_tpr_at_fpr_scorer(target_fpr=0.05)
        assert callable(scorer)

    def test_scorer_with_cross_validation(self):
        """Scorer should work with cross-validation."""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
        clf = LogisticRegression(random_state=42)
        scorer = make_tpr_at_fpr_scorer(target_fpr=0.05)

        scores = cross_val_score(clf, X, y, cv=3, scoring=scorer)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_different_fpr_targets(self):
        """Different FPR targets should produce different scorers."""
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
        clf = LogisticRegression(random_state=42)

        scorer_5 = make_tpr_at_fpr_scorer(target_fpr=0.05)
        scorer_10 = make_tpr_at_fpr_scorer(target_fpr=0.10)

        scores_5 = cross_val_score(clf, X, y, cv=3, scoring=scorer_5)
        scores_10 = cross_val_score(clf, X, y, cv=3, scoring=scorer_10)

        assert np.mean(scores_10) >= np.mean(scores_5)


class TestGetScorer:
    """Test get_scorer function."""

    def test_custom_scorer_tpr_at_fpr(self):
        """Should return custom scorer for tpr_at_fpr."""
        scorer = get_scorer("tpr_at_fpr", target_fpr=0.05)
        assert callable(scorer)

    def test_custom_scorer_default_fpr(self):
        """Should use default target_fpr=0.05 when not specified."""
        scorer = get_scorer("tpr_at_fpr")
        assert callable(scorer)

        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = LogisticRegression(random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring=scorer)
        assert len(scores) == 3

    def test_standard_sklearn_scorer(self):
        """Should pass through standard sklearn scorer names."""
        scorer = get_scorer("roc_auc")
        assert scorer == "roc_auc"

        scorer = get_scorer("accuracy")
        assert scorer == "accuracy"

        scorer = get_scorer("average_precision")
        assert scorer == "average_precision"

    def test_custom_fpr_value(self):
        """Should handle custom target_fpr values."""
        scorer = get_scorer("tpr_at_fpr", target_fpr=0.10)
        assert callable(scorer)

        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = LogisticRegression(random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring=scorer)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestScorerIntegration:
    """Integration tests for scorers with real ML workflows."""

    def test_scorer_with_imbalanced_data(self):
        """Scorer should work with imbalanced datasets."""
        _rng = np.random.RandomState(42)  # noqa: F841
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            weights=[0.95, 0.05],
            random_state=42,
        )
        clf = LogisticRegression(random_state=42)
        scorer = make_tpr_at_fpr_scorer(target_fpr=0.05)

        scores = cross_val_score(clf, X, y, cv=5, scoring=scorer)
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_scorer_stability(self):
        """Scorer should produce consistent results with same random state."""
        from sklearn.model_selection import StratifiedKFold

        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, random_state=42)
        clf = LogisticRegression(random_state=42)
        scorer = make_tpr_at_fpr_scorer(target_fpr=0.05)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        scores1 = cross_val_score(clf, X, y, cv=cv, scoring=scorer)
        scores2 = cross_val_score(clf, X, y, cv=cv, scoring=scorer)

        np.testing.assert_array_equal(scores1, scores2)


class TestScorerEdgeCases:
    """Test edge cases in scorer behavior."""

    def test_single_class_all_negatives(self):
        """Scorer should return NaN for all-negative labels (single-class guard)."""
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        # Returns NaN for consistency with other metrics (AUROC, PR-AUC, etc.)
        result = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert np.isnan(result)

    def test_single_class_all_positives(self):
        """Scorer should return NaN for all-positive labels (single-class guard)."""
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])

        # Returns NaN for consistency with other metrics
        result = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert np.isnan(result)

    def test_very_small_dataset(self):
        """Scorer should handle very small datasets."""
        y_true = np.array([0, 1])
        y_score = np.array([0.3, 0.7])

        result = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert 0.0 <= result <= 1.0

    def test_ties_in_scores(self):
        """Scorer should handle tied prediction scores."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # All tied scores should still compute
        result = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert 0.0 <= result <= 1.0

    def test_extreme_fpr_values(self):
        """Scorer should handle extreme FPR values."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        # FPR=0.0 (perfect specificity)
        result_zero = tpr_at_fpr_score(y_true, y_score, target_fpr=0.0)
        assert 0.0 <= result_zero <= 1.0

        # FPR=1.0 (no specificity constraint)
        result_one = tpr_at_fpr_score(y_true, y_score, target_fpr=1.0)
        assert result_one == 1.0  # Can achieve perfect sensitivity

    def test_inverted_labels(self):
        """Scorer should handle inverted prediction scores."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Inverted: low scores for positives, high for negatives
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        # Should still compute without error (just poor performance)
        result = tpr_at_fpr_score(y_true, y_score, target_fpr=0.05)
        assert 0.0 <= result <= 1.0
