"""
Unit tests for multi-objective Optuna optimization.

Tests cover:
- Config validation
- Multi-objective study creation
- Pareto frontier selection strategies
- DataFrame export
- Backward compatibility
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ced_ml.config.schema import OptunaConfig
from ced_ml.models.optuna_search import OptunaSearchCV


@pytest.fixture
def toy_data():
    """Generate small toy dataset for fast testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        class_sep=1.5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def simple_param_dist():
    """Simple parameter distribution for LogisticRegression."""
    return {
        "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
    }


class TestOptunaConfigValidation:
    """Test multi-objective configuration validation."""

    def test_multi_objective_requires_two_objectives(self):
        """Multi-objective=True with <2 objectives should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2 objectives"):
            OptunaConfig(multi_objective=True, objectives=["roc_auc"])

    def test_single_objective_allowed(self):
        """Single objective with multi_objective=False should work."""
        config = OptunaConfig(multi_objective=False, objectives=["roc_auc"])
        assert config.multi_objective is False

    def test_unsupported_objective_raises_error(self):
        """Unsupported objective should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported objective"):
            OptunaConfig(multi_objective=True, objectives=["roc_auc", "unsupported_metric"])

    def test_supported_objectives(self):
        """All supported objectives should validate."""
        config = OptunaConfig(multi_objective=True, objectives=["roc_auc", "neg_brier_score"])
        assert config.objectives == ["roc_auc", "neg_brier_score"]

    def test_cmaes_with_multiobjective_warns(self, caplog):
        """CMA-ES with multi-objective should log warning."""
        import logging

        # Re-enable propagation on ced_ml logger so caplog can capture
        ced_ml_logger = logging.getLogger("ced_ml")
        orig_propagate = ced_ml_logger.propagate
        ced_ml_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="ced_ml.config.schema"):
                OptunaConfig(
                    multi_objective=True,
                    objectives=["roc_auc", "neg_brier_score"],
                    sampler="cmaes",
                )
        finally:
            ced_ml_logger.propagate = orig_propagate
        assert "CMA-ES sampler with multi-objective may be unstable" in caplog.text


class TestOptunaSearchCVMultiObjective:
    """Test OptunaSearchCV with multi-objective optimization."""

    def test_init_with_multi_objective_params(self):
        """OptunaSearchCV should accept multi-objective parameters."""
        estimator = LogisticRegression()
        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions={"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=5,
            multi_objective=True,
            objectives=["roc_auc", "neg_brier_score"],
            pareto_selection="knee",
        )
        assert search.multi_objective is True
        assert search.objectives == ["roc_auc", "neg_brier_score"]
        assert search.pareto_selection == "knee"

    def test_multi_objective_fit(self, toy_data, simple_param_dist):
        """Multi-objective fit should work and return Pareto frontier."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=5,
            cv=2,
            random_state=42,
            multi_objective=True,
            objectives=["roc_auc", "neg_brier_score"],
            pareto_selection="knee",
        )

        search.fit(X, y)

        # Check multi-objective results
        assert hasattr(search, "pareto_frontier_")
        assert len(search.pareto_frontier_) > 0
        assert hasattr(search, "selected_trial_")
        assert search.selected_trial_ is not None
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")

    def test_get_directions(self, simple_param_dist):
        """get_optimization_directions should return correct optimization directions."""
        from ced_ml.models.optuna_utils import get_optimization_directions

        objectives = ["roc_auc", "neg_brier_score"]
        directions = get_optimization_directions(objectives)
        assert directions == ["maximize", "maximize"]

    def test_multi_objective_cv_score_returns_tuple(self, toy_data):
        """_multi_objective_cv_score should return tuple of (AUROC, -Brier)."""
        from sklearn.model_selection import StratifiedKFold

        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions={},
            n_trials=1,
            multi_objective=True,
        )

        scores = search._multi_objective_cv_score(estimator, X, y, cv)

        assert isinstance(scores, tuple)
        assert len(scores) == 2
        assert isinstance(scores[0], float)  # AUROC
        assert isinstance(scores[1], float)  # -Brier
        assert 0 <= scores[0] <= 1  # AUROC in [0, 1]
        assert scores[1] <= 0  # -Brier should be negative


class TestParetoFrontierSelection:
    """Test Pareto frontier selection strategies."""

    def test_knee_point_selection(self, toy_data, simple_param_dist):
        """Knee point selection should find balanced tradeoff."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=10,
            cv=2,
            random_state=42,
            multi_objective=True,
            pareto_selection="knee",
        )

        search.fit(X, y)

        # Knee point should be selected
        assert search.selected_trial_ is not None
        assert search.selected_trial_ in search.pareto_frontier_

    def test_extreme_auroc_selection(self, toy_data, simple_param_dist):
        """Extreme AUROC selection should select trial with highest AUROC."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=10,
            cv=2,
            random_state=42,
            multi_objective=True,
            pareto_selection="extreme_auroc",
        )

        search.fit(X, y)

        # Should select trial with max AUROC
        max_auroc_trial = max(search.pareto_frontier_, key=lambda t: t.values[0])
        assert search.selected_trial_ == max_auroc_trial

    def test_balanced_selection(self, toy_data, simple_param_dist):
        """Balanced selection should maximize sum of normalized objectives."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=10,
            cv=2,
            random_state=42,
            multi_objective=True,
            pareto_selection="balanced",
        )

        search.fit(X, y)

        # Should select a trial from Pareto frontier
        assert search.selected_trial_ in search.pareto_frontier_

    def test_invalid_pareto_selection_raises_error(self, toy_data, simple_param_dist):
        """Invalid pareto_selection strategy should raise ValueError."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=5,
            cv=2,
            random_state=42,
            multi_objective=True,
            pareto_selection="invalid_strategy",
        )

        with pytest.raises(ValueError, match="Unknown pareto_selection"):
            search.fit(X, y)


class TestParetoFrontierDataFrame:
    """Test Pareto frontier DataFrame export."""

    def test_get_pareto_frontier_returns_dataframe(self, toy_data, simple_param_dist):
        """get_pareto_frontier should return DataFrame with correct structure."""
        import pandas as pd

        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=10,
            cv=2,
            random_state=42,
            multi_objective=True,
            pareto_selection="knee",
        )

        search.fit(X, y)
        df = search.get_pareto_frontier()

        assert isinstance(df, pd.DataFrame)
        assert "trial_number" in df.columns
        assert "auroc" in df.columns
        assert "brier_score" in df.columns
        assert "params" in df.columns
        assert "is_selected" in df.columns
        assert len(df) == len(search.pareto_frontier_)
        assert df["is_selected"].sum() == 1  # Exactly one trial selected

    def test_get_pareto_frontier_single_objective_raises_error(self, toy_data):
        """get_pareto_frontier on single-objective study should raise ValueError."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions={"C": {"type": "float", "low": 0.01, "high": 10.0}},
            n_trials=5,
            cv=2,
            random_state=42,
            multi_objective=False,  # Single-objective
        )

        search.fit(X, y)

        with pytest.raises(ValueError, match="only available for multi-objective"):
            search.get_pareto_frontier()

    def test_get_pareto_frontier_before_fit_raises_error(self, simple_param_dist):
        """get_pareto_frontier before fit should raise ValueError."""
        estimator = LogisticRegression()

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=5,
            multi_objective=True,
        )

        with pytest.raises(ValueError, match="Call fit\\(\\) first"):
            search.get_pareto_frontier()


class TestBackwardCompatibility:
    """Test backward compatibility with single-objective optimization."""

    def test_single_objective_still_works(self, toy_data, simple_param_dist):
        """Single-objective optimization should work unchanged."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=5,
            cv=2,
            random_state=42,
            multi_objective=False,  # Explicitly single-objective
            direction="maximize",
        )

        search.fit(X, y)

        # Single-objective results
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert hasattr(search, "study_")
        assert not hasattr(search, "pareto_frontier_") or len(search.pareto_frontier_) == 0

    def test_default_is_single_objective(self, toy_data, simple_param_dist):
        """Default behavior should be single-objective (backward compatible)."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=5,
            cv=2,
            random_state=42,
            # No multi_objective parameter - should default to False
        )

        search.fit(X, y)

        assert search.multi_objective is False
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skip(reason="Difficult to reliably trigger all trials failing in multi-objective")
    def test_all_trials_fail(self, toy_data):
        """All failed trials should raise RuntimeError.

        Note: This test is skipped because it's difficult to reliably create
        a scenario where all multi-objective trials fail without depending on
        implementation details of specific estimators.
        """
        pass

    def test_pareto_frontier_with_min_trials(self, toy_data, simple_param_dist):
        """Pareto frontier should work with minimal number of trials."""
        X, y = toy_data
        estimator = LogisticRegression(max_iter=500)

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=simple_param_dist,
            n_trials=2,  # Minimal trials
            cv=2,
            random_state=42,
            multi_objective=True,
        )

        search.fit(X, y)

        assert len(search.pareto_frontier_) >= 1
        assert search.selected_trial_ is not None
