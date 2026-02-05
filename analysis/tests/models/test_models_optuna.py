"""
Basic tests for Optuna hyperparameter search module.

Tests cover:
- OptunaSearchCV basic instantiation
- Basic fitting workflow

Note: Comprehensive Optuna tests should be added in future iterations.
This provides basic smoke testing to prevent regressions.
"""

import logging

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ced_ml.models.optuna_search import OptunaSearchCV
from ced_ml.models.optuna_utils import DEFAULT_SEED_FALLBACK


class TestOptunaSearchCVBasic:
    """Basic smoke tests for OptunaSearchCV."""

    def test_instantiation(self):
        """OptunaSearchCV can be instantiated."""
        estimator = LogisticRegression()
        param_distributions = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=5,
            random_state=42,
        )

        assert search.estimator is estimator
        assert search.n_trials == 5
        assert search.random_state == 42

    def test_fit_basic(self):
        """OptunaSearchCV can fit on toy data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )

        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=3,
            cv=2,
            random_state=42,
        )

        search.fit(X, y)

        # Check basic sklearn interface attributes exist
        assert hasattr(search, "best_estimator_")
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert search.best_score_ >= 0.0

    def test_different_samplers(self):
        """OptunaSearchCV supports different sampler types."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        for sampler in ["tpe", "random"]:
            search = OptunaSearchCV(
                estimator=estimator,
                param_distributions=param_distributions,
                n_trials=2,
                cv=2,
                sampler=sampler,
                random_state=42,
            )
            search.fit(X, y)
            assert search.best_score_ >= 0.0

    def test_pruning(self):
        """OptunaSearchCV supports pruning."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=5,
            cv=3,
            pruner="median",
            pruner_n_startup_trials=2,
            random_state=42,
        )
        search.fit(X, y)
        assert search.best_score_ >= 0.0

    def test_seed_fallback_logs_warning(self, caplog):
        """OptunaSearchCV logs warning when no seed is provided."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        # Create search without random_state or sampler_seed
        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=2,
            cv=2,
            # No random_state or sampler_seed provided
        )

        # Re-enable propagation on ced_ml logger so caplog can capture
        ced_ml_logger = logging.getLogger("ced_ml")
        orig_propagate = ced_ml_logger.propagate
        ced_ml_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="ced_ml.models.optuna_search"):
                search.fit(X, y)
        finally:
            ced_ml_logger.propagate = orig_propagate

        # Verify warning was logged about seed fallback
        assert any(
            "sampler_seed and random_state are None" in record.message
            and f"seed={DEFAULT_SEED_FALLBACK}" in record.message
            for record in caplog.records
        ), "Expected warning about seed fallback not found in logs"

        # Search should still complete successfully
        assert search.best_score_ >= 0.0

    def test_no_seed_warning_with_random_state(self, caplog):
        """OptunaSearchCV does not warn when random_state is provided."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=2,
            cv=2,
            random_state=42,  # Explicit random_state
        )

        with caplog.at_level(logging.WARNING):
            search.fit(X, y)

        # Verify no warning about seed fallback
        seed_warnings = [
            r for r in caplog.records if "sampler_seed and random_state are None" in r.message
        ]
        assert len(seed_warnings) == 0, "Unexpected seed fallback warning"

    def test_no_seed_warning_with_sampler_seed(self, caplog):
        """OptunaSearchCV does not warn when sampler_seed is provided."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        estimator = LogisticRegression(max_iter=100)
        param_distributions = {
            "C": {"type": "float", "low": 0.1, "high": 1.0},
        }

        search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_trials=2,
            cv=2,
            sampler_seed=123,  # Explicit sampler_seed
        )

        with caplog.at_level(logging.WARNING):
            search.fit(X, y)

        # Verify no warning about seed fallback
        seed_warnings = [
            r for r in caplog.records if "sampler_seed and random_state are None" in r.message
        ]
        assert len(seed_warnings) == 0, "Unexpected seed fallback warning"
