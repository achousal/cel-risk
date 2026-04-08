"""
Optuna-based hyperparameter search wrapper.

Provides OptunaSearchCV - a drop-in replacement for sklearn's RandomizedSearchCV
that uses Optuna for more efficient hyperparameter optimization.

Features:
- Compatible sklearn interface (best_estimator_, best_params_, best_score_, cv_results_)
- Supports TPE, Random, CMA-ES, and Grid samplers
- Supports Median, Percentile, and Hyperband pruners
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .hyperparams_common import resolve_class_weights_in_params
from .optuna_callbacks import (
    check_tpe_hyperband_trials,
    create_trial_callback,
    log_optimization_summary,
    validate_study_compatibility,
)
from .optuna_utils import (
    create_pruner,
    create_sampler,
    get_optimization_directions,
    select_from_pareto_frontier,
    suggest_params,
)

# Suppress convergence warnings to prevent heavy .err files
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)


class OptunaSearchCV(BaseEstimator):
    """
    Optuna-based hyperparameter search with sklearn-compatible interface.

    Drop-in replacement for RandomizedSearchCV using Optuna's efficient
    hyperparameter optimization algorithms (TPE, CMA-ES, etc.).

    Parameters
    ----------
    estimator : BaseEstimator
        The sklearn estimator or pipeline to tune.
    param_distributions : dict
        Parameter search space. Each value should be a dict with:
        - type: "int", "float", or "categorical"
        - For int/float: low, high, log (optional bool)
        - For categorical: choices (list)
    n_trials : int, default=100
        Number of optimization trials.
    timeout : float, optional
        Stop study after this many seconds.
    scoring : str or callable, default="accuracy"
        Scoring metric for cross-validation.
    cv : int or CV splitter, default=5
        Cross-validation strategy.
    n_jobs : int, default=1
        Number of parallel jobs for CV (not study parallelization).
    random_state : int, optional
        Random seed for CV splitting. Also used as sampler seed if sampler_seed
        is not explicitly provided.
    refit : bool, default=True
        Whether to refit best estimator on full training data.
    direction : {"minimize", "maximize"}, default="maximize"
        Optimization direction.
    sampler : {"tpe", "random", "cmaes", "grid"}, default="tpe"
        Optuna sampler type.
    sampler_seed : int, optional
        Seed for the Optuna sampler. If None, uses random_state instead.
        Setting this explicitly allows different seeds for CV splitting vs
        hyperparameter sampling.
    pruner : {"median", "percentile", "hyperband", "none"}, default="hyperband"
        Optuna pruner type. HyperbandPruner is recommended for TPE sampler.
    pruner_n_startup_trials : int, default=5
        Number of trials before pruning starts.
    pruner_percentile : float, default=25.0
        Percentile threshold for PercentilePruner.
    storage : str, optional
        Optuna storage URL (e.g., "sqlite:///study.db").
    study_name : str, optional
        Name for the Optuna study.
    load_if_exists : bool, default=False
        Load existing study if it exists.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    best_params_ : dict
        Best hyperparameters found.
    best_score_ : float
        Best CV score achieved.
    best_estimator_ : BaseEstimator
        Estimator fitted with best parameters (if refit=True).
    cv_results_ : dict
        Dictionary with trial results (compatible with sklearn).
    study_ : optuna.Study
        The underlying Optuna study object.
    n_trials_ : int
        Number of completed trials.

    Notes
    -----
    Pruning limitation: This implementation uses sklearn's cross_val_score,
    which does not report intermediate values during CV fold evaluation.
    As a result, pruning only takes effect between full CV evaluations (i.e.,
    between trials), not within a trial's CV folds. For full pruning benefits,
    consider using Optuna's native integration or reporting intermediate
    values manually.

    TPE + Hyperband: When using TPE sampler with HyperbandPruner, Optuna
    recommends at least 40 trials for the TPE startup to gather sufficient
    observations. A warning is logged if n_trials < 40 with this combination.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any],
        *,
        n_trials: int = 100,
        timeout: float | None = None,
        scoring: str | Callable = "accuracy",
        cv: int | Any = 5,
        n_jobs: int = 1,
        random_state: int | None = None,
        refit: bool = True,
        direction: Literal["minimize", "maximize"] = "maximize",
        sampler: Literal["tpe", "random", "cmaes", "grid"] = "tpe",
        sampler_seed: int | None = None,
        pruner: Literal["median", "percentile", "hyperband", "none"] = "hyperband",
        pruner_n_startup_trials: int = 5,
        pruner_percentile: float = 25.0,
        storage: str | None = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
        verbose: int = 0,
        multi_objective: bool = False,
        objectives: list[str] | None = None,
        pareto_selection: str = "knee",
        storage_backend: str = "none",
        user_attrs: dict[str, Any] | None = None,
        warm_start_params_file: str | None = None,
        warm_start_top_k: int = 5,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_trials = n_trials
        self.timeout = timeout
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.refit = refit
        self.direction = direction
        self.sampler = sampler
        self.sampler_seed = sampler_seed
        self.pruner = pruner
        self.pruner_n_startup_trials = pruner_n_startup_trials
        self.pruner_percentile = pruner_percentile
        self.storage = storage
        self.study_name = study_name
        self.load_if_exists = load_if_exists
        self.verbose = verbose

        # Multi-objective optimization parameters
        self.multi_objective = multi_objective
        self.objectives = objectives if objectives is not None else ["roc_auc", "neg_brier_score"]
        self.pareto_selection = pareto_selection

        # Study persistence and metadata (Enhancements 1-3)
        self.storage_backend = storage_backend
        self.user_attrs = user_attrs
        self.warm_start_params_file = warm_start_params_file
        self.warm_start_top_k = warm_start_top_k

        # Attributes set during fit
        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = np.nan
        self.best_estimator_: BaseEstimator | None = None
        self.cv_results_: dict[str, list] = {}
        self.study_: optuna.Study | None = None
        self.n_trials_: int = 0

        # Multi-objective attributes (set during fit if multi_objective=True)
        self.pareto_frontier_: list = []
        self.selected_trial_: optuna.Trial | None = None

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        return create_sampler(
            self.sampler,
            self.sampler_seed,
            self.random_state,
            self.param_distributions,
        )

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration."""
        return create_pruner(
            self.pruner,
            self.pruner_n_startup_trials,
            self.pruner_percentile,
        )

    def _resolve_storage(self) -> Any:
        """Build Optuna storage object from config.

        Returns None for in-memory (no persistence), a JournalStorage for
        append-only concurrent access, or a raw URL string for sqlite.
        """
        if not self.storage or self.storage_backend == "none":
            return None
        if self.storage_backend == "journal":
            from pathlib import Path

            from optuna.storages import JournalFileStorage, JournalStorage

            # Ensure parent directory exists
            Path(self.storage).parent.mkdir(parents=True, exist_ok=True)
            lock_obj = optuna.storages.JournalFileOpenLock(self.storage)
            return JournalStorage(JournalFileStorage(self.storage, lock_obj=lock_obj))
        # sqlite or raw URL passthrough
        return self.storage

    def _enqueue_warm_start_trials(self) -> None:
        """Enqueue pre-computed scout trials as starting points.

        Reads a JSON file mapping model names to lists of param dicts.
        Filters to params that exist in the current search space and
        enqueues up to warm_start_top_k trials.
        """
        import json

        if not self.warm_start_params_file or self.study_ is None:
            return

        try:
            with open(self.warm_start_params_file) as f:
                all_params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("[optuna] Failed to load warm-start params: %s", e)
            return

        # Match by model name in user_attrs
        model = (self.user_attrs or {}).get("model")
        if not model or model not in all_params:
            logger.debug("[optuna] No warm-start params for model '%s'", model)
            return

        enqueued = 0
        for params in all_params[model][: self.warm_start_top_k]:
            # Filter to params that exist in current search space
            valid = {k: v for k, v in params.items() if k in self.param_distributions}
            if valid:
                self.study_.enqueue_trial(valid)
                enqueued += 1

        if enqueued:
            logger.info("[optuna] Enqueued %d warm-start trials for model '%s'", enqueued, model)

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest hyperparameters for a trial based on param_distributions."""
        return suggest_params(trial, self.param_distributions)

    def fit(self, X, y, **fit_params) -> OptunaSearchCV:  # noqa: ARG002
        """
        Run Optuna hyperparameter optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **fit_params : dict
            Additional fit parameters (currently unused).

        Returns
        -------
        self : OptunaSearchCV
            Fitted instance.
        """
        # Keep X as-is (DataFrame or array) for pipeline compatibility
        # ColumnTransformer with string column names requires DataFrame input
        X_arr = X
        # Convert y to array for safe indexing
        y_arr = np.asarray(y)

        # Setup CV splitter
        if isinstance(self.cv, int):
            cv_splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv_splitter = self.cv

        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Warn about TPE + Hyperband needing sufficient trials for startup
        check_tpe_hyperband_trials(self.sampler, self.pruner, self.n_trials)

        # Set optuna verbosity
        if self.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        # Resolve storage backend
        resolved_storage = self._resolve_storage()

        # Create or load study (multi-objective or single-objective)
        if self.multi_objective:
            directions = get_optimization_directions(self.objectives)
            self.study_ = optuna.create_study(
                study_name=self.study_name,
                storage=resolved_storage,
                load_if_exists=self.load_if_exists,
                directions=directions,  # List for multi-objective
                sampler=sampler,
                pruner=pruner,
            )
        else:
            self.study_ = optuna.create_study(
                study_name=self.study_name,
                storage=resolved_storage,
                load_if_exists=self.load_if_exists,
                direction=self.direction,
                sampler=sampler,
                pruner=pruner,
            )

        # Tag study with factorial metadata (Enhancement 2)
        if self.user_attrs:
            for key, value in self.user_attrs.items():
                self.study_.set_user_attr(key, value)

        # Enqueue warm-start trials from scout (Enhancement 1)
        self._enqueue_warm_start_trials()

        # Validate existing study compatibility
        validate_study_compatibility(
            self.study_,
            self.multi_objective,
            self.direction,
            self.load_if_exists,
        )

        # Create objective with CV splitter
        if self.multi_objective:
            # Worst-case sentinel for failed trials. TrialPruned stores None
            # values that crash Optuna TPE sampler (>=4.0) in
            # _calculate_weights_below_for_multi_objective, so return explicit
            # worst-case scores: AUROC=0.0 (worst), neg_brier=-1.0 (worst).
            _WORST_MO: tuple[float, float] = (0.0, -1.0)

            def objective(trial: optuna.Trial) -> tuple[float, float]:
                params = resolve_class_weights_in_params(
                    self._suggest_params(trial),
                    y_arr,
                )
                estimator = clone(self.estimator)
                try:
                    estimator.set_params(**params)
                except ValueError as e:
                    logger.warning(f"[optuna] Invalid params {params}: {e}")
                    return _WORST_MO

                try:
                    means, auroc_folds, brier_folds = self._multi_objective_cv_score(
                        estimator, X_arr, y_arr, cv_splitter
                    )
                    # Store per-fold metrics as trial user attrs (Enhancement 4)
                    trial.set_user_attr("fold_aurocs", [float(s) for s in auroc_folds])
                    trial.set_user_attr("fold_briers", [float(s) for s in brier_folds])
                    trial.set_user_attr("auroc_std", float(np.std(auroc_folds)))
                    trial.set_user_attr("brier_std", float(np.std(brier_folds)))
                    return means
                except Exception as e:
                    logger.warning(f"[optuna] CV failed for params {params}: {e}")
                    return _WORST_MO

        else:

            def objective(trial: optuna.Trial) -> float:
                params = resolve_class_weights_in_params(
                    self._suggest_params(trial),
                    y_arr,
                )
                estimator = clone(self.estimator)
                try:
                    estimator.set_params(**params)
                except ValueError as e:
                    logger.warning(f"[optuna] Invalid params {params}: {e}")
                    raise optuna.TrialPruned() from e

                try:
                    scores = cross_val_score(
                        estimator,
                        X_arr,
                        y_arr,
                        cv=cv_splitter,
                        scoring=self.scoring,
                        n_jobs=self.n_jobs,
                    )
                    # Store per-fold scores as trial user attrs (Enhancement 4)
                    trial.set_user_attr("fold_scores", [float(s) for s in scores])
                    trial.set_user_attr("score_std", float(np.std(scores)))
                    return float(np.mean(scores))
                except Exception as e:
                    logger.warning(f"[optuna] CV failed for params {params}: {e}")
                    raise optuna.TrialPruned() from e

        # Add trial callback for progress logging
        logger.info(
            f"Hyperparameter search: {self.n_trials} trials (sampler={self.sampler}, pruner={self.pruner})"
        )

        trial_callback_fn = create_trial_callback(self.n_trials, self.multi_objective)

        # Run optimization
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose > 0,
            callbacks=[trial_callback_fn] if logger.isEnabledFor(logging.INFO) else None,
        )

        # Extract results
        self.n_trials_ = len(self.study_.trials)

        # Check if any trials completed successfully
        completed_trials = [
            t for t in self.study_.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            raise RuntimeError(
                f"All {self.n_trials_} Optuna trials failed. "
                "Check logs for error messages. This may indicate incompatible "
                "hyperparameters, insufficient data, or other issues."
            )

        # Extract best parameters (Pareto frontier selection for multi-objective)
        if self.multi_objective:
            self._select_from_pareto_frontier()
        else:
            self.best_params_ = self.study_.best_params
            self.best_score_ = self.study_.best_value

        # Build cv_results_ for sklearn compatibility
        self.cv_results_ = self._build_cv_results()

        # Refit best estimator
        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            refit_params = resolve_class_weights_in_params(self.best_params_, y_arr)
            self.best_estimator_.set_params(**refit_params)
            self.best_estimator_.fit(X_arr, y_arr)

        # Log summary with hyperparameter importance
        log_optimization_summary(self.study_, self.best_score_, self.multi_objective)

        return self

    def _build_cv_results(self) -> dict[str, list]:
        """Build sklearn-compatible cv_results_ from Optuna study.

        For multi-objective studies, uses the first objective (AUROC) as
        the primary score for ranking purposes.
        """
        if self.study_ is None:
            return {}

        results: dict[str, list] = {
            "mean_test_score": [],
            "rank_test_score": [],
            "params": [],
        }

        # Add param columns
        for param_name in self.param_distributions.keys():
            results[f"param_{param_name}"] = []

        # Populate from trials
        for trial in self.study_.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Handle multi-objective (use first objective as primary score)
                if self.multi_objective:
                    results["mean_test_score"].append(trial.values[0])  # AUROC
                else:
                    results["mean_test_score"].append(trial.value)

                results["params"].append(trial.params)

                for param_name in self.param_distributions.keys():
                    results[f"param_{param_name}"].append(trial.params.get(param_name))

        # Compute ranks
        if results["mean_test_score"]:
            scores = np.array(results["mean_test_score"])
            if self.multi_objective:
                # For multi-objective, first objective is always maximized
                ranks = np.argsort(np.argsort(-scores)) + 1
            elif self.direction == "maximize":
                ranks = np.argsort(np.argsort(-scores)) + 1
            else:
                ranks = np.argsort(np.argsort(scores)) + 1
            results["rank_test_score"] = ranks.tolist()

        return results

    def predict(self, X):
        """Predict using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError(
                f"The estimator {type(self.best_estimator_).__name__} does not have a predict_proba method."
            )
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        """Return the score of the best estimator on the given data."""
        if self.best_estimator_ is None:
            raise ValueError("Estimator not fitted. Call fit() first.")
        return self.best_estimator_.score(X, y)

    def get_trials_dataframe(self):
        """Return trials as a pandas DataFrame (convenience wrapper)."""
        if self.study_ is None:
            raise ValueError("Study not created. Call fit() first.")
        return self.study_.trials_dataframe()

    def _multi_objective_cv_score(
        self, estimator, X, y, cv_splitter
    ) -> tuple[tuple[float, float], list[float], list[float]]:
        """Compute AUROC and Brier score across CV folds.

        Performs manual CV loop to compute both metrics, required because
        sklearn's cross_val_score only supports single scoring metric.

        Parameters
        ----------
        estimator : BaseEstimator
            Unfitted sklearn estimator or pipeline.
        X : array-like
            Training features.
        y : array-like
            Training labels.
        cv_splitter : CV splitter
            Cross-validation strategy.

        Returns
        -------
        tuple[tuple[float, float], list[float], list[float]]
            ((auroc_mean, neg_brier_mean), auroc_per_fold, brier_per_fold).
            Note: Brier score in the mean tuple is negated for Optuna maximization;
            brier_per_fold contains raw (positive) Brier scores.

        Raises
        ------
        ValueError
            If all CV folds have single class (cannot compute metrics).
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score

        auroc_scores = []
        brier_scores = []

        for train_idx, val_idx in cv_splitter.split(X, y):
            # Handle DataFrame and array indexing
            if hasattr(X, "iloc"):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone and fit estimator for this fold
            estimator_fold = clone(estimator)
            estimator_fold.fit(X_train, y_train)

            # Predict probabilities
            y_pred = estimator_fold.predict_proba(X_val)[:, 1]

            # Compute metrics (skip single-class folds)
            try:
                auroc = roc_auc_score(y_val, y_pred)
                brier = brier_score_loss(y_val, y_pred)
                auroc_scores.append(auroc)
                brier_scores.append(brier)
            except ValueError:
                # Single-class fold, skip
                continue

        if not auroc_scores:
            raise ValueError("All CV folds had single class, cannot compute metrics")

        # Return means + per-fold arrays
        means = (float(np.mean(auroc_scores)), -float(np.mean(brier_scores)))
        return means, auroc_scores, brier_scores

    def _select_from_pareto_frontier(self):
        """Select best model from Pareto frontier using configured strategy.

        Extracts Pareto-optimal trials from multi-objective study and selects
        a single "best" trial based on pareto_selection strategy.

        Sets attributes:
        - best_params_: Parameters of selected trial
        - best_score_: AUROC of selected trial (primary metric)
        - pareto_frontier_: List of all Pareto-optimal trials
        - selected_trial_: The selected trial object

        Raises
        ------
        RuntimeError
            If no completed trials in Pareto frontier.
        ValueError
            If unknown pareto_selection strategy.
        """
        pareto_trials = [
            t for t in self.study_.best_trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if not pareto_trials:
            raise RuntimeError(
                f"No completed trials in Pareto frontier. "
                f"All {self.n_trials_} trials may have failed or been pruned."
            )

        logger.info(
            f"[optuna] Pareto frontier has {len(pareto_trials)} trials. "
            f"Selecting using strategy: {self.pareto_selection}"
        )

        selected = select_from_pareto_frontier(pareto_trials, self.pareto_selection)

        self.best_params_ = selected.params
        self.best_score_ = selected.values[0]  # AUROC as primary metric
        self.pareto_frontier_ = pareto_trials
        self.selected_trial_ = selected

        logger.info(
            f"[optuna] Selected trial {selected.number}: "
            f"AUROC={selected.values[0]:.4f}, Brier={-selected.values[1]:.4f}"
        )

    def get_pareto_frontier(self):
        """Get Pareto frontier as DataFrame for analysis/visualization.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - trial_number: Trial index
            - auroc: AUROC score
            - brier_score: Brier score (positive, lower is better)
            - params: Dictionary of hyperparameters
            - is_selected: Whether this trial was selected by pareto_selection

        Raises
        ------
        ValueError
            If called on single-objective study or before fit().
        """
        import pandas as pd

        if not self.multi_objective:
            raise ValueError("get_pareto_frontier() only available for multi-objective studies")

        if not self.pareto_frontier_:
            raise ValueError("No Pareto frontier available. Call fit() first.")

        trials = self.pareto_frontier_
        return pd.DataFrame(
            {
                "trial_number": [t.number for t in trials],
                "auroc": [t.values[0] for t in trials],
                "brier_score": [-t.values[1] for t in trials],
                "params": [t.params for t in trials],
                "is_selected": [t.number == self.selected_trial_.number for t in trials],
            }
        )
