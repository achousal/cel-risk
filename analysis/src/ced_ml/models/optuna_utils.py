"""
Optuna utility functions for parameter space and Pareto frontier selection.

Provides helper functions for parameter space handling, multi-objective
optimization, and Pareto frontier selection strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PercentilePruner,
)
from optuna.samplers import (
    CmaEsSampler,
    GridSampler,
    RandomSampler,
    TPESampler,
)

logger = logging.getLogger(__name__)

# Default seed used when neither sampler_seed nor random_state is provided.
# Named constant to avoid magic number and make the behavior explicit.
DEFAULT_SEED_FALLBACK = 0


def create_sampler(
    sampler_type: str,
    sampler_seed: int | None,
    random_state: int | None,
    param_distributions: dict[str, Any] | None = None,
) -> optuna.samplers.BaseSampler:
    """
    Create Optuna sampler based on configuration.

    Parameters
    ----------
    sampler_type : str
        Type of sampler ("tpe", "random", "cmaes", "grid").
    sampler_seed : int, optional
        Explicit seed for sampler.
    random_state : int, optional
        Random state (used as fallback if sampler_seed is None).
    param_distributions : dict, optional
        Parameter distributions (required for grid sampler).

    Returns
    -------
    optuna.samplers.BaseSampler
        Configured sampler instance.

    Raises
    ------
    ValueError
        If sampler_type is unknown or if grid sampler is requested
        without param_distributions.
    """
    # Determine seed with explicit precedence
    if sampler_seed is not None:
        seed = sampler_seed
    elif random_state is not None:
        seed = random_state
    else:
        seed = DEFAULT_SEED_FALLBACK
        logger.warning(
            "Both sampler_seed and random_state are None. "
            "Defaulting to seed=%d for determinism. "
            "Consider setting random_state explicitly for reproducibility.",
            DEFAULT_SEED_FALLBACK,
        )

    if sampler_type == "tpe":
        return TPESampler(seed=seed)
    elif sampler_type == "random":
        return RandomSampler(seed=seed)
    elif sampler_type == "cmaes":
        return CmaEsSampler(seed=seed)
    elif sampler_type == "grid":
        if param_distributions is None:
            raise ValueError("Grid sampler requires param_distributions")
        # Grid sampler requires explicit search space
        search_space = build_grid_search_space(param_distributions)
        return GridSampler(search_space, seed=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_type}")


def create_pruner(
    pruner_type: str,
    pruner_n_startup_trials: int = 5,
    pruner_percentile: float = 25.0,
) -> optuna.pruners.BasePruner:
    """
    Create Optuna pruner based on configuration.

    Parameters
    ----------
    pruner_type : str
        Type of pruner ("median", "percentile", "hyperband", "none").
    pruner_n_startup_trials : int, default=5
        Number of trials before pruning starts.
    pruner_percentile : float, default=25.0
        Percentile threshold for PercentilePruner.

    Returns
    -------
    optuna.pruners.BasePruner
        Configured pruner instance.

    Raises
    ------
    ValueError
        If pruner_type is unknown.
    """
    if pruner_type == "median":
        return MedianPruner(n_startup_trials=pruner_n_startup_trials)
    elif pruner_type == "percentile":
        return PercentilePruner(
            percentile=pruner_percentile,
            n_startup_trials=pruner_n_startup_trials,
        )
    elif pruner_type == "hyperband":
        return HyperbandPruner()
    elif pruner_type == "none":
        return NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_type}")


def build_grid_search_space(param_distributions: dict[str, Any]) -> dict[str, list]:
    """
    Build search space for GridSampler from param_distributions.

    Parameters
    ----------
    param_distributions : dict
        Parameter distributions with type specifications.

    Returns
    -------
    dict[str, list]
        Grid search space with discrete values for each parameter.

    Raises
    ------
    ValueError
        If parameter type is incompatible with grid search or if
        log scale is used with non-positive values.
    """
    search_space = {}
    for name, spec in param_distributions.items():
        if spec.get("type") == "categorical":
            search_space[name] = spec["choices"]
        elif spec.get("type") in ("int", "float"):
            # Create a small grid for continuous params
            low, high = spec["low"], spec["high"]
            if spec.get("log", False):
                if low <= 0:
                    raise ValueError(
                        f"Cannot use log scale for param {name}: low={low} must be > 0"
                    )
                values = np.logspace(np.log10(low), np.log10(high), num=5).tolist()
            else:
                values = np.linspace(low, high, num=5).tolist()
            if spec["type"] == "int":
                values = sorted({int(v) for v in values})
            search_space[name] = values
        else:
            raise ValueError(f"Cannot build grid for param {name}: {spec}")
    return search_space


def suggest_params(trial: optuna.Trial, param_distributions: dict[str, Any]) -> dict[str, Any]:
    """
    Suggest hyperparameters for a trial based on param_distributions.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    param_distributions : dict
        Parameter distributions with type specifications.

    Returns
    -------
    dict[str, Any]
        Suggested parameter values for the trial.

    Raises
    ------
    ValueError
        If parameter type is unknown.
    """
    params = {}
    for name, spec in param_distributions.items():
        param_type = spec.get("type", "categorical")

        if param_type == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            # Handle unhashable types (dicts) by converting to/from tuples
            choices = spec["choices"]
            hashable_choices = []
            for choice in choices:
                if isinstance(choice, dict):
                    # Convert dict to tuple of tuples for hashing
                    hashable_choices.append(tuple(sorted(choice.items())))
                else:
                    hashable_choices.append(choice)

            suggested = trial.suggest_categorical(name, hashable_choices)

            # Convert back to dict if needed
            if isinstance(suggested, tuple) and all(
                isinstance(item, tuple) and len(item) == 2 for item in suggested
            ):
                params[name] = dict(suggested)
            else:
                params[name] = suggested
        else:
            raise ValueError(f"Unknown param type for {name}: {param_type}")

    return params


def get_optimization_directions(objectives: list[str]) -> list[str]:
    """
    Get optimization directions for each objective.

    Parameters
    ----------
    objectives : list[str]
        List of objective names.

    Returns
    -------
    list[str]
        List of "maximize" or "minimize" for each objective.
    """
    direction_map = {
        "roc_auc": "maximize",
        "neg_brier_score": "maximize",  # Negative Brier, so maximize
        "average_precision": "maximize",
    }
    return [direction_map[obj] for obj in objectives]


def find_knee_point(trials: list[optuna.Trial]) -> optuna.Trial:
    """
    Find knee point in Pareto frontier (closest to ideal point).

    Normalizes objectives to [0, 1] and finds trial with minimum Euclidean
    distance to the ideal point (AUROC=1, Brier=0).

    Parameters
    ----------
    trials : list[optuna.Trial]
        Pareto-optimal trials.

    Returns
    -------
    optuna.Trial
        Trial at knee point.
    """
    auroc_vals = np.array([t.values[0] for t in trials])
    brier_vals = np.array([-t.values[1] for t in trials])  # Convert back to positive

    # Normalize to [0, 1]
    auroc_range = auroc_vals.max() - auroc_vals.min() + 1e-10
    brier_range = brier_vals.max() - brier_vals.min() + 1e-10

    auroc_norm = (auroc_vals - auroc_vals.min()) / auroc_range
    brier_norm = 1 - (brier_vals - brier_vals.min()) / brier_range  # Invert (lower better)

    # Distance from ideal (AUROC=1, Brier=0)
    distances = np.sqrt((1 - auroc_norm) ** 2 + (1 - brier_norm) ** 2)
    knee_idx = np.argmin(distances)

    return trials[knee_idx]


def find_balanced_point(trials: list[optuna.Trial]) -> optuna.Trial:
    """
    Find balanced point in Pareto frontier (maximum sum of normalized objectives).

    Normalizes both objectives and selects trial with maximum sum, giving
    equal weight to AUROC and calibration quality.

    Parameters
    ----------
    trials : list[optuna.Trial]
        Pareto-optimal trials.

    Returns
    -------
    optuna.Trial
        Trial with best balanced performance.
    """
    auroc_vals = np.array([t.values[0] for t in trials])
    brier_vals = np.array([-t.values[1] for t in trials])

    # Normalize
    auroc_range = auroc_vals.max() - auroc_vals.min() + 1e-10
    brier_range = brier_vals.max() - brier_vals.min() + 1e-10

    auroc_norm = (auroc_vals - auroc_vals.min()) / auroc_range
    brier_norm = 1 - (brier_vals - brier_vals.min()) / brier_range

    # Equal weight sum
    scores = auroc_norm + brier_norm
    best_idx = np.argmax(scores)

    return trials[best_idx]


def select_from_pareto_frontier(
    trials: list[optuna.Trial],
    selection_strategy: str = "knee",
) -> optuna.Trial:
    """
    Select best model from Pareto frontier using configured strategy.

    Parameters
    ----------
    trials : list[optuna.Trial]
        Pareto-optimal trials.
    selection_strategy : str, default="knee"
        Selection strategy: "knee", "extreme_auroc", or "balanced".

    Returns
    -------
    optuna.Trial
        Selected trial.

    Raises
    ------
    ValueError
        If selection_strategy is unknown.
    """
    if selection_strategy == "knee":
        return find_knee_point(trials)
    elif selection_strategy == "extreme_auroc":
        return max(trials, key=lambda t: t.values[0])
    elif selection_strategy == "balanced":
        return find_balanced_point(trials)
    else:
        raise ValueError(f"Unknown pareto_selection: {selection_strategy}")
