"""
Optuna trial callbacks and logging utilities.

Provides callbacks for trial progress logging and result tracking
during hyperparameter optimization.
"""

from __future__ import annotations

import logging

import optuna

logger = logging.getLogger(__name__)


def create_trial_callback(n_trials: int, multi_objective: bool = False):
    """
    Create trial callback for progress logging.

    Parameters
    ----------
    n_trials : int
        Total number of trials for percentage calculations.
    multi_objective : bool, default=False
        Whether this is a multi-objective optimization.

    Returns
    -------
    callable
        Trial callback function that logs every 10th trial.
    """

    def trial_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """Log progress every 10th trial."""
        trial_number = trial.number + 1  # 1-indexed for logging

        # Log every 10th trial
        if trial_number % 10 == 0:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if multi_objective:
                    score_str = f"AUROC={trial.values[0]:.3f}, Brier={trial.values[1]:.3f}"
                else:
                    score_str = f"score={trial.value:.3f}"

                # Format params concisely
                param_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])
                logger.info(
                    f"  Trial {trial_number}/{n_trials}: {score_str}, params={{{param_str}}}"
                )

    return trial_callback


def log_optimization_summary(
    study: optuna.Study,
    best_score: float,
    multi_objective: bool = False,
):
    """
    Log optimization summary with trial statistics and hyperparameter importance.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    best_score : float
        Best score achieved.
    multi_objective : bool, default=False
        Whether this is a multi-objective optimization.
    """
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    logger.info(
        f"  Completed {len(completed_trials)} trials ({n_pruned} pruned): best_score={best_score:.3f}"
    )

    # Log hyperparameter importance if enough trials (single-objective only)
    if len(completed_trials) >= 20 and not multi_objective:
        try:
            importance = optuna.importance.get_param_importances(study)
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            importance_str = ", ".join([f"{k} ({v:.2f})" for k, v in top_params])
            logger.info(f"  Top hyperparameter importance: {importance_str}")
        except Exception as e:
            logger.debug("Hyperparameter importance calculation skipped: %s", e)


def check_tpe_hyperband_trials(sampler: str, pruner: str, n_trials: int):
    """
    Warn if TPE + Hyperband combination has insufficient trials.

    TPE sampler with HyperbandPruner typically needs 40+ trials for
    effective optimization to gather sufficient observations.

    Parameters
    ----------
    sampler : str
        Sampler type ("tpe", "random", etc.).
    pruner : str
        Pruner type ("hyperband", "median", etc.).
    n_trials : int
        Number of trials configured.
    """
    if sampler == "tpe" and pruner == "hyperband" and n_trials < 40:
        logger.warning(
            f"[optuna] TPE sampler with HyperbandPruner typically needs 40+ trials "
            f"for effective optimization (n_trials={n_trials}). "
            "Consider increasing n_trials or using sampler='random' for fewer trials."
        )


def validate_study_compatibility(
    study: optuna.Study,
    multi_objective: bool,
    direction: str | None = None,
    load_if_exists: bool = False,
):
    """
    Validate that loaded study is compatible with requested configuration.

    Parameters
    ----------
    study : optuna.Study
        Loaded or newly created study.
    multi_objective : bool
        Whether multi-objective optimization is requested.
    direction : str, optional
        Expected direction for single-objective ("maximize" or "minimize").
    load_if_exists : bool, default=False
        Whether study was loaded from storage.

    Raises
    ------
    ValueError
        If loaded study is incompatible with requested configuration.
    """
    if not load_if_exists or len(study.trials) == 0:
        return

    # Check if loaded study is multi-objective when we expect single-objective
    study_is_multi = hasattr(study, "directions")
    if study_is_multi and not multi_objective:
        raise ValueError(
            "[optuna] Incompatible study loaded: study is multi-objective but "
            "multi_objective=False was requested. Please use a different study_name, "
            "set load_if_exists=False, or delete the existing study database."
        )

    # Check if loaded study is single-objective when we expect multi-objective
    if not study_is_multi and multi_objective:
        raise ValueError(
            "[optuna] Incompatible study loaded: study is single-objective but "
            "multi_objective=True was requested. Please use a different study_name, "
            "set load_if_exists=False, or delete the existing study database."
        )

    # For single-objective, check direction compatibility
    if not multi_objective and direction and study.direction.name.lower() != direction:
        logger.warning(
            f"[optuna] Loaded study has direction={study.direction.name.lower()}, "
            f"but requested direction={direction}. Using existing study's direction."
        )
