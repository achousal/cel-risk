"""
Collection phase for aggregate_splits.

Gathers metrics, predictions, features, and hyperparameters across splits.
Returns consolidated data structures without writing outputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ced_ml.cli.aggregation.collection import (
    collect_best_hyperparams,
    collect_ensemble_hyperparams,
    collect_ensemble_predictions,
    collect_feature_reports,
    collect_metrics,
    collect_predictions,
)


@dataclass
class CollectedData:
    """Container for all collected data across splits."""

    pooled_test_df: pd.DataFrame
    pooled_val_df: pd.DataFrame
    pooled_train_oof_df: pd.DataFrame
    test_metrics: pd.DataFrame
    val_metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    best_params: pd.DataFrame
    ensemble_params: pd.DataFrame
    all_feature_reports: pd.DataFrame
    all_models: list[str]
    optuna_trials_combined: pd.DataFrame | None
    ensemble_metadata_raw: dict[str, Any]


def collect_all_predictions(
    split_dirs: list[Path],
    ensemble_dirs: list[Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collect and merge predictions from base models and ensemble.

    Args:
        split_dirs: List of split directories
        ensemble_dirs: List of ensemble split directories
        logger: Logger instance

    Returns:
        Tuple of (pooled_test_df, pooled_val_df, pooled_train_oof_df)
    """
    pooled_test_df = collect_predictions(split_dirs, "test", logger)
    pooled_val_df = collect_predictions(split_dirs, "val", logger)
    pooled_train_oof_df = collect_predictions(split_dirs, "train_oof", logger)

    if ensemble_dirs:
        ensemble_test_df = collect_ensemble_predictions(ensemble_dirs, "test", logger)
        ensemble_val_df = collect_ensemble_predictions(ensemble_dirs, "val", logger)
        ensemble_oof_df = collect_ensemble_predictions(ensemble_dirs, "train_oof", logger)

        if not ensemble_test_df.empty:
            pooled_test_df = pd.concat([pooled_test_df, ensemble_test_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE test predictions: {len(ensemble_test_df)} samples")

        if not ensemble_val_df.empty:
            pooled_val_df = pd.concat([pooled_val_df, ensemble_val_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE val predictions: {len(ensemble_val_df)} samples")

        if not ensemble_oof_df.empty:
            pooled_train_oof_df = pd.concat(
                [pooled_train_oof_df, ensemble_oof_df], ignore_index=True
            )
            logger.info(f"Merged ENSEMBLE OOF predictions: {len(ensemble_oof_df)} samples")

    return pooled_test_df, pooled_val_df, pooled_train_oof_df


def collect_all_metrics(
    split_dirs: list[Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Collect test, val, and CV metrics across splits.

    Args:
        split_dirs: List of split directories
        logger: Logger instance

    Returns:
        Tuple of (test_metrics, val_metrics, cv_metrics)
    """
    test_metrics = collect_metrics(split_dirs, "core/test_metrics.csv", logger=logger)
    val_metrics = collect_metrics(split_dirs, "core/val_metrics.csv", logger=logger)
    cv_metrics = collect_metrics(split_dirs, "cv/cv_repeat_metrics.csv", logger=logger)

    if cv_metrics.empty:
        logger.info("No CV metrics found (optional)")

    return test_metrics, val_metrics, cv_metrics


def collect_all_hyperparams(
    split_dirs: list[Path],
    ensemble_dirs: list[Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect hyperparameters from base models and ensemble.

    Args:
        split_dirs: List of split directories
        ensemble_dirs: List of ensemble split directories
        logger: Logger instance

    Returns:
        Tuple of (best_params, ensemble_params)
    """
    best_params = collect_best_hyperparams(split_dirs, logger=logger)

    if best_params.empty:
        logger.info("No hyperparameters found (Optuna may not be enabled)")

    ensemble_params = pd.DataFrame()
    if ensemble_dirs:
        ensemble_params = collect_ensemble_hyperparams(ensemble_dirs, logger=logger)

    return best_params, ensemble_params


def collect_optuna_trials(
    split_dirs: list[Path],
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """
    Aggregate Optuna hyperparameter tuning trials across splits.

    Args:
        split_dirs: List of split directories
        logger: Logger instance

    Returns:
        Combined Optuna trials DataFrame or None if no trials found
    """
    try:
        optuna_trials_combined = None
        n_optuna_trials = 0

        for split_dir in split_dirs:
            cv_dir = split_dir / "cv"
            if not cv_dir.exists():
                continue

            optuna_files = list(cv_dir.glob("*__optuna_trials.csv"))

            if optuna_files:
                optuna_csv = optuna_files[0]
                try:
                    df = pd.read_csv(optuna_csv)
                    if optuna_trials_combined is None:
                        optuna_trials_combined = df
                    else:
                        optuna_trials_combined = pd.concat(
                            [optuna_trials_combined, df], ignore_index=True
                        )
                    n_optuna_trials += 1
                except Exception as e:
                    logger.warning(f"Failed to load optuna trials from {optuna_csv}: {e}")

        if optuna_trials_combined is not None:
            logger.info(f"Aggregated {n_optuna_trials} Optuna trial sets")
        else:
            logger.info("No Optuna trials found (optional - depends on config.optuna.enabled)")

        return optuna_trials_combined

    except Exception as e:
        logger.warning(f"Failed to aggregate Optuna trials: {e}")
        return None


def collect_ensemble_raw_metadata(
    ensemble_dirs: list[Path],
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Collect raw ensemble metadata from split directories.

    Args:
        ensemble_dirs: List of ensemble split directories
        logger: Logger instance

    Returns:
        Dictionary with ensemble metadata including coefficients
    """
    coefs_per_split: dict[int, dict[str, float]] = {}
    base_models_list = []
    meta_penalty = "l2"
    meta_C = 1.0

    for ed in ensemble_dirs:
        settings_path = ed / "core" / "run_settings.json"
        config_path = ed / "config.yaml"

        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
                meta_coef = settings.get("meta_coef", {})
                if meta_coef:
                    seed = settings.get("split_seed", 0)
                    coefs_per_split[seed] = meta_coef
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Could not read ensemble settings from {ed}: {e}")

        if config_path.exists() and not base_models_list:
            try:
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "ensemble" in config:
                    ensemble_cfg = config["ensemble"]
                    base_models_list = ensemble_cfg.get("base_models", [])
                    meta_learner_cfg = ensemble_cfg.get("meta_model", {})
                    meta_penalty = meta_learner_cfg.get("penalty", "l2")
                    meta_C = meta_learner_cfg.get("C", 1.0)
            except Exception as e:
                logger.debug(f"Could not read ensemble config from {config_path}: {e}")

    return {
        "coefs_per_split": coefs_per_split,
        "base_models": base_models_list,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
    }


def run_collection_phase(
    split_dirs: list[Path],
    ensemble_dirs: list[Path],
    logger: logging.Logger,
) -> CollectedData:
    """
    Run the complete collection phase across all splits.

    Args:
        split_dirs: List of split directories
        ensemble_dirs: List of ensemble split directories
        logger: Logger instance

    Returns:
        CollectedData container with all collected artifacts
    """
    from ced_ml.utils.logging import log_section

    log_section(logger, "Collecting Pooled Predictions")
    pooled_test_df, pooled_val_df, pooled_train_oof_df = collect_all_predictions(
        split_dirs, ensemble_dirs, logger
    )

    log_section(logger, "Aggregating Per-Split Metrics")
    test_metrics, val_metrics, cv_metrics = collect_all_metrics(split_dirs, logger)

    log_section(logger, "Aggregating Hyperparameters")
    best_params, ensemble_params = collect_all_hyperparams(split_dirs, ensemble_dirs, logger)

    log_section(logger, "Aggregating Feature Reports")
    all_feature_reports = collect_feature_reports(split_dirs, logger=logger)

    if all_feature_reports.empty:
        logger.info("No feature reports found (optional - depends on feature selection)")

    log_section(logger, "Aggregating Optuna Trials")
    optuna_trials_combined = collect_optuna_trials(split_dirs, logger)

    test_models = (
        pooled_test_df["model"].unique().tolist()
        if not pooled_test_df.empty and "model" in pooled_test_df.columns
        else []
    )
    val_models = (
        pooled_val_df["model"].unique().tolist()
        if not pooled_val_df.empty and "model" in pooled_val_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models))

    ensemble_metadata_raw = {}
    if ensemble_dirs:
        ensemble_metadata_raw = collect_ensemble_raw_metadata(ensemble_dirs, logger)

    return CollectedData(
        pooled_test_df=pooled_test_df,
        pooled_val_df=pooled_val_df,
        pooled_train_oof_df=pooled_train_oof_df,
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        cv_metrics=cv_metrics,
        best_params=best_params,
        ensemble_params=ensemble_params,
        all_feature_reports=all_feature_reports,
        all_models=all_models,
        optuna_trials_combined=optuna_trials_combined,
        ensemble_metadata_raw=ensemble_metadata_raw,
    )
