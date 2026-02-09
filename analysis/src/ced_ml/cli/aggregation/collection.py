"""
Collection module for aggregate_splits.

Collects metrics, predictions, hyperparameters, and feature reports from split directories.
Handles both base models and ENSEMBLE models.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ced_ml.data.io_helpers import read_feature_report, read_metrics, read_predictions


def collect_ensemble_predictions(
    ensemble_dirs: list[Path],
    pred_type: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect predictions from ENSEMBLE model directories.

    Args:
        ensemble_dirs: List of ensemble split directories
        pred_type: One of "test", "val", "train_oof"
        logger: Optional logger instance

    Returns:
        DataFrame with pooled ensemble predictions
    """
    pred_subdir_map = {
        "test": "preds",
        "val": "preds",
        "train_oof": "preds",
    }

    if pred_type not in pred_subdir_map:
        raise ValueError(
            f"Unknown pred_type: {pred_type}. Must be one of {list(pred_subdir_map.keys())}"
        )

    subdir = pred_subdir_map[pred_type]
    all_preds = []

    for ensemble_dir in ensemble_dirs:
        # Extract seed from directory name
        seed = int(ensemble_dir.name.replace("split_seed", ""))

        pred_dir = ensemble_dir / subdir

        if not pred_dir.exists():
            if logger:
                logger.debug(f"No {pred_type} predictions dir in ENSEMBLE/{ensemble_dir.name}")
            continue

        pred_file_patterns = {
            "test": "test_preds__ENSEMBLE.csv",
            "val": "val_preds__ENSEMBLE.csv",
            "train_oof": "train_oof__ENSEMBLE.csv",
        }
        file_pattern = pred_file_patterns[pred_type]
        csv_files = list(pred_dir.glob(file_pattern))
        if not csv_files:
            if logger:
                logger.debug(f"No ENSEMBLE CSV files in {pred_dir}")
            continue

        for csv_path in csv_files:
            try:
                df = read_predictions(csv_path)
                df["split_seed"] = seed
                df["source_file"] = csv_path.name
                df["model"] = "ENSEMBLE"
                all_preds.append(df)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to read {csv_path}: {e}")

    if not all_preds:
        return pd.DataFrame()

    return pd.concat(all_preds, ignore_index=True)


def collect_metrics(
    split_dirs: list[Path],
    metrics_file: str = "core/test_metrics.csv",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect metrics from all split directories.

    Args:
        split_dirs: List of split subdirectory paths
        metrics_file: Relative path to metrics file within each split dir
        logger: Optional logger instance

    Returns:
        DataFrame with all metrics, indexed by split_seed
    """
    all_metrics = []

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))
        metrics_path = split_dir / metrics_file

        if not metrics_path.exists():
            if logger:
                logger.debug(f"Metrics file not found: {metrics_path}")
            continue

        try:
            df = read_metrics(metrics_path)
            df["split_seed"] = seed
            all_metrics.append(df)
            if logger:
                logger.debug(f"Loaded metrics from {split_dir.name}/{metrics_file}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read {metrics_path}: {e}")

    if not all_metrics:
        if logger:
            logger.debug(f"No metrics collected from {metrics_file}")
        return pd.DataFrame()

    df_combined = pd.concat(all_metrics, ignore_index=True)
    if logger:
        logger.info(
            f"Collected {len(df_combined)} rows from {len(all_metrics)} splits ({metrics_file})"
        )

    return df_combined


def collect_best_hyperparams(
    split_dirs: list[Path],
    optuna_file: str = "cv/best_params_optuna.csv",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect best hyperparameters from all split directories.

    Args:
        split_dirs: List of split subdirectory paths
        optuna_file: Relative path to best_params file within each split dir
        logger: Optional logger instance

    Returns:
        DataFrame with all best hyperparameters, indexed by split_seed
    """
    all_params = []

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))
        params_path = split_dir / optuna_file

        if not params_path.exists():
            if logger:
                logger.debug(f"Best params file not found: {params_path}")
            continue

        try:
            df = pd.read_csv(params_path)
            df["split_seed"] = seed

            # Parse JSON best_params column if present
            if "best_params" in df.columns:
                params_list = []
                for _, row in df.iterrows():
                    try:
                        params_dict = json.loads(row["best_params"])
                        row_data = {
                            "split_seed": row["split_seed"],
                            "model": row["model"],
                            "repeat": row["repeat"],
                            "outer_split": row["outer_split"],
                            "best_score_inner": row.get("best_score_inner", np.nan),
                        }
                        # Flatten nested params dict
                        for k, v in params_dict.items():
                            row_data[k] = v
                        params_list.append(row_data)
                    except (json.JSONDecodeError, KeyError) as e:
                        if logger:
                            logger.warning(f"Failed to parse params in {split_dir.name}: {e}")
                        continue

                if params_list:
                    df = pd.DataFrame(params_list)
                    all_params.append(df)
            else:
                all_params.append(df)

            if logger:
                logger.debug(f"Loaded best params from {split_dir.name}/{optuna_file}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read {params_path}: {e}")

    if not all_params:
        if logger:
            logger.debug(f"No hyperparameters collected from {optuna_file}")
        return pd.DataFrame()

    df_combined = pd.concat(all_params, ignore_index=True)
    if logger:
        logger.info(
            f"Collected {len(df_combined)} hyperparameter sets from {len(all_params)} splits"
        )

    return df_combined


def collect_ensemble_hyperparams(
    ensemble_dirs: list[Path],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect hyperparameters from ENSEMBLE split directories.

    Args:
        ensemble_dirs: List of ENSEMBLE split subdirectory paths
        logger: Optional logger instance

    Returns:
        DataFrame with ENSEMBLE hyperparameters
    """
    all_params = []

    for split_dir in ensemble_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))

        # Ensemble meta-learner config is in run_settings.json (canonical source)
        settings_path = split_dir / "core" / "run_settings.json"
        config_path = split_dir / "config.yaml"

        # Prefer run_settings.json (matches ensemble_metadata.py pattern)
        if settings_path.exists():
            try:
                import json

                with open(settings_path) as f:
                    settings = json.load(f)

                # Extract ensemble settings from JSON
                ensemble_config = settings.get("ensemble", {})
                meta_model = ensemble_config.get("meta_model", {})

                row = {
                    "split_seed": seed,
                    "model": "ENSEMBLE",
                    "method": ensemble_config.get("method", "stacking"),
                    "base_models": ", ".join(ensemble_config.get("base_models", [])),
                    "meta_model_type": meta_model.get("type", "logistic_regression"),
                    "meta_model_penalty": meta_model.get("penalty", "l2"),
                    "meta_model_C": meta_model.get("C", 1.0),
                }
                all_params.append(row)
                continue
            except Exception as e:
                if logger:
                    logger.debug(f"Could not read run_settings.json from {split_dir.name}: {e}")

        # Fallback to config.yaml for backward compatibility
        if config_path.exists():
            try:
                import yaml

                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Extract ensemble settings
                ensemble_config = config.get("ensemble", {})
                meta_model = ensemble_config.get("meta_model", {})

                row = {
                    "split_seed": seed,
                    "model": "ENSEMBLE",
                    "method": ensemble_config.get("method", "stacking"),
                    "base_models": ", ".join(ensemble_config.get("base_models", [])),
                    "meta_model_type": meta_model.get("type", "logistic_regression"),
                    "meta_model_penalty": meta_model.get("penalty", "l2"),
                    "meta_model_C": meta_model.get("C", 1.0),
                }
                all_params.append(row)
                if logger:
                    logger.debug(
                        f"Using config.yaml fallback for {split_dir.name} (run_settings.json not found)"
                    )
                continue
            except Exception as e:
                if logger:
                    logger.debug(f"Could not read config.yaml from {split_dir.name}: {e}")

        if logger:
            logger.debug(f"No config files found in {split_dir.name}")

    if not all_params:
        if logger:
            logger.debug("No ensemble hyperparameters collected")
        return pd.DataFrame()

    df_combined = pd.DataFrame(all_params)
    if logger:
        logger.info(f"Collected ensemble configs from {len(all_params)} splits")

    return df_combined


def collect_predictions(
    split_dirs: list[Path],
    pred_type: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect predictions from all splits and add split_seed and model columns.

    Args:
        split_dirs: List of split subdirectory paths
        pred_type: One of "test", "val", "train_oof"
        logger: Optional logger instance

    Returns:
        DataFrame with pooled predictions including split_seed and model columns
    """
    pred_subdir_map = {
        "test": "preds",
        "val": "preds",
        "train_oof": "preds",
    }

    if pred_type not in pred_subdir_map:
        raise ValueError(
            f"Unknown pred_type: {pred_type}. Must be one of {list(pred_subdir_map.keys())}"
        )

    subdir = pred_subdir_map[pred_type]
    all_preds = []

    # Define file patterns for each prediction type to avoid collecting auxiliary files
    pred_file_patterns = {
        "test": "test_preds__*.csv",
        "val": "val_preds__*.csv",
        "train_oof": "train_oof__*.csv",
    }
    file_pattern = pred_file_patterns[pred_type]

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))
        pred_dir = split_dir / subdir

        if not pred_dir.exists():
            if logger:
                logger.debug(f"No {pred_type} predictions dir in {split_dir.name}")
            continue

        csv_files = list(pred_dir.glob(file_pattern))
        if not csv_files:
            if logger:
                logger.debug(f"No {file_pattern} files in {pred_dir}")
            continue

        for csv_path in csv_files:
            try:
                df = read_predictions(csv_path)
                df["split_seed"] = seed
                df["source_file"] = csv_path.name

                # Extract model name from filename pattern: {prefix}__{MODEL}.csv
                # e.g., "test_preds__LR_EN.csv" -> "LR_EN"
                # e.g., "train_oof__RF.csv" -> "RF"
                filename_stem = csv_path.stem  # without .csv
                if "__" in filename_stem:
                    model_name = filename_stem.split("__", 1)[1]
                else:
                    model_name = "unknown"
                df["model"] = model_name

                all_preds.append(df)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to read {csv_path}: {e}")

    if not all_preds:
        return pd.DataFrame()

    return pd.concat(all_preds, ignore_index=True)


def collect_feature_reports(
    split_dirs: list[Path],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Collect feature reports from all split directories.

    Args:
        split_dirs: List of split subdirectory paths
        logger: Optional logger instance

    Returns:
        DataFrame with all feature reports, including split_seed column
    """
    all_reports = []

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))
        reports_dir = split_dir / "panels"

        if not reports_dir.exists():
            if logger:
                logger.debug(f"No feature_reports dir in {split_dir.name}")
            continue

        csv_files = list(reports_dir.glob("*__feature_report_train.csv"))
        if not csv_files:
            if logger:
                logger.debug(f"No feature report CSV files in {reports_dir}")
            continue

        for csv_path in csv_files:
            try:
                df = read_feature_report(csv_path)
                df["split_seed"] = seed
                all_reports.append(df)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to read {csv_path}: {e}")

    if not all_reports:
        return pd.DataFrame()

    return pd.concat(all_reports, ignore_index=True)
