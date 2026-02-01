"""
Ensemble metadata collection and aggregation.

Extracts ensemble-specific metadata including meta-learner coefficients,
configurations, and performance comparisons.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ced_ml.data.schema import METRIC_AUROC, METRIC_BRIER, METRIC_PRAUC


def collect_ensemble_metadata(
    ensemble_dirs: list[Path],
    all_models: list[str],
    pooled_test_metrics: dict[str, dict[str, float]],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Collect and aggregate ensemble-specific metadata.

    Args:
        ensemble_dirs: List of ensemble split directories
        all_models: All discovered model names
        pooled_test_metrics: Pooled test metrics for all models
        logger: Optional logger

    Returns:
        Dictionary with ensemble metadata including:
        - meta_learner_coefficients: Aggregated stats and per-split values
        - ensemble_config: Base models and meta-learner settings
        - performance: ENSEMBLE vs best base model comparison
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensemble_metadata: dict[str, Any] = {}

    # Return early if no ENSEMBLE model found
    if "ENSEMBLE" not in all_models or not ensemble_dirs:
        return ensemble_metadata

    ensemble_coefs_agg: dict[int, dict[str, float]] = {}
    ensemble_configs: dict[int, dict[str, Any]] = {}

    # Collect per-split coefficients and configurations
    for ed in ensemble_dirs:
        settings_path = ed / "core" / "run_settings.json"
        config_path = ed / "config.yaml"

        # Extract coefficients
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    settings = json.load(f)
                meta_coef = settings.get("meta_coef", {})
                split_seed = settings.get("split_seed", 0)
                if meta_coef:
                    ensemble_coefs_agg[split_seed] = meta_coef
            except Exception as e:
                logger.debug(f"Could not read ensemble settings from {ed}: {e}")

        # Extract base model list and meta-learner config
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                if "ensemble" in config:
                    ensemble_cfg = config["ensemble"]
                    split_seed = int(ed.name.replace("split_seed", ""))
                    ensemble_configs[split_seed] = {
                        "base_models": ensemble_cfg.get("base_models", []),
                        "meta_penalty": ensemble_cfg.get("meta_model", {}).get("penalty", "l2"),
                        "meta_C": ensemble_cfg.get("meta_model", {}).get("C", 1.0),
                    }
            except Exception as e:
                logger.debug(f"Could not read ensemble config from {config_path}: {e}")

    # Aggregate coefficients across splits
    if ensemble_coefs_agg:
        all_coef_names = set()
        for coef_dict in ensemble_coefs_agg.values():
            all_coef_names.update(coef_dict.keys())

        coef_stats = {}
        for name in all_coef_names:
            vals = [cd.get(name) for cd in ensemble_coefs_agg.values() if name in cd]
            if vals:
                coef_stats[name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_splits": len(vals),
                }

        ensemble_metadata["meta_learner_coefficients"] = {
            "aggregated_stats": coef_stats,
            "per_split": ensemble_coefs_agg,
            "n_splits_with_coefs": len(ensemble_coefs_agg),
        }

    # Add ensemble configuration metadata
    if ensemble_configs:
        # Get base models from first available config
        first_config = next(iter(ensemble_configs.values()), {})
        ensemble_metadata["ensemble_config"] = {
            "base_models": first_config.get("base_models", []),
            "n_base_models": len(first_config.get("base_models", [])),
            "meta_penalty": first_config.get("meta_penalty", "l2"),
            "meta_C": first_config.get("meta_C", 1.0),
            "n_splits_with_config": len(ensemble_configs),
        }

    # Add ENSEMBLE model performance vs best single model
    if "ENSEMBLE" in pooled_test_metrics and len(all_models) > 1:
        ensemble_test = pooled_test_metrics["ENSEMBLE"]
        base_models_test = {m: pooled_test_metrics[m] for m in all_models if m != "ENSEMBLE"}

        if base_models_test:
            best_base_auroc = max(
                (m.get(METRIC_AUROC, 0) for m in base_models_test.values()), default=0
            )
            ensemble_auroc = ensemble_test.get(METRIC_AUROC, 0)

            if best_base_auroc > 0:
                improvement = ((ensemble_auroc - best_base_auroc) / best_base_auroc) * 100
                ensemble_metadata["performance"] = {
                    "test_AUROC": ensemble_auroc,
                    "best_base_model_AUROC": best_base_auroc,
                    "AUROC_improvement_percent": improvement,
                    "test_PR_AUC": ensemble_test.get(METRIC_PRAUC),
                    "test_Brier": ensemble_test.get(METRIC_BRIER),
                }

    return ensemble_metadata
