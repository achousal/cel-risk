"""Utility functions for CeD-ML."""

from ced_ml.utils.logging import log_section, setup_logger
from ced_ml.utils.math_utils import (
    EPSILON_BOUNDS,
    EPSILON_LOGIT,
    EPSILON_LOGLOSS,
    EPSILON_PREVALENCE,
    JEFFREYS_ALPHA,
    inv_logit,
    jeffreys_smooth,
    logit,
)
from ced_ml.utils.metadata import (
    build_aggregated_metadata,
    build_oof_metadata,
    build_plot_metadata,
)
from ced_ml.utils.paths import ensure_dir
from ced_ml.utils.random import (
    apply_seed_global,
    create_rng,
    get_cv_seed,
    set_random_seed,
)
from ced_ml.utils.serialization import load_joblib, load_json, save_joblib, save_json

__all__ = [
    "setup_logger",
    "log_section",
    "build_plot_metadata",
    "build_oof_metadata",
    "build_aggregated_metadata",
    "ensure_dir",
    "set_random_seed",
    "create_rng",
    "apply_seed_global",
    "get_cv_seed",
    "save_joblib",
    "load_joblib",
    "save_json",
    "load_json",
    "logit",
    "inv_logit",
    "jeffreys_smooth",
    "EPSILON_LOGIT",
    "EPSILON_LOGLOSS",
    "EPSILON_PREVALENCE",
    "EPSILON_BOUNDS",
    "JEFFREYS_ALPHA",
]
