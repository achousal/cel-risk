"""
Default configuration values matching current implementation exactly.

This module serves as the single source of truth for all default parameter values,
ensuring behavioral equivalence with the legacy codebase.
"""

from typing import Any

from ..data.schema import ModelName

# Valid scenario names
VALID_SCENARIOS = [
    "IncidentOnly",
    "PrevalentOnly",
    "IncidentPlusPrevalent",
]

# Valid model names
VALID_MODELS = [m.value for m in ModelName]

# Default split configuration (matches save_splits.py)
DEFAULT_SPLITS_CONFIG: dict[str, Any] = {
    "mode": "development",
    "scenarios": ["IncidentOnly"],
    "n_splits": 1,
    "val_size": 0.0,
    "test_size": 0.30,
    "holdout_size": 0.30,
    "seed_start": 0,
    "prevalent_train_only": False,
    "prevalent_train_frac": 1.0,
    "train_control_per_case": None,
    "eval_control_per_case": None,
    "temporal_split": False,
    "temporal_col": "CeD_date",
    "temporal_cutoff": None,
    "outdir": "splits",
}

# Default CV configuration (matches celiacML_faith.py)
DEFAULT_CV_CONFIG: dict[str, Any] = {
    "folds": 5,
    "repeats": 3,
    "inner_folds": 5,
    "scoring": "average_precision",
    "n_iter": None,  # None = use per-model n_iter; set value to override all
    "random_state": 0,
    "grid_randomize": False,
}

# Default feature selection configuration
DEFAULT_FEATURE_CONFIG: dict[str, Any] = {
    "feature_selection_strategy": "hybrid_stability",  # Use new parameter (not deprecated feature_select)
    "screen_method": "mannwhitney",
    "screen_top_n": 0,
    "kbest_max": 500,
    "k_grid": [50, 100, 200, 500],
    "stability_thresh": 0.70,
    "stable_corr_thresh": 0.80,
}

# Default threshold configuration
DEFAULT_THRESHOLD_CONFIG: dict[str, Any] = {
    "objective": "max_f1",
    "fbeta": 1.0,
    "fixed_spec": 0.90,
    "fixed_ppv": 0.10,
    "threshold_source": "val",
    "target_prevalence_source": "test",
    "target_prevalence_fixed": None,
    "risk_prob_source": "test",
}

# Default evaluation configuration
DEFAULT_EVALUATION_CONFIG: dict[str, Any] = {
    "test_ci_bootstrap": True,
    "n_boot": 500,
    "boot_random_state": 0,
    "learning_curve": False,
    "lc_train_sizes": [0.1, 0.25, 0.5, 0.75, 1.0],
    "feature_reports": True,
    "feature_report_max": 100,
    "control_spec_targets": [0.90, 0.95, 0.99],
    "toprisk_fracs": [0.01, 0.05, 0.10],
}

# Default DCA configuration
DEFAULT_DCA_CONFIG: dict[str, Any] = {
    "compute_dca": False,
    "dca_threshold_min": 0.0005,
    "dca_threshold_max": 1.0,
    "dca_threshold_step": 0.001,
    "dca_report_points": [0.01, 0.05, 0.10, 0.20],
}

# Default output configuration
DEFAULT_OUTPUT_CONFIG: dict[str, Any] = {
    # Prediction artifacts
    "save_train_preds": False,
    "save_train_oof": True,
    "save_val_preds": True,
    "save_test_preds": True,
    # Calibration artifacts
    "save_calibration": True,
    "calib_bins": 10,
    # Feature artifacts
    "save_feature_importance": True,
    "feature_reports": True,
    # Optuna artifacts
    "save_optuna_study": True,
    "save_optuna_trials": True,
    # Master plot controls
    "save_plots": True,
    "max_plot_splits": 0,
    "plot_format": "png",
    "plot_dpi": 300,
    # Individual plot type controls
    "plot_roc": True,
    "plot_pr": True,
    "plot_calibration": True,
    "plot_risk_distribution": True,
    "plot_dca": True,
    "plot_learning_curve": True,
    "plot_oof_combined": True,
    "plot_optuna": True,
    "optuna_plot_format": "html",
    # Ensemble-specific plots
    "plot_ensemble_weights": True,
    "plot_ensemble_comparison": True,
    "plot_base_correlations": True,
    # Aggregation-specific
    "save_pooled_preds": True,
    "save_summary_csv": True,
    "save_thresholds": True,
    "save_individual": False,
    # Panel optimization
    "save_panel_csv": True,
    "save_rfe_curve": True,
    "save_stability_ranks": True,
    "save_consensus_ranks": True,
}

# Default strictness configuration
DEFAULT_STRICTNESS_CONFIG: dict[str, Any] = {
    "level": "warn",
    "check_split_overlap": True,
    "check_prevalent_in_eval": True,
    "check_threshold_source": True,
    "check_prevalence_adjustment": True,
    "check_feature_leakage": True,
}

# Model-specific hyperparameter defaults (matching celiacML_faith.py get_param_distributions)
DEFAULT_LR_CONFIG: dict[str, Any] = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C_min": 0.001,
    "C_max": 10.0,
    "C_points": 5,
    "l1_ratio": [0.1, 0.5, 0.9],
    "solver": "saga",
    "max_iter": 1000,
    "class_weight_options": "balanced",
    "random_state": 0,
}

DEFAULT_SVM_CONFIG: dict[str, Any] = {
    "C_min": 0.01,
    "C_max": 10.0,
    "C_points": 4,
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto", 0.001, 0.01],
    "class_weight_options": "balanced",
    "max_iter": 5000,
    "probability": True,
    "random_state": 0,
}

DEFAULT_RF_CONFIG: dict[str, Any] = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 0,
}

DEFAULT_XGBOOST_CONFIG: dict[str, Any] = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
    "scale_pos_weight": "auto",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 0,
}

DEFAULT_CALIBRATION_CONFIG: dict[str, Any] = {
    "method": "sigmoid",
    "cv": 5,
    "ensemble": False,
}

# Default Optuna configuration
DEFAULT_OPTUNA_CONFIG: dict[str, Any] = {
    "enabled": False,
    "n_trials": 100,
    "timeout": None,
    "sampler": "tpe",
    "sampler_seed": None,
    "pruner": "median",
    "pruner_n_startup_trials": 5,
    "pruner_percentile": 25.0,
    "n_jobs": 1,
    "storage": None,
    "study_name": None,
    "load_if_exists": False,
    "save_study": True,
    "save_trials_csv": True,
    "direction": None,
}

# Default aggregate configuration
DEFAULT_AGGREGATE_CONFIG: dict[str, Any] = {
    "results_dir": "results",
    "outdir": "results_aggregated",
    "split_pattern": "split_seed*",
    "predictions_method": "median",
    "save_individual": False,
    "summary_stats": ["mean", "std", "median", "ci95"],
    "group_by": ["scenario", "model"],
    "min_stability": 0.7,
    "corr_method": "spearman",
    "corr_threshold": 0.80,
    "save_pooled_preds": True,
    "save_summary_csv": True,
    "save_plots": True,
}

# Default holdout evaluation configuration
DEFAULT_HOLDOUT_CONFIG: dict[str, Any] = {
    "compute_dca": True,
    "save_preds": True,
    "toprisk_fracs": [0.01, 0.05, 0.10],
    "subgroup_min_n": 40,
    "dca_threshold_min": 0.0005,
    "dca_threshold_max": 1.0,
    "dca_threshold_step": 0.001,
    "dca_report_points": [0.01, 0.05, 0.10, 0.20],
    "dca_use_target_prevalence": False,
    "clinical_threshold_points": [],
    "target_prevalence": None,
}
