"""
Configuration schema for CeD-ML pipeline.

Defines Pydantic models for all pipeline configuration parameters (~200 total).
All defaults match the current implementation exactly for behavioral equivalence.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..data.schema import ModelName

# ============================================================================
# Data and Split Configuration
# ============================================================================


class ColumnsConfig(BaseModel):
    """Configuration for metadata column selection."""

    mode: Literal["auto", "explicit"] = "auto"
    numeric_metadata: list[str] | None = None
    categorical_metadata: list[str] | None = None
    warn_missing_defaults: bool = True


class SplitsConfig(BaseModel):
    """Configuration for data split generation."""

    mode: Literal["development", "holdout"] = "development"
    scenarios: list[str] = Field(default_factory=lambda: ["IncidentOnly"])
    n_splits: int = Field(default=1, ge=1)
    val_size: float = Field(default=0.0, ge=0.0, le=1.0)
    test_size: float = Field(default=0.30, ge=0.0, le=1.0)
    holdout_size: float = Field(default=0.30, ge=0.0, le=1.0)
    seed_start: int = Field(default=0, ge=0)

    # Prevalent case handling
    prevalent_train_only: bool = False
    prevalent_train_frac: float = Field(default=1.0, ge=0.0, le=1.0)

    # Control downsampling
    train_control_per_case: float | None = Field(default=None, ge=1.0)
    eval_control_per_case: float | None = Field(default=None, ge=1.0)

    # Temporal split
    temporal_split: bool = False
    temporal_col: str = "CeD_date"
    temporal_cutoff: str | None = None

    # Output
    outdir: Path = Field(default=Path("../splits"))
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_split_sizes(self):
        """Validate that split sizes don't exceed 1.0."""
        if self.mode == "development":
            total = self.val_size + self.test_size
            if total >= 1.0:
                raise ValueError(
                    f"val_size ({self.val_size}) + test_size ({self.test_size}) >= 1.0. "
                    "No data left for training."
                )
        return self


# ============================================================================
# Cross-Validation Configuration
# ============================================================================


class CVConfig(BaseModel):
    """Configuration for cross-validation structure."""

    folds: int = Field(default=5, ge=2)
    repeats: int = Field(default=3, ge=1)
    inner_folds: int = Field(default=5, ge=2)
    scoring: str = "average_precision"
    scoring_target_fpr: float | None = Field(default=0.05, ge=0.0, le=1.0)
    n_iter: int | None = Field(
        default=None,
        ge=1,
        description="Global n_iter override. If set, overrides all per-model n_iter values.",
    )
    random_state: int = 0
    grid_randomize: bool = False


# ============================================================================
# Feature Selection Configuration
# ============================================================================


class FeatureConfig(BaseModel):
    """Configuration for feature selection methods.

    Two mutually exclusive strategies:
    1. hybrid_stability (default): screen → kbest (tuned) → stability → model
       - Robust, interpretable, uses k_grid tuning
       - Best for: production models, reproducibility
    2. rfecv: screen → light kbest cap → RFECV → model
       - Automatic size discovery, can churn across folds
       - Best for: scientific discovery, understanding feature stability
    """

    model_config = ConfigDict(protected_namespaces=())

    # Feature selection strategy (mutually exclusive paths)
    feature_selection_strategy: Literal["hybrid_stability", "rfecv", "fixed_panel", "none"] = Field(
        default="hybrid_stability",
        description=(
            "Feature selection strategy:\n"
            "  - hybrid_stability: screen → kbest (tuned) → stability → model\n"
            "  - rfecv: screen → light kbest cap → RFECV → model\n"
            "  - fixed_panel: use pre-specified feature panel from CSV file\n"
            "  - none: no feature selection"
        ),
    )

    # Screening (common to both strategies)
    screen_method: Literal["mannwhitney", "f_classif"] = "mannwhitney"
    screen_top_n: int = Field(default=0, ge=0)

    # Hybrid+Stability strategy parameters (used when strategy="hybrid_stability")
    kbest_max: int = Field(default=500, ge=1)
    k_grid: list[int] = Field(
        default_factory=lambda: [50, 100, 200, 500],
        description="k values for SelectKBest tuning (hybrid_stability strategy only)",
    )
    stability_thresh: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Stability threshold for post-hoc panel building",
    )
    stable_corr_thresh: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Correlation threshold for pruning"
    )
    stable_corr_method: Literal["pearson", "spearman"] = Field(
        default="spearman",
        description="Correlation method for pruning (spearman recommended for proteomics)",
    )

    # RFECV strategy parameters (used when strategy="rfecv")
    rfe_target_size: int = Field(
        default=50,
        ge=5,
        description="Minimum features for RFECV (stops elimination at rfe_target_size // 2)",
    )
    rfe_step_strategy: Literal["adaptive", "linear", "geometric"] | int = Field(
        default="adaptive",
        description=(
            "RFECV step strategy: adaptive (10% per iter), linear (1 per iter), "
            "geometric, or integer for fixed step size"
        ),
    )
    rfe_min_auroc_frac: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description=(
            "Early stop if AUROC drops below this fraction of max "
            "(currently unused in nested RFECV)"
        ),
    )
    rfe_consensus_thresh: float = Field(
        default=0.80,
        ge=0.5,
        le=1.0,
        description=(
            "Consensus panel threshold: include features selected in >= "
            "this fraction of CV folds"
        ),
    )
    rfe_cv_folds: int = Field(
        default=3, ge=2, description="Internal CV folds for RFECV (within each outer fold)"
    )
    rfe_kbest_prefilter: bool = Field(
        default=True,
        description=(
            "Apply k-best univariate pre-filter before RFECV to reduce "
            "computational cost (~5× speedup)"
        ),
    )
    rfe_kbest_k: int = Field(
        default=100,
        ge=10,
        description=(
            "Maximum features to retain before RFECV "
            "(reduces ~300 proteins → ~100 for 5× speedup)"
        ),
    )

    # RF permutation importance
    rf_use_permutation: bool = False
    rf_perm_repeats: int = Field(default=5, ge=1)
    rf_perm_min_importance: float = Field(default=0.0, ge=0.0)
    rf_perm_top_n: int = Field(default=100, ge=1)

    # Model-specific selector (hybrid_stability only, opt-in)
    model_selector: bool = Field(
        default=False,
        description=(
            "Enable model-specific feature selection step between KBest and classifier. "
            "Only active under hybrid_stability strategy. Each model type uses its own "
            "inductive bias (L1 coefs, tree importances) to further prune features."
        ),
    )
    model_selector_threshold: str = Field(
        default="median",
        description="Importance threshold for SelectFromModel (e.g. 'median', 'mean', or float).",
    )
    model_selector_max_features: int | None = Field(
        default=None,
        ge=1,
        description="Maximum features to retain from model selector. None means no cap.",
    )
    model_selector_min_features: int = Field(
        default=10,
        ge=1,
        description="Minimum features to retain (floor to prevent empty selection).",
    )

    # Coefficient threshold for linear model feature extraction
    coef_threshold: float = Field(default=0.01, ge=0.0)

    # Fixed panel configuration (used when strategy="fixed_panel")
    fixed_panel_csv: str | None = Field(
        default=None,
        description=(
            "Path to CSV file containing fixed panel features. "
            "Relative paths are resolved from configs/ directory. "
            "CSV should have 'protein' column or features in first column. "
            "Only used when feature_selection_strategy='fixed_panel'."
        ),
    )


# ============================================================================
# Model-Specific Hyperparameter Configurations
# ============================================================================


class LRConfig(BaseModel):
    """Logistic Regression hyperparameters.

    C_min/C_max/C_points define log-spaced C values for RandomizedSearchCV.
    Optuna range fields (optuna_*) are used when Optuna is enabled.
    """

    penalty: list[str] = Field(default_factory=lambda: ["l1", "l2", "elasticnet"])
    C_min: float = 0.0001
    C_max: float = 100.0
    C_points: int = 7
    l1_ratio: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    solver: str = "saga"
    max_iter: int = 1000
    class_weight_options: str = "balanced"
    random_state: int = 0
    n_iter: int | None = Field(default=None, ge=1, description="Override cv.n_iter for LR")

    # Optuna-specific ranges
    optuna_C: tuple[float, float] | None = Field(
        default=None, description="(min, max) for C (log-scale), e.g. (1e-5, 100)"
    )
    optuna_l1_ratio: tuple[float, float] | None = Field(
        default=None, description="(min, max) for l1_ratio, e.g. (0.0, 1.0)"
    )


class SVMConfig(BaseModel):
    """Support Vector Machine hyperparameters.

    C_min/C_max/C_points define log-spaced C values for RandomizedSearchCV.
    Optuna range fields (optuna_*) are used when Optuna is enabled.
    """

    C_min: float = 0.01
    C_max: float = 10.0
    C_points: int = 4
    kernel: list[str] = Field(default_factory=lambda: ["linear", "rbf"])
    gamma: list[str | float] = Field(default_factory=lambda: ["scale", "auto", 0.001, 0.01])
    class_weight_options: str = "balanced"
    max_iter: int = 5000
    probability: bool = True
    random_state: int = 0
    n_iter: int | None = Field(default=None, ge=1, description="Override cv.n_iter for SVM")

    # Optuna-specific ranges
    optuna_C: tuple[float, float] | None = Field(
        default=None, description="(min, max) for C (log-scale), e.g. (1e-3, 100)"
    )


class RFConfig(BaseModel):
    """Random Forest hyperparameters.

    Grid fields (*_grid) are used for RandomizedSearchCV.
    Optuna range fields (optuna_*) are used when Optuna is enabled.
    If Optuna range fields are not specified, ranges are derived from grid fields.
    """

    # Grid-based parameters for RandomizedSearchCV
    n_estimators_grid: list[int] = Field(default_factory=lambda: [100, 300, 500])
    max_depth_grid: list[int | None] = Field(default_factory=lambda: [None, 10, 20, 30])
    min_samples_split_grid: list[int] = Field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf_grid: list[int] = Field(default_factory=lambda: [1, 2, 4])
    max_features_grid: list[str | float] = Field(default_factory=lambda: ["sqrt", "log2", 0.5])
    class_weight_options: str = "balanced"

    # Optuna-specific ranges (wider, for better exploration)
    # These override grid-derived ranges when Optuna is enabled
    optuna_n_estimators: tuple[int, int] | None = Field(
        default=None, description="(min, max) for n_estimators, e.g. (50, 500)"
    )
    optuna_max_depth: tuple[int, int] | None = Field(
        default=None, description="(min, max) for max_depth, e.g. (3, 20)"
    )
    optuna_min_samples_split: tuple[int, int] | None = Field(
        default=None, description="(min, max) for min_samples_split, e.g. (2, 20)"
    )
    optuna_min_samples_leaf: tuple[int, int] | None = Field(
        default=None, description="(min, max) for min_samples_leaf, e.g. (1, 10)"
    )
    optuna_max_features: tuple[float, float] | None = Field(
        default=None, description="(min, max) fraction for max_features, e.g. (0.1, 1.0)"
    )

    # Fixed parameters
    n_jobs: int = -1
    random_state: int = 0
    n_iter: int | None = Field(default=None, ge=1, description="Override cv.n_iter for RF")


class XGBoostConfig(BaseModel):
    """XGBoost hyperparameters.

    Grid fields (*_grid) are used for RandomizedSearchCV.
    Optuna range fields (optuna_*) are used when Optuna is enabled.
    If Optuna range fields are not specified, ranges are derived from grid fields.
    """

    # Grid-based parameters for RandomizedSearchCV
    n_estimators_grid: list[int] = Field(default_factory=lambda: [100, 300, 500])
    max_depth_grid: list[int] = Field(default_factory=lambda: [3, 5, 7, 10])
    learning_rate_grid: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.3])
    min_child_weight_grid: list[int] = Field(default_factory=lambda: [1, 3, 5])
    gamma_grid: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.3])
    subsample_grid: list[float] = Field(default_factory=lambda: [0.7, 0.8, 1.0])
    colsample_bytree_grid: list[float] = Field(default_factory=lambda: [0.7, 0.8, 1.0])
    reg_alpha_grid: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.1])
    reg_lambda_grid: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])
    scale_pos_weight_grid: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])

    # Optuna-specific ranges (wider, with proper log-scale sampling)
    # These override grid-derived ranges when Optuna is enabled
    optuna_n_estimators: tuple[int, int] | None = Field(
        default=None, description="(min, max) for n_estimators, e.g. (50, 500)"
    )
    optuna_max_depth: tuple[int, int] | None = Field(
        default=None, description="(min, max) for max_depth, e.g. (2, 12)"
    )
    optuna_learning_rate: tuple[float, float] | None = Field(
        default=None, description="(min, max) for learning_rate (log-scale), e.g. (0.001, 0.3)"
    )
    optuna_min_child_weight: tuple[float, float] | None = Field(
        default=None, description="(min, max) for min_child_weight (log-scale), e.g. (0.1, 10)"
    )
    optuna_gamma: tuple[float, float] | None = Field(
        default=None, description="(min, max) for gamma, e.g. (0.0, 1.0)"
    )
    optuna_subsample: tuple[float, float] | None = Field(
        default=None, description="(min, max) for subsample, e.g. (0.5, 1.0)"
    )
    optuna_colsample_bytree: tuple[float, float] | None = Field(
        default=None, description="(min, max) for colsample_bytree, e.g. (0.5, 1.0)"
    )
    optuna_reg_alpha: tuple[float, float] | None = Field(
        default=None, description="(min, max) for reg_alpha (log-scale), e.g. (1e-8, 1.0)"
    )
    optuna_reg_lambda: tuple[float, float] | None = Field(
        default=None, description="(min, max) for reg_lambda (log-scale), e.g. (1e-8, 10.0)"
    )

    # Fixed parameters
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 0
    n_iter: int | None = Field(default=None, ge=1, description="Override cv.n_iter for XGBoost")


class CalibrationConfig(BaseModel):
    """Calibration wrapper configuration.

    Attributes:
        enabled: Whether calibration is enabled at all.
        strategy: Calibration strategy to use:
            - "per_fold": Apply CalibratedClassifierCV inside each CV fold
                (default, current behavior).
            - "oof_posthoc": Collect raw OOF predictions, then fit a single
                calibrator post-hoc.
            - "none": No calibration applied.
        method: Calibration method ("sigmoid" for Platt scaling,
            "isotonic" for isotonic regression).
        cv: Number of CV folds for per_fold calibration.
        per_model: Optional per-model strategy overrides. Keys are model names
            (e.g., "LR_EN"), values are strategy names ("per_fold", "oof_posthoc",
            "none").
    """

    enabled: bool = True
    strategy: Literal["per_fold", "oof_posthoc", "none"] = "per_fold"
    method: Literal["sigmoid", "isotonic"] = "isotonic"
    cv: int = 5
    per_model: dict[str, Literal["per_fold", "oof_posthoc", "none"]] | None = None

    def get_strategy_for_model(self, model_name: str) -> str:
        """Get the effective calibration strategy for a specific model.

        Args:
            model_name: Name of the model (e.g., "LR_EN", "RF").

        Returns:
            The calibration strategy to use for this model.
        """
        if not self.enabled:
            return "none"
        if self.per_model and model_name in self.per_model:
            return self.per_model[model_name]
        return self.strategy


class OptunaConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization."""

    enabled: bool = False
    n_trials: int = Field(default=100, ge=1)
    timeout: float | None = Field(default=None, ge=0)
    sampler: Literal["tpe", "random", "cmaes", "grid"] = "tpe"
    sampler_seed: int | None = None
    pruner: Literal["median", "percentile", "hyperband", "none"] = "median"
    pruner_n_startup_trials: int = Field(default=5, ge=0)
    pruner_percentile: float = Field(default=25.0, ge=0, le=100)
    n_jobs: int = Field(default=1, ge=-1)  # -1 means use all available CPUs
    storage: str | None = None
    study_name: str | None = None
    load_if_exists: bool = False
    save_study: bool = True
    save_trials_csv: bool = True
    direction: Literal["minimize", "maximize"] | None = None

    # Multi-objective optimization settings
    multi_objective: bool = False
    objectives: list[str] = Field(
        default_factory=lambda: ["roc_auc", "neg_brier_score"],
        description="Metrics to optimize (requires 2+ for multi-objective)",
    )
    pareto_selection: Literal["knee", "extreme_auroc", "balanced"] = "knee"

    @model_validator(mode="after")
    def validate_multi_objective(self) -> "OptunaConfig":
        """Validate multi-objective configuration."""
        if self.multi_objective and len(self.objectives) < 2:
            raise ValueError(
                "multi_objective=True requires at least 2 objectives, "
                f"got {len(self.objectives)}"
            )

        supported = ["roc_auc", "neg_brier_score", "average_precision"]
        for obj in self.objectives:
            if obj not in supported:
                raise ValueError(
                    f"Unsupported objective: {obj}. " f"Supported objectives: {supported}"
                )

        if self.multi_objective and self.sampler == "cmaes":
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "CMA-ES sampler with multi-objective may be unstable. "
                "Consider sampler='tpe' for multi-objective optimization."
            )

        return self


# ============================================================================
# Threshold Selection Configuration
# ============================================================================


class ThresholdConfig(BaseModel):
    """Configuration for threshold selection."""

    objective: Literal["max_f1", "max_fbeta", "youden", "fixed_spec", "fixed_ppv"] = "max_f1"
    fbeta: float = Field(default=1.0, gt=0.0)
    fixed_spec: float = Field(default=0.90, ge=0.0, le=1.0)
    fixed_ppv: float = Field(default=0.10, ge=0.0, le=1.0)
    threshold_source: Literal["val", "test", "train_oof"] = "val"
    target_prevalence_source: Literal["val", "test", "train", "fixed"] = "test"
    target_prevalence_fixed: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_prob_source: Literal["val", "test"] = "test"


# ============================================================================
# Evaluation and Reporting Configuration
# ============================================================================


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and reporting."""

    # Bootstrap confidence intervals
    test_ci_bootstrap: bool = True
    n_boot: int = Field(default=500, ge=100)
    boot_random_state: int = 0
    bootstrap_min_samples: int = Field(
        default=100,
        ge=10,
        description="Compute bootstrap CI only when test set has fewer than this many samples",
    )

    # Learning curves
    learning_curve: bool = False
    lc_train_sizes: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])

    # Feature importance
    feature_reports: bool = True
    feature_report_max: int = 100

    # Specificity/sensitivity targets
    control_spec_targets: list[float] = Field(default_factory=lambda: [0.90, 0.95, 0.99])
    toprisk_fracs: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10])


# ============================================================================
# Decision Curve Analysis Configuration
# ============================================================================


class DCAConfig(BaseModel):
    """Configuration for decision curve analysis."""

    compute_dca: bool = False
    dca_threshold_min: float = Field(default=0.0005, ge=0.0, le=1.0)
    dca_threshold_max: float = Field(default=1.0, ge=0.0, le=1.0)
    dca_threshold_step: float = Field(default=0.001, gt=0.0)
    dca_report_points: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])


# ============================================================================
# Output Control Configuration
# ============================================================================


class OutputConfig(BaseModel):
    """Configuration for output file generation."""

    model_config = ConfigDict(extra="forbid")

    save_train_preds: bool = False
    save_train_oof: bool = True
    save_val_preds: bool = True
    save_test_preds: bool = True
    save_calibration: bool = True
    calib_bins: int = Field(default=10, ge=2)
    save_feature_importance: bool = True
    feature_reports: bool = True
    save_plots: bool = True
    max_plot_splits: int = Field(default=0, ge=0)
    plot_format: str = "png"
    plot_dpi: int = 300

    # Individual plot type controls
    plot_roc: bool = True
    plot_pr: bool = True
    plot_calibration: bool = True
    plot_risk_distribution: bool = True
    plot_dca: bool = True
    plot_learning_curve: bool = True
    plot_oof_combined: bool = True
    plot_optuna: bool = True


# ============================================================================
# Strictness and Validation Configuration
# ============================================================================


class StrictnessConfig(BaseModel):
    """Configuration for validation strictness."""

    level: Literal["off", "warn", "error"] = "warn"
    check_split_overlap: bool = True
    check_prevalent_in_eval: bool = True
    check_threshold_source: bool = True
    check_prevalence_adjustment: bool = True
    check_feature_leakage: bool = True


class ComputeConfig(BaseModel):
    """Configuration for compute resources."""

    cpus: int = Field(default_factory=lambda: os.cpu_count() or 1)
    tune_n_jobs: int | None = None


# ============================================================================
# Master Training Configuration
# ============================================================================


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    # Data
    infile: Path | None = None
    split_dir: Path | None = None
    scenario: str | None = None  # Auto-detect from split files
    split_seed: int = 0

    # Model selection
    model: str = ModelName.LR_EN

    # Sub-configurations
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dca: DCAConfig = Field(default_factory=DCAConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    strictness: StrictnessConfig = Field(default_factory=StrictnessConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    # Model-specific hyperparameters
    lr: LRConfig = Field(default_factory=LRConfig)
    svm: SVMConfig = Field(default_factory=SVMConfig)
    rf: RFConfig = Field(default_factory=RFConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)

    # Output
    outdir: Path = Field(default=Path("../results"))
    run_name: str | None = None
    run_id: str | None = None

    # Resources
    n_jobs: int = -1
    verbose: int = 1

    @model_validator(mode="after")
    def validate_config(self):
        """Cross-field validation."""
        # Ensure threshold source is available
        if self.thresholds.threshold_source == "val" and self.cv.folds < 2:
            raise ValueError("threshold_source='val' requires val_size > 0")

        # Check prevalence source consistency
        if self.thresholds.target_prevalence_source == "fixed":
            if self.thresholds.target_prevalence_fixed is None:
                raise ValueError(
                    "target_prevalence_source='fixed' requires target_prevalence_fixed"
                )

        return self


# ============================================================================
# Master Configuration (All Subcommands)
# ============================================================================


class AggregateConfig(BaseModel):
    """Configuration for aggregate-splits command."""

    # Input/output
    results_dir: Path = Field(default=Path("../results"))
    outdir: Path = Field(default=Path("../results_aggregated"))

    # Discovery
    split_pattern: str = "split_seed*"

    # Pooling settings
    predictions_method: Literal["median", "mean", "vote"] = "median"
    save_individual: bool = False

    # Summary statistics
    summary_stats: list[str] = Field(default_factory=lambda: ["mean", "std", "median", "ci95"])
    group_by: list[str] = Field(default_factory=lambda: ["scenario", "model"])

    # Consensus panels
    min_stability: float = Field(default=0.7, ge=0.0, le=1.0)
    corr_method: Literal["pearson", "spearman"] = "pearson"
    corr_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # Output control
    save_pooled_preds: bool = True
    save_summary_csv: bool = True
    save_plots: bool = True
    save_thresholds: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300

    # Individual plot type controls
    plot_roc: bool = True
    plot_pr: bool = True
    plot_calibration: bool = True
    plot_risk_distribution: bool = True
    plot_dca: bool = True
    plot_oof_combined: bool = True
    plot_learning_curve: bool = True


class PanelOptimizeConfig(BaseModel):
    """Configuration for panel size optimization via RFE.

    Used by `ced optimize-panel` command to find minimum viable protein panels
    through Recursive Feature Elimination after training.
    """

    model_config = ConfigDict(protected_namespaces=())

    # Required paths
    infile: Path | None = Field(
        default=None,
        description="Path to input data file (Parquet/CSV)",
    )
    split_dir: Path | None = Field(
        default=None,
        description="Directory containing split indices",
    )
    model_path: Path | None = Field(
        default=None,
        description="Path to trained model bundle (.joblib)",
    )

    # Split and scenario
    split_seed: int = Field(
        default=0,
        ge=0,
        description="Split seed to use (must match training split)",
    )
    scenario: str | None = Field(
        default=None,
        description="Scenario name (auto-detected from model if not specified)",
    )

    # RFE parameters
    start_size: int = Field(
        default=100,
        ge=5,
        description="Starting panel size (top N from stability ranking)",
    )
    min_size: int = Field(
        default=5,
        ge=1,
        description="Minimum panel size to evaluate",
    )
    min_auroc_frac: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Early stop if AUROC drops below this fraction of max",
    )

    # Cross-validation
    cv_folds: int = Field(
        default=5,
        ge=0,
        description="CV folds for OOF AUROC estimation (0=skip CV, use train AUROC)",
    )

    # Elimination strategy
    step_strategy: Literal["adaptive", "linear", "geometric"] = Field(
        default="adaptive",
        description="Feature elimination strategy: adaptive (10%/iter), linear (1/iter), geometric",
    )

    # Feature initialization
    use_stability_panel: bool = Field(
        default=True,
        description="If True, start from stability ranking; else use all proteins",
    )

    # Output
    outdir: Path | None = Field(
        default=None,
        description="Output directory (default: alongside model in optimize_panel/)",
    )

    # Verbosity
    verbose: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Verbosity level: 0=warnings, 1=info, 2=debug",
    )

    @model_validator(mode="after")
    def validate_sizes(self) -> "PanelOptimizeConfig":
        """Validate that min_size <= start_size."""
        if self.min_size > self.start_size:
            raise ValueError(
                f"min_size ({self.min_size}) cannot be greater than "
                f"start_size ({self.start_size})"
            )
        return self


class HoldoutEvalConfig(BaseModel):
    """Configuration for holdout evaluation."""

    model_config = ConfigDict(protected_namespaces=())

    # Data paths
    infile: Path
    holdout_idx: Path
    model_artifact: Path
    outdir: Path = Field(default=Path("../holdout_results"))

    # Evaluation settings
    scenario: str | None = None
    compute_dca: bool = True
    save_preds: bool = True
    toprisk_fracs: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10])
    subgroup_min_n: int = Field(default=40, ge=1)

    # DCA settings
    dca_threshold_min: float = Field(default=0.0005, ge=0.0)
    dca_threshold_max: float = Field(default=1.0, ge=0.0, le=1.0)
    dca_threshold_step: float = Field(default=0.001, gt=0.0)
    dca_report_points: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    dca_use_target_prevalence: bool = False

    # Clinical thresholds
    clinical_threshold_points: list[float] = Field(default_factory=list)
    target_prevalence: float | None = Field(default=None, ge=0.0, le=1.0)


# ============================================================================
# HPC Configuration
# ============================================================================


class HPCResourceConfig(BaseModel):
    """HPC resource allocation for a single job type."""

    queue: str = "medium"
    cores: int = Field(default=4, ge=1)
    mem_per_core: int = Field(default=4000, ge=1, description="Memory per core in MB")
    walltime: str = Field(default="24:00", description="Wall time limit as HH:MM")


class HPCConfig(BaseModel):
    """HPC-specific configuration.

    Attributes:
        project: HPC project allocation code (e.g., acc_elahi).
        scheduler: Scheduler type (currently only 'lsf' supported).
        queue: Default queue for job submission.
        cores: Default number of CPU cores per job.
        mem_per_core: Default memory per core in MB.
        walltime: Default wall time limit as HH:MM string.
        training: Optional resource override for training jobs.
        postprocessing: Optional resource override for aggregation/ensemble jobs.
        optimization: Optional resource override for panel optimization jobs.
    """

    project: str = Field(
        ...,
        description="HPC project allocation code (e.g., acc_elahi, required)",
    )
    scheduler: str = Field(default="lsf", description="Scheduler type (lsf only)")
    queue: str = Field(default="medium", description="Default queue name")
    cores: int = Field(default=4, ge=1, description="Default number of CPU cores")
    mem_per_core: int = Field(default=4000, ge=1, description="Default memory per core in MB")
    walltime: str = Field(default="24:00", description="Default wall time limit as HH:MM")

    # Optional per-stage resource overrides
    training: HPCResourceConfig | None = None
    postprocessing: HPCResourceConfig | None = None
    optimization: HPCResourceConfig | None = None

    @model_validator(mode="after")
    def validate_project(self) -> "HPCConfig":
        """Validate that project is not a placeholder."""
        placeholders = {"YOUR_PROJECT_ALLOCATION", "YOUR_ALLOCATION"}
        if self.project in placeholders:
            raise ValueError(
                f"HPC project not configured. Got placeholder '{self.project}'. "
                "Update 'hpc.project' in pipeline_hpc.yaml"
            )
        return self

    def get_resources(self, stage: str = "default") -> dict[str, int | str]:
        """Get resource config for a specific pipeline stage.

        Args:
            stage: Pipeline stage ('training', 'postprocessing', 'optimization', 'default').

        Returns:
            Dict with keys: queue, cores, mem_per_core, walltime.
        """
        if stage == "training" and self.training:
            return {
                "queue": self.training.queue,
                "cores": self.training.cores,
                "mem_per_core": self.training.mem_per_core,
                "walltime": self.training.walltime,
            }
        elif stage == "postprocessing" and self.postprocessing:
            return {
                "queue": self.postprocessing.queue,
                "cores": self.postprocessing.cores,
                "mem_per_core": self.postprocessing.mem_per_core,
                "walltime": self.postprocessing.walltime,
            }
        elif stage == "optimization" and self.optimization:
            return {
                "queue": self.optimization.queue,
                "cores": self.optimization.cores,
                "mem_per_core": self.optimization.mem_per_core,
                "walltime": self.optimization.walltime,
            }
        else:
            return {
                "queue": self.queue,
                "cores": self.cores,
                "mem_per_core": self.mem_per_core,
                "walltime": self.walltime,
            }


class RootConfig(BaseModel):
    """Root configuration for all CeD-ML commands."""

    # Common
    random_state: int = 0
    verbose: int = 1

    # Sub-configs (populated based on command)
    splits: SplitsConfig | None = None
    training: TrainingConfig | None = None
    aggregate: AggregateConfig | None = None
    holdout: HoldoutEvalConfig | None = None
    panel_optimize: PanelOptimizeConfig | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
