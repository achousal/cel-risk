"""Feature selection configuration schema for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    """Configuration for feature selection methods.

    Two mutually exclusive strategies:
    1. hybrid_stability (default): screen -> kbest (tuned) -> stability -> model
       - Robust, interpretable, uses k_grid tuning
       - Best for: production models, reproducibility
    2. rfecv: screen -> light kbest cap -> RFECV -> model
       - Automatic size discovery, can churn across folds
       - Best for: scientific discovery, understanding feature stability
    """

    model_config = ConfigDict(protected_namespaces=())

    # Feature selection strategy (mutually exclusive paths)
    feature_selection_strategy: Literal["hybrid_stability", "rfecv", "fixed_panel", "none"] = Field(
        default="hybrid_stability",
        description=(
            "Feature selection strategy:\n"
            "  - hybrid_stability: screen -> kbest (tuned) -> stability -> model\n"
            "  - rfecv: screen -> light kbest cap -> RFECV -> model\n"
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
        default=3,
        ge=2,
        description="Internal CV folds for RFECV (within each outer fold)",
    )
    rfe_kbest_prefilter: bool = Field(
        default=True,
        description=(
            "Apply k-best univariate pre-filter before RFECV to reduce "
            "computational cost (~5x speedup)"
        ),
    )
    rfe_kbest_k: int = Field(
        default=100,
        ge=10,
        description=(
            "Maximum features to retain before RFECV "
            "(reduces ~300 proteins -> ~100 for 5x speedup)"
        ),
    )

    # RF permutation importance
    rf_use_permutation: bool = False
    rf_perm_repeats: int = Field(default=5, ge=1)
    rf_perm_min_importance: float = Field(default=0.0, ge=0.0)
    rf_perm_top_n: int = Field(default=100, ge=1)

    # OOF Importance configuration
    compute_oof_importance: bool = Field(
        default=True, description="Enable OOF importance computation during training"
    )
    pfi_n_repeats: int = Field(
        default=30, ge=1, description="Number of permutation repeats for tree model PFI"
    )
    grouped_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Correlation threshold for grouping features in grouped PFI",
    )
    include_builtin: bool = Field(
        default=True, description="Also save Gini/gain importance for tree models"
    )

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
            "Relative paths are resolved from data/ directory. "
            "CSV should have 'protein' column or features in first column. "
            "Only used when feature_selection_strategy='fixed_panel'."
        ),
    )
