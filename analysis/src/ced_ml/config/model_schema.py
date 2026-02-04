"""Model-specific hyperparameter configuration schemas for CeD-ML pipeline."""

from pydantic import BaseModel, Field


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
        default=None,
        description="(min, max) fraction for max_features, e.g. (0.1, 1.0)",
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
        default=None,
        description="(min, max) for learning_rate (log-scale), e.g. (0.001, 0.3)",
    )
    optuna_min_child_weight: tuple[float, float] | None = Field(
        default=None,
        description="(min, max) for min_child_weight (log-scale), e.g. (0.1, 10)",
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
        default=None,
        description="(min, max) for reg_alpha (log-scale), e.g. (1e-8, 1.0)",
    )
    optuna_reg_lambda: tuple[float, float] | None = Field(
        default=None,
        description="(min, max) for reg_lambda (log-scale), e.g. (1e-8, 10.0)",
    )

    # Fixed parameters
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 0
    n_iter: int | None = Field(default=None, ge=1, description="Override cv.n_iter for XGBoost")
