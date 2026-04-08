"""Cross-validation and Optuna configuration schemas for CeD-ML pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    storage_backend: Literal["journal", "sqlite", "none"] = Field(
        default="none",
        description="Storage backend selector: 'journal' for JournalStorage, 'sqlite' for SQLite, 'none' for in-memory",
    )
    user_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="User attributes to tag on the Optuna study (factorial metadata)",
    )
    warm_start_params_file: str | None = Field(
        default=None,
        description="Path to JSON file of scout top-K params per model for warm-starting",
    )
    warm_start_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of scout params to enqueue per model",
    )
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
