"""Evaluation and DCA configuration schemas for CeD-ML pipeline."""

from pydantic import BaseModel, Field


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


class DCAConfig(BaseModel):
    """Configuration for decision curve analysis."""

    compute_dca: bool = False
    dca_threshold_min: float = Field(default=0.0005, ge=0.0, le=1.0)
    dca_threshold_max: float = Field(default=1.0, ge=0.0, le=1.0)
    dca_threshold_step: float = Field(default=0.001, gt=0.0)
    dca_report_points: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
