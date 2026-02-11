"""Aggregate, panel optimization, and holdout evaluation configuration schemas for CeD-ML pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .output_schema import OutputConfig


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

    # Output configuration (loaded from output_config.yaml)
    output: OutputConfig = Field(default_factory=OutputConfig)


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
    step_strategy: Literal["adaptive", "linear", "geometric", "fine"] = Field(
        default="fine",
        description="Feature elimination strategy: fine (geometric+quarter-steps), geometric, adaptive, linear",
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
