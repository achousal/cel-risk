"""Training configuration schema for CeD-ML pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..data.schema import ModelName
from .calibration_schema import CalibrationConfig, ThresholdConfig
from .compute_schema import ComputeConfig
from .cv_schema import CVConfig, OptunaConfig
from .data_schema import ColumnsConfig
from .evaluation_schema import DCAConfig, EvaluationConfig
from .features_schema import FeatureConfig
from .model_schema import LRConfig, RFConfig, SVMConfig, XGBoostConfig
from .output_schema import OutputConfig, StrictnessConfig


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

    # Evaluation flags
    allow_test_thresholding: bool = False  # Explicit override for threshold-on-test

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
