"""Data and split configuration schemas for CeD-ML pipeline."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ColumnsConfig(BaseModel):
    """Configuration for metadata column selection."""

    mode: Literal["auto", "explicit"] = "auto"
    numeric_metadata: list[str] | None = None
    categorical_metadata: list[str] | None = None
    warn_missing_defaults: bool = True


class SplitsConfig(BaseModel):
    """Configuration for data split generation."""

    mode: Literal["development", "holdout"] = "development"
    scenarios: list[str] = Field(default_factory=lambda: ["IncidentPlusPrevalent"])
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
