"""Calibration and threshold configuration schemas for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


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
