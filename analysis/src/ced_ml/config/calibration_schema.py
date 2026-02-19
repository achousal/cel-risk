"""Calibration and threshold configuration schemas for CeD-ML pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Supported OOF calibration method names (kept in sync with OOFCalibrator).
CalibrationMethodLiteral = Literal[
    "isotonic",  # Isotonic regression (non-parametric, high variance).
    "logistic_full",  # Two-parameter Platt: logit(Y=1) = a + b*logit(p).
    "logistic_intercept",  # Intercept-only: logit(Y=1) = a + logit(p). Lowest variance.
    "beta",  # Beta calibration: logit(q) = a*log(p) + b*log(1-p) + c.
]

CalibrationStrategyLiteral = Literal["per_fold", "oof_posthoc", "none"]


class PerModelCalibrationConfig(BaseModel):
    """Per-model calibration overrides.

    Attributes:
        strategy: Override the global calibration strategy for this model.
        method: Override the global calibration method for this model.
    """

    strategy: CalibrationStrategyLiteral | None = None
    method: CalibrationMethodLiteral | None = None


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
        method: Calibration method for OOF post-hoc calibration. One of:
            "isotonic", "logistic_full", "logistic_intercept", "beta".
        cv: Number of CV folds for per_fold calibration.
        per_model: Optional per-model overrides. Keys are model names
            (e.g., "LR_EN"), values are PerModelCalibrationConfig with optional
            strategy and/or method overrides.
    """

    enabled: bool = True
    strategy: CalibrationStrategyLiteral = "per_fold"
    method: CalibrationMethodLiteral = "isotonic"
    cv: int = 5
    per_model: dict[str, PerModelCalibrationConfig] | None = None

    @field_validator("per_model", mode="before")
    @classmethod
    def _coerce_per_model(cls, v: Any) -> dict[str, PerModelCalibrationConfig] | None:
        """Coerce legacy string-valued per_model entries.

        Allows YAML configs to pass plain strategy strings
        (e.g. {"LR_EN": "oof_posthoc"}) instead of full PerModelCalibrationConfig
        dicts. String values are treated as strategy overrides with no method override.
        """
        if v is None:
            return v
        if not isinstance(v, dict):
            return v  # Let Pydantic raise the type error normally.
        coerced: dict[str, PerModelCalibrationConfig] = {}
        for model_name, val in v.items():
            if isinstance(val, str):
                coerced[model_name] = PerModelCalibrationConfig(strategy=val)
            elif isinstance(val, PerModelCalibrationConfig):
                coerced[model_name] = val
            else:
                coerced[model_name] = PerModelCalibrationConfig(**val)
        return coerced

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
            override = self.per_model[model_name]
            if override.strategy is not None:
                return override.strategy
        return self.strategy

    def get_method_for_model(self, model_name: str) -> str:
        """Get the effective calibration method for a specific model.

        Args:
            model_name: Name of the model (e.g., "LR_EN", "RF").

        Returns:
            The calibration method to use for this model.
        """
        if self.per_model and model_name in self.per_model:
            override = self.per_model[model_name]
            if override.method is not None:
                return override.method
        return self.method


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
