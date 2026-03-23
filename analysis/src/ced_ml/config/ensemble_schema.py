"""Ensemble configuration schema for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class EnsembleConfig(BaseModel):
    """Configuration for the stacking ensemble meta-learner.

    Attributes:
        meta_penalty: Regularization type for the LR meta-learner.
        meta_c: Inverse regularization strength. Larger C = less regularization.
            With only 4 base-model features and n~900, high C avoids probability
            compression from over-regularization.
        calibrate_meta: Whether to wrap the LR meta-learner in CalibratedClassifierCV.
            Defaults to False because base models are already OOF-calibrated and the
            logistic regression meta-learner is inherently calibrated via its logistic
            link function.  Set to True only when base models are uncalibrated or when
            you have a specific reason to apply an additional calibration layer (and
            accept that double calibration may distort probabilities).
        meta_calibration_method: sklearn CalibratedClassifierCV method when
            calibrate_meta=True. 'isotonic' or 'sigmoid' (Platt scaling).
        calibration_cv: Number of CV folds for meta-learner calibration.
            Only used when calibrate_meta=True.
    """

    meta_penalty: Literal["l1", "l2", "none"] = Field(
        default="l2",
        description="Meta-learner regularization type.",
    )
    meta_c: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Inverse regularization strength for meta-learner. "
            "Higher values = less regularization. With few features (4 base models) "
            "and ample samples, high C prevents probability compression."
        ),
    )
    calibrate_meta: bool = Field(
        default=False,
        description=(
            "Wrap meta-learner in CalibratedClassifierCV. "
            "Default False: base models are already OOF-calibrated and LR calibrates "
            "naturally via the logistic link."
        ),
    )
    meta_calibration_method: Literal["isotonic", "sigmoid"] = Field(
        default="isotonic",
        description="Calibration method when calibrate_meta=True.",
    )
    calibration_cv: int = Field(
        default=5,
        ge=2,
        description="CV folds for meta-learner calibration (only used when calibrate_meta=True).",
    )
