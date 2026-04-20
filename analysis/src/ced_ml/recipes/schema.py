"""Pydantic v2 models for the recipe manifest.

The manifest declares how panels are derived (trunk, ordering, size rule)
and which factorial factors to cross for cell tuning.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderingType(str, Enum):
    """Supported protein ordering strategies."""

    consensus_score_descending = "consensus_score_descending"
    stream_balanced = "stream_balanced"
    abs_coefficient_descending = "abs_coefficient_descending"
    oof_importance = "oof_importance"
    rfe_elimination = "rfe_elimination"
    purity_ordering = "purity_ordering"


class SizeRuleType(str, Enum):
    """Supported panel size derivation rules."""

    three_criterion = "three_criterion"
    three_criterion_unanimous = "three_criterion_unanimous"
    stability = "stability"
    significance_count = "significance_count"
    fixed = "fixed"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class OrderingRule(BaseModel):
    """How proteins within a recipe are ordered."""

    type: OrderingType
    params: dict[str, Any] = Field(default_factory=dict)


class SizeRule(BaseModel):
    """How optimal panel size is determined."""

    type: SizeRuleType
    params: dict[str, Any] = Field(default_factory=dict)


class TrunkConfig(BaseModel):
    """Source data references for a derivation trunk."""

    id: str = Field(description="Unique trunk identifier (e.g. T1, T2)")
    description: str = ""
    proteins_csv: Path = Field(
        description="CSV with protein identifiers and scores (e.g. RRA q-values)"
    )
    sweep_csv: Path | None = Field(
        default=None,
        description="Aggregated sweep results CSV (required for three_criterion size rule)",
    )
    feature_csv: Path | None = Field(
        default=None,
        description="Feature consistency CSV (required for stability size rule)",
    )


class RecipeConfig(BaseModel):
    """A single recipe: trunk + ordering + size rule → derived panel."""

    id: str = Field(description="Unique recipe identifier (e.g. R1_consensus)")
    trunk_id: str = Field(description="Reference to a trunk defined in trunks[]")
    description: str = ""
    ordering: OrderingRule
    size_rule: SizeRule

    # Optional filter applied to sweep_csv before size derivation
    sweep_filter: dict[str, Any] | None = Field(
        default=None,
        description="Column filters for sweep CSV (e.g. {order: rra})",
    )

    # Model-specific recipes: pin to a single model.
    # Factorial crosses only calibration × weighting × downsampling (not models).
    pinned_model: str | None = Field(
        default=None,
        description="If set, this recipe is model-specific. Factorial uses only this model.",
    )

    # Nested panel expansion: auto-generate sub-recipes at every size from
    # the derived plateau down to core_size (inclusive). Each sub-recipe
    # uses the same ordering, just truncated to top-N.
    expand_to_core: int | None = Field(
        default=None,
        description=(
            "If set, expand this recipe into nested sub-recipes from the "
            "derived optimal size down to this value (e.g. 4 for the significance core). "
            "Each sub-recipe is named {recipe_id}_p{N}."
        ),
    )


class FactorialConfig(BaseModel):
    """Factorial factors for cell tuning."""

    models: list[str] = Field(description="Model identifiers (e.g. LinSVM_cal, XGBoost)")
    calibration: list[str] = Field(description="Calibration methods")
    weighting: list[str] = Field(description="Class weight strategies (log, sqrt, none)")
    downsampling: list[float] = Field(
        description="Majority downsampling ratios (1.0 = no downsampling)",
    )

    @property
    def n_cells(self) -> int:
        """Total number of factorial cells."""
        return (
            len(self.models) * len(self.calibration) * len(self.weighting) * len(self.downsampling)
        )


class OptunaOverrides(BaseModel):
    """Optuna settings applied to every factorial cell."""

    n_trials: int = 200
    storage_path: str | None = Field(
        default=None,
        description=(
            "Shared filesystem directory for Optuna study persistence. "
            "Each recipe gets its own file: {storage_path}/{recipe_id}.optuna.journal"
        ),
    )
    storage_backend: Literal["journal", "sqlite", "none"] = Field(
        default="none",
        description="Storage backend: 'journal' (append-only, lock-free), 'sqlite', or 'none' (in-memory)",
    )
    warm_start_params: Path | None = Field(
        default=None,
        description="JSON file of scout top-K params per model (output of ced_ml.utils.optuna_warmstart.extract_top_params)",
    )
    warm_start_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of scout params to enqueue per model",
    )


class SharedSplits(BaseModel):
    """Shared split configuration across all cells."""

    seed_start: int = 100
    seed_end: int = 129

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_end + 1))

    @property
    def n_seeds(self) -> int:
        return self.seed_end - self.seed_start + 1


# ---------------------------------------------------------------------------
# V0 Gate: training strategy comparison
# ---------------------------------------------------------------------------


class TrainingStrategy(BaseModel):
    """One V0 training strategy configuration.

    Each strategy produces a splits_config overlay that changes how
    prevalent cases are handled during training.
    """

    name: str = Field(
        description="Strategy identifier (e.g. IncidentOnly, IncidentPlusPrevalent_0.5)"
    )
    scenarios: list[str] = Field(
        description="Scenario list for splits_config (e.g. ['IncidentOnly'])"
    )
    prevalent_train_only: bool = Field(
        default=True,
        description="Whether prevalent cases appear only in TRAIN (not VAL/TEST)",
    )
    prevalent_train_frac: float | None = Field(
        default=None,
        description="Fraction of prevalent cases to include in training (only for IncidentPlusPrevalent)",
    )

    def to_splits_overlay(self, control_ratio: int | None = None) -> dict[str, Any]:
        """Generate the splits_config.yaml overlay dict for this strategy.

        Parameters
        ----------
        control_ratio : int, optional
            If provided, overrides train_control_per_case in the splits config.
        """
        overlay: dict[str, Any] = {
            "scenarios": self.scenarios,
            "prevalent_train_only": self.prevalent_train_only,
        }
        if self.prevalent_train_frac is not None:
            overlay["prevalent_train_frac"] = self.prevalent_train_frac
        if control_ratio is not None:
            overlay["train_control_per_case"] = control_ratio
            overlay["eval_control_per_case"] = control_ratio
        return overlay


class V0GateConfig(BaseModel):
    """V0 gate configuration: training strategy comparison before main factorial."""

    strategies: list[TrainingStrategy] = Field(
        description="Training strategies to compare",
    )
    control_ratios: list[int] = Field(
        default=[5],
        description="Control-per-case ratios to test (e.g. [1, 5, 10]). Crosses with strategies.",
    )
    representative_recipes: list[str] = Field(
        description="Recipe IDs to test (e.g. ['R1_sig', 'R1_plateau'])",
    )
    optuna_n_trials: int = Field(
        default=100,
        description="Optuna budget for gate (sweep-level, not final)",
    )


# ---------------------------------------------------------------------------
# Top-level manifest
# ---------------------------------------------------------------------------


class Manifest(BaseModel):
    """Top-level recipe manifest — the single source of truth for panel derivation."""

    trunks: list[TrunkConfig]
    recipes: list[RecipeConfig]
    factorial: FactorialConfig
    optuna: OptunaOverrides = Field(default_factory=OptunaOverrides)
    splits: SharedSplits = Field(default_factory=SharedSplits)
    v0_gate: V0GateConfig | None = Field(
        default=None,
        description="V0 gate config. If set, generate_v0_configs() produces strategy comparison cells.",
    )

    # Base config paths for YAML inheritance
    base_training_config: Path = Field(
        description="Path to base training_config.yaml for _base inheritance",
    )
    base_pipeline_config: Path = Field(
        description="Path to base pipeline_hpc.yaml for _base inheritance",
    )
    base_splits_config: Path = Field(
        default=Path("splits_config.yaml"),
        description="Path to base splits_config.yaml for V0 overlay",
    )

    @model_validator(mode="after")
    def validate_trunk_references(self) -> Manifest:
        """Ensure every recipe references a defined trunk."""
        trunk_ids = {t.id for t in self.trunks}
        for recipe in self.recipes:
            if recipe.trunk_id not in trunk_ids:
                raise ValueError(
                    f"Recipe '{recipe.id}' references trunk '{recipe.trunk_id}' "
                    f"which is not defined. Available: {trunk_ids}"
                )
        return self

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Manifest:
        """Ensure trunk and recipe IDs are unique."""
        trunk_ids = [t.id for t in self.trunks]
        if len(trunk_ids) != len(set(trunk_ids)):
            raise ValueError(f"Duplicate trunk IDs: {trunk_ids}")
        recipe_ids = [r.id for r in self.recipes]
        if len(recipe_ids) != len(set(recipe_ids)):
            raise ValueError(f"Duplicate recipe IDs: {recipe_ids}")
        return self

    def get_trunk(self, trunk_id: str) -> TrunkConfig:
        """Look up a trunk by ID."""
        for trunk in self.trunks:
            if trunk.id == trunk_id:
                return trunk
        raise KeyError(f"Trunk '{trunk_id}' not found")
