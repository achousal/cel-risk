"""Pydantic v2 models for cellml ExperimentSpec.

An ExperimentSpec is the single declarative source of truth for one
cellml experiment. It names:
  - base configs (training / pipeline / splits) to inherit from
  - panels (derived from a trunk, from a fixed CSV, or from a prior experiment)
  - factorial axes (model x calibration x weighting x downsampling x
    scenario x feature_selection x control_ratio)
  - seeds, optuna, and LSF resource defaults.

This complements (does not replace) ced_ml.recipes.schema.Manifest. A
resolved ExperimentSpec is converted into a minimal Manifest-equivalent
for re-use of recipes.config_gen.generate_factorial_configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ced_ml.recipes.schema import (
    OrderingRule,
    SizeRule,
    TrainingStrategy,
)

# ---------------------------------------------------------------------------
# Panel sources
# ---------------------------------------------------------------------------

PanelSource = Literal["derived", "fixed_csv", "reference"]


class PanelSpec(BaseModel):
    """One panel entry in the experiment spec.

    Three source modes:
      - derived: derived via ced_ml.recipes.derive from a trunk + ordering + size_rule
      - fixed_csv: an existing CSV of protein IDs, copied verbatim
      - reference: extracted from a prior registered experiment
    """

    id: str = Field(description="Unique panel identifier within this spec.")
    source: PanelSource
    pinned_model: str | None = Field(
        default=None,
        description="If set, collapses the model axis to this single model for this panel.",
    )

    # derived source fields
    trunk_id: str | None = None
    ordering: OrderingRule | None = None
    size_rule: SizeRule | None = None
    sweep_filter: dict[str, Any] | None = None
    expand_to_core: int | None = None

    # fixed_csv source fields
    csv: Path | None = None

    # reference source fields
    experiment: str | None = None
    extract: Literal["best_prauc", "union", "intersection"] | None = None

    @model_validator(mode="after")
    def _validate_source_fields(self) -> PanelSpec:
        if self.source == "derived":
            if self.trunk_id is None or self.ordering is None or self.size_rule is None:
                raise ValueError(
                    f"Panel '{self.id}': source=derived requires trunk_id, ordering, size_rule"
                )
        elif self.source == "fixed_csv":
            if self.csv is None:
                raise ValueError(f"Panel '{self.id}': source=fixed_csv requires csv path")
        elif self.source == "reference":
            if self.experiment is None:
                raise ValueError(f"Panel '{self.id}': source=reference requires experiment name")
        return self


class TrunkSpec(BaseModel):
    """Trunk declaration — matches recipes.schema.TrunkConfig but local to a spec."""

    id: str
    description: str = ""
    proteins_csv: Path
    sweep_csv: Path | None = None
    feature_csv: Path | None = None


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


class AxesSpec(BaseModel):
    """Factorial axes for the experiment.

    Any axis may be omitted or given as a single-element list; the plan
    stage expands the cartesian product. scenario entries are
    TrainingStrategy-compatible dicts (see recipes.schema.TrainingStrategy).
    """

    model: list[str]
    calibration: list[str]
    weighting: list[str]
    downsampling: list[float]
    scenario: list[TrainingStrategy] = Field(default_factory=list)
    feature_selection: list[str] = Field(default_factory=lambda: ["fixed_panel"])
    control_ratio: list[int] = Field(default_factory=lambda: [5])


# ---------------------------------------------------------------------------
# Resources + supporting structs
# ---------------------------------------------------------------------------


class BaseConfigs(BaseModel):
    training: Path
    pipeline: Path
    splits: Path


class SeedRange(BaseModel):
    start: int = 100
    end: int = 119

    @property
    def seeds(self) -> list[int]:
        return list(range(self.start, self.end + 1))

    @property
    def n_seeds(self) -> int:
        return self.end - self.start + 1


class OptunaSpec(BaseModel):
    n_trials: int = 200
    storage_path: str | None = None
    storage_backend: Literal["journal", "sqlite", "none"] = "none"
    warm_start_params: Path | None = None
    warm_start_top_k: int = 5


class ResourcesSpec(BaseModel):
    wall: str = "48:00"
    cores: int = 12
    mem_mb_per_core: int = 8000
    queue: str = "premium"
    project: str = "acc_vascbrain"


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


class ExperimentSpec(BaseModel):
    """A complete experiment specification."""

    name: str = Field(description="Unique experiment name (used for registry + LSF job name)")
    description: str = ""

    base_configs: BaseConfigs
    trunks: list[TrunkSpec] = Field(default_factory=list)
    panels: list[PanelSpec]
    axes: AxesSpec

    seeds: SeedRange = Field(default_factory=SeedRange)
    optuna: OptunaSpec = Field(default_factory=OptunaSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)

    @model_validator(mode="after")
    def _validate_trunk_refs(self) -> ExperimentSpec:
        trunk_ids = {t.id for t in self.trunks}
        for panel in self.panels:
            if panel.source == "derived" and panel.trunk_id not in trunk_ids:
                raise ValueError(
                    f"Panel '{panel.id}' references trunk '{panel.trunk_id}' "
                    f"which is not defined. Available: {sorted(trunk_ids)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> ExperimentSpec:
        panel_ids = [p.id for p in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError(f"Duplicate panel IDs: {panel_ids}")
        trunk_ids = [t.id for t in self.trunks]
        if len(trunk_ids) != len(set(trunk_ids)):
            raise ValueError(f"Duplicate trunk IDs: {trunk_ids}")
        return self


class ResolvedSpec(BaseModel):
    """A spec after the resolution stage.

    Carries the original spec plus any inferred defaults and a log of
    decisions made by ced_ml.cellml.resolve.resolve_semantic_decisions.
    """

    spec: ExperimentSpec
    decisions: list[str] = Field(default_factory=list)


class SemanticResolutionError(ValueError):
    """Raised when the resolution stage cannot disambiguate a spec."""
