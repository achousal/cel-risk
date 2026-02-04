"""Root configuration schema for CeD-ML pipeline."""

from pydantic import BaseModel, ConfigDict

from .aggregate_schema import AggregateConfig, HoldoutEvalConfig, PanelOptimizeConfig
from .data_schema import SplitsConfig
from .training_schema import TrainingConfig


class RootConfig(BaseModel):
    """Root configuration for all CeD-ML commands."""

    # Common
    random_state: int = 0
    verbose: int = 1

    # Sub-configs (populated based on command)
    splits: SplitsConfig | None = None
    training: TrainingConfig | None = None
    aggregate: AggregateConfig | None = None
    holdout: HoldoutEvalConfig | None = None
    panel_optimize: PanelOptimizeConfig | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
