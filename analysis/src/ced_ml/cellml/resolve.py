"""Semantic resolution stage.

Fills defaults / sanity-checks the ExperimentSpec before any files are
written. Writes a resolved_spec.yaml for provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ced_ml.cellml.schema import (
    ExperimentSpec,
    ResolvedSpec,
    SemanticResolutionError,
)

logger = logging.getLogger(__name__)


def resolve_semantic_decisions(spec: ExperimentSpec) -> ResolvedSpec:
    """Apply default-fills and raise on ambiguous specs.

    Current checks:
      - feature_selection values must be 'fixed_panel' in v1 (see
        generate.py for the actual runtime rejection). Non-fixed values
        are recorded as decisions so v2 can react to them.
      - reference-source panels with no extract rule get ``best_prauc``
        filled in (logged as a decision).
      - At least one axis value for each required axis.

    Parameters
    ----------
    spec : ExperimentSpec

    Returns
    -------
    ResolvedSpec
        Carries the (possibly updated) spec plus a decision log.

    Raises
    ------
    SemanticResolutionError
        If the spec cannot be disambiguated.
    """
    decisions: list[str] = []

    # Required axes sanity
    for axis_name in ("model", "calibration", "weighting", "downsampling"):
        if not getattr(spec.axes, axis_name):
            raise SemanticResolutionError(f"axes.{axis_name} must have at least one value")

    # Panel references
    for panel in spec.panels:
        if panel.source == "reference" and panel.extract is None:
            panel.extract = "best_prauc"
            decisions.append(f"panel '{panel.id}': filled extract=best_prauc (default)")

    # feature_selection v1 note
    fs = spec.axes.feature_selection or ["fixed_panel"]
    for value in fs:
        if value != "fixed_panel":
            decisions.append(
                f"feature_selection='{value}' is a v2 feature — generate will reject it"
            )

    for d in decisions:
        logger.info("resolve: %s", d)

    return ResolvedSpec(spec=spec, decisions=decisions)


def write_resolved_spec(resolved: ResolvedSpec, path: Path) -> None:
    """Dump ResolvedSpec to YAML for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": resolved.spec.model_dump(mode="json"),
        "decisions": resolved.decisions,
    }
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
