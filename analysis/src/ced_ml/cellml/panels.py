"""Panel resolution: produce a panel.csv for every PanelSpec.

Three source modes:
  - derived: build a one-recipe Manifest and call ced_ml.recipes.derive
  - fixed_csv: copy the provided CSV verbatim
  - reference: extract a panel from a prior registered experiment (v1 stub)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ced_ml.cellml.schema import ExperimentSpec, PanelSpec

logger = logging.getLogger(__name__)


def _resolve_derived_panel(spec: ExperimentSpec, panel: PanelSpec, output_dir: Path) -> Path:
    """Derive a single panel using ced_ml.recipes.derive.

    Builds a minimal one-recipe Manifest in memory and delegates to the
    existing derive_all_recipes machinery — no reimplementation.
    """
    from ced_ml.recipes.derive import derive_all_recipes
    from ced_ml.recipes.schema import (
        FactorialConfig,
        Manifest,
        RecipeConfig,
        TrunkConfig,
    )

    # Locate the declared trunk
    trunk_spec = next((t for t in spec.trunks if t.id == panel.trunk_id), None)
    if trunk_spec is None:
        raise ValueError(f"Panel '{panel.id}': trunk '{panel.trunk_id}' not in spec")

    trunk = TrunkConfig(
        id=trunk_spec.id,
        description=trunk_spec.description,
        proteins_csv=trunk_spec.proteins_csv,
        sweep_csv=trunk_spec.sweep_csv,
        feature_csv=trunk_spec.feature_csv,
    )

    recipe = RecipeConfig(
        id=panel.id,
        trunk_id=panel.trunk_id,  # type: ignore[arg-type]
        description=f"Derived via cellml.panels for experiment '{spec.name}'",
        ordering=panel.ordering,  # type: ignore[arg-type]
        size_rule=panel.size_rule,  # type: ignore[arg-type]
        sweep_filter=panel.sweep_filter,
        pinned_model=panel.pinned_model,
        expand_to_core=panel.expand_to_core,
    )

    mini_manifest = Manifest(
        trunks=[trunk],
        recipes=[recipe],
        factorial=FactorialConfig(
            models=spec.axes.model,
            calibration=spec.axes.calibration,
            weighting=spec.axes.weighting,
            downsampling=spec.axes.downsampling,
        ),
        base_training_config=spec.base_configs.training,
        base_pipeline_config=spec.base_configs.pipeline,
        base_splits_config=spec.base_configs.splits,
    )

    derive_all_recipes(mini_manifest, data_df=None, output_dir=output_dir)
    panel_path = output_dir / panel.id / "panel.csv"
    if not panel_path.exists():
        raise RuntimeError(f"Derived panel.csv missing for '{panel.id}' at {panel_path}")
    return panel_path


def _resolve_fixed_csv(panel: PanelSpec, output_dir: Path) -> Path:
    """Copy a fixed CSV into the experiment panels/ directory."""
    assert panel.csv is not None
    src = Path(panel.csv)
    if not src.exists():
        raise FileNotFoundError(f"Panel '{panel.id}': fixed_csv not found: {src}")
    dest_dir = output_dir / panel.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "panel.csv"
    shutil.copyfile(src, dest)
    logger.info("Panel '%s': copied fixed_csv %s -> %s", panel.id, src, dest)
    return dest


def _resolve_reference(panel: PanelSpec, output_dir: Path) -> Path:
    """Extract a panel from a prior registered experiment.

    v1 behavior: if ``extract=best_prauc`` and the referenced experiment
    has a compiled.csv that names a single winning cell's panel, we can
    copy it. Otherwise we refuse — no silent guessing.
    """
    # TODO(cellml v2): implement best_prauc / union / intersection
    # extraction by walking the referenced experiment's compiled.csv,
    # picking the winning row, and re-reading that cell's panel.csv.
    # For now, raise cleanly so the caller knows this is stubbed.
    raise NotImplementedError(
        f"Panel '{panel.id}': source=reference (extract={panel.extract}) "
        "is not implemented in v1. Use source=fixed_csv with an explicit "
        "CSV path, or wait for v2."
    )


def resolve_panels(spec: ExperimentSpec, output_dir: Path) -> dict[str, Path]:
    """Materialize a panel.csv for every PanelSpec in the experiment.

    Parameters
    ----------
    spec : ExperimentSpec
        The experiment spec.
    output_dir : Path
        Root panels/ directory (e.g. experiments/<name>/panels/).

    Returns
    -------
    dict[panel_id, Path]
        Mapping panel id -> path to the materialized panel.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for panel in spec.panels:
        if panel.source == "derived":
            out[panel.id] = _resolve_derived_panel(spec, panel, output_dir)
        elif panel.source == "fixed_csv":
            out[panel.id] = _resolve_fixed_csv(panel, output_dir)
        elif panel.source == "reference":
            out[panel.id] = _resolve_reference(panel, output_dir)
        else:  # pragma: no cover - pydantic Literal guards this
            raise ValueError(f"Unknown panel source: {panel.source}")
    return out
