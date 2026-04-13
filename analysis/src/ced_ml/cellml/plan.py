"""Cell expansion: ExperimentSpec -> list of cell descriptors.

Pure function, no I/O. The plan stage is unit-testable and forms the
basis for both dry-run cell counting and the generate stage.
"""

from __future__ import annotations

import itertools
from typing import Any

from ced_ml.cellml.schema import ExperimentSpec


def _cell_name(
    panel_id: str,
    model: str,
    calibration: str,
    weighting: str,
    downsampling: float,
    scenario: str | None,
    feature_selection: str,
    control_ratio: int,
) -> str:
    """Deterministic cell directory name.

    When scenario is None, preserves the legacy naming used by
    recipes.config_gen: {model}_{cal}_{weight}_ds{ratio}.
    """
    ds_str = f"ds{downsampling:g}"
    parts = [model, calibration, weighting, ds_str]
    if scenario is not None:
        parts.append(scenario)
    # feature_selection + control_ratio are quiet suffixes — emitted
    # only when they deviate from defaults so the legacy naming survives.
    if feature_selection != "fixed_panel":
        parts.append(f"fs-{feature_selection}")
    if control_ratio != 5:
        parts.append(f"ctrl{control_ratio}")
    return "_".join(parts)


def expand_cells(spec: ExperimentSpec) -> list[dict[str, Any]]:
    """Expand the factorial into a flat list of cell descriptors.

    The cartesian product is over:
      panel x model x calibration x weighting x downsampling
            x scenario x feature_selection x control_ratio

    Pinned models on a panel collapse the model axis to that single
    model. An empty ``axes.scenario`` list means "no scenario axis" —
    cells carry ``scenario=None`` and preserve legacy naming.

    Parameters
    ----------
    spec : ExperimentSpec
        Validated experiment spec.

    Returns
    -------
    list[dict]
        One dict per cell with keys: cell_id, panel_id, model,
        calibration, weighting, downsampling, scenario,
        feature_selection, control_ratio, cell_name.
    """
    axes = spec.axes
    cells: list[dict[str, Any]] = []
    cell_id = 0

    # When no scenarios are given, emit a single None placeholder so
    # itertools still produces one cell per (panel, model, cal, ...).
    scenarios: list[Any] = list(axes.scenario) if axes.scenario else [None]

    for panel in spec.panels:
        models = [panel.pinned_model] if panel.pinned_model else axes.model
        for model, cal, wt, ds, scen, fs, ctrl in itertools.product(
            models,
            axes.calibration,
            axes.weighting,
            axes.downsampling,
            scenarios,
            axes.feature_selection,
            axes.control_ratio,
        ):
            cell_id += 1
            scen_name = scen.name if scen is not None else None
            cells.append(
                {
                    "cell_id": cell_id,
                    "panel_id": panel.id,
                    "model": model,
                    "calibration": cal,
                    "weighting": wt,
                    "downsampling": ds,
                    "scenario": scen_name,
                    "feature_selection": fs,
                    "control_ratio": ctrl,
                    "cell_name": _cell_name(panel.id, model, cal, wt, ds, scen_name, fs, ctrl),
                }
            )
    return cells


def cell_count(spec: ExperimentSpec) -> int:
    """Count cells without materializing them — convenience for CLI dry-run."""
    return len(expand_cells(spec))
