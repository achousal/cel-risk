"""Generate stage: resolve -> derive panels -> write cell configs.

Wraps ced_ml.recipes.config_gen.generate_factorial_configs with the
experiment-level ExperimentSpec, materializing:

  experiments/<name>/
    spec.yaml                 (user input)
    resolved_spec.yaml        (after resolve stage)
    panels/<panel_id>/panel.csv
    recipes/<panel_id>/<cell_name>/training_config.yaml
    recipes/<panel_id>/<cell_name>/pipeline_hpc.yaml
    recipes/<panel_id>/<cell_name>/splits_config.yaml   (if scenarios axis)
    recipes/cell_manifest.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ced_ml.cellml.panels import resolve_panels
from ced_ml.cellml.registry import register, update_status
from ced_ml.cellml.resolve import resolve_semantic_decisions, write_resolved_spec
from ced_ml.cellml.schema import ExperimentSpec, ResolvedSpec
from ced_ml.recipes.config_gen import generate_factorial_configs
from ced_ml.recipes.schema import (
    FactorialConfig,
    Manifest,
    OptunaOverrides,
    RecipeConfig,
    SharedSplits,
    TrunkConfig,
)

logger = logging.getLogger(__name__)


def _spec_to_manifest(spec: ExperimentSpec) -> Manifest:
    """Build a minimal recipes.Manifest from an ExperimentSpec.

    The generated Manifest is only used to feed config_gen — panels are
    already materialized by resolve_panels, so the recipe entries here
    are stubs (one per panel). Nothing persists this Manifest.
    """
    trunks = [
        TrunkConfig(
            id=t.id,
            description=t.description,
            proteins_csv=t.proteins_csv,
            sweep_csv=t.sweep_csv,
            feature_csv=t.feature_csv,
        )
        for t in spec.trunks
    ]
    # Stub recipes — one per panel, carrying only id + pinned_model.
    # config_gen iterates recipe_panels dict (passed separately), so
    # ordering/size_rule on these stubs is never consulted.
    from ced_ml.recipes.schema import OrderingRule, OrderingType, SizeRule, SizeRuleType

    stub_ordering = OrderingRule(type=OrderingType.consensus_score_descending)
    stub_size = SizeRule(type=SizeRuleType.significance_count)
    recipes = []
    for p in spec.panels:
        if p.source == "derived":
            # derived: real values already used during panel resolution
            recipes.append(
                RecipeConfig(
                    id=p.id,
                    trunk_id=p.trunk_id or (trunks[0].id if trunks else "unknown"),
                    ordering=p.ordering or stub_ordering,
                    size_rule=p.size_rule or stub_size,
                    pinned_model=p.pinned_model,
                )
            )
        else:
            # fixed_csv / reference: need a trunk_id to satisfy the
            # Manifest validator, so point at the first declared trunk
            # or a fabricated one.
            if not trunks:
                trunks.append(
                    TrunkConfig(
                        id="_stub_trunk",
                        description="Stub — fixed_csv / reference panels",
                        proteins_csv=Path("/dev/null"),
                    )
                )
            recipes.append(
                RecipeConfig(
                    id=p.id,
                    trunk_id=trunks[0].id,
                    ordering=stub_ordering,
                    size_rule=stub_size,
                    pinned_model=p.pinned_model,
                )
            )

    return Manifest(
        trunks=trunks,
        recipes=recipes,
        factorial=FactorialConfig(
            models=spec.axes.model,
            calibration=spec.axes.calibration,
            weighting=spec.axes.weighting,
            downsampling=spec.axes.downsampling,
        ),
        optuna=OptunaOverrides(
            n_trials=spec.optuna.n_trials,
            storage_path=spec.optuna.storage_path,
            storage_backend=spec.optuna.storage_backend,
            warm_start_params=spec.optuna.warm_start_params,
            warm_start_top_k=spec.optuna.warm_start_top_k,
        ),
        splits=SharedSplits(seed_start=spec.seeds.start, seed_end=spec.seeds.end),
        base_training_config=spec.base_configs.training,
        base_pipeline_config=spec.base_configs.pipeline,
        base_splits_config=spec.base_configs.splits,
    )


def generate_experiment(
    spec: ExperimentSpec,
    experiment_dir: Path,
    spec_path: Path,
) -> dict[str, Any]:
    """Run resolve -> panels -> cells for one experiment.

    Parameters
    ----------
    spec : ExperimentSpec
    experiment_dir : Path
        Root experiment output dir (e.g. experiments/my_exp/).
    spec_path : Path
        Path to the original spec.yaml (recorded in the registry).

    Returns
    -------
    dict
        Summary with keys: resolved (ResolvedSpec), panels, cells, manifest_csv.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve
    resolved: ResolvedSpec = resolve_semantic_decisions(spec)
    write_resolved_spec(resolved, experiment_dir / "resolved_spec.yaml")

    # 2. Panels
    panels_dir = experiment_dir / "panels"
    panel_paths = resolve_panels(spec, panels_dir)

    # 3. Cell configs — reuse config_gen
    mini_manifest = _spec_to_manifest(spec)
    recipes_dir = experiment_dir / "recipes"
    recipes_dir.mkdir(parents=True, exist_ok=True)

    scenarios = spec.axes.scenario if spec.axes.scenario else None
    feature_selection = spec.axes.feature_selection

    all_cells = generate_factorial_configs(
        mini_manifest,
        panel_paths,
        recipes_dir,
        pinned_models={p.id: p.pinned_model for p in spec.panels},
        scenarios=scenarios,
        feature_selection=feature_selection,
    )

    total_cells = sum(len(cs) for cs in all_cells.values())
    manifest_csv = recipes_dir / "cell_manifest.csv"

    # Register / update
    register(
        spec,
        spec_path=spec_path,
        cells=total_cells,
        recipes=len(spec.panels),
    )
    update_status(spec.name, status="generated")

    return {
        "resolved": resolved,
        "panels": panel_paths,
        "cells": all_cells,
        "manifest_csv": manifest_csv,
        "total_cells": total_cells,
    }
