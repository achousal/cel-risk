"""Factorial cross: recipe × (model × calibration × weighting × downsampling) → YAML pairs.

For each recipe, generates 108 cell directories, each containing a
fully-merged training_config.yaml and pipeline_hpc.yaml (no _base chains).
"""

from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path
from typing import Any

import yaml

from ced_ml.config.loader import _deep_merge, load_yaml
from ced_ml.recipes.schema import Manifest

logger = logging.getLogger(__name__)

# Maps model name → config section key for class_weight_options.
# Registered models are derived from MODEL_REGISTRY.weight_key; entries here
# only exist for models not yet in the registry (e.g. LR_L2, bare LinSVM).
_MODEL_WEIGHT_KEY_EXTRA = {
    "LR_L2": "lr",
    "LinSVM": "svm",
}


def _build_model_weight_key() -> dict[str, str]:
    """Build model→weight-key lookup from MODEL_REGISTRY + legacy extras."""
    from ced_ml.models.registry import MODEL_REGISTRY

    out = {spec.name.value: spec.weight_key for spec in MODEL_REGISTRY.values() if spec.weight_key}
    out.update(_MODEL_WEIGHT_KEY_EXTRA)
    return out


_MODEL_WEIGHT_KEY = _build_model_weight_key()


def generate_factorial_configs(
    manifest: Manifest,
    recipe_panels: dict[str, Path],
    output_dir: Path,
    pinned_models: dict[str, str | None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate YAML config pairs for all recipes × factorial cells.

    Produces fully-merged configs (no _base inheritance) to avoid
    directory-traversal constraints in load_yaml.

    For model-specific recipes (pinned_model set), only crosses
    calibration × weighting × downsampling (27 cells instead of 108).

    Parameters
    ----------
    manifest : Manifest
        Validated manifest with factorial config.
    recipe_panels : dict[str, Path]
        Mapping recipe_id → path to derived panel.csv.
    output_dir : Path
        Root output directory (configs/recipes/).
    pinned_models : dict[str, str | None], optional
        Mapping recipe_id → pinned model name (or None for shared).
        Includes expanded nested recipes not in the manifest.

    Returns
    -------
    dict[str, list[dict]]
        Per-recipe list of cell descriptors.
    """
    factorial = manifest.factorial
    all_cells: dict[str, list[dict[str, Any]]] = {}
    global_cell_id = 0

    # Pre-load base configs once (with full _base chain resolution)
    base_training = load_yaml(manifest.base_training_config)
    base_pipeline = load_yaml(manifest.base_pipeline_config)

    # Build pinned_model lookup: manifest recipes + any overrides
    recipe_lookup = {r.id: r.pinned_model for r in manifest.recipes}
    if pinned_models:
        recipe_lookup.update(pinned_models)

    for recipe_id, panel_path in recipe_panels.items():
        recipe_dir = output_dir / recipe_id
        recipe_dir.mkdir(parents=True, exist_ok=True)

        cells: list[dict[str, Any]] = []

        # Determine model list: pinned or full factorial
        pinned = recipe_lookup.get(recipe_id)
        if pinned:
            models = [pinned]
        else:
            models = factorial.models

        for model, cal, weight, ds in itertools.product(
            models,
            factorial.calibration,
            factorial.weighting,
            factorial.downsampling,
        ):
            global_cell_id += 1
            cell_name = _cell_dir_name(model, cal, weight, ds)
            cell_dir = recipe_dir / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)

            # Generate fully-merged training config
            training_cfg = _build_training_config(
                base=base_training,
                manifest=manifest,
                panel_csv_path=panel_path,
                model=model,
                calibration=cal,
                weighting=weight,
                downsampling=ds,
                recipe_id=recipe_id,
                cell_name=cell_name,
                cell_id=global_cell_id,
            )
            training_path = cell_dir / "training_config.yaml"
            _write_yaml(training_cfg, training_path)

            # Generate fully-merged pipeline config
            pipeline_cfg = _build_pipeline_config(
                base=base_pipeline,
                model=model,
            )
            pipeline_path = cell_dir / "pipeline_hpc.yaml"
            _write_yaml(pipeline_cfg, pipeline_path)

            cells.append(
                {
                    "cell_id": global_cell_id,
                    "recipe_id": recipe_id,
                    "model": model,
                    "calibration": cal,
                    "weighting": weight,
                    "downsampling": ds,
                    "cell_name": cell_name,
                    "training_config": str(training_path),
                    "pipeline_config": str(pipeline_path),
                }
            )

        # Write cell manifest CSV
        manifest_csv = recipe_dir / "cell_manifest.csv"
        _write_cell_manifest(cells, manifest_csv)
        all_cells[recipe_id] = cells

        logger.info(
            "Generated %d cells for recipe '%s' in %s",
            len(cells),
            recipe_id,
            recipe_dir,
        )

    # Write global cell manifest across all recipes
    all_flat = [c for cells in all_cells.values() for c in cells]
    global_manifest_csv = output_dir / "cell_manifest.csv"
    _write_cell_manifest(all_flat, global_manifest_csv)
    logger.info("Global cell manifest: %d cells → %s", len(all_flat), global_manifest_csv)

    return all_cells


def _cell_dir_name(model: str, calibration: str, weighting: str, downsampling: float) -> str:
    """Generate cell directory name: {model}_{calibration}_{weight}_ds{ratio}."""
    ds_str = f"ds{downsampling:g}"
    return f"{model}_{calibration}_{weighting}_{ds_str}"


def _build_training_config(
    base: dict[str, Any],
    manifest: Manifest,
    panel_csv_path: Path,
    model: str,
    calibration: str,
    weighting: str,
    downsampling: float,
    *,
    recipe_id: str,
    cell_name: str,
    cell_id: int,
) -> dict[str, Any]:
    """Build a fully-merged training config for one factorial cell."""
    overlay: dict[str, Any] = {}

    # Fixed panel (absolute path for portability)
    overlay["features"] = {
        "feature_selection_strategy": "fixed_panel",
        "fixed_panel_csv": str(Path(panel_csv_path).resolve()),
    }

    # Calibration method (global override, clear per-model overrides)
    overlay["calibration"] = {
        "method": calibration,
        "per_model": None,
    }

    # Model-specific class weight
    weight_section = _MODEL_WEIGHT_KEY.get(model)
    if weight_section:
        weight_value = "None" if weighting == "none" else weighting
        overlay[weight_section] = {
            "class_weight_options": weight_value,
        }

    # Downsampling: training-time majority downsampling ratio
    if downsampling != 1.0:
        overlay["data"] = {
            "downsample_majority_ratio": downsampling,
        }

    # Optuna overrides
    optuna_overlay: dict[str, Any] = {
        "n_trials": manifest.optuna.n_trials,
    }

    # Shared storage (Enhancement 3)
    if manifest.optuna.storage_backend != "none" and manifest.optuna.storage_path:
        storage_dir = Path(manifest.optuna.storage_path).resolve()
        recipe_storage = storage_dir / f"{recipe_id}.optuna.journal"
        if manifest.optuna.storage_backend == "journal":
            optuna_overlay["storage"] = str(recipe_storage)
            optuna_overlay["storage_backend"] = "journal"
        elif manifest.optuna.storage_backend == "sqlite":
            optuna_overlay["storage"] = f"sqlite:///{recipe_storage.with_suffix('.db')}"
            optuna_overlay["storage_backend"] = "sqlite"

        # Per-cell study name with {seed} placeholder resolved at runtime
        optuna_overlay["study_name"] = f"{recipe_id}__{cell_name}__seed{{seed}}"
        optuna_overlay["load_if_exists"] = True

    # Factorial metadata as study user_attrs (Enhancement 2)
    optuna_overlay["user_attrs"] = {
        "recipe_id": recipe_id,
        "cell_id": cell_id,
        "cell_name": cell_name,
        "model": model,
        "calibration": calibration,
        "weighting": weighting,
        "downsampling": downsampling,
    }

    # Warm-start params (Enhancement 1)
    if manifest.optuna.warm_start_params:
        optuna_overlay["warm_start_params_file"] = str(
            Path(manifest.optuna.warm_start_params).resolve()
        )
        optuna_overlay["warm_start_top_k"] = manifest.optuna.warm_start_top_k

    overlay["optuna"] = optuna_overlay

    return _deep_merge(base, overlay)


def _build_pipeline_config(
    base: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Build a fully-merged pipeline config for one factorial cell."""
    overlay = {
        "pipeline": {
            "models": [model],
            "ensemble": False,
            "consensus": False,
            "optimize_panel": False,
            "permutation_test": False,
        },
    }
    return _deep_merge(base, overlay)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a YAML file with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _write_cell_manifest(cells: list[dict[str, Any]], path: Path) -> None:
    """Write cell manifest as CSV."""
    if not cells:
        return
    fieldnames = list(cells[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cells)


def generate_v0_configs(
    manifest: Manifest,
    recipe_panels: dict[str, Path],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate V0 gate configs: strategy × model × representative recipe.

    Each cell gets its own splits_config.yaml with a strategy-specific
    overlay, plus the standard training and pipeline configs. The pipeline
    config's ``configs.splits`` field points to the per-cell splits config.

    Parameters
    ----------
    manifest : Manifest
        Validated manifest with v0_gate config.
    recipe_panels : dict[str, Path]
        Mapping recipe_id → path to derived panel.csv.
    output_dir : Path
        Root output directory for V0 cells (e.g. configs/recipes/v0/).

    Returns
    -------
    list[dict]
        V0 cell descriptors (one per strategy × model × recipe).

    Raises
    ------
    ValueError
        If v0_gate is not configured in the manifest.
    """
    if manifest.v0_gate is None:
        raise ValueError("v0_gate not configured in manifest")

    v0 = manifest.v0_gate
    factorial = manifest.factorial

    # Load base configs
    base_training = load_yaml(manifest.base_training_config)
    base_pipeline = load_yaml(manifest.base_pipeline_config)
    base_splits = load_yaml(manifest.base_splits_config)

    # Default downstream factors for V0 (not the focus — use sensible defaults)
    default_cal = factorial.calibration[0]  # logistic_intercept
    default_weight = factorial.weighting[-1]  # none
    default_ds = factorial.downsampling[0]  # 1.0

    cells: list[dict[str, Any]] = []
    cell_id = 0

    for strategy in v0.strategies:
        for control_ratio in v0.control_ratios:
            for model in factorial.models:
                for recipe_id in v0.representative_recipes:
                    if recipe_id not in recipe_panels:
                        logger.warning("V0 recipe '%s' not in derived panels, skipping", recipe_id)
                        continue

                    cell_id += 1
                    cell_name = f"{strategy.name}_ctrl{control_ratio}_{model}"
                    cell_dir = output_dir / recipe_id / cell_name
                    cell_dir.mkdir(parents=True, exist_ok=True)

                    # 1. Per-cell splits_config.yaml: base + strategy overlay + control ratio
                    splits_overlay = strategy.to_splits_overlay(control_ratio=control_ratio)
                    cell_splits = _deep_merge(base_splits, splits_overlay)
                    splits_path = cell_dir / "splits_config.yaml"
                    _write_yaml(cell_splits, splits_path)

                    # 2. Training config (same as factorial, with gate Optuna budget)
                    training_cfg = _build_training_config(
                        base=base_training,
                        manifest=manifest,
                        panel_csv_path=recipe_panels[recipe_id],
                        model=model,
                        calibration=default_cal,
                        weighting=default_weight,
                        downsampling=default_ds,
                        recipe_id=recipe_id,
                        cell_name=cell_name,
                        cell_id=cell_id,
                    )
                    # Override Optuna budget for gate
                    if "optuna" in training_cfg:
                        training_cfg["optuna"]["n_trials"] = v0.optuna_n_trials
                    # Add V0-specific user_attrs
                    if "optuna" in training_cfg and "user_attrs" in training_cfg["optuna"]:
                        training_cfg["optuna"]["user_attrs"]["v0_strategy"] = strategy.name
                        training_cfg["optuna"]["user_attrs"]["v0_control_ratio"] = control_ratio
                        training_cfg["optuna"]["user_attrs"]["v0_gate"] = True

                    training_path = cell_dir / "training_config.yaml"
                    _write_yaml(training_cfg, training_path)

                    # 3. Pipeline config: point configs.splits to per-cell splits
                    pipeline_cfg = _build_pipeline_config(
                        base=base_pipeline,
                        model=model,
                    )
                    # Inject per-cell splits config path
                    if "configs" not in pipeline_cfg:
                        pipeline_cfg["configs"] = {}
                    pipeline_cfg["configs"]["splits"] = str(splits_path.resolve())

                    pipeline_path = cell_dir / "pipeline_hpc.yaml"
                    _write_yaml(pipeline_cfg, pipeline_path)

                    cells.append(
                        {
                            "cell_id": cell_id,
                            "recipe_id": recipe_id,
                            "strategy": strategy.name,
                            "control_ratio": control_ratio,
                            "model": model,
                            "calibration": default_cal,
                            "weighting": default_weight,
                            "downsampling": default_ds,
                            "cell_name": cell_name,
                            "training_config": str(training_path),
                            "pipeline_config": str(pipeline_path),
                            "splits_config": str(splits_path),
                        }
                    )

    # Write V0 cell manifest
    v0_manifest_csv = output_dir / "v0_cell_manifest.csv"
    _write_cell_manifest(cells, v0_manifest_csv)
    logger.info("V0 gate: %d cells → %s", len(cells), v0_manifest_csv)

    return cells


def generate_scout_manifest(
    all_cells: dict[str, list[dict[str, Any]]],
    manifest: Manifest,
    output_path: Path,
    *,
    scout_recipe: str | None = None,
    default_calibration: str = "logistic_intercept",
    default_weighting: str = "log",
    default_downsampling: float = 1.0,
) -> list[dict[str, Any]]:
    """Select representative scout cells for warm-start scouting.

    Picks one cell per model from a shared recipe using default factorial
    settings. Scout results provide warm-start params for the full factorial.

    Parameters
    ----------
    all_cells : dict[str, list[dict]]
        Output of generate_factorial_configs().
    manifest : Manifest
        Validated manifest.
    output_path : Path
        Where to write scout_manifest.csv.
    scout_recipe : str, optional
        Recipe to scout. Defaults to the largest shared recipe.
    default_calibration, default_weighting, default_downsampling
        Default factorial settings for scout cells.

    Returns
    -------
    list[dict]
        Scout cell descriptors.
    """
    # Find largest shared recipe if not specified
    if scout_recipe is None:
        shared_recipes = [r for r in manifest.recipes if r.pinned_model is None]
        if not shared_recipes:
            logger.warning("No shared recipes found for scouting")
            return []
        # Pick recipe with most proteins (largest panel = hardest search space)
        scout_recipe = shared_recipes[-1].id  # last by convention (plateau)
        logger.info("Auto-selected scout recipe: %s", scout_recipe)

    if scout_recipe not in all_cells:
        logger.warning("Scout recipe '%s' not found in generated cells", scout_recipe)
        return []

    # Select one cell per model with default settings
    scout_cells = []
    for cell in all_cells[scout_recipe]:
        for model in manifest.factorial.models:
            expected = _cell_dir_name(
                model, default_calibration, default_weighting, default_downsampling
            )
            if cell["cell_name"] == expected:
                scout_cells.append(cell)

    _write_cell_manifest(scout_cells, output_path)
    logger.info("Scout manifest: %d cells → %s", len(scout_cells), output_path)
    return scout_cells
