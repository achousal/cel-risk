"""Orchestrator: manifest + source data → derived panels + audit logs.

Load manifest → validate → for each recipe: derive ordering, derive size →
emit panel.csv + derivation JSONs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ced_ml.recipes.ordering_rules import dispatch_ordering
from ced_ml.recipes.schema import Manifest, RecipeConfig, SizeRuleType, TrunkConfig
from ced_ml.recipes.size_rules import (
    derive_size_significance_count,
    derive_size_stability,
    derive_size_three_criterion,
)

logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str | Path) -> Manifest:
    """Load and validate a manifest YAML file.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest.yaml.

    Returns
    -------
    Manifest
        Validated manifest object.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        raw = yaml.safe_load(f)

    # Resolve relative paths against manifest directory
    manifest_dir = manifest_path.parent
    _resolve_trunk_paths(raw, manifest_dir)

    return Manifest(**raw)


def derive_all_recipes(
    manifest: Manifest,
    data_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Derive panels for all recipes in the manifest.

    Parameters
    ----------
    manifest : Manifest
        Validated manifest.
    data_df : pd.DataFrame, optional
        Training data matrix (required for stream_balanced ordering).
    output_dir : Path, optional
        Root output directory. If None, defaults to configs/recipes/.
    dry_run : bool
        If True, print derived panels without writing files.

    Returns
    -------
    dict[str, dict]
        Per-recipe results: {recipe_id: {panel, size, ordering_log, size_log}}.
    """
    results: dict[str, dict[str, Any]] = {}

    for recipe in manifest.recipes:
        trunk = manifest.get_trunk(recipe.trunk_id)
        logger.info("Deriving recipe '%s' (trunk: %s)", recipe.id, trunk.id)

        try:
            result = _derive_single_recipe(recipe, trunk, data_df)
        except Exception as exc:
            logger.error("Recipe '%s' failed: %s", recipe.id, exc)
            continue

        results[recipe.id] = result

        if dry_run:
            _print_recipe_summary(recipe.id, result)
        elif output_dir is not None:
            _write_recipe_artifacts(recipe.id, result, output_dir)

        # Expand nested sub-recipes from plateau down to core
        if recipe.expand_to_core is not None:
            nested = _expand_nested_recipes(recipe, result)
            for sub_id, sub_result in nested.items():
                results[sub_id] = sub_result
                if dry_run:
                    _print_recipe_summary(sub_id, sub_result)
                elif output_dir is not None:
                    _write_recipe_artifacts(sub_id, sub_result, output_dir)

    return results


def _derive_single_recipe(
    recipe: RecipeConfig,
    trunk: TrunkConfig,
    data_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Derive panel for a single recipe.

    Returns
    -------
    dict with keys: ordered_proteins, optimal_size, panel, size_log, ordering_log
    """
    # Load trunk source data
    proteins_df = pd.read_csv(trunk.proteins_csv)

    # Derive ordering
    ordering_params = dict(recipe.ordering.params)
    ordered_proteins = dispatch_ordering(
        ordering_type=recipe.ordering.type.value,
        proteins_df=proteins_df,
        params=ordering_params,
        data_df=data_df,
    )

    ordering_log = {
        "ordering_type": recipe.ordering.type.value,
        "params": ordering_params,
        "n_proteins_ordered": len(ordered_proteins),
        "protein_order": ordered_proteins,
    }

    # Derive size
    # For model-specific recipes, filter sweep data to pinned model
    model_filter = recipe.pinned_model

    if recipe.size_rule.type in (
        SizeRuleType.three_criterion,
        SizeRuleType.three_criterion_unanimous,
    ):
        if trunk.sweep_csv is None:
            raise ValueError(
                f"Recipe '{recipe.id}' uses {recipe.size_rule.type.value} size rule "
                f"but trunk '{trunk.id}' has no sweep_csv"
            )
        sweep_df = pd.read_csv(trunk.sweep_csv)

        order_filter = None
        if recipe.sweep_filter and "order" in recipe.sweep_filter:
            order_filter = recipe.sweep_filter["order"]

        min_criteria = 3 if recipe.size_rule.type == SizeRuleType.three_criterion_unanimous else 2

        optimal_size, size_log = derive_size_three_criterion(
            sweep_df,
            order_filter=order_filter,
            model_filter=model_filter,
            min_criteria=min_criteria,
            **recipe.size_rule.params,
        )

    elif recipe.size_rule.type == SizeRuleType.stability:
        if trunk.feature_csv is None:
            raise ValueError(
                f"Recipe '{recipe.id}' uses stability size rule "
                f"but trunk '{trunk.id}' has no feature_csv"
            )
        feature_df = pd.read_csv(trunk.feature_csv)
        optimal_size, size_log = derive_size_stability(feature_df)

    elif recipe.size_rule.type == SizeRuleType.significance_count:
        optimal_size, size_log = derive_size_significance_count(
            proteins_df, **recipe.size_rule.params
        )

    else:
        raise ValueError(f"Unknown size rule type: {recipe.size_rule.type}")

    # Truncate ordered proteins to optimal size
    n_available = len(ordered_proteins)
    if n_available < optimal_size:
        logger.warning(
            "Recipe '%s': optimal size %d but only %d proteins available "
            "(panel will be %d proteins)",
            recipe.id,
            optimal_size,
            n_available,
            n_available,
        )
    panel = ordered_proteins[:optimal_size]

    return {
        "ordered_proteins": ordered_proteins,
        "optimal_size": optimal_size,
        "available_proteins": n_available,
        "panel": panel,
        "size_log": size_log,
        "ordering_log": ordering_log,
    }


def _expand_nested_recipes(
    recipe: RecipeConfig,
    base_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Generate nested sub-recipes by truncating the base panel.

    For each size from (optimal_size - 1) down to expand_to_core,
    create a sub-recipe with the top-N proteins from the same ordering.
    Skips sizes that equal the base recipe's optimal_size.

    Returns
    -------
    dict[str, dict]
        {sub_recipe_id: result_dict} for each nested size.
    """
    core_size = recipe.expand_to_core
    optimal_size = base_result["optimal_size"]
    ordered = base_result["ordered_proteins"]

    if core_size >= optimal_size:
        return {}

    nested: dict[str, dict[str, Any]] = {}

    for p in range(optimal_size - 1, core_size - 1, -1):
        sub_id = f"{recipe.id}_p{p}"
        panel = ordered[:p]

        nested[sub_id] = {
            "ordered_proteins": ordered,
            "optimal_size": p,
            "available_proteins": len(ordered),
            "panel": panel,
            "size_log": {
                "rule": "nested_expansion",
                "parent_recipe": recipe.id,
                "parent_optimal_size": optimal_size,
                "nested_size": p,
                "core_size": core_size,
            },
            "ordering_log": base_result["ordering_log"],
            "pinned_model": recipe.pinned_model,
        }

    logger.info(
        "Expanded '%s' (p=%d) → %d nested recipes down to p=%d",
        recipe.id,
        optimal_size,
        len(nested),
        core_size,
    )
    return nested


def _write_recipe_artifacts(
    recipe_id: str,
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write panel CSV and derivation JSONs for one recipe."""
    recipe_dir = output_dir / recipe_id
    recipe_dir.mkdir(parents=True, exist_ok=True)

    # Panel CSV
    panel_path = recipe_dir / "panel.csv"
    panel_df = pd.DataFrame({"protein": result["panel"]})
    panel_df.to_csv(panel_path, index=False)
    logger.info("Wrote %s (%d proteins)", panel_path, len(result["panel"]))

    # Size derivation log
    size_path = recipe_dir / "size_derivation.json"
    with open(size_path, "w") as f:
        json.dump(result["size_log"], f, indent=2, default=str)

    # Ordering derivation log
    ordering_path = recipe_dir / "ordering_derivation.json"
    with open(ordering_path, "w") as f:
        json.dump(result["ordering_log"], f, indent=2, default=str)


def _print_recipe_summary(recipe_id: str, result: dict[str, Any]) -> None:
    """Print dry-run summary for one recipe."""
    panel = result["panel"]
    print(f"\n{'=' * 60}")
    print(f"Recipe: {recipe_id}")
    print(f"Optimal size: {result['optimal_size']}")
    print(f"Panel ({len(panel)} proteins):")
    for i, p in enumerate(panel, 1):
        print(f"  {i:3d}. {p}")
    print(f"{'=' * 60}")


def _resolve_trunk_paths(raw: dict, manifest_dir: Path) -> None:
    """Resolve relative paths in trunk configs and recipe params against manifest directory."""
    for trunk in raw.get("trunks", []):
        for key in ("proteins_csv", "sweep_csv", "feature_csv"):
            if key in trunk and trunk[key] is not None:
                p = Path(trunk[key])
                if not p.is_absolute():
                    trunk[key] = str((manifest_dir / p).resolve())

    # Resolve paths in recipe ordering params (for oof_importance, rfe_elimination)
    _PATH_PARAMS = ("importance_csv", "rfe_csv")
    for recipe in raw.get("recipes", []):
        ordering = recipe.get("ordering", {})
        params = ordering.get("params", {})
        for key in _PATH_PARAMS:
            if key in params and params[key] is not None:
                p = Path(params[key])
                if not p.is_absolute():
                    params[key] = str((manifest_dir / p).resolve())

    # Also resolve base config paths
    for key in ("base_training_config", "base_pipeline_config", "base_splits_config"):
        if key in raw and raw[key] is not None:
            p = Path(raw[key])
            if not p.is_absolute():
                raw[key] = str((manifest_dir / p).resolve())
