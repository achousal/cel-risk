"""Backcompat guard: generate_factorial_configs with no scenarios arg
must still produce the same columns and cell count for the production
manifest.

On first run this test generates a golden snapshot next to the test
file; subsequent runs compare against it. Update by deleting the
snapshot and re-running the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ced_ml.recipes.config_gen import generate_factorial_configs
from ced_ml.recipes.derive import derive_all_recipes, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROD_MANIFEST = REPO_ROOT / "operations" / "cellml" / "configs" / "manifest.yaml"
GOLDEN = Path(__file__).parent / "_golden_factorial_snapshot.json"


@pytest.mark.skipif(not PROD_MANIFEST.exists(), reason="production manifest absent")
def test_backcompat_no_scenarios_arg(tmp_path):
    manifest = load_manifest(PROD_MANIFEST)

    # Derive panels into a tmp output dir (skip I/O-heavy configs)
    derive_all_recipes(manifest, data_df=None, output_dir=tmp_path)
    recipe_panels = {}
    pinned = {}
    for recipe_id in [r.id for r in manifest.recipes]:
        panel_csv = tmp_path / recipe_id / "panel.csv"
        if panel_csv.exists():
            recipe_panels[recipe_id] = panel_csv

    # No scenarios, no feature_selection -> legacy behavior
    all_cells = generate_factorial_configs(manifest, recipe_panels, tmp_path, pinned_models=pinned)
    total_cells = sum(len(cs) for cs in all_cells.values())

    # Column set from one cell
    any_recipe = next(iter(all_cells.values()))
    columns = sorted(any_recipe[0].keys()) if any_recipe else []

    snapshot = {"total_cells": total_cells, "columns": columns}

    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        pytest.skip(f"wrote golden snapshot: {GOLDEN}")

    golden = json.loads(GOLDEN.read_text())
    assert snapshot["total_cells"] == golden["total_cells"], (
        f"cell count changed: {snapshot['total_cells']} vs golden " f"{golden['total_cells']}"
    )
    assert (
        snapshot["columns"] == golden["columns"]
    ), f"column set changed: {snapshot['columns']} vs golden {golden['columns']}"
