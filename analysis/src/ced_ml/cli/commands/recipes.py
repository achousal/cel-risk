"""
Recipe derivation and factorial config generation commands.

Commands:
  - derive-recipes: Derive panels from manifest and generate factorial cell configs
  - generate-v0: Generate V0 gate configs (strategy × imbalance-probe × model × recipe)

V0 gate note (rb-v0.2.0): ``generate-v0`` now crosses strategies with the
``imbalance_probes`` axis instead of the retired ``control_ratios`` axis.
See ``operations/cellml/rulebook/protocols/v0-strategy.md``.
"""

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command("derive-recipes")
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to manifest.yaml defining recipes and factorial design.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to training data (Parquet/CSV). Required for stream_balanced ordering.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for derived panels and configs. Default: configs/recipes/",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate manifest and print derived panels without writing files.",
)
@click.option(
    "--skip-configs",
    is_flag=True,
    default=False,
    help="Only derive panels, skip factorial config generation.",
)
@click.pass_context
def derive_recipes(ctx, manifest, data_path, output_dir, dry_run, skip_configs):
    """Derive protein panels from a recipe manifest and generate factorial cell configs.

    Each recipe in the manifest declares a trunk (source data), ordering rule,
    and size rule. Panels are generated artifacts — fully reproducible from the
    manifest + source data.

    After panel derivation, generates factorial cell configs (model × calibration
    × weighting × downsampling) for each recipe, ready for SLURM array submission.
    """
    import pandas as pd

    from ced_ml.recipes.config_gen import generate_factorial_configs
    from ced_ml.recipes.derive import derive_all_recipes, load_manifest

    log_level = ctx.obj.get("log_level", logging.INFO) if ctx.obj else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Load manifest
    manifest_obj = load_manifest(manifest)
    click.echo(
        f"Loaded manifest: {len(manifest_obj.recipes)} recipes, "
        f"{len(manifest_obj.trunks)} trunks"
    )

    # Count cells accounting for pinned models and nested expansion
    n_cal = len(manifest_obj.factorial.calibration)
    n_wt = len(manifest_obj.factorial.weighting)
    n_ds = len(manifest_obj.factorial.downsampling)
    n_shared = manifest_obj.factorial.n_cells  # 108
    n_pinned = n_cal * n_wt * n_ds  # 27

    n_shared_recipes = sum(1 for r in manifest_obj.recipes if not r.pinned_model)
    n_pinned_recipes = sum(1 for r in manifest_obj.recipes if r.pinned_model)
    manifest_cells = sum(n_pinned if r.pinned_model else n_shared for r in manifest_obj.recipes)

    # Estimate nested expansion (exact count known after derivation)
    n_expand = 0
    for r in manifest_obj.recipes:
        if r.expand_to_core is not None and r.pinned_model:
            # Expansion adds (plateau - core) sub-recipes; exact plateau unknown pre-derivation
            n_expand += 1  # flag that expansion exists

    expand_note = " (+ nested expansion after derivation)" if n_expand > 0 else ""
    click.echo(
        f"Factorial: {n_shared_recipes} shared ({n_shared} cells each) + "
        f"{n_pinned_recipes} model-specific ({n_pinned} cells each) = "
        f"{manifest_cells} manifest cells{expand_note}"
    )

    # Load training data if provided (needed for stream_balanced)
    data_df = None
    if data_path is not None:
        data_path_obj = Path(data_path)
        if data_path_obj.suffix == ".parquet":
            data_df = pd.read_parquet(data_path_obj)
        else:
            data_df = pd.read_csv(data_path_obj)
        click.echo(f"Loaded data: {data_df.shape[0]} samples × {data_df.shape[1]} features")

    # Resolve output directory
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(manifest).parent / "recipes"

    # Derive panels
    results = derive_all_recipes(manifest_obj, data_df=data_df, output_dir=out, dry_run=dry_run)

    if dry_run:
        click.echo(
            f"\n[DRY RUN] {len(results)} total recipes (including nested). No files written."
        )
        return

    click.echo(f"\nDerived {len(results)} recipe panels (including nested) → {out}")

    # Generate factorial configs
    if skip_configs:
        click.echo("Skipping factorial config generation (--skip-configs).")
        return

    recipe_panels = {}
    pinned_models = {}
    for recipe_id, result in results.items():
        panel_path = out / recipe_id / "panel.csv"
        if panel_path.exists():
            recipe_panels[recipe_id] = panel_path
            # Propagate pinned_model for expanded nested recipes
            if "pinned_model" in result and result["pinned_model"]:
                pinned_models[recipe_id] = result["pinned_model"]
        else:
            click.echo(f"WARNING: panel.csv missing for {recipe_id}, skipping configs")

    all_cells = generate_factorial_configs(manifest_obj, recipe_panels, out, pinned_models)
    total = sum(len(cells) for cells in all_cells.values())
    click.echo(f"Generated {total} factorial cell configs → {out}")
    click.echo(f"Global manifest: {out / 'cell_manifest.csv'}")


@click.command("generate-v0")
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to manifest.yaml with v0_gate section.",
)
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to training data (required if representative recipes use stream_balanced).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for V0 cell configs. Default: configs/recipes/v0/",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print V0 cell plan without writing files.",
)
@click.pass_context
def generate_v0(ctx, manifest, data_path, output_dir, dry_run):
    """Generate V0 gate configs: strategy × model × representative recipe.

    Produces per-cell splits_config.yaml overlays for each training strategy,
    along with training and pipeline configs. Ready for SLURM submission.

    Requires the v0_gate section in manifest.yaml.
    """
    import pandas as pd

    from ced_ml.recipes.config_gen import generate_v0_configs
    from ced_ml.recipes.derive import derive_all_recipes, load_manifest

    log_level = ctx.obj.get("log_level", logging.INFO) if ctx.obj else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    manifest_obj = load_manifest(manifest)
    if manifest_obj.v0_gate is None:
        raise click.UsageError("Manifest has no v0_gate section. Add it to manifest.yaml.")

    v0 = manifest_obj.v0_gate
    n_strategies = len(v0.strategies)
    n_probes = len(v0.imbalance_probes)
    n_models = len(manifest_obj.factorial.models)
    n_recipes = len(v0.representative_recipes)
    n_cells = n_strategies * n_probes * n_models * n_recipes

    click.echo(
        f"V0 gate: {n_strategies} strategies × {n_probes} imbalance probes × "
        f"{n_models} models × {n_recipes} recipes = {n_cells} cells"
    )
    click.echo(f"Optuna budget: {v0.optuna_n_trials} trials/cell (sweep-level)")
    click.echo(f"Imbalance probes: {v0.imbalance_probes}")

    for s in v0.strategies:
        click.echo(
            f"  {s.name}: scenarios={s.scenarios}, " f"prevalent_frac={s.prevalent_train_frac}"
        )

    if dry_run:
        click.echo(f"\n[DRY RUN] Would generate {n_cells} V0 cells. No files written.")
        return

    # Load data if needed
    data_df = None
    if data_path is not None:
        data_path_obj = Path(data_path)
        if data_path_obj.suffix == ".parquet":
            data_df = pd.read_parquet(data_path_obj)
        else:
            data_df = pd.read_csv(data_path_obj)

    # Derive panels for representative recipes only
    click.echo("\nDeriving panels for representative recipes...")
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(manifest).parent / "recipes" / "v0"

    derive_all_recipes(manifest_obj, data_df=data_df, output_dir=out.parent)

    # Build recipe_panels for representative recipes
    recipe_panels = {}
    for recipe_id in v0.representative_recipes:
        panel_path = out.parent / recipe_id / "panel.csv"
        if panel_path.exists():
            recipe_panels[recipe_id] = panel_path
        else:
            click.echo(f"WARNING: panel.csv missing for {recipe_id}")

    if not recipe_panels:
        raise click.UsageError("No representative recipe panels found. Run derive-recipes first.")

    # Generate V0 configs
    cells = generate_v0_configs(manifest_obj, recipe_panels, out)
    click.echo(f"\nGenerated {len(cells)} V0 cells → {out}")
    click.echo(f"V0 manifest: {out / 'v0_cell_manifest.csv'}")
    click.echo(
        f"\nNext: bash operations/cellml/submit_experiment.sh "
        f"--experiment v0_gate --manifest {out / 'v0_cell_manifest.csv'} "
        f"--results-root results/v0_gate --seeds 100-119"
    )
