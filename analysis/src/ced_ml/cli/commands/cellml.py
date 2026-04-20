"""``ced cellml`` CLI group.

Experiment-level factorial orchestration. Subcommands:

    init      scaffold a new experiment from a template
    plan      load a spec + print cell count
    resolve   run the semantic resolution stage
    generate  resolve + derive panels + write cell configs
    submit    write runner scripts and (optionally) bsub an LSF array
    status    bjobs-based state counts
    compile   gather per-cell results into one CSV
    validate  run validate_tree.R and parse V1-V4 readouts
    list      show the experiment registry as a table
    show      inspect one experiment
    run       full pipeline: generate -> submit -> monitor -> compile -> validate

Each subcommand delegates to ced_ml.cellml.* library code.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


def _experiment_dir(name: str) -> Path:
    return Path("experiments") / name


def _repo_root() -> Path:
    """Repo root = CWD (CLI is always invoked from the repo)."""
    return Path.cwd()


def _templates_dir() -> Path:
    return Path("experiments") / "templates"


@click.group("cellml")
def cellml_group() -> None:
    """Experiment-level factorial orchestration (init/plan/generate/submit/...)."""


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cellml_group.command("init")
@click.argument("name")
@click.option(
    "--template",
    type=click.Choice(["minimal", "svm", "full"]),
    default="minimal",
    help="Which bundled template to seed the spec from.",
)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing spec.")
def cellml_init(name: str, template: str, force: bool) -> None:
    """Scaffold a new experiment from a bundled template."""
    from ced_ml.cellml.registry import register
    from ced_ml.cellml.spec_io import load_spec

    tmpl_path = _templates_dir() / f"{template}.yaml"
    if not tmpl_path.exists():
        raise click.UsageError(f"Template not found: {tmpl_path}")
    exp_dir = _experiment_dir(name)
    dest = exp_dir / "spec.yaml"
    if dest.exists() and not force:
        raise click.UsageError(f"{dest} already exists; use --force to overwrite")
    exp_dir.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(tmpl_path.read_text()) or {}
    data["name"] = name
    dest.write_text(yaml.safe_dump(data, sort_keys=False))
    click.echo(f"Wrote {dest}")
    spec = load_spec(dest)
    register(spec, spec_path=dest)
    click.echo(f"Registered experiment '{name}' in experiments/_registry.csv")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@cellml_group.command("plan")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--override", "-o", multiple=True, help="key=value spec overrides.")
@click.option("--dry-run", is_flag=True, default=False, help="Alias for plan itself — prints only.")
def cellml_plan(spec: str, override: tuple[str, ...], dry_run: bool) -> None:
    """Load a spec, apply overrides, print cell count and a sample."""
    from ced_ml.cellml.plan import expand_cells
    from ced_ml.cellml.spec_io import load_spec

    spec_obj = load_spec(Path(spec), list(override))
    cells = expand_cells(spec_obj)
    click.echo(f"Experiment:  {spec_obj.name}")
    click.echo(f"Panels:      {len(spec_obj.panels)}")
    click.echo(f"Total cells: {len(cells)}")
    click.echo("Sample (first 3):")
    for cell in cells[:3]:
        click.echo(f"  {cell['cell_id']:>4d}  {cell['cell_name']}")


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


@cellml_group.command("resolve")
@click.argument("spec", type=click.Path(exists=True))
def cellml_resolve(spec: str) -> None:
    """Run the semantic resolution stage and dump resolved_spec.yaml."""
    from ced_ml.cellml.resolve import resolve_semantic_decisions, write_resolved_spec
    from ced_ml.cellml.spec_io import load_spec

    spec_path = Path(spec)
    spec_obj = load_spec(spec_path)
    resolved = resolve_semantic_decisions(spec_obj)
    out_path = _experiment_dir(spec_obj.name) / "resolved_spec.yaml"
    write_resolved_spec(resolved, out_path)
    click.echo(f"Wrote {out_path}")
    for d in resolved.decisions:
        click.echo(f"  decision: {d}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cellml_group.command("generate")
@click.argument("spec", type=click.Path(exists=True))
def cellml_generate(spec: str) -> None:
    """Resolve + derive panels + write per-cell configs."""
    from ced_ml.cellml.generate import generate_experiment
    from ced_ml.cellml.spec_io import load_spec

    spec_path = Path(spec)
    spec_obj = load_spec(spec_path)
    exp_dir = _experiment_dir(spec_obj.name)
    summary = generate_experiment(spec_obj, exp_dir, spec_path=spec_path)
    click.echo(f"Generated {summary['total_cells']} cells")
    click.echo(f"Manifest: {summary['manifest_csv']}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


@cellml_group.command("submit")
@click.argument("name")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--cells", default=None, help="Cell range lo-hi (default: all)")
@click.option("--wall", default=None, help="LSF wall time override")
@click.option("--queue", default=None, help="LSF queue override")
@click.option("--project", default=None, help="LSF project override")
def cellml_submit(
    name: str,
    dry_run: bool,
    cells: str | None,
    wall: str | None,
    queue: str | None,
    project: str | None,
) -> None:
    """Write runner scripts and (optionally) submit an LSF array."""
    from ced_ml.cellml.spec_io import load_spec
    from ced_ml.cellml.submit import submit_experiment

    exp_dir = _experiment_dir(name)
    spec_obj = load_spec(exp_dir / "spec.yaml")
    result = submit_experiment(
        spec_obj,
        exp_dir,
        repo_root=_repo_root(),
        cell_range=cells,
        wall=wall,
        queue=queue,
        project=project,
        dry_run=dry_run,
    )
    click.echo(f"Runners dir: {result.runners_dir}")
    click.echo(f"bsub: {' '.join(result.bsub_cmd)}")
    if dry_run:
        click.echo(f"[dry-run] would submit {result.n_cells} cells")
    else:
        click.echo(f"Submitted job {result.job_id} ({result.n_cells} cells)")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cellml_group.command("status")
@click.argument("name")
def cellml_status(name: str) -> None:
    """Print LSF state counts for an experiment."""
    from ced_ml.cellml.monitor import get_status

    result = get_status(name)
    if result.error:
        click.echo(f"ERROR: {result.error}", err=True)
    click.echo(f"Experiment: {name}")
    for state, count in result.counts.items():
        click.echo(f"  {state:>6s}: {count}")


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


@cellml_group.command("compile")
@click.argument("name")
@click.option(
    "--results-root",
    type=click.Path(),
    default=None,
    help="Per-cell results root (default: results/cellml/<name>).",
)
def cellml_compile(name: str, results_root: str | None) -> None:
    """Gather per-cell test metrics into experiments/<name>/compiled.csv."""
    from ced_ml.cellml.compile import compile_experiment

    exp_dir = _experiment_dir(name)
    root = Path(results_root) if results_root else _repo_root() / "results" / "cellml" / name
    df = compile_experiment(name, exp_dir, root, repo_root=_repo_root())
    click.echo(f"Compiled {len(df)} rows -> {exp_dir / 'compiled.csv'}")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cellml_group.command("validate")
@click.argument("name")
def cellml_validate(name: str) -> None:
    """Run validate_tree.R and print V1-V4 readouts."""
    from ced_ml.cellml.validate import validate_experiment

    exp_dir = _experiment_dir(name)
    report = validate_experiment(name, exp_dir, repo_root=_repo_root())
    click.echo(f"rc={report.returncode}")
    for key in ("V1", "V2", "V3", "V4"):
        if key in report.readouts:
            click.echo(f"  {key}: {report.readouts[key]}")
    if report.stderr.strip():
        click.echo(f"stderr: {report.stderr.strip()}", err=True)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cellml_group.command("list")
def cellml_list() -> None:
    """List all registered experiments."""
    from ced_ml.cellml.registry import list_all

    rows = list_all()
    if not rows:
        click.echo("(no experiments registered)")
        return
    header = f"{'name':<30s} {'status':<12s} {'cells':>6s} {'job_id':<12s} {'created':<20s}"
    click.echo(header)
    click.echo("-" * len(header))
    for row in rows:
        click.echo(
            f"{row['name']:<30s} {row['status']:<12s} "
            f"{row['cells']:>6s} {row['job_id']:<12s} {row['created']:<20s}"
        )


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@cellml_group.command("show")
@click.argument("name")
def cellml_show(name: str) -> None:
    """Show one experiment's registry row + top 5 cells by PRAUC if compiled."""
    import pandas as pd

    from ced_ml.cellml.registry import get

    row = get(name)
    click.echo(f"Experiment: {name}")
    for k, v in row.items():
        click.echo(f"  {k}: {v}")
    compiled = _experiment_dir(name) / "compiled.csv"
    if not compiled.exists():
        return
    df = pd.read_csv(compiled)
    for metric_col in ("summary_prauc_mean", "summary_auprc_mean", "summary_auroc_mean"):
        if metric_col in df.columns:
            top = df.nlargest(5, metric_col)
            click.echo(f"\nTop 5 by {metric_col}:")
            cols = [c for c in ["recipe_id", "cell_name", metric_col] if c in df.columns]
            click.echo(top[cols].to_string(index=False))
            return


# ---------------------------------------------------------------------------
# run (full pipeline)
# ---------------------------------------------------------------------------


@cellml_group.command("run")
@click.argument("name")
@click.option("--dry-run", is_flag=True, default=False)
def cellml_run(name: str, dry_run: bool) -> None:
    """Full pipeline: generate -> submit -> (caller waits) -> compile -> validate.

    With ``--dry-run``, runs generate + submit dry-run only.
    """
    from ced_ml.cellml.generate import generate_experiment
    from ced_ml.cellml.spec_io import load_spec
    from ced_ml.cellml.submit import submit_experiment

    exp_dir = _experiment_dir(name)
    spec_path = exp_dir / "spec.yaml"
    if not spec_path.exists():
        raise click.UsageError(
            f"Spec missing — init first: ced cellml init {name} --template minimal"
        )
    spec_obj = load_spec(spec_path)
    summary = generate_experiment(spec_obj, exp_dir, spec_path=spec_path)
    click.echo(f"[1/4] generate: {summary['total_cells']} cells")
    submit_result = submit_experiment(spec_obj, exp_dir, repo_root=_repo_root(), dry_run=dry_run)
    click.echo(f"[2/4] submit:   {'(dry-run)' if dry_run else submit_result.job_id}")
    if dry_run:
        click.echo("(dry-run) skipping compile + validate")
        return
    click.echo("[3/4] compile:  (invoke `ced cellml compile` after jobs finish)")
    click.echo("[4/4] validate: (invoke `ced cellml validate` after compile)")
    _ = shutil  # keep import for future compile/validate scheduling
