"""
Adaptive sweep runner CLI command.

Commands:
  - sweep: Run an adaptive sweep from a spec file
"""

import json
import logging
import sys
from pathlib import Path

import click

from ced_ml.utils.paths import get_project_root

logger = logging.getLogger(__name__)


@click.command("sweep")
@click.option(
    "--spec",
    "-s",
    type=click.Path(exists=True),
    required=True,
    help="Path to sweep spec YAML file.",
)
@click.option(
    "--max-iter",
    type=int,
    default=None,
    help="Override max iterations from spec.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs without submitting to SLURM.",
)
@click.option(
    "--poll-interval",
    type=int,
    default=60,
    help="Seconds between SLURM status polls.",
)
@click.option(
    "--params",
    type=str,
    default=None,
    help="JSON dict of parameters for a single manual iteration (e.g. '{\"downsampling_ratio\": 5.0}').",
)
@click.option(
    "--status",
    is_flag=True,
    help="Print sweep status and exit (no new iteration).",
)
@click.pass_context
def sweep(ctx, spec, max_iter, dry_run, poll_interval, params, status):
    """Run an adaptive sweep iteration from a spec file.

    The sweep runner reads a YAML spec declaring the question, parameter
    space, constraints, and evaluation criteria. It submits experiments
    to Minerva, polls for completion, evaluates results, and logs to
    an append-only ledger.

    \b
    Examples:
      ced sweep --spec specs/09_downsampling.yaml --dry-run
      ced sweep --spec specs/02_bootstrap.yaml --params '{"n_boot": 2000}'
      ced sweep --spec specs/09_downsampling.yaml --status
    """
    # Import here to avoid circular deps and keep CLI fast
    sys.path.insert(0, str(get_project_root() / "operations" / "cellml" / "sweeps"))
    from sweep_orchestrator import SweepOrchestrator, load_sweep_spec

    spec_path = Path(spec)
    sweep_spec = load_sweep_spec(spec_path)

    # Override max iterations if requested
    if max_iter is not None:
        sweep_spec.constraints.max_iterations = max_iter

    project_root = get_project_root()
    orchestrator = SweepOrchestrator(
        spec=sweep_spec,
        project_root=project_root,
        dry_run=dry_run,
    )

    # Status mode: print and exit
    if status:
        summary = orchestrator.summary()
        click.echo(f"Sweep: {summary['sweep_id']}")
        click.echo(f"  Iterations: {summary['iterations_completed']}/{summary['max_iterations']}")
        click.echo(f"  Running best: {summary['running_best']}")
        click.echo(f"  Baseline: {summary['baseline']}")
        click.echo(f"  No-improve streak: {summary['consecutive_no_improve']}")
        click.echo(f"  Wall hours used: {summary['total_wall_hours']}")
        return

    # Check prerequisites
    missing = orchestrator.check_prerequisites()
    if missing:
        click.echo("Missing prerequisites:", err=True)
        for m in missing:
            click.echo(f"  - {m}", err=True)
        raise click.Abort()

    # Check budget
    can_go, reason = orchestrator.can_continue()
    if not can_go:
        click.echo(f"Sweep cannot continue: {reason.value}", err=True)
        raise click.Abort()

    # Parse params
    if params:
        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON: {e}") from e
    else:
        # For now, require explicit params. Agent integration will propose them.
        click.echo(
            "No --params provided. Use --params '{...}' to specify iteration parameters, "
            "or integrate with sweep_agent.py for autonomous proposals.",
            err=True,
        )
        raise click.Abort()

    # Run iteration
    click.echo(f"\nSweep: {sweep_spec.id}")
    click.echo(f"  Question: {sweep_spec.question}")
    click.echo(f"  Params: {param_dict}")
    click.echo(f"  Dry run: {dry_run}")
    click.echo()

    eval_result, decision = orchestrator.run_iteration(
        params=param_dict,
        poll_interval=poll_interval,
    )

    # Report
    click.echo(f"\nDecision: {decision}")
    if eval_result:
        click.echo(f"  Metric: {eval_result.metric_value:.6f}")
        if eval_result.delta_baseline is not None:
            click.echo(f"  Delta vs baseline: {eval_result.delta_baseline:+.6f}")
        if eval_result.delta_previous is not None:
            click.echo(f"  Delta vs previous: {eval_result.delta_previous:+.6f}")
        click.echo(f"  Running best: {eval_result.running_best:.6f}")

    summary = orchestrator.summary()
    click.echo(f"\n  Progress: {summary['iterations_completed']}/{summary['max_iterations']}")
    click.echo(f"  Ledger: {orchestrator.ledger.path}")
