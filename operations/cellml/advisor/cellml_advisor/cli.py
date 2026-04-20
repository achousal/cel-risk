"""`cellml-advisor` Click CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="cellml-advisor")
def main() -> None:
    """LLM advisor for the CellML factorial gate workflow (scaffold)."""


# -----------------------------------------------------------------------------
# rulebook
# -----------------------------------------------------------------------------


@main.group()
def rulebook() -> None:
    """Inspect and validate the rulebook."""


@rulebook.command("show")
@click.option(
    "--filter",
    "filter_type",
    type=click.Choice(["equation", "condensate", "protocol"], case_sensitive=False),
    default=None,
    help="Restrict output to a single entry type.",
)
def rulebook_show(filter_type: str | None) -> None:
    """List rulebook entries, optionally filtered by type."""
    click.echo(f"NotImplementedError: rulebook show --filter={filter_type} (scaffold: implement in v0.2)")


@rulebook.command("validate")
def rulebook_validate() -> None:
    """Walk wiki-links in the rulebook; report dangling or cyclic references."""
    click.echo("NotImplementedError: rulebook validate (scaffold: implement in v0.2)")


# -----------------------------------------------------------------------------
# ledger
# -----------------------------------------------------------------------------


@main.group()
def ledger() -> None:
    """Draft and validate gate ledger.md artifacts."""


@ledger.command("draft")
@click.argument("project", type=str)
@click.argument("gate", type=str)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the draft to PATH (default: print to stdout).",
)
def ledger_draft(project: str, gate: str, out_path: Path | None) -> None:
    """Draft a ledger.md for <project> <gate> from the protocol and prior locks."""
    click.echo(
        f"NotImplementedError: ledger draft {project} {gate} --out={out_path} "
        "(scaffold: implement in v0.2)"
    )


@ledger.command("validate")
@click.argument("ledger_path", type=click.Path(exists=False, dir_okay=False, path_type=Path))
def ledger_validate(ledger_path: Path) -> None:
    """Validate sections, frontmatter, and links for a ledger.md on disk."""
    click.echo(f"NotImplementedError: ledger validate {ledger_path} (scaffold: implement in v0.2)")


# -----------------------------------------------------------------------------
# decision
# -----------------------------------------------------------------------------


@main.group()
def decision() -> None:
    """Draft gate decision.md artifacts."""


@decision.command("draft")
@click.argument("project", type=str)
@click.argument("gate", type=str)
def decision_draft(project: str, gate: str) -> None:
    """Draft a decision.md from the ledger + observation for <project> <gate>."""
    click.echo(
        f"NotImplementedError: decision draft {project} {gate} (scaffold: implement in v0.2)"
    )


# -----------------------------------------------------------------------------
# tension
# -----------------------------------------------------------------------------


@main.group()
def tension() -> None:
    """Detect tensions between ledger predictions and observed metrics."""


@tension.command("detect")
@click.argument("project", type=str)
@click.argument("gate", type=str)
def tension_detect(project: str, gate: str) -> None:
    """Diff predicted vs observed for <project> <gate>; emit tensions.md."""
    click.echo(
        f"NotImplementedError: tension detect {project} {gate} (scaffold: implement in v0.2)"
    )


if __name__ == "__main__":
    main()
