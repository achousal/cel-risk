"""Configuration management CLI commands (validate, diff)."""

from pathlib import Path

import click

from ced_ml.cli.config_tools import run_config_diff, run_config_validate


@click.group("config")
@click.pass_context
def config_group(ctx):
    """Configuration management tools (validate, diff)."""


@config_group.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--command",
    type=click.Choice(["save-splits", "train"]),
    default="train",
    help="Command type (default: train)",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as errors",
)
@click.pass_context
def config_validate(ctx, config_file, command, strict):
    """Validate configuration file and report issues."""
    run_config_validate(
        config_file=Path(config_file),
        command=command,
        strict=strict,
        log_level=ctx.obj.get("log_level"),
    )


@config_group.command("diff")
@click.argument("config_file1", type=click.Path(exists=True))
@click.argument("config_file2", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for diff report",
)
@click.pass_context
def config_diff(ctx, config_file1, config_file2, output):
    """Compare two configuration files."""
    run_config_diff(
        config_file1=Path(config_file1),
        config_file2=Path(config_file2),
        output_file=Path(output) if output else None,
        log_level=ctx.obj.get("log_level"),
    )
