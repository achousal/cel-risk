"""Common CLI options used across multiple commands."""

import click


def run_id_option(
    required: bool = False,
    help_text: str = "Shared run ID (default: auto-generated timestamp). Use to group multiple splits/models under one run.",
):
    """
    Reusable --run-id option decorator.

    Args:
        required: If True, makes the option required
        help_text: Custom help text for the option

    Returns:
        Click option decorator
    """
    return click.option(
        "--run-id",
        type=str,
        required=required,
        default=None if not required else ...,
        help=help_text,
    )


def run_id_required_option(
    help_text: str = "Run ID to process (e.g., 20260127_115115). Auto-discovers all models.",
):
    """Convenience wrapper for required run-id option."""
    return run_id_option(required=True, help_text=help_text)


config_option = click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)


dry_run_option = click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be executed without running",
)


log_level_option = click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging level (default: info). Use 'debug' for detailed algorithm insights.",
)
