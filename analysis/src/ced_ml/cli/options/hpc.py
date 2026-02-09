"""HPC-related CLI options."""

import click

hpc_option = click.option(
    "--hpc",
    is_flag=True,
    default=False,
    help="Submit LSF jobs to HPC cluster instead of running locally",
)


queue_option = click.option(
    "--queue",
    type=str,
    default="short",
    help="LSF queue name (default: short)",
)


memory_option = click.option(
    "--memory",
    type=int,
    default=16000,
    help="Memory per job in MB (default: 16000)",
)


runtime_option = click.option(
    "--runtime",
    type=str,
    default="4:00",
    help="Wall time limit (default: 4:00 hours)",
)
