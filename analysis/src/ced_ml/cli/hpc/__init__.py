"""
HPC job submission utilities for CLI commands.

Centralizes HPC-related logic including:
- HPC config loading and validation
- Job script building
- Job submission (with dry-run support)
- Common patterns used by train, etc.
"""

from ced_ml.cli.hpc.submission import (
    load_hpc_config_with_fallback,
    setup_hpc_environment,
    submit_train_jobs,
)

__all__ = [
    "load_hpc_config_with_fallback",
    "setup_hpc_environment",
    "submit_train_jobs",
]
