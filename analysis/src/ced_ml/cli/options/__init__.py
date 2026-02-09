"""
Reusable Click option decorators for CLI commands.

Provides centralized option definitions to eliminate duplication across commands.
Options are organized by category:
- common: run_id, config, dry_run
- paths: infile, outdir, results_dir, splits_dir
- hpc: hpc, queue, memory, runtime
- models: model, split_seed, n_splits
- feature_selection: feature methods
- evaluation: metrics and reporting
"""

from ced_ml.cli.options.common import (
    config_option,
    dry_run_option,
    log_level_option,
    run_id_option,
    run_id_required_option,
)
from ced_ml.cli.options.hpc import (
    hpc_option,
    memory_option,
    queue_option,
    runtime_option,
)
from ced_ml.cli.options.models import (
    model_option,
    n_splits_option,
    split_seed_option,
)
from ced_ml.cli.options.paths import (
    infile_option,
    outdir_option,
    results_dir_option,
    splits_dir_option,
)

__all__ = [
    # Common
    "run_id_option",
    "run_id_required_option",
    "config_option",
    "dry_run_option",
    "log_level_option",
    # Paths
    "infile_option",
    "outdir_option",
    "results_dir_option",
    "splits_dir_option",
    # HPC
    "hpc_option",
    "queue_option",
    "memory_option",
    "runtime_option",
    # Models
    "model_option",
    "split_seed_option",
    "n_splits_option",
]
