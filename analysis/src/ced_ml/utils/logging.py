"""
Consistent logging setup for CeD-ML pipeline.

Replaces print() statements with proper logging throughout the codebase.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "ced_ml",
    level: int = logging.INFO,
    log_file: Path | None = None,
    format_string: str | None = None,
    use_live_log: bool = False,
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    USAGE PATTERN:
        - CLI entrypoints: Call this function to create a logger with handlers
        - Library modules: Use logging.getLogger(__name__) directly (no handlers)
        - Child loggers automatically propagate to parent logger with handlers

    Args:
        name: Logger name (typically "ced_ml" for main CLI)
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Custom format string (default: timestamp + level + message)
        use_live_log: If True, log to .live file and rename on completion

    Returns:
        Configured logger instance with handlers attached
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Design: Only CLI entrypoints call setup_logger() which adds handlers.
    # All library modules use logging.getLogger(__name__) with no handlers.
    # To prevent duplicate output, disable propagation on loggers with handlers.
    # Child loggers (from library modules) keep propagate=True by default,
    # so they bubble up to their parent logger which has the handlers.
    logger.propagate = False

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # If use_live_log, write to .live file first
        if use_live_log:
            live_log_file = log_file.with_suffix(".live")
            file_handler = logging.FileHandler(live_log_file, mode="a")
            # Store the final log path for later renaming
            file_handler._final_log_path = log_file
            file_handler._live_log_path = live_log_file
        else:
            file_handler = logging.FileHandler(log_file, mode="a")

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Removed get_logger() - modules should use logging.getLogger(__name__) directly
# Only CLI entrypoints should call setup_logger() to configure the root "ced_ml" logger


def log_hpc_context(logger: logging.Logger) -> None:
    """Log HPC scheduler metadata if running inside a batch job.

    Reads environment variables from LSF (LSB_*) and SLURM (SLURM_*)
    to emit job ID, hostname, allocated resources, and queue/partition.
    No-op when no scheduler variables are detected (local runs).
    """
    import os
    import socket

    # LSF
    lsf_job_id = os.environ.get("LSB_JOBID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    if lsf_job_id:
        logger.info("--- HPC context (LSF) ---")
        logger.info(f"  Job ID:    {lsf_job_id}")
        logger.info(f"  Node:      {socket.gethostname()}")
        logger.info(f"  Queue:     {os.environ.get('LSB_QUEUE', 'n/a')}")
        logger.info(f"  Project:   {os.environ.get('LSB_PROJECT_NAME', 'n/a')}")
        logger.info(f"  CPUs:      {os.environ.get('LSB_DJOB_NUMPROC', 'n/a')}")
        logger.info("--- end HPC context ---")
    elif slurm_job_id:
        logger.info("--- HPC context (SLURM) ---")
        logger.info(f"  Job ID:    {slurm_job_id}")
        logger.info(f"  Node:      {socket.gethostname()}")
        logger.info(f"  Partition: {os.environ.get('SLURM_JOB_PARTITION', 'n/a')}")
        logger.info(f"  Account:   {os.environ.get('SLURM_JOB_ACCOUNT', 'n/a')}")
        logger.info(f"  CPUs:      {os.environ.get('SLURM_CPUS_ON_NODE', 'n/a')}")
        logger.info(f"  Mem (MB):  {os.environ.get('SLURM_MEM_PER_NODE', 'n/a')}")
        logger.info("--- end HPC context ---")


def auto_log_path(
    command: str,
    outdir: Path | str = "results",
    run_id: str | None = None,
    model: str | None = None,
    split_seed: int | None = None,
) -> Path:
    """Build an automatic log file path based on command context.

    Log directory structure:
        logs/
          training/run_{ID}/{model}_seed{N}.log
          ensemble/run_{ID}/ENSEMBLE_seed{N}.log
          aggregation/run_{ID}/{model}.log
          optimization/run_{ID}/{model}.log
          consensus/run_{ID}/consensus.log
          pipeline/run_{ID}.log

    Args:
        command: CLI command name (train, train-ensemble, aggregate-splits,
                 optimize-panel, consensus-panel, run-pipeline).
        outdir: Results output directory (used to resolve logs/ sibling).
        run_id: Run identifier (falls back to "unknown" if None).
        model: Model name (used for per-model log files).
        split_seed: Split seed (used for per-seed log files).

    Returns:
        Absolute Path for the log file. Parent directories are NOT created
        here -- callers should use ``setup_logger(log_file=...)`` which
        handles that.
    """
    outdir = Path(outdir).resolve()

    # Find CeliacRisks project root by traversing up
    # This is more reliable than looking for logs/results which may exist at multiple levels
    current = outdir
    project_root = None

    for parent in [current] + list(current.parents):
        if parent.name == "CeliacRisks":
            project_root = parent
            break

    if project_root is None:
        # Fallback: place logs as sibling to outdir (works in test environments)
        logs_root = outdir.parent / "logs"
    else:
        logs_root = project_root / "logs"
    rid = run_id or "unknown"

    if command == "train":
        model_part = model or "unknown"
        seed_part = f"_seed{split_seed}" if split_seed is not None else ""
        return logs_root / "training" / f"run_{rid}" / f"{model_part}{seed_part}.log"

    if command == "train-ensemble":
        seed_part = f"_seed{split_seed}" if split_seed is not None else ""
        return logs_root / "ensemble" / f"run_{rid}" / f"ENSEMBLE{seed_part}.log"

    if command == "aggregate-splits":
        model_part = model or "all"
        return logs_root / "aggregation" / f"run_{rid}" / f"{model_part}.log"

    if command == "optimize-panel":
        model_part = model or "all"
        seed_part = f"_seed{split_seed}" if split_seed is not None else ""
        return logs_root / "optimization" / f"run_{rid}" / f"{model_part}{seed_part}.log"

    if command == "consensus-panel":
        return logs_root / "consensus" / f"run_{rid}" / "consensus.log"

    if command == "run-pipeline":
        return logs_root / "pipeline" / f"run_{rid}.log"

    # Fallback
    return logs_root / "misc" / f"{command}_{rid}.log"


def setup_command_logger(
    command: str,
    log_level: int = logging.INFO,
    outdir: Path | str = "results",
    run_id: str | None = None,
    model: str | None = None,
    split_seed: int | None = None,
    logger_name: str = "ced_ml",
) -> logging.Logger:
    """Set up a logger for a CLI command with automatic log file path.

    Combines auto_log_path() + setup_logger() + info message into a single call.
    This replaces 6-10 lines of boilerplate in every CLI command.

    Args:
        command: CLI command name (train, train-ensemble, aggregate-splits, etc.)
        log_level: Logging level (default: INFO)
        outdir: Results output directory
        run_id: Run identifier
        model: Model name (optional)
        split_seed: Split seed (optional)
        logger_name: Logger name (default: "ced_ml")

    Returns:
        Configured logger instance with file and console handlers.
    """
    log_path = auto_log_path(command, outdir, run_id, model, split_seed)
    logger = setup_logger(logger_name, log_level, log_file=log_path)
    logger.info(f"Logging to file: {log_path}")
    return logger


def log_section(logger: logging.Logger, title: str, width: int = 80, char: str = "="):
    """Log a section header."""
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_fold_header(
    logger: logging.Logger,
    fold_num: int,
    total_folds: int,
    repeat_num: int,
    width: int = 72,
) -> None:
    """Log a visually distinct fold header using dashes.

    Example output::

        --- Fold 3/15 (repeat=0) -------------------------------------------
    """
    label = f" Fold {fold_num}/{total_folds} (repeat={repeat_num}) "
    line = f"---{label}{'-' * max(0, width - 3 - len(label))}"
    logger.info(line)
