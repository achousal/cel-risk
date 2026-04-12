"""
Path utilities for standardized file/directory operations.

This module provides path resolution for CLI commands.

Project structure:
    cel-risk/                    <- Project root (run ced here)
    ├── data/                       <- Input data
    ├── splits/                     <- Split indices
    ├── results/                    <- Model outputs (namespaced by experiment)
    │   ├── pipeline/               <- Ad-hoc pipeline runs
    │   ├── cellml/                 <- CellML factorial experiment
    │   └── incident-validation/    <- Incident validation experiment
    ├── logs/                       <- Runtime logs (mirrors results/ namespace)
    │   ├── pipeline/
    │   ├── cellml/
    │   └── incident-validation/
    ├── operations/                 <- Orchestration code, configs, scripts, post-hoc analysis
    │   ├── cellml/                  (concrete experiments materialize under results/cellml/)
    │   └── incident-validation/     (concrete experiments materialize under results/incident-validation/)
    └── analysis/                   <- Analysis package
        ├── configs/               <- Base configuration files (hand-authored)
        ├── src/ced_ml/            <- Package source
        └── tests/                  <- Test suite

Path resolution:
    - Run from cel-risk/: paths like "data/" resolve correctly
    - Run from cel-risk/analysis/: root auto-detected by walking up
    - Paths in configs are relative to analysis/ directory
"""

from datetime import datetime
from pathlib import Path

# Maximum number of parent directories to search when locating the project root.
_MAX_SEARCH_DEPTH = 5


def _is_project_root(path: Path) -> bool:
    """Return True if *path* looks like the cel-risk project root."""
    return (path / "data").is_dir() and (path / "analysis").is_dir()


# ============================================================================
# Path Resolution
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory (cel-risk/).

    Resolution order:
        1. Current working directory (CWD)
        2. Walk up from CWD (up to 5 levels) looking for a directory
           that contains both ``data/`` and ``analysis/`` subdirectories.

    Returns:
        Resolved Path to the project root.

    Raises:
        RuntimeError: If the project root cannot be found.
    """
    cwd = Path.cwd()

    # Fast path: CWD is the project root
    if _is_project_root(cwd):
        return cwd

    # Walk up the directory tree
    candidate = cwd
    for _ in range(_MAX_SEARCH_DEPTH):
        candidate = candidate.parent
        if candidate == candidate.parent:
            break  # reached filesystem root
        if _is_project_root(candidate):
            return candidate

    raise RuntimeError(
        f"Could not locate project root (cel-risk/).\n"
        f"Current directory: {cwd}\n"
        f"Searched {_MAX_SEARCH_DEPTH} levels up.\n"
        f"Expected structure: data/, analysis/, splits/, results/"
    )


def get_analysis_dir() -> Path:
    """Get the analysis/ directory path."""
    return get_project_root() / "analysis"


def get_default_paths() -> dict:
    """
    Get default paths for common directories.

    Returns:
        Dict with keys: project_root, analysis, data, splits, results, configs
    """
    root = get_project_root()
    return {
        "project_root": root,
        "analysis": root / "analysis",
        "data": root / "data",
        "splits": root / "splits",
        "results": root / "results",
        "configs": root / "analysis" / "configs",
        "logs": root / "logs",
    }


def make_run_id(experiment_tag: str | None = None) -> str:
    """Generate a timestamped run identifier.

    Args:
        experiment_tag: Optional prefix that names the experiment context
            (e.g. ``"cellml_v0"``, ``"incval_lr"``). When provided, the run ID
            becomes ``"{tag}_{timestamp}"`` so that ``ls results/cellml/v0_gate/``
            is immediately readable without consulting a registry.

    Returns:
        Run ID string, e.g. ``"cellml_v0_20260412_123456"`` or ``"20260412_123456"``.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_tag:
        return f"{experiment_tag}_{ts}"
    return ts


def derive_logs_dir(outdir: Path, project_root: Path) -> Path:
    """Derive the logs namespace that mirrors the results namespace.

    When ``outdir`` is under ``{project_root}/results/``, the relative path
    segment (e.g. ``cellml/v0_gate``) is transplanted into ``logs/`` so that
    log files co-locate with their result artifacts structurally:

        results/cellml/v0_gate/run_{id}/  →  logs/cellml/v0_gate/run_{id}/

    If ``outdir`` is not under ``results/`` (e.g. tests or ad-hoc paths), the
    flat ``logs/`` root is returned unchanged.

    Args:
        outdir: Resolved results output directory.
        project_root: Resolved project root directory.

    Returns:
        Resolved logs directory (namespace stem only — caller appends ``run_{id}``).
    """
    results_root = (project_root / "results").resolve()
    outdir_resolved = outdir.resolve()
    try:
        namespace = outdir_resolved.relative_to(results_root)
        if str(namespace) != ".":
            return (project_root / "logs" / namespace).resolve()
    except ValueError:
        pass
    return (project_root / "logs").resolve()
