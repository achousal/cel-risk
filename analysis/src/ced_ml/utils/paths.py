"""
Path utilities for standardized file/directory operations.

This module provides path resolution for CLI commands.

Project structure:
    CeliacRisks/                    <- Project root (run ced here)
    ├── data/                       <- Input data
    ├── splits/                     <- Split indices
    ├── results/                    <- Model outputs
    ├── analysis/                   <- Analysis package
    │   ├── configs/               <- Configuration files
    │   ├── src/ced_ml/            <- Package source
    │   └── tests/                  <- Test suite

Path resolution:
    - Run from CeliacRisks/: paths like "data/" resolve correctly
    - Run from CeliacRisks/analysis/: root auto-detected by walking up
    - Paths in configs are relative to analysis/ directory
"""

from pathlib import Path

# Maximum number of parent directories to search when locating the project root.
_MAX_SEARCH_DEPTH = 5


def _is_project_root(path: Path) -> bool:
    """Return True if *path* looks like the CeliacRisks project root."""
    return (path / "data").is_dir() and (path / "analysis").is_dir()


# ============================================================================
# Path Resolution
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory (CeliacRisks/).

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
        f"Could not locate project root (CeliacRisks/).\n"
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
