"""
Path utilities for standardized file/directory operations.

This module provides path resolution for CLI commands.

**IMPORTANT**: All CLI commands must be run from the project root (CeliacRisks/).

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
    - Paths in configs are relative to analysis/ directory
"""

from pathlib import Path

# ============================================================================
# Path Resolution (must run from project root)
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory (CeliacRisks/).

    **IMPORTANT**: This assumes you are running from the project root.
    If cwd is not the project root, raises an error.

    Returns:
        Path to project root (current working directory)

    Raises:
        RuntimeError: If current directory doesn't look like project root
    """
    cwd = Path.cwd()

    # Check for expected project structure
    has_data = (cwd / "data").is_dir()
    has_analysis = (cwd / "analysis").is_dir()

    if not (has_data and has_analysis):
        raise RuntimeError(
            f"Must run 'ced' commands from project root (CeliacRisks/).\n"
            f"Current directory: {cwd}\n"
            f"Expected structure: data/, analysis/, splits/, results/"
        )

    return cwd


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
