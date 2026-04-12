"""Experiment registry: append-only provenance log for all runs.

Writes one row per run to ``results/experiment_registry.csv`` at the project
root. The registry is the authoritative cross-experiment index — given any
run_id, this file resolves experiment, phase, git SHA, config path, and outdir
without consulting per-run metadata JSON.

Schema
------
run_id          : str  — unique run identifier (may carry experiment prefix)
experiment      : str  — experiment namespace (cellml, incident-validation, pipeline, …)
phase           : str  — sub-phase within the experiment (v0_gate, main, discovery, …)
started_at      : str  — ISO-8601 timestamp when the run was registered
git_sha         : str  — short git SHA of the working tree (or "dirty" suffix)
config_path     : str  — absolute path to the training config used
outdir          : str  — absolute path to the results output directory for this run
notes           : str  — optional free-text annotation

Usage
-----
    from ced_ml.utils.registry import register_run

    register_run(
        run_id="cellml_v0_20260412_123456",
        experiment="cellml",
        phase="v0_gate",
        config_path=config_file,
        outdir=outdir,
    )
"""

from __future__ import annotations

import csv
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "experiment_registry.csv"

_COLUMNS = [
    "run_id",
    "experiment",
    "phase",
    "started_at",
    "git_sha",
    "config_path",
    "outdir",
    "notes",
]


def _get_git_sha() -> str:
    """Return short git SHA of HEAD, with '+dirty' if working tree is modified."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}+dirty" if dirty != 0 else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _registry_path(results_dir: Path) -> Path:
    """Resolve the registry CSV path from the results directory."""
    # Walk up to find the results/ root (the direct child of project root)
    results_dir = results_dir.resolve()
    candidate = results_dir
    while candidate.parent != candidate:
        if candidate.name == "results" and (candidate.parent / "analysis").is_dir():
            return candidate / _REGISTRY_FILENAME
        candidate = candidate.parent
    # Fallback: place it next to outdir
    return results_dir / _REGISTRY_FILENAME


def register_run(
    run_id: str,
    experiment: str,
    phase: str,
    outdir: Path | str,
    config_path: Path | str | None = None,
    notes: str = "",
) -> Path:
    """Append a run entry to the experiment registry CSV.

    Creates the registry file with headers if it does not exist.
    Uses append mode so concurrent writes from different HPC jobs are safe
    (each row is a single atomic ``csv.writer.writerow`` call).

    Args:
        run_id: Unique run identifier (e.g. ``"cellml_v0_20260412_123456"``).
        experiment: Experiment namespace (e.g. ``"cellml"``, ``"incident-validation"``).
        phase: Sub-phase (e.g. ``"v0_gate"``, ``"main"``, ``"lr"``).
        outdir: Absolute path to the results directory for this run.
        config_path: Path to the training config YAML (optional).
        notes: Free-text annotation (optional).

    Returns:
        Path to the registry CSV file.
    """
    outdir = Path(outdir).resolve()
    registry = _registry_path(outdir)

    needs_header = not registry.exists()
    registry.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": run_id,
        "experiment": experiment,
        "phase": phase,
        "started_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _get_git_sha(),
        "config_path": str(config_path) if config_path else "",
        "outdir": str(outdir),
        "notes": notes,
    }

    with open(registry, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    logger.debug("Registry entry written: run_id=%s → %s", run_id, registry)
    return registry
