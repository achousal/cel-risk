"""Analysis runner -- Python harness for R script execution + artifact logging.

Manages the agent's R script generation/execution loop:
- Creates workspace directories
- Executes R scripts and captures output
- Tracks all generated artifacts in analysis_log.json
- Optionally pauses for human review before execution
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE = Path(__file__).parent
SCRIPTS_DIR = WORKSPACE / "scripts"
FIGURES_DIR = WORKSPACE / "figures"
TABLES_DIR = WORKSPACE / "tables"
LOG_PATH = WORKSPACE / "analysis_log.json"


def init_workspace() -> None:
    """Create workspace directories if they don't exist."""
    for d in (SCRIPTS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("[]")


def execute_r(script_path: Path, timeout: int = 300) -> tuple[str, str, int]:
    """Execute an R script and return (stdout, stderr, exit_code).

    Parameters
    ----------
    script_path
        Path to the .R script.
    timeout
        Maximum seconds to wait for the script.

    Returns
    -------
    (stdout, stderr, exit_code)
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    result = subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(WORKSPACE),
    )

    logger.info(
        "Rscript %s exited with code %d",
        script_path.name, result.returncode,
    )
    if result.returncode != 0:
        logger.warning("R stderr: %s", result.stderr[:500])

    return result.stdout, result.stderr, result.returncode


def log_artifact(
    script: str,
    figures: list[str] | None = None,
    tables: list[str] | None = None,
    notes: str = "",
    exit_code: int = 0,
) -> None:
    """Append an entry to analysis_log.json.

    Parameters
    ----------
    script
        Name of the R script that was executed.
    figures
        List of figure filenames produced.
    tables
        List of table filenames produced.
    notes
        Agent's interpretation notes for this step.
    exit_code
        R script exit code.
    """
    init_workspace()

    log = json.loads(LOG_PATH.read_text())
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "figures": figures or [],
        "tables": tables or [],
        "exit_code": exit_code,
        "notes": notes,
    }
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def list_artifacts() -> list[dict]:
    """Read the analysis log."""
    if not LOG_PATH.exists():
        return []
    return json.loads(LOG_PATH.read_text())


def list_figures() -> list[Path]:
    """List all figure files in the figures directory."""
    if not FIGURES_DIR.exists():
        return []
    return sorted(FIGURES_DIR.glob("*.*"))


def list_tables() -> list[Path]:
    """List all table files in the tables directory."""
    if not TABLES_DIR.exists():
        return []
    return sorted(TABLES_DIR.glob("*.csv"))


def run_analysis_step(
    script_name: str,
    review_before_execute: bool = False,
) -> tuple[str, str, int]:
    """Execute one analysis step: run script, log artifacts.

    Parameters
    ----------
    script_name
        Name of the R script in scripts/ (e.g. 'v1_01_recipe_heatmap.R').
    review_before_execute
        If True, print the script and wait for confirmation before running.

    Returns
    -------
    (stdout, stderr, exit_code)
    """
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if review_before_execute:
        print(f"\n{'='*60}")
        print(f"REVIEW: {script_name}")
        print(f"{'='*60}")
        print(script_path.read_text())
        print(f"{'='*60}")
        response = input("Execute? [y/N] ").strip().lower()
        if response != "y":
            print("Skipped.")
            return "", "Skipped by user", -1

    # Snapshot figures/tables before execution
    figs_before = set(f.name for f in list_figures())
    tbls_before = set(t.name for t in list_tables())

    stdout, stderr, exit_code = execute_r(script_path)

    # Detect new artifacts
    figs_after = set(f.name for f in list_figures())
    tbls_after = set(t.name for t in list_tables())
    new_figs = sorted(figs_after - figs_before)
    new_tbls = sorted(tbls_after - tbls_before)

    log_artifact(
        script=script_name,
        figures=new_figs,
        tables=new_tbls,
        exit_code=exit_code,
    )

    return stdout, stderr, exit_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run factorial analysis R scripts")
    parser.add_argument("script", help="R script name in scripts/")
    parser.add_argument("--review", action="store_true", help="Review script before execution")
    args = parser.parse_args()

    init_workspace()
    stdout, stderr, rc = run_analysis_step(args.script, review_before_execute=args.review)
    if stdout:
        print(stdout)
    if stderr and rc != 0:
        print(stderr, file=sys.stderr)
    sys.exit(rc)
