"""LSF monitoring via bjobs shell-out.

Different from operations/cellml/monitor_factorial.py, which queries
Optuna storage. This one watches the LSF queue by job name.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STATES = ("PEND", "RUN", "DONE", "EXIT", "UNKWN")


@dataclass
class MonitorResult:
    experiment: str
    counts: dict[str, int]
    error: str | None = None


def _bjobs_available() -> bool:
    return shutil.which("bjobs") is not None


def _parse_bjobs_wide(stdout: str, experiment: str) -> dict[str, int]:
    """Parse ``bjobs -w`` output, counting jobs whose name starts with experiment[.

    Column order for ``bjobs -w``:
      JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME ... SUBMIT_TIME
    """
    counts: dict[str, int] = dict.fromkeys(STATES, 0)
    lines = stdout.strip().splitlines()
    if not lines:
        return counts
    # Skip header if present
    if lines[0].lstrip().startswith("JOBID"):
        lines = lines[1:]
    for line in lines:
        cols = line.split()
        if len(cols) < 7:
            continue
        stat = cols[2]
        job_name = cols[6]
        if not (job_name == experiment or job_name.startswith(f"{experiment}[")):
            continue
        if stat in counts:
            counts[stat] += 1
        else:
            counts["UNKWN"] += 1
    return counts


def get_status(
    experiment: str,
    *,
    runner=subprocess.run,
    env: Mapping[str, str] | None = None,
) -> MonitorResult:
    """Return counts of jobs for this experiment by LSF state.

    Parameters
    ----------
    experiment : str
        Experiment name (matches the LSF -J prefix).
    runner : callable, optional
        subprocess.run-compatible; injected for tests.
    env : mapping, optional
        Passed through to bjobs ($USER used for -u).
    """
    if not _bjobs_available() and runner is subprocess.run:
        return MonitorResult(
            experiment=experiment,
            counts=dict.fromkeys(STATES, 0),
            error="bjobs not available on this host",
        )

    user = (env or os.environ).get("USER", "")
    cmd = ["bjobs", "-w"]
    if user:
        cmd += ["-u", user]
    try:
        result = runner(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        return MonitorResult(
            experiment=experiment,
            counts=dict.fromkeys(STATES, 0),
            error=str(e),
        )

    stdout = getattr(result, "stdout", "") or ""
    rc = getattr(result, "returncode", 0)
    if rc != 0 and not stdout:
        return MonitorResult(
            experiment=experiment,
            counts=dict.fromkeys(STATES, 0),
            error=getattr(result, "stderr", "") or f"bjobs returned {rc}",
        )
    counts = _parse_bjobs_wide(stdout, experiment)
    return MonitorResult(experiment=experiment, counts=counts)
