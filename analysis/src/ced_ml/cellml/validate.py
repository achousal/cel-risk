"""Shell out to operations/cellml/validate_tree.R and parse V1-V4 readouts."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_READOUT_RE = re.compile(r"^\s*(V[1-4])\s*[:=\-]\s*(.+?)\s*$")


@dataclass
class ValidationReport:
    experiment: str
    returncode: int
    stdout: str
    stderr: str
    readouts: dict[str, str] = field(default_factory=dict)


def validate_experiment(
    name: str,
    experiment_dir: Path,
    *,
    repo_root: Path,
    rscript_runner=subprocess.run,
) -> ValidationReport:
    """Run validate_tree.R against experiments/<name>/compiled.csv.

    Does NOT reimplement the R analysis — just captures stdout and
    extracts V1-V4 lines into a structured dict.
    """
    compiled_csv = experiment_dir / "compiled.csv"
    if not compiled_csv.exists():
        raise FileNotFoundError(f"compiled.csv missing — run `ced cellml compile {name}` first")
    script = repo_root / "operations" / "cellml" / "validate_tree.R"
    if not script.exists():
        raise FileNotFoundError(f"validate_tree.R not found: {script}")

    rscript = shutil.which("Rscript") or "Rscript"
    cmd = [rscript, str(script), "--input", str(compiled_csv)]
    result = rscript_runner(cmd, capture_output=True, text=True, check=False)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    rc = getattr(result, "returncode", 0)

    readouts: dict[str, str] = {}
    for line in stdout.splitlines():
        match = _READOUT_RE.match(line)
        if match:
            readouts[match.group(1)] = match.group(2)

    return ValidationReport(
        experiment=name,
        returncode=rc,
        stdout=stdout,
        stderr=stderr,
        readouts=readouts,
    )
