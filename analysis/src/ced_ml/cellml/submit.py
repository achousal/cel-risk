"""LSF submission via Python subprocess.

Ports the runner-script + bsub logic from
operations/cellml/submit_experiment.sh into a library function. The
legacy shell script is preserved untouched for backward compatibility.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ced_ml.cellml.registry import update_status
from ced_ml.cellml.schema import ExperimentSpec

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"Job <(\d+)> is submitted")


@dataclass
class SubmitResult:
    experiment: str
    runners_dir: Path
    bsub_cmd: list[str]
    job_id: str | None
    dry_run: bool
    n_cells: int


def _load_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with open(manifest_csv, newline="") as f:
        return list(csv.DictReader(f))


def _write_runner_script(
    runner_path: Path,
    *,
    cell_row: dict[str, str],
    experiment: ExperimentSpec,
    results_root: Path,
    cores: int,
    submit_ts: str,
    seeds: tuple[int, int],
    repo_root: Path,
) -> None:
    """Port of the per-cell runner section of submit_experiment.sh."""
    cell_id = int(cell_row["cell_id"])
    recipe_id = cell_row["recipe_id"]
    model = cell_row["model"]
    cell_name = cell_row["cell_name"]
    pipeline_config = cell_row["pipeline_config"]

    run_id = f"{recipe_id}__{cell_name}"
    outdir = results_root / recipe_id / cell_name
    seeds_start, seeds_end = seeds

    manifest_entry: dict[str, object] = {
        "cell_id": cell_id,
        "experiment": experiment.name,
        "recipe_id": recipe_id,
        "model": model,
        "cell_name": cell_name,
        "pipeline_config": pipeline_config,
        "seeds": [seeds_start, seeds_end],
        "submit_timestamp": submit_ts,
    }
    for optional in ("scenario", "calibration", "weighting", "downsampling", "control_ratio"):
        if optional in cell_row and cell_row[optional] != "":
            manifest_entry[optional] = cell_row[optional]
    manifest_json = json.dumps(manifest_entry)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {repo_root}
source analysis/.venv/bin/activate
export OPENBLAS_NUM_THREADS=${{LSB_DJOB_NUMPROC:-{cores}}}

RECIPE_ID="{recipe_id}"
CELL_NAME="{cell_name}"
RUN_ID="{run_id}"
OUTDIR="{outdir}"
PIPELINE_CONFIG="{pipeline_config}"

echo "============================================================"
echo "Cell {cell_id}: ${{CELL_NAME}}"
echo "Recipe: ${{RECIPE_ID}} | Run ID: ${{RUN_ID}}"
echo "Output: ${{OUTDIR}}/run_${{RUN_ID}}/"
echo "Seeds: {seeds_start}-{seeds_end}"
echo "============================================================"

mkdir -p "${{OUTDIR}}/run_${{RUN_ID}}"
cat > "${{OUTDIR}}/run_${{RUN_ID}}/cell_manifest_entry.json" <<'MANIFESTEOF'
{manifest_json}
MANIFESTEOF

SEEDS_CSV="$(seq -s, {seeds_start} {seeds_end})"
echo "--- Cell {cell_id} (${{CELL_NAME}}) Seeds ${{SEEDS_CSV}} ---"
ced run-pipeline \\
    --experiment "{experiment.name}" \\
    --pipeline-config "${{PIPELINE_CONFIG}}" \\
    --split-seeds "${{SEEDS_CSV}}" \\
    --run-id "${{RUN_ID}}" \\
    --outdir "${{OUTDIR}}"

echo "Cell {cell_id} complete."
"""
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(script)
    runner_path.chmod(0o755)


def _parse_cell_range(cell_range: str | None, n_cells: int) -> tuple[int, int]:
    if cell_range is None:
        return 1, n_cells
    if "-" not in cell_range:
        raise ValueError(f"--cells must be RANGE like 1-10, got: {cell_range}")
    lo, hi = cell_range.split("-", 1)
    return int(lo), int(hi)


def submit_experiment(
    spec: ExperimentSpec,
    experiment_dir: Path,
    *,
    repo_root: Path,
    results_root: Path | None = None,
    cell_range: str | None = None,
    wall: str | None = None,
    queue: str | None = None,
    project: str | None = None,
    cores: int | None = None,
    mem_mb_per_core: int | None = None,
    dry_run: bool = False,
    runner_factory=_write_runner_script,
    bsub_runner=subprocess.run,
) -> SubmitResult:
    """Generate per-cell runners and submit an LSF array job.

    Parameters
    ----------
    spec : ExperimentSpec
        The (resolved) experiment spec — supplies defaults.
    experiment_dir : Path
        experiments/<name>/ directory with the recipes/cell_manifest.csv.
    repo_root : Path
        Repository root to cd into inside the runner. Passed explicitly
        so tests don't depend on CWD.
    results_root : Path, optional
        Where per-cell results go. Default: results/cellml/<name>/.
    cell_range : str, optional
        "lo-hi" range to submit. Defaults to all cells.
    wall, queue, project, cores, mem_mb_per_core : overrides
        Override individual resource values from spec.resources.
    dry_run : bool
        If True, write runners + print bsub command without calling bsub.
    runner_factory, bsub_runner : injection points for tests
        Default to internal implementations; tests can mock these.

    Returns
    -------
    SubmitResult
    """
    manifest_csv = experiment_dir / "recipes" / "cell_manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Cell manifest missing — run `ced cellml generate` first: {manifest_csv}"
        )

    rows = _load_manifest(manifest_csv)
    n_cells = len(rows)
    lo, hi = _parse_cell_range(cell_range, n_cells)

    resources = spec.resources
    _wall = wall or resources.wall
    _queue = queue or resources.queue
    _project = project or resources.project
    _cores = cores or resources.cores
    _mem = mem_mb_per_core or resources.mem_mb_per_core

    runners_dir = experiment_dir / "logs" / f"{spec.name}_runners"
    runners_dir.mkdir(parents=True, exist_ok=True)

    _results_root = results_root or (repo_root / "results" / "cellml" / spec.name)
    _results_root.mkdir(parents=True, exist_ok=True)

    submit_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for row in rows:
        cid = int(row["cell_id"])
        if cid < lo or cid > hi:
            continue
        runner_path = runners_dir / f"cell_{cid}.sh"
        runner_factory(
            runner_path,
            cell_row=row,
            experiment=spec,
            results_root=_results_root,
            cores=_cores,
            submit_ts=submit_ts,
            seeds=(spec.seeds.start, spec.seeds.end),
            repo_root=repo_root,
        )

    log_dir = experiment_dir / "logs"
    bsub_cmd = [
        "bsub",
        "-J",
        f"{spec.name}[{lo}-{hi}]",
        "-P",
        _project,
        "-q",
        _queue,
        "-n",
        str(_cores),
        "-W",
        _wall,
        "-R",
        f"rusage[mem={_mem}] span[hosts=1]",
        "-o",
        str(log_dir / f"{spec.name}_%J_%I.out"),
        "-e",
        str(log_dir / f"{spec.name}_%J_%I.err"),
        str(runners_dir / "cell_$LSB_JOBINDEX.sh"),
    ]

    if dry_run:
        logger.info("[dry-run] would submit: %s", " ".join(bsub_cmd))
        return SubmitResult(
            experiment=spec.name,
            runners_dir=runners_dir,
            bsub_cmd=bsub_cmd,
            job_id=None,
            dry_run=True,
            n_cells=hi - lo + 1,
        )

    # Real submit
    result = bsub_runner(bsub_cmd, capture_output=True, text=True, check=False)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    rc = getattr(result, "returncode", 1)
    if rc != 0:
        raise RuntimeError(f"bsub failed (rc={rc}): {stderr.strip()}")
    match = _JOB_ID_RE.search(stdout)
    job_id = match.group(1) if match else None
    logger.info("bsub stdout: %s", stdout.strip())

    update_status(
        spec.name,
        status="submitted",
        submitted_at=submit_ts,
        job_id=job_id or "",
    )

    return SubmitResult(
        experiment=spec.name,
        runners_dir=runners_dir,
        bsub_cmd=bsub_cmd,
        job_id=job_id,
        dry_run=False,
        n_cells=hi - lo + 1,
    )
