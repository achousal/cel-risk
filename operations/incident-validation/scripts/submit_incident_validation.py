#!/usr/bin/env python3
"""Manifest-driven LSF launcher for the incident-validation operation.

Reads operations/incident-validation/manifest.yaml and submits one 14-job
chain per model (feat -> 12 CV combos -> aggregate) with per-model LSF
resource profiles.

Usage (on Minerva login node, from repo root):
    python operations/incident-validation/scripts/submit_incident_validation.py
    python operations/incident-validation/scripts/submit_incident_validation.py --smoke
    python operations/incident-validation/scripts/submit_incident_validation.py --models LR_EN,XGB
    python operations/incident-validation/scripts/submit_incident_validation.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path("/sc/arion/projects/vascbrain/andres/cel-risk")
MANIFEST_PATH = REPO_ROOT / "operations/incident-validation/manifest.yaml"
SCRIPT_PATH = REPO_ROOT / "operations/incident-validation/scripts/run_lr.py"
DATAFILE = REPO_ROOT / "data/Celiac_dataset_proteomics_w_demo.parquet"
LOGDIR = REPO_ROOT / "logs/incident-validation"

ACTIVATE = (
    f"cd {REPO_ROOT} && module load python/3.12.5 "
    f"&& source analysis/venv/bin/activate "
    f"&& unset PYTHONPATH"
)


def resource_profile(manifest: dict, model: str) -> dict:
    """Merge default resources with per-model overrides."""
    defaults = manifest["resources"]["default"]
    override = manifest["resources"].get(model) or {}
    return {**defaults, **override}


def submit(cmd: list[str], dry_run: bool) -> str:
    """Run bsub, return job id."""
    if dry_run:
        print("  [dry-run]", " ".join(cmd))
        return "DRYRUN"
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match = re.search(r"Job <(\d+)>", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job id from: {result.stdout}")
    return match.group(1)


def build_bsub(
    *,
    project: str,
    queue: str,
    wall: str,
    cores: int,
    mem: int,
    job_name: str,
    stdout_path: Path,
    stderr_path: Path,
    depend_on: str | None,
    command: str,
) -> list[str]:
    cmd = [
        "bsub",
        "-P", project,
        "-q", queue,
        "-W", wall,
        "-n", str(cores),
        "-R", f"rusage[mem={mem}] span[hosts=1]",
        "-J", job_name,
        "-o", str(stdout_path),
        "-e", str(stderr_path),
    ]
    if depend_on:
        cmd += ["-w", depend_on]
    cmd += ["bash", "-c", command]
    return cmd


def submit_model_chain(
    manifest: dict, model: str, smoke: bool, dry_run: bool
) -> list[str]:
    """Submit one 14-job chain for a single model. Returns job ids."""
    res = resource_profile(manifest, model)
    project = res["project"]
    queue = res["queue"]

    strategies = manifest["strategies"]
    weights = manifest["weights"]

    prefix = f"CeD_iv_{model}" + ("_smoke" if smoke else "")
    smoke_flag = " --smoke" if smoke else ""
    common_args = (
        f"--model {model} --data-path {DATAFILE}{smoke_flag}"
    )

    # Wall times shrink in smoke mode
    feat_wall = "00:30" if smoke else res["feat_wall"]
    cv_wall = "00:30" if smoke else res["cv_wall"]
    agg_wall = "00:15" if smoke else res["agg_wall"]

    job_ids: list[str] = []

    print(f"=== Submitting {model} ===")
    print(f"  Cores: feat={res['feat_cores']} cv={res['cv_cores']} agg={res['agg_cores']}")
    print(f"  Mem:   feat={res['feat_mem']} cv={res['cv_mem']} agg={res['agg_mem']}")
    print(f"  Combos: {len(strategies)} x {len(weights)} = {len(strategies)*len(weights)} jobs")

    # --- Feature selection ---
    feat_job = submit(
        build_bsub(
            project=project,
            queue=queue,
            wall=feat_wall,
            cores=res["feat_cores"],
            mem=res["feat_mem"],
            job_name=f"{prefix}_feat",
            stdout_path=LOGDIR / f"{prefix}_feat_%J.stdout",
            stderr_path=LOGDIR / f"{prefix}_feat_%J.stderr",
            depend_on=None,
            command=f"{ACTIVATE} && python {SCRIPT_PATH} --phase features {common_args}",
        ),
        dry_run,
    )
    job_ids.append(feat_job)
    print(f"  Feature selection: job {feat_job}")

    # --- CV combos (parallel, each depends on feat job) ---
    cv_jobs: list[str] = []
    for strat in strategies:
        for wt in weights:
            job_name = f"{prefix}_{strat}_{wt}"
            cv_job = submit(
                build_bsub(
                    project=project,
                    queue=queue,
                    wall=cv_wall,
                    cores=res["cv_cores"],
                    mem=res["cv_mem"],
                    job_name=job_name,
                    stdout_path=LOGDIR / f"{job_name}_%J.stdout",
                    stderr_path=LOGDIR / f"{job_name}_%J.stderr",
                    depend_on=f"done({feat_job})" if feat_job != "DRYRUN" else None,
                    command=(
                        f"{ACTIVATE} && python {SCRIPT_PATH} "
                        f"--phase cv --strategy {strat} --weight-scheme {wt} {common_args}"
                    ),
                ),
                dry_run,
            )
            cv_jobs.append(cv_job)
            job_ids.append(cv_job)
            print(f"  {strat} + {wt}: job {cv_job}")

    # --- Aggregation (depends on all CV jobs) ---
    depend_expr = (
        " && ".join(f"done({j})" for j in cv_jobs)
        if all(j != "DRYRUN" for j in cv_jobs)
        else None
    )
    agg_job = submit(
        build_bsub(
            project=project,
            queue=queue,
            wall=agg_wall,
            cores=res["agg_cores"],
            mem=res["agg_mem"],
            job_name=f"{prefix}_agg",
            stdout_path=LOGDIR / f"{prefix}_agg_%J.stdout",
            stderr_path=LOGDIR / f"{prefix}_agg_%J.stderr",
            depend_on=depend_expr,
            command=f"{ACTIVATE} && python {SCRIPT_PATH} --phase aggregate {common_args}",
        ),
        dry_run,
    )
    job_ids.append(agg_job)
    print(f"  Aggregation: job {agg_job}")
    print(f"  Kill all {model}: bkill {' '.join(job_ids)}")
    print()
    return job_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of models (default: all from manifest)",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke run")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print bsub commands without executing them",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Path to manifest yaml (default: {MANIFEST_PATH})",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    all_models = manifest["models"]
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in requested if m not in all_models]
        if missing:
            print(f"ERROR: unknown model(s) {missing}. Manifest has: {all_models}", file=sys.stderr)
            return 2
        models = requested
    else:
        models = all_models

    # Pre-create log dir (run_lr.py creates per-model results dirs itself)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {args.manifest}")
    print(f"Models:   {models}")
    print(f"Smoke:    {args.smoke}")
    print(f"Dry run:  {args.dry_run}")
    print()

    all_ids: list[str] = []
    for model in models:
        all_ids.extend(submit_model_chain(manifest, model, args.smoke, args.dry_run))

    print("=" * 60)
    print(f"Total jobs submitted: {len(all_ids)}")
    print("Monitor: bjobs -w | grep CeD_iv_")
    return 0


if __name__ == "__main__":
    sys.exit(main())
