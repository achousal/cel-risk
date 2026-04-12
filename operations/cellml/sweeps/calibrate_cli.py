"""Standalone calibration CLI.

Loads a sweep spec, runs pre-sweep calibration against the real dataset
referenced by `spec.data_path`, prints the result, and exits. Does NOT
touch the ledger, does NOT submit to Minerva, does NOT mutate the spec
file. Intended for smoke-testing calibration against real data before
trusting it in a production sweep.

Usage
-----
    python -m operations.cellml.sweeps.calibrate_cli \
        --spec operations/cellml/sweeps/specs/09_downsampling_ratio.yaml

    # Force a fresh run even if a cached result exists
    python -m operations.cellml.sweeps.calibrate_cli --spec <path> --force

    # Override the project root (default: auto-detected from spec path)
    python -m operations.cellml.sweeps.calibrate_cli --spec <path> --project-root /.../cel-risk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .sweep_calibration import CalibrationError
from .sweep_orchestrator import SweepOrchestrator, load_sweep_spec

logger = logging.getLogger("calibrate_cli")


def _auto_project_root(spec_path: Path) -> Path:
    """Walk up from the spec path until we find a cel-risk project root.

    Heuristic: the directory containing both `operations/` and `data/`.
    """
    for parent in spec_path.resolve().parents:
        if (parent / "operations").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError(
        f"Could not auto-detect project root from {spec_path}. "
        f"Pass --project-root explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pre-sweep calibration against real data (smoke test).",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Path to sweep spec YAML")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="cel-risk project root (auto-detected if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore calibration cache and rerun from scratch",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.spec.exists():
        print(f"ERROR: spec not found: {args.spec}", file=sys.stderr)
        return 2

    project_root = args.project_root or _auto_project_root(args.spec)
    logger.info("Project root: %s", project_root)

    spec = load_sweep_spec(args.spec)
    logger.info("Loaded spec: %s (dim=%d)", spec.id, len(spec.parameter_space))

    # Use a temp ledger dir so we never touch the real one.
    tmp_ledger = project_root / ".calibrate_cli_tmp" / spec.id / "ledger"
    tmp_ledger.mkdir(parents=True, exist_ok=True)

    orch = SweepOrchestrator(
        spec=spec,
        project_root=project_root,
        ledger_dir=tmp_ledger,
    )

    try:
        result = orch.ensure_calibration(force_rerun=args.force)
    except CalibrationError as exc:
        print(f"\nCALIBRATION FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}", file=sys.stderr)
        raise

    print("\n" + "=" * 60)
    print(f"Calibration complete: {result.calibration_id}")
    print("=" * 60)
    print(f"  sweep_id              : {result.sweep_id}")
    print(f"  trials executed       : {result.n_calib_executed} / {result.n_calib_requested}")
    print(f"  wall seconds          : {result.wall_seconds_total:.1f}")
    print(f"  subsample rows        : {result.subsample_rows_used}")
    print(f"  plateau trial         : {result.plateau_trial}")
    print(f"  plateau value         : {result.plateau_value}")
    print(f"  noise sigma           : {result.noise_sigma}")
    print(f"  noise warning         : {result.noise_warning}")
    if result.proposed is not None:
        p = result.proposed
        print(f"\n  PROPOSED (not promoted):")
        print(f"    n_trials_cap        : {p.n_trials_cap}")
        print(f"    patience            : {p.patience}")
        print(f"    confidence          : {p.confidence.value}")
        print(f"    warm_start_points   : {len(p.warm_start_points)}")
        for i, point in enumerate(p.warm_start_points):
            print(f"      [{i}] obj={point.objective:.4f} params={json.dumps(point.params)}")
    print("\nArtifacts:")
    print(f"  {orch.calibration_dir / (result.calibration_id + '.json')}")
    print(f"  {orch.calibration_dir / (result.calibration_id + '.parquet')}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
