#!/usr/bin/env python3
"""Compile all cel-risk results into a single analysis-ready CSV.

Scans every run_* directory under results/, extracts per-seed metrics,
aggregated metrics, config metadata, and timing. Outputs one row per
(run_id, model, seed) for per-seed data, and one row per (run_id, model)
for aggregated data.

Works for any cel-risk result directory layout -- sweep runs, phase2/phase3
holdout runs, or arbitrary run_ids.

Usage:
    python compile_results.py                           # default: ../results → compiled_results.csv
    python compile_results.py --results-dir ../results  # explicit
    python compile_results.py --output metrics.csv      # custom output
    python compile_results.py --run-filter sweep_rra    # only matching runs
    python compile_results.py --aggregated-only         # skip per-seed rows
    python compile_results.py --per-seed-only            # skip aggregated rows

Outputs:
    compiled_results_per_seed.csv   -- one row per (run, model, seed)
    compiled_results_aggregated.csv -- one row per (run, model)
    compiled_results_ensemble.csv   -- one row per run (ensemble metrics)
    compile_manifest.json           -- compilation metadata (timestamp, counts, errors)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

MODELS = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]


def discover_runs(results_dir: Path, run_filter: str | None = None) -> list[Path]:
    """Find all run_* directories, optionally filtered."""
    runs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if run_filter:
        runs = [r for r in runs if run_filter in r.name]
    return runs


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Load JSON file, return None on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def load_csv_row(path: Path) -> dict | None:
    """Load first row of a CSV as dict, return None on error."""
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            return next(reader)
    except (FileNotFoundError, StopIteration, csv.Error, PermissionError):
        return None


def load_csv_all(path: Path) -> list[dict]:
    """Load all rows of a CSV as list of dicts."""
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, csv.Error, PermissionError):
        return []


def parse_run_id(run_dir: Path) -> dict[str, str]:
    """Extract structured metadata from run_id naming conventions.

    Handles:
        run_sweep_rra_5p        → order=rra, panel_size=5, experiment=sweep
        run_phase3_holdout      → experiment=phase3_holdout
        run_phase2_val_4protein → experiment=phase2_val, panel_tag=4protein
        run_20260217_194153     → experiment=dated, timestamp=20260217_194153
    """
    name = run_dir.name.removeprefix("run_")
    meta: dict[str, str] = {"run_id": run_dir.name}

    # Sweep runs: sweep_{order}_{size}p
    if name.startswith("sweep_"):
        parts = name.split("_")
        # sweep_rra_5p, sweep_importance_10p, sweep_pathway_25p
        meta["experiment"] = "sweep"
        meta["order"] = parts[1]
        for p in parts[2:]:
            if p.endswith("p") and p[:-1].isdigit():
                meta["panel_size"] = p[:-1]
                break
    elif name.startswith("phase"):
        meta["experiment"] = name
    elif name[0].isdigit():
        meta["experiment"] = "dated"
        meta["run_timestamp"] = name
    else:
        meta["experiment"] = name

    return meta


def _load_yaml_permissive(path: Path) -> dict | None:
    """Load YAML that may contain Python-specific tags (!!python/tuple etc)."""
    try:
        import yaml

        class _PermissiveLoader(yaml.SafeLoader):
            pass

        # Handle !!python/tuple and other custom tags by converting to lists/strings
        def _tuple_constructor(loader, node):
            return list(loader.construct_sequence(node))

        def _generic_constructor(loader, tag_suffix, node):
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            return loader.construct_scalar(node)

        _PermissiveLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)
        _PermissiveLoader.add_multi_constructor("tag:yaml.org,2002:python/", _generic_constructor)

        with open(path) as f:
            return yaml.load(f, Loader=_PermissiveLoader)
    except Exception:
        return None


def extract_panel_proteins(run_dir: Path, model: str, seed: int) -> list[str] | None:
    """Extract the protein features used from the fixed panel CSV or OOF importance.

    Priority: fixed_panel_csv (ground truth for what was fed to the model)
    > OOF importance (what the model found useful -- may drop zero-weight features).
    """
    # Try training_config.yaml for fixed_panel_csv path
    config_path = run_dir / model / "splits" / f"split_seed{seed}" / "training_config.yaml"
    try:
        cfg = _load_yaml_permissive(config_path)
        panel_csv = (cfg or {}).get("features", {}).get("fixed_panel_csv")
        if panel_csv:
            project_root = run_dir.parent.parent
            analysis_dir = project_root / "analysis"
            # Search multiple locations (matches runtime resolution quirks)
            candidates = [
                project_root / "data" / panel_csv,          # data/ (feature_stage.py default)
                analysis_dir / "configs" / panel_csv,       # analysis/configs/
                analysis_dir / panel_csv,                   # analysis/ relative
                project_root / panel_csv,                   # project root
                Path(panel_csv),                            # absolute path
            ]
            for panel_path in candidates:
                if panel_path.exists():
                    proteins = [
                        line.strip() for line in panel_path.read_text().splitlines()
                        if line.strip() and not line.startswith("#")
                    ]
                    if proteins:
                        return proteins
                    break
    except Exception:
        pass

    # Fallback: OOF importance (reflects what the model actually used after any
    # internal filtering like screening, kbest, stability thresholds)
    imp_file = run_dir / model / "splits" / f"split_seed{seed}" / "cv" / f"oof_importance__{model}.csv"
    imp_rows = load_csv_all(imp_file)
    if imp_rows:
        imp_proteins = [r["feature"] for r in imp_rows if r.get("feature", "").endswith("_resid")]
        if imp_proteins:
            return imp_proteins
    return None


def extract_config_metadata(run_dir: Path, model: str, seed: int) -> dict:
    """Extract training config metadata for a specific (model, seed) pair."""
    config_meta = load_json(
        run_dir / model / "splits" / f"split_seed{seed}" / "config_metadata.json"
    )
    if not config_meta:
        return {}

    # Select fields relevant to reproducibility and analysis
    keys = [
        "feature_selection_strategy", "folds", "repeats", "inner_folds",
        "scoring", "screen_method", "screen_top_n", "n_proteins",
        "n_train", "n_val", "n_test", "scenario", "pipeline_version",
        "calibrate_final_models", "threshold_source", "target_prevalence",
        "train_prevalence", "timestamp",
    ]
    return {f"config_{k}": config_meta.get(k) for k in keys if k in config_meta}


def extract_run_settings(run_dir: Path, model: str, seed: int) -> dict:
    """Extract run_settings.json for timing and environment."""
    settings = load_json(
        run_dir / model / "splits" / f"split_seed{seed}" / "core" / "run_settings.json"
    )
    if not settings:
        return {}
    out = {}
    if "cv_elapsed_sec" in settings:
        out["cv_elapsed_sec"] = settings["cv_elapsed_sec"]
    if "columns" in settings:
        out["n_proteins_available"] = settings["columns"].get("n_proteins")
    return out


def extract_training_config_yaml(run_dir: Path, model: str, seed: int) -> dict:
    """Extract key fields from the training_config.yaml saved with results."""
    config_path = run_dir / model / "splits" / f"split_seed{seed}" / "training_config.yaml"
    try:
        cfg = _load_yaml_permissive(config_path)
        if not cfg:
            return {}
        out = {}
        # Fixed panel CSV path (tells us which panel was used)
        features = cfg.get("features", {})
        if features.get("fixed_panel_csv"):
            out["fixed_panel_csv"] = features["fixed_panel_csv"]
        # Optuna trials
        optuna = cfg.get("optuna", {})
        if optuna.get("n_trials"):
            out["optuna_n_trials"] = optuna["n_trials"]
        return out
    except Exception:
        return {}


def extract_cv_metrics(run_dir: Path, model: str, seed: int) -> dict:
    """Extract cross-validation OOF metrics."""
    cv_file = run_dir / model / "splits" / f"split_seed{seed}" / "cv" / "cv_repeat_metrics.csv"
    rows = load_csv_all(cv_file)
    if not rows:
        return {}

    # Average across repeats
    aurocs = [float(r["AUROC_oof"]) for r in rows if "AUROC_oof" in r]
    praucs = [float(r["PR_AUC_oof"]) for r in rows if "PR_AUC_oof" in r]
    briers = [float(r["Brier_oof"]) for r in rows if "Brier_oof" in r]

    out = {}
    if aurocs:
        out["cv_auroc_mean"] = sum(aurocs) / len(aurocs)
        out["cv_auroc_std"] = (sum((x - out["cv_auroc_mean"])**2 for x in aurocs) / len(aurocs)) ** 0.5
    if praucs:
        out["cv_prauc_mean"] = sum(praucs) / len(praucs)
    if briers:
        out["cv_brier_mean"] = sum(briers) / len(briers)
    return out


# ---------------------------------------------------------------------------
# Per-seed compilation
# ---------------------------------------------------------------------------

def compile_per_seed(run_dir: Path, errors: list[str]) -> list[dict]:
    """Compile one row per (model, seed) from a single run directory."""
    run_meta = parse_run_id(run_dir)
    run_metadata_json = load_json(run_dir / "run_metadata.json") or {}
    rows = []

    for model in MODELS:
        splits_dir = run_dir / model / "splits"
        if not splits_dir.exists():
            continue

        for seed_dir in sorted(splits_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("split_seed"):
                continue

            seed_str = seed_dir.name.removeprefix("split_seed")
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            row: dict[str, Any] = {}
            row.update(run_meta)
            row["model"] = model
            row["seed"] = seed

            # Test metrics (primary)
            test_metrics = load_csv_row(seed_dir / "core" / "test_metrics.csv")
            if test_metrics:
                for k, v in test_metrics.items():
                    if k not in ("scenario", "model"):
                        row[f"test_{k}"] = _safe_float(v)
            else:
                errors.append(f"{run_dir.name}/{model}/seed{seed}: missing test_metrics.csv")

            # Val metrics
            val_metrics = load_csv_row(seed_dir / "core" / "val_metrics.csv")
            if val_metrics:
                for k, v in val_metrics.items():
                    if k not in ("scenario", "model"):
                        row[f"val_{k}"] = _safe_float(v)

            # CV OOF metrics
            row.update(extract_cv_metrics(run_dir, model, seed))

            # Config metadata
            row.update(extract_config_metadata(run_dir, model, seed))

            # Run settings (timing)
            row.update(extract_run_settings(run_dir, model, seed))

            # Training config YAML
            row.update(extract_training_config_yaml(run_dir, model, seed))

            # Panel proteins (both sources for transparency)
            proteins = extract_panel_proteins(run_dir, model, seed)
            if proteins:
                row["n_panel_proteins"] = len(proteins)
                row["panel_proteins"] = ";".join(proteins)

            # Also record OOF importance protein count (what model actually used)
            imp_file = run_dir / model / "splits" / f"split_seed{seed}" / "cv" / f"oof_importance__{model}.csv"
            imp_rows = load_csv_all(imp_file)
            if imp_rows:
                imp_proteins = [r["feature"] for r in imp_rows if r.get("feature", "").endswith("_resid")]
                row["n_active_proteins"] = len(imp_proteins)

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Aggregated compilation
# ---------------------------------------------------------------------------

def compile_aggregated(run_dir: Path, errors: list[str]) -> list[dict]:
    """Compile one row per model from aggregated results."""
    run_meta = parse_run_id(run_dir)
    rows = []

    for model in MODELS:
        agg_dir = run_dir / model / "aggregated"
        if not agg_dir.exists():
            continue

        row: dict[str, Any] = {}
        row.update(run_meta)
        row["model"] = model

        # Pooled test metrics
        pooled_test = load_csv_row(agg_dir / "metrics" / "pooled_test_metrics.csv")
        if pooled_test:
            for k, v in pooled_test.items():
                if k != "model":
                    row[f"pooled_test_{k}"] = _safe_float(v)
        else:
            errors.append(f"{run_dir.name}/{model}: missing pooled_test_metrics.csv")

        # Pooled val metrics
        pooled_val = load_csv_row(agg_dir / "metrics" / "pooled_val_metrics.csv")
        if pooled_val:
            for k, v in pooled_val.items():
                if k != "model":
                    row[f"pooled_val_{k}"] = _safe_float(v)

        # Summary stats (mean, std, CI across seeds)
        summary = load_csv_row(agg_dir / "metrics" / "test_metrics_summary.csv")
        if summary:
            for k, v in summary.items():
                if k not in ("scenario", "model", "n_splits"):
                    row[f"summary_{k}"] = _safe_float(v)
            if "n_splits" in summary:
                row["n_splits"] = int(summary["n_splits"])

        # Aggregation metadata
        agg_meta = load_json(agg_dir / "aggregation_metadata.json")
        if agg_meta:
            row["agg_timestamp"] = agg_meta.get("timestamp")
            row["agg_n_splits"] = agg_meta.get("n_splits")
            row["agg_n_boot"] = agg_meta.get("n_boot")
            seeds = agg_meta.get("split_seeds", [])
            row["agg_seeds"] = ";".join(str(s) for s in seeds)

        # Panel info (from first available seed)
        splits_dir = run_dir / model / "splits"
        if splits_dir.exists():
            for seed_dir in sorted(splits_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith("split_seed"):
                    seed = int(seed_dir.name.removeprefix("split_seed"))
                    proteins = extract_panel_proteins(run_dir, model, seed)
                    if proteins:
                        row["n_panel_proteins"] = len(proteins)
                        row["panel_proteins"] = ";".join(proteins)
                    # Timing from first seed
                    settings = extract_run_settings(run_dir, model, seed)
                    row.update(settings)
                    # Config
                    cfg = extract_config_metadata(run_dir, model, seed)
                    row.update(cfg)
                    yaml_cfg = extract_training_config_yaml(run_dir, model, seed)
                    row.update(yaml_cfg)
                    break

        rows.append(row)

    return rows


def compile_ensemble(run_dir: Path, errors: list[str]) -> dict | None:
    """Compile ensemble metrics for a single run."""
    ens_dir = run_dir / "ENSEMBLE" / "aggregated"
    if not ens_dir.exists():
        return None

    run_meta = parse_run_id(run_dir)
    row: dict[str, Any] = {}
    row.update(run_meta)
    row["model"] = "ENSEMBLE"

    pooled_test = load_csv_row(ens_dir / "metrics" / "pooled_test_metrics.csv")
    if pooled_test:
        for k, v in pooled_test.items():
            if k != "model":
                row[f"pooled_test_{k}"] = _safe_float(v)
    else:
        return None

    pooled_val = load_csv_row(ens_dir / "metrics" / "pooled_val_metrics.csv")
    if pooled_val:
        for k, v in pooled_val.items():
            if k != "model":
                row[f"pooled_val_{k}"] = _safe_float(v)

    return row


# ---------------------------------------------------------------------------
# Orchestrator / log metadata
# ---------------------------------------------------------------------------

def extract_log_metadata(run_dir: Path) -> dict:
    """Extract timing and job metadata from orchestrator logs."""
    run_name = run_dir.name  # e.g., run_sweep_rra_5p
    project_root = run_dir.parent.parent
    log_dir = project_root / "logs" / run_name

    meta: dict[str, Any] = {}

    # Orchestrator log -- extract start/end time
    orch_log = log_dir / "orchestrator.log"
    if orch_log.exists():
        try:
            lines = orch_log.read_text().strip().splitlines()
            # Find first and last timestamp lines
            timestamps = []
            for line in lines:
                if line.startswith("[") and "]" in line:
                    ts_str = line[1:line.index("]")]
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                        timestamps.append(ts)
                    except ValueError:
                        pass
            if len(timestamps) >= 2:
                meta["orchestrator_start"] = timestamps[0].isoformat()
                meta["orchestrator_end"] = timestamps[-1].isoformat()
                meta["orchestrator_duration_sec"] = (timestamps[-1] - timestamps[0]).total_seconds()

            # Check for completion
            if lines:
                meta["orchestrator_complete"] = "Orchestrator complete" in lines[-1]
        except (PermissionError, OSError):
            pass

    # Submission log -- extract job counts
    sub_log = log_dir / "submission.log"
    if sub_log.exists():
        try:
            text = sub_log.read_text()
            for line in text.splitlines():
                if "Training jobs:" in line:
                    # "Training jobs:    40 (4 models x 10 seeds)"
                    parts = line.split("Training jobs:")
                    if len(parts) > 1:
                        n = parts[1].strip().split()[0]
                        meta["n_training_jobs"] = int(n)
                if "HPC:" in line:
                    # "HPC: acc_vascbrain / premium / 24:00 / 12c / 8000MB"
                    meta["hpc_config"] = line.split("HPC:")[1].strip()
        except (PermissionError, OSError):
            pass

    return meta


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> Any:
    """Convert to float if possible, otherwise return as-is."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def write_csv(rows: list[dict], output_path: Path) -> int:
    """Write rows to CSV, auto-detecting all columns. Returns row count."""
    if not rows:
        return 0

    # Collect all keys in order of first appearance
    seen: dict[str, None] = {}
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = None
    fieldnames = list(seen.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile all cel-risk results into analysis-ready CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent.parent.parent / "results",
        help="Path to results/ directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for CSVs (default: results_dir)",
    )
    parser.add_argument(
        "--run-filter", type=str, default=None,
        help="Only include runs matching this substring (e.g., 'sweep_rra')",
    )
    parser.add_argument(
        "--aggregated-only", action="store_true",
        help="Only compile aggregated (cross-seed) metrics",
    )
    parser.add_argument(
        "--per-seed-only", action="store_true",
        help="Only compile per-seed metrics",
    )
    parser.add_argument(
        "--include-logs", action="store_true", default=True,
        help="Include orchestrator/submission log metadata (default: True)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    runs = discover_runs(results_dir, args.run_filter)
    print(f"Found {len(runs)} run directories in {results_dir}")

    errors: list[str] = []
    per_seed_rows: list[dict] = []
    aggregated_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    for i, run_dir in enumerate(runs):
        run_name = run_dir.name
        print(f"  [{i+1}/{len(runs)}] {run_name}...", end="", flush=True)

        # Log metadata (shared across per-seed and aggregated)
        log_meta = extract_log_metadata(run_dir) if args.include_logs else {}

        # Per-seed
        if not args.aggregated_only:
            seed_rows = compile_per_seed(run_dir, errors)
            for row in seed_rows:
                row.update(log_meta)
            per_seed_rows.extend(seed_rows)

        # Aggregated
        if not args.per_seed_only:
            agg_rows = compile_aggregated(run_dir, errors)
            for row in agg_rows:
                row.update(log_meta)
            aggregated_rows.extend(agg_rows)

        # Ensemble
        ens_row = compile_ensemble(run_dir, errors)
        if ens_row:
            ens_row.update(log_meta)
            ensemble_rows.append(ens_row)

        n_seeds = len([r for r in (per_seed_rows if not args.aggregated_only else []) if r.get("run_id") == run_name])
        n_models = len([r for r in (aggregated_rows if not args.per_seed_only else []) if r.get("run_id") == run_name])
        print(f" {n_seeds or n_models} rows")

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    n_per_seed = 0
    n_agg = 0
    n_ens = 0

    if per_seed_rows:
        out = output_dir / "compiled_results_per_seed.csv"
        n_per_seed = write_csv(per_seed_rows, out)
        print(f"\nWrote {n_per_seed} rows → {out}")

    if aggregated_rows:
        out = output_dir / "compiled_results_aggregated.csv"
        n_agg = write_csv(aggregated_rows, out)
        print(f"Wrote {n_agg} rows → {out}")

    if ensemble_rows:
        out = output_dir / "compiled_results_ensemble.csv"
        n_ens = write_csv(ensemble_rows, out)
        print(f"Wrote {n_ens} rows → {out}")

    # Write manifest
    manifest = {
        "compiled_at": datetime.now().isoformat(),
        "results_dir": str(results_dir),
        "run_filter": args.run_filter,
        "n_runs": len(runs),
        "n_per_seed_rows": n_per_seed,
        "n_aggregated_rows": n_agg,
        "n_ensemble_rows": n_ens,
        "n_errors": len(errors),
        "errors": errors[:50],  # cap at 50 to avoid huge manifests
        "runs_compiled": [r.name for r in runs],
    }
    manifest_path = output_dir / "compile_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest → {manifest_path}")

    if errors:
        print(f"\n{len(errors)} warnings:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more (see manifest)")

    print(f"\nDone. {n_per_seed + n_agg + n_ens} total rows across 3 files.")


if __name__ == "__main__":
    main()
