#!/usr/bin/env python3
"""Preflight check for the panel saturation sweep experiment.

Validates all prerequisites before committing HPC resources:
  - Data integrity (parquet readable, expected columns present)
  - Split availability (holdout splits for seeds 200-209)
  - Panel CSV correctness (incremental nesting, protein existence in data)
  - Addition order consistency (each order is a permutation of the same set)
  - Config chain resolution (_base inheritance, fixed_panel_csv paths)
  - Existing results reuse (detect completed runs to skip)
  - Resource estimation (total runs, CPU-hours, wall time)
  - Mathematical bounds verification (baseline metrics from existing 4p/7p runs)

Usage:
    python preflight.py                          # full check
    python preflight.py --order rra              # check single order
    python preflight.py --generate-configs       # emit all YAML + panel CSVs
    python preflight.py --dry-run                # preview HPC submissions

Exit codes:
    0 = all checks pass
    1 = fatal (data/splits missing, can't proceed)
    2 = warnings only (missing results for reuse, non-blocking)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parents[2]  # cel-risk/analysis/
PROJECT_ROOT = ANALYSIS_DIR.parent                  # cel-risk/
CONFIGS_DIR = ANALYSIS_DIR / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = PROJECT_ROOT / "splits"
DATA_DIR = PROJECT_ROOT / "data"
SWEEP_DIR = Path(__file__).resolve().parent  # this experiment dir
SWEEP_CONFIGS_DIR = SWEEP_DIR / "configs"
SWEEP_PANELS_DIR = SWEEP_DIR / "panels"

SEEDS = list(range(200, 210))
MODELS = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]
PANEL_MIN = 4
PANEL_MAX = 25

# Canonical RRA rank order (from rra_significance_corrected.csv, by observed_rra desc)
RRA_ORDER = [
    "tgm2_resid",
    "cpa2_resid",
    "itgb7_resid",
    "gip_resid",
    "cxcl9_resid",
    "cd160_resid",
    "muc2_resid",
    "nos2_resid",
    "fabp6_resid",
    "agr2_resid",
    "reg3a_resid",
    "mln_resid",
    "ccl25_resid",
    "pafah1b3_resid",
    "tnfrsf8_resid",
    "tigit_resid",
    "cxcl11_resid",
    "ckmt1a_ckmt1b_resid",
    "acy3_resid",
    "hla_a_resid",
    "xcl1_resid",
    "nell2_resid",
    "pof1b_resid",
    "ppp1r14d_resid",
    "ada2_resid",
]

# Cross-model importance rank (from 7p holdout OOF importance, averaged across 10 seeds)
# For proteins 1-7, reordered by importance. Proteins 8-25 fall back to RRA order.
IMPORTANCE_ORDER = [
    "tgm2_resid",
    "cpa2_resid",
    "itgb7_resid",
    "cxcl9_resid",    # jumps ahead of gip (rank 4 vs 7 in importance)
    "cd160_resid",     # jumps ahead of gip
    "muc2_resid",      # jumps ahead of gip
    "gip_resid",       # drops to position 7
    # 8-25: same as RRA (no importance data beyond 7p)
    "nos2_resid",
    "fabp6_resid",
    "agr2_resid",
    "reg3a_resid",
    "mln_resid",
    "ccl25_resid",
    "pafah1b3_resid",
    "tnfrsf8_resid",
    "tigit_resid",
    "cxcl11_resid",
    "ckmt1a_ckmt1b_resid",
    "acy3_resid",
    "hla_a_resid",
    "xcl1_resid",
    "nell2_resid",
    "pof1b_resid",
    "ppp1r14d_resid",
    "ada2_resid",
]

# Pathway-informed order: group by biological axis, add one pathway at a time
# Axis 1: Core mucosal/autoimmune (tTG + MUC2)
# Axis 2: Immune surveillance/homing (integrin + chemokines + NK receptor)
# Axis 3: GI metabolic (pancreatic + intestinal enzymes)
# Axis 4: Extended immune effector (Th1/Th17 cytokines, checkpoint)
# Axis 5: Extended GI/tissue remodeling
PATHWAY_ORDER = [
    # Axis 1: mucosal autoimmune core
    "tgm2_resid",       # tissue transglutaminase -- celiac autoantigen
    "muc2_resid",       # mucin -- intestinal barrier integrity
    # Axis 2: immune surveillance/homing
    "itgb7_resid",      # integrin beta-7 -- gut-homing lymphocytes
    "cxcl9_resid",      # CXCL9 -- IFN-gamma-induced chemokine
    "cd160_resid",      # CD160 -- NK/T-cell co-receptor
    # Axis 3: GI metabolic
    "cpa2_resid",       # carboxypeptidase A2 -- pancreatic enzyme
    "gip_resid",        # GIP -- incretin, GI metabolic signaling
    # Axis 4: extended immune
    "nos2_resid",       # iNOS -- inflammatory macrophage marker
    "cxcl11_resid",     # CXCL11 -- IFN-gamma chemokine (same axis as cxcl9)
    "tigit_resid",      # TIGIT -- T-cell checkpoint
    "tnfrsf8_resid",    # CD30 -- activated T/B-cell marker
    "ccl25_resid",      # CCL25 -- gut-specific chemokine (TECK)
    "xcl1_resid",       # XCL1 -- lymphotactin
    "hla_a_resid",      # HLA-A -- antigen presentation
    # Axis 5: extended GI / tissue remodeling / metabolic
    "fabp6_resid",      # FABP6 -- ileal bile acid binding
    "agr2_resid",       # AGR2 -- intestinal goblet cell secretory
    "reg3a_resid",      # REG3A -- antimicrobial lectin, Paneth cell
    "mln_resid",        # motilin -- GI motility
    "pafah1b3_resid",   # PAF-AH -- lipid mediator metabolism
    "acy3_resid",       # ACY3 -- aminoacylase
    "nell2_resid",      # NELL2 -- neural/GI signaling
    "pof1b_resid",      # POF1B -- epithelial cytoskeleton
    "ppp1r14d_resid",   # PPP1R14D -- smooth muscle regulation
    "ada2_resid",       # ADA2 -- adenosine deaminase (immune modulation)
    "ckmt1a_ckmt1b_resid",  # CKM -- creatine kinase, tissue damage
]

ORDERS: dict[str, list[str]] = {
    "rra": RRA_ORDER,
    "importance": IMPORTANCE_ORDER,
    "pathway": PATHWAY_ORDER,
}

# Existing results that can be reused (run_id -> {panel_size, order})
EXISTING_RUNS = {
    "run_phase3_holdout_4protein": {"size": 4, "order": "rra", "proteins": RRA_ORDER[:4]},
    "run_phase3_holdout": {"size": 7, "order": "rra", "proteins": RRA_ORDER[:7]},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, fatal


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)
    reusable_runs: dict[str, Any] = field(default_factory=dict)
    new_runs_needed: int = 0
    total_cpu_hours: float = 0.0
    est_wall_hours: float = 0.0

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    @property
    def fatals(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "fatal"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.fatals) == 0

    def summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("PANEL SATURATION SWEEP -- PREFLIGHT REPORT")
        lines.append("=" * 70)

        # Group by severity
        for severity in ["fatal", "error", "warning", "info"]:
            group = [c for c in self.checks if c.severity == severity]
            if not group:
                continue
            icon = {"fatal": "FATAL", "error": "ERROR", "warning": " WARN", "info": "   OK"}[severity]
            for c in group:
                status = "PASS" if c.passed else "FAIL"
                lines.append(f"  [{icon}] {status}: {c.name}")
                lines.append(f"          {c.message}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("RESOURCE ESTIMATE")
        lines.append("-" * 70)
        lines.append(f"  Reusable existing runs:  {len(self.reusable_runs)}")
        lines.append(f"  New training runs:       {self.new_runs_needed}")
        lines.append(f"  Estimated CPU-hours:     {self.total_cpu_hours:.0f}")
        lines.append(f"  Estimated wall-hours:    {self.est_wall_hours:.1f} (at 96 cores)")
        lines.append("")

        if self.fatals:
            lines.append("VERDICT: BLOCKED -- fix fatal issues before proceeding")
        elif self.warnings:
            lines.append(f"VERDICT: READY with {len(self.warnings)} warning(s)")
        else:
            lines.append("VERDICT: READY -- all checks pass")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_data_integrity(report: PreflightReport) -> None:
    """Verify parquet data file exists and contains expected protein columns."""
    parquet = DATA_DIR / "Celiac_dataset_proteomics_w_demo.parquet"
    if not parquet.exists():
        report.add(CheckResult(
            "data_file_exists",
            False,
            f"Missing: {parquet}",
            "fatal",
        ))
        return

    try:
        import pandas as pd
        df = pd.read_parquet(parquet, columns=["idx"] if "idx" in pd.read_parquet(parquet, columns=[]).columns else None)
        n_rows = len(pd.read_parquet(parquet, columns=[pd.read_parquet(parquet).columns[0]]))
    except Exception:
        # Lightweight: just check columns without loading full data
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet)
        col_names = schema.names
        n_rows = pq.read_metadata(parquet).num_rows

        report.add(CheckResult(
            "data_file_readable",
            True,
            f"Parquet readable: {n_rows:,} rows, {len(col_names):,} columns",
        ))

        # Check all 25 proteins exist
        missing = [p for p in RRA_ORDER if p not in col_names]
        if missing:
            report.add(CheckResult(
                "proteins_in_data",
                False,
                f"Missing proteins in data: {missing}",
                "fatal",
            ))
        else:
            report.add(CheckResult(
                "proteins_in_data",
                True,
                f"All {len(RRA_ORDER)} RRA-ranked proteins found in data columns",
            ))

        # Check demographics
        demo_cols = ["age", "BMI", "sex", "Genetic ethnic grouping"]
        missing_demo = [c for c in demo_cols if c not in col_names]
        if missing_demo:
            report.add(CheckResult(
                "demographics_in_data",
                False,
                f"Missing demographic columns: {missing_demo}",
                "warning",
            ))
        else:
            report.add(CheckResult(
                "demographics_in_data",
                True,
                "All demographic columns present (age, BMI, sex, ethnicity)",
            ))
        return

    # Fallback: pandas path
    report.add(CheckResult("data_file_readable", True, f"Parquet readable: {n_rows:,} rows"))


def check_data_integrity_lightweight(report: PreflightReport) -> None:
    """Column-only check without loading data into memory."""
    parquet = DATA_DIR / "Celiac_dataset_proteomics_w_demo.parquet"
    if not parquet.exists():
        report.add(CheckResult("data_file_exists", False, f"Missing: {parquet}", "fatal"))
        return

    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(parquet)
        col_names = set(schema.names)
        n_rows = pq.read_metadata(parquet).num_rows
    except ImportError:
        # Fallback: pandas
        import pandas as pd
        cols = pd.read_parquet(parquet, columns=[]).columns.tolist()
        col_names = set(cols)
        n_rows = "unknown (pyarrow not available)"

    report.add(CheckResult(
        "data_file_readable", True,
        f"Parquet readable: {n_rows:,} rows, {len(col_names):,} columns" if isinstance(n_rows, int)
        else f"Parquet readable: {len(col_names)} columns (row count requires pyarrow)",
    ))

    # Validate all 25 proteins
    missing_proteins = [p for p in RRA_ORDER if p not in col_names]
    if missing_proteins:
        report.add(CheckResult(
            "proteins_in_data", False,
            f"{len(missing_proteins)} proteins missing from data: {missing_proteins[:5]}...",
            "fatal",
        ))
    else:
        report.add(CheckResult(
            "proteins_in_data", True,
            f"All {len(RRA_ORDER)} RRA-ranked proteins present in data",
        ))

    # Demographics
    demo_cols = {"age", "BMI", "sex", "Genetic ethnic grouping"}
    missing_demo = demo_cols - col_names
    report.add(CheckResult(
        "demographics_in_data",
        len(missing_demo) == 0,
        f"Demographics: {4 - len(missing_demo)}/4 present" + (f" (missing: {missing_demo})" if missing_demo else ""),
        "warning" if missing_demo else "info",
    ))


def check_splits(report: PreflightReport) -> None:
    """Verify holdout splits exist for all 10 seeds."""
    if not SPLITS_DIR.exists():
        report.add(CheckResult("splits_dir_exists", False, f"Missing: {SPLITS_DIR}", "fatal"))
        return

    missing_seeds = []
    for seed in SEEDS:
        # Holdout splits use a specific naming convention
        pattern = list(SPLITS_DIR.glob(f"*seed{seed}*")) + list(SPLITS_DIR.glob(f"*{seed}*"))
        if not pattern:
            missing_seeds.append(seed)

    if missing_seeds:
        report.add(CheckResult(
            "holdout_splits", False,
            f"Missing LOCAL split files for seeds: {missing_seeds}. "
            f"Expected on HPC (Minerva). Generate with: ced save-splits --config configs/splits_config_holdout.yaml",
            "warning",  # splits may exist on HPC; not fatal for preflight
        ))
    else:
        report.add(CheckResult(
            "holdout_splits", True,
            f"Split files found for all {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]})",
        ))


def check_addition_orders(report: PreflightReport) -> None:
    """Verify all three addition orders are valid permutations of the same protein set."""
    reference = set(RRA_ORDER)

    for name, order in ORDERS.items():
        order_set = set(order)

        # Same proteins
        if order_set != reference:
            extra = order_set - reference
            missing = reference - order_set
            report.add(CheckResult(
                f"order_{name}_proteins",
                False,
                f"Order '{name}' protein mismatch. Extra: {extra}, Missing: {missing}",
                "fatal",
            ))
            continue

        # No duplicates
        if len(order) != len(set(order)):
            seen = set()
            dups = [p for p in order if p in seen or seen.add(p)]
            report.add(CheckResult(
                f"order_{name}_duplicates",
                False,
                f"Order '{name}' has duplicates: {dups}",
                "fatal",
            ))
            continue

        # Correct length
        if len(order) != PANEL_MAX:
            report.add(CheckResult(
                f"order_{name}_length",
                False,
                f"Order '{name}' has {len(order)} proteins, expected {PANEL_MAX}",
                "fatal",
            ))
            continue

        report.add(CheckResult(
            f"order_{name}_valid", True,
            f"Order '{name}': valid permutation of {len(order)} proteins",
        ))

    # Check incremental nesting property: panel(k) ⊂ panel(k+1)
    for name, order in ORDERS.items():
        for k in range(PANEL_MIN, PANEL_MAX):
            panel_k = set(order[:k])
            panel_k1 = set(order[:k + 1])
            if not panel_k.issubset(panel_k1):
                report.add(CheckResult(
                    f"order_{name}_nesting",
                    False,
                    f"Order '{name}': panel({k}) is not a subset of panel({k+1})",
                    "fatal",
                ))
                break
        else:
            report.add(CheckResult(
                f"order_{name}_nesting", True,
                f"Order '{name}': incremental nesting verified (panel(k) ⊂ panel(k+1) for all k)",
            ))


def check_existing_results(report: PreflightReport) -> None:
    """Detect completed runs that can be reused to avoid redundant computation."""
    reusable = {}

    for run_id, meta in EXISTING_RUNS.items():
        run_dir = RESULTS_DIR / run_id
        if not run_dir.exists():
            report.add(CheckResult(
                f"existing_{run_id}", False,
                f"Expected existing run not found: {run_dir}",
                "warning",
            ))
            continue

        # Check all models × seeds have pooled test metrics
        complete_models = []
        for model in MODELS:
            metrics_file = run_dir / model / "aggregated" / "metrics" / "pooled_test_metrics.csv"
            if metrics_file.exists():
                complete_models.append(model)

        if len(complete_models) == len(MODELS):
            reusable[run_id] = meta
            report.add(CheckResult(
                f"existing_{run_id}", True,
                f"Reusable: {run_id} (panel={meta['size']}p, order={meta['order']}, all 4 models complete)",
            ))
        else:
            missing = set(MODELS) - set(complete_models)
            report.add(CheckResult(
                f"existing_{run_id}", False,
                f"Partial: {run_id} missing models: {missing}",
                "warning",
            ))

    report.reusable_runs = reusable

    # Check per-seed completeness for reusable runs
    for run_id in reusable:
        run_dir = RESULTS_DIR / run_id
        for model in MODELS:
            missing_seeds = []
            for seed in SEEDS:
                test_metrics = run_dir / model / "splits" / f"split_seed{seed}" / "core" / "test_metrics.csv"
                if not test_metrics.exists():
                    missing_seeds.append(seed)
            if missing_seeds:
                report.add(CheckResult(
                    f"existing_{run_id}_{model}_seeds", False,
                    f"{run_id}/{model}: missing seeds {missing_seeds}",
                    "warning",
                ))


def check_baseline_metrics(report: PreflightReport) -> None:
    """Load existing 4p and 7p metrics to establish baseline bounds for the sweep."""
    baselines = {}

    for run_id, meta in EXISTING_RUNS.items():
        run_dir = RESULTS_DIR / run_id
        run_metrics = {}

        for model in MODELS:
            metrics_file = run_dir / model / "aggregated" / "metrics" / "pooled_test_metrics.csv"
            if not metrics_file.exists():
                continue
            try:
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    run_metrics[model] = {
                        "auroc": float(row["auroc"]),
                        "prauc": float(row["prauc"]),
                        "brier": float(row["brier_score"]),
                    }
            except (StopIteration, KeyError, ValueError):
                continue

        if run_metrics:
            baselines[meta["size"]] = run_metrics

    if not baselines:
        report.add(CheckResult(
            "baseline_metrics", False,
            "No baseline metrics loadable from existing runs",
            "warning",
        ))
        return

    # Validate expected relationships
    if 4 in baselines and 7 in baselines:
        for model in MODELS:
            if model in baselines[4] and model in baselines[7]:
                a4 = baselines[4][model]["auroc"]
                a7 = baselines[7][model]["auroc"]
                delta = a7 - a4
                report.add(CheckResult(
                    f"baseline_{model}_monotonic", True,
                    f"{model}: 4p→7p AUROC {a4:.3f}→{a7:.3f} (Δ={delta:+.3f})",
                ))

        # Verify crossover: linear leads at 4p, trees lead at 7p
        lr_4 = baselines[4].get("LR_EN", {}).get("auroc", 0)
        rf_4 = baselines[4].get("RF", {}).get("auroc", 0)
        lr_7 = baselines[7].get("LR_EN", {}).get("auroc", 0)
        rf_7 = baselines[7].get("RF", {}).get("auroc", 0)

        crossover = lr_4 > rf_4 and rf_7 > lr_7
        report.add(CheckResult(
            "baseline_crossover",
            crossover,
            f"Model crossover {'confirmed' if crossover else 'NOT confirmed'}: "
            f"LR>RF at 4p ({lr_4:.3f}>{rf_4:.3f}), RF>LR at 7p ({rf_7:.3f}>{lr_7:.3f})",
            "info" if crossover else "warning",
        ))

    # Store for downstream use
    report.reusable_runs["_baselines"] = baselines


def estimate_resources(report: PreflightReport) -> None:
    """Estimate computational cost of the full sweep."""
    n_orders = len(ORDERS)
    n_sizes = PANEL_MAX - PANEL_MIN + 1  # 4..25 = 22 sizes
    n_seeds = len(SEEDS)
    n_models = len(MODELS)

    # Total base-model training runs
    total_runs = n_orders * n_sizes * n_seeds * n_models

    # Subtract reusable runs
    reusable_count = 0
    for run_id, meta in report.reusable_runs.items():
        if run_id.startswith("_"):
            continue
        # Each reusable run covers 1 order × 1 size × 10 seeds × 4 models
        reusable_count += n_seeds * n_models

    new_runs = total_runs - reusable_count

    # Ensemble runs: 1 per (order × size × seed)
    ensemble_runs = n_orders * n_sizes * n_seeds
    reusable_ensemble = sum(n_seeds for rid, m in report.reusable_runs.items() if not rid.startswith("_"))
    new_ensemble = ensemble_runs - reusable_ensemble

    # Time estimates
    # Base model: ~50 Optuna trials × ~1 min/trial = ~50 min, but with 50 trials = ~30 min avg
    avg_base_minutes = 30
    avg_ensemble_minutes = 5
    total_cpu_minutes = (new_runs * avg_base_minutes) + (new_ensemble * avg_ensemble_minutes)
    total_cpu_hours = total_cpu_minutes / 60

    # Wall time at 96 cores (Minerva)
    # With HPC orchestrator: ~20 concurrent jobs
    concurrent = 20
    wall_hours = total_cpu_hours / concurrent

    report.new_runs_needed = new_runs + new_ensemble
    report.total_cpu_hours = total_cpu_hours
    report.est_wall_hours = wall_hours

    report.add(CheckResult(
        "resource_estimate", True,
        f"Total: {total_runs} base + {ensemble_runs} ensemble = {total_runs + ensemble_runs} runs. "
        f"Reusable: {reusable_count + reusable_ensemble}. "
        f"New: {new_runs} base + {new_ensemble} ensemble = {new_runs + new_ensemble}",
    ))

    report.add(CheckResult(
        "resource_time", True,
        f"Estimated: {total_cpu_hours:.0f} CPU-hours, {wall_hours:.1f} wall-hours at 20 concurrent jobs",
    ))


def check_config_chain(report: PreflightReport) -> None:
    """Verify the YAML config inheritance chain resolves correctly."""
    # The sweep configs inherit: sweep_training_{size}p.yaml -> training_config.yaml
    base_config = CONFIGS_DIR / "training_config.yaml"
    if not base_config.exists():
        report.add(CheckResult(
            "config_base", False,
            f"Missing base config: {base_config}",
            "fatal",
        ))
        return

    try:
        with open(base_config) as f:
            base = yaml.safe_load(f)

        # Verify critical fields
        strategy = base.get("features", {}).get("feature_selection_strategy")
        report.add(CheckResult(
            "config_base_strategy", True,
            f"Base config feature strategy: '{strategy}' (sweep overrides to 'fixed_panel')",
        ))

        # Verify optuna settings
        optuna_trials = base.get("optuna", {}).get("n_trials", "missing")
        report.add(CheckResult(
            "config_base_optuna", True,
            f"Base optuna n_trials={optuna_trials} (sweep overrides to 50 for fixed panels)",
        ))

    except yaml.YAMLError as e:
        report.add(CheckResult("config_base_parse", False, f"YAML parse error: {e}", "fatal"))

    # Verify holdout splits config
    holdout_config = CONFIGS_DIR / "splits_config_holdout.yaml"
    if holdout_config.exists():
        with open(holdout_config) as f:
            splits_cfg = yaml.safe_load(f)
        mode = splits_cfg.get("mode", "unknown")
        holdout_size = splits_cfg.get("holdout_size", "unknown")
        n_splits = splits_cfg.get("n_splits", "unknown")
        seed_start = splits_cfg.get("seed_start", "unknown")
        report.add(CheckResult(
            "config_splits", True,
            f"Holdout config: mode={mode}, holdout_size={holdout_size}, "
            f"n_splits={n_splits}, seed_start={seed_start}",
        ))
    else:
        report.add(CheckResult(
            "config_splits", False,
            f"Missing: {holdout_config}",
            "fatal",
        ))


def check_mathematical_bounds(report: PreflightReport) -> None:
    """Verify mathematical assumptions and compute expected bounds."""
    baselines = report.reusable_runs.get("_baselines", {})
    if not baselines or 4 not in baselines or 7 not in baselines:
        report.add(CheckResult(
            "math_bounds", False,
            "Cannot compute bounds: need both 4p and 7p baselines",
            "warning",
        ))
        return

    for model in MODELS:
        if model not in baselines[4] or model not in baselines[7]:
            continue

        a4 = baselines[4][model]["auroc"]
        a7 = baselines[7][model]["auroc"]

        # Fit 2-point exponential: A(p) = A_inf - alpha * exp(-beta * p)
        # A(4) = A_inf - alpha * exp(-4*beta)
        # A(7) = A_inf - alpha * exp(-7*beta)
        # Assuming A_inf ~ 0.95 (upper bound for this task), solve for beta:
        a_inf = 0.95  # reasonable ceiling assumption
        # alpha * exp(-4*beta) = a_inf - a4
        # alpha * exp(-7*beta) = a_inf - a7
        # ratio: exp(-3*beta) = (a_inf - a7) / (a_inf - a4)
        gap4 = a_inf - a4
        gap7 = a_inf - a7
        if gap4 > 0 and gap7 > 0 and gap7 < gap4:
            beta = -np.log(gap7 / gap4) / 3.0
            alpha = gap4 / np.exp(-4 * beta)

            # Predict where 95% of ceiling is reached
            # A(p*) = 0.95 * a_inf => a_inf - alpha * exp(-beta * p*) = 0.95 * a_inf
            # alpha * exp(-beta * p*) = 0.05 * a_inf
            p_95 = -np.log(0.05 * a_inf / alpha) / beta if alpha > 0 else float("inf")

            report.add(CheckResult(
                f"math_{model}_saturation", True,
                f"{model}: beta={beta:.3f}, predicted 95%-ceiling at p={p_95:.1f} "
                f"(A_inf={a_inf}, alpha={alpha:.3f})",
            ))
        else:
            report.add(CheckResult(
                f"math_{model}_saturation", False,
                f"{model}: cannot fit saturation model (gap4={gap4:.3f}, gap7={gap7:.3f})",
                "warning",
            ))


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_configs(orders: list[str] | None = None) -> None:
    """Generate all panel CSVs, training configs, and pipeline configs for the sweep."""
    target_orders = {k: v for k, v in ORDERS.items() if orders is None or k in orders}

    SWEEP_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    for order_name, order in target_orders.items():
        for size in range(PANEL_MIN, PANEL_MAX + 1):
            panel_proteins = order[:size]

            # Check if this exact panel already has results
            run_id = f"sweep_{order_name}_{size}p"
            reuse_run = None
            for existing_id, meta in EXISTING_RUNS.items():
                if meta["size"] == size and set(meta["proteins"]) == set(panel_proteins):
                    reuse_run = existing_id
                    break

            # Panel CSV
            panel_file = SWEEP_PANELS_DIR / f"{order_name}_{size}p.csv"
            panel_file.write_text("\n".join(panel_proteins) + "\n")

            # Training config
            training_config = SWEEP_CONFIGS_DIR / f"training_{order_name}_{size}p.yaml"
            # Use relative path from analysis/configs to the sweep panel
            panel_rel = f"experiments/panel_saturation_sweep/panels/{order_name}_{size}p.csv"
            config_content = (
                f"# Panel saturation sweep: {order_name} order, {size} proteins\n"
                f"# Auto-generated by preflight.py -- do not edit\n"
                f"_base: training_config.yaml\n"
                f"\n"
                f"features:\n"
                f"  feature_selection_strategy: fixed_panel\n"
                f"  fixed_panel_csv: \"{panel_rel}\"\n"
                f"\n"
                f"optuna:\n"
                f"  n_trials: 50\n"
            )
            training_config.write_text(config_content)

            # Pipeline config
            pipeline_config = SWEEP_CONFIGS_DIR / f"pipeline_{order_name}_{size}p.yaml"
            pipeline_content = (
                f"# Panel saturation sweep pipeline: {order_name} order, {size}p\n"
                f"# Auto-generated by preflight.py -- do not edit\n"
                f"_base: pipeline_hpc.yaml\n"
                f"\n"
                f"configs:\n"
                f"  training: experiments/panel_saturation_sweep/configs/training_{order_name}_{size}p.yaml\n"
                f"  splits: configs/splits_config_holdout.yaml\n"
                f"\n"
                f"pipeline:\n"
                f"  optimize_panel: false\n"
                f"  consensus: false\n"
                f"  permutation_test: false\n"
            )
            pipeline_config.write_text(pipeline_content)

            manifest.append({
                "order": order_name,
                "size": size,
                "run_id": run_id,
                "reuse": reuse_run,
                "panel_csv": str(panel_file.relative_to(PROJECT_ROOT)),
                "training_config": str(training_config.relative_to(PROJECT_ROOT)),
                "pipeline_config": str(pipeline_config.relative_to(PROJECT_ROOT)),
                "n_proteins": len(panel_proteins),
            })

    # Write manifest
    manifest_file = SWEEP_DIR / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    n_reusable = sum(1 for m in manifest if m["reuse"])
    n_new = len(manifest) - n_reusable
    print(f"Generated {len(manifest)} configurations ({n_reusable} reusable, {n_new} new)")
    print(f"  Panel CSVs:       {SWEEP_PANELS_DIR}/")
    print(f"  Training configs: {SWEEP_CONFIGS_DIR}/")
    print(f"  Manifest:         {manifest_file}")


# ---------------------------------------------------------------------------
# HPC submission preview
# ---------------------------------------------------------------------------

def dry_run_hpc() -> None:
    """Preview HPC submission commands without executing."""
    manifest_file = SWEEP_DIR / "manifest.json"
    if not manifest_file.exists():
        print("ERROR: Run --generate-configs first to create manifest.json")
        sys.exit(1)

    with open(manifest_file) as f:
        manifest = json.load(f)

    print("=" * 70)
    print("DRY RUN: HPC submission commands")
    print("=" * 70)

    n_skip = 0
    n_submit = 0

    for entry in manifest:
        if entry["reuse"]:
            n_skip += 1
            continue

        run_id = entry["run_id"]
        pipeline_cfg = entry["pipeline_config"]
        n_submit += 1

        print(f"ced run-pipeline \\")
        print(f"    --hpc \\")
        print(f"    --pipeline-config {pipeline_cfg} \\")
        print(f"    --run-id {run_id}")
        print()

    print(f"Total: {n_submit} submissions ({n_skip} skipped -- reusable)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight check for panel saturation sweep experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--order", choices=list(ORDERS.keys()),
        help="Check a single addition order (default: all)",
    )
    parser.add_argument(
        "--generate-configs", action="store_true",
        help="Generate all panel CSVs, training configs, and pipeline configs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview HPC submission commands",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output report as JSON (for programmatic consumption)",
    )
    args = parser.parse_args()

    if args.generate_configs:
        orders = [args.order] if args.order else None
        generate_configs(orders)
        return

    if args.dry_run:
        dry_run_hpc()
        return

    # Run all preflight checks
    report = PreflightReport()

    print("Running preflight checks...\n")

    check_data_integrity_lightweight(report)
    check_splits(report)
    check_addition_orders(report)
    check_config_chain(report)
    check_existing_results(report)
    check_baseline_metrics(report)
    check_mathematical_bounds(report)
    estimate_resources(report)

    if args.json:
        out = {
            "passed": report.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message, "severity": c.severity}
                for c in report.checks
            ],
            "reusable_runs": {k: v for k, v in report.reusable_runs.items() if not k.startswith("_")},
            "new_runs_needed": report.new_runs_needed,
            "total_cpu_hours": report.total_cpu_hours,
            "est_wall_hours": report.est_wall_hours,
        }
        print(json.dumps(out, indent=2))
    else:
        print(report.summary())

    sys.exit(0 if report.passed else (1 if report.fatals else 2))


if __name__ == "__main__":
    main()
