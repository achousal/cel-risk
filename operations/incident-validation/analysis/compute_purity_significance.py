#!/usr/bin/env python3
"""Bootstrap-based significance test for purity-ranked panel AUPRC.

For each (model, panel_size) in the purity ordering, tests whether
test AUPRC is significantly greater than the random-classifier baseline
(AUPRC_null = incident prevalence).

Significance method
-------------------
SE is estimated from the 1000-sample bootstrap 95% CI already stored
in saturation_all_models.csv:

    SE  = (test_auprc_hi - test_auprc_lo) / 3.92
    z   = (test_auprc - prevalence) / SE
    p   = 1 - Phi(z)   [one-sided: better than chance]

BH-FDR is applied across panel sizes within each model.

Optimal panel size
------------------
Uses the three-criterion rule from ced_ml.recipes.size_rules
(same algorithm as the main cel-risk pipeline, applied to AUPRC):

  C1 Non-inferior  : AUPRC(N) >= AUPRC(best) - delta  (one-sided z-test)
  C2 Within 1 SE   : AUPRC(N) >= AUPRC(best) - SE(best)
  C3 Marginal gain : AUPRC(N -> N+1) not significant after Holm correction

  pareto_opt = smallest N where >= min_criteria criteria pass.

SE for the three-criterion rule is recovered from the bootstrap CI:
  SE = (ci_hi - ci_lo) / 3.92  →  std = SE * sqrt(n_bootstrap)
fed to derive_size_three_criterion with n_seeds = n_bootstrap.

Config
------
Parameters are read from manifest.yaml (purity_significance section).
All values can be overridden via CLI flags.

Outputs (under --out)
---------------------
    purity_significance.csv
    purity_significance_audit.json   (per-model three-criterion audit logs)
    fig_purity_significance.{pdf,png}

Usage
-----
    python compute_purity_significance.py --config operations/incident-validation/manifest.yaml
    python compute_purity_significance.py --delta 0.01 --min-criteria 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from ced_ml.recipes.size_rules import derive_size_three_criterion

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CEL_ROOT = Path(__file__).resolve().parents[3]

SAT_PATH = CEL_ROOT / "operations/incident-validation/analysis/out/saturation_all_models.csv"
DATA_PATH = CEL_ROOT / "data/Celiac_dataset_proteomics_w_demo.parquet"
MANIFEST_PATH = CEL_ROOT / "operations/incident-validation/manifest.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(manifest_path: Path) -> dict:
    """Load purity_significance section from manifest.yaml."""
    import yaml
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    return manifest.get("purity_significance", {})


# ---------------------------------------------------------------------------
# Prevalence
# ---------------------------------------------------------------------------


def estimate_prevalence(data_path: Path) -> float:
    log.info("Estimating prevalence from %s", data_path)
    df = pd.read_parquet(data_path, columns=["status"])
    n_incident = (df["status"] == "incident").sum()
    n_total = len(df)
    prev = n_incident / n_total
    log.info("Incident: %d / %d = %.5f", n_incident, n_total, prev)
    return float(prev)


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    adjusted = pvals * n / rank
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return np.clip(adjusted, 0, 1)


# ---------------------------------------------------------------------------
# Three-criterion adapter
# ---------------------------------------------------------------------------


def _to_sweep_df(grp: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    """Convert bootstrap CI data to the sweep_df format expected by derive_size_three_criterion.

    SE is recovered from the 95% CI: SE = (hi - lo) / 3.92.
    std is back-calculated as SE * sqrt(n_bootstrap) so the function
    recovers the original SE when it computes std / sqrt(n_seeds).
    """
    se = (grp["test_auprc_hi"] - grp["test_auprc_lo"]) / 3.92
    return pd.DataFrame({
        "panel_size": grp["panel_size"].values,
        "auprc_mean": grp["test_auprc"].values,
        "auprc_std":  se.values * np.sqrt(n_bootstrap),
    })


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_significance(
    df: pd.DataFrame,
    prevalence: float,
    alpha: float,
    delta: float,
    n_bootstrap: int,
    min_criteria: int,
) -> tuple[pd.DataFrame, dict]:
    purity = df[df["ordering"] == "purity"].copy()
    if purity.empty:
        raise ValueError("No 'purity' rows found in saturation_all_models.csv")

    purity["se"] = (purity["test_auprc_hi"] - purity["test_auprc_lo"]) / 3.92
    purity["z"] = (purity["test_auprc"] - prevalence) / purity["se"]
    purity["p_value"] = norm.sf(purity["z"])

    results = []
    audits = {}

    for model, grp in purity.groupby("model"):
        grp = grp.sort_values("panel_size").copy()
        grp["p_adj"] = bh_fdr(grp["p_value"].values)
        grp["significant"] = grp["p_adj"] < alpha

        # Three-criterion optimal panel size
        sweep_df = _to_sweep_df(grp, n_bootstrap)
        optimal_p, audit = derive_size_three_criterion(
            sweep_df,
            auroc_col="auprc_mean",
            auroc_std_col="auprc_std",
            n_seeds=n_bootstrap,
            delta=delta,
            min_criteria=min_criteria,
        )
        grp["pareto_opt"] = optimal_p
        audits[model] = audit
        log.info("Model %s: three-criterion pareto_opt=%d (delta=%.3f, min_criteria=%d)",
                 model, optimal_p, delta, min_criteria)

        results.append(grp)

    out = pd.concat(results, ignore_index=True)

    # Pareto minimum (significance vs null): smallest N where p_adj < alpha
    pareto_sig = (
        out[out["significant"]]
        .groupby("model")["panel_size"]
        .min()
        .rename("pareto_min")
    )
    out = out.merge(pareto_sig, on="model", how="left")

    col_order = [
        "model", "panel_size", "n_features_used",
        "test_auprc", "test_auprc_lo", "test_auprc_hi",
        "se", "z", "p_value", "p_adj", "significant", "pareto_min", "pareto_opt",
    ]
    return out[col_order].sort_values(["model", "panel_size"]).reset_index(drop=True), audits


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot(
    results: pd.DataFrame,
    prevalence: float,
    alpha: float,
    delta: float,
    out_dir: Path,
) -> None:
    models = sorted(results["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = results[results["model"] == model].sort_values("panel_size")
        pareto_min = sub["pareto_min"].iloc[0]
        pareto_opt = sub["pareto_opt"].iloc[0]
        sig = sub["significant"].values

        xs = sub["panel_size"].values
        ys = sub["test_auprc"].values
        lo = sub["test_auprc_lo"].values
        hi = sub["test_auprc_hi"].values

        ax.fill_between(xs, lo, hi, alpha=0.15, color="steelblue")
        ax.plot(xs[~sig], ys[~sig], "o", color="gray", ms=6, zorder=3,
                label="p_adj ≥ {:.2f}".format(alpha))
        ax.plot(xs[sig], ys[sig], "o", color="steelblue", ms=7, zorder=4,
                label="p_adj < {:.2f}".format(alpha))
        ax.plot(xs, ys, "-", color="steelblue", lw=1.2, zorder=2)

        ax.axhline(prevalence, color="tomato", lw=1.2, ls="--",
                   label=f"null (prevalence={prevalence:.4f})")

        if not np.isnan(pareto_min):
            ax.axvline(pareto_min, color="darkgreen", lw=1.5, ls=":",
                       label=f"pareto sig N={int(pareto_min)}")

        if not np.isnan(pareto_opt):
            ax.axvline(pareto_opt, color="purple", lw=1.5, ls="--",
                       label=f"pareto opt N={int(pareto_opt)} (3-crit δ={delta:.3f})")

        ax.set_xlabel("Panel size (N proteins)", fontsize=10)
        ax.set_ylabel("Test AUPRC (95% CI)", fontsize=10)
        ax.set_title(f"{model} — purity ordering", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle(
        f"Purity panel: AUPRC significance vs chance (BH-FDR) | 3-criterion δ={delta:.3f}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"fig_purity_significance.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved %s", p)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=MANIFEST_PATH,
                        help="Path to manifest.yaml (reads purity_significance section)")
    parser.add_argument("--sat", type=Path, default=SAT_PATH,
                        help="Path to saturation_all_models.csv")
    parser.add_argument("--prevalence", type=float, default=None,
                        help="Incident prevalence (overrides config; if neither set, computed from --data)")
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help="Parquet dataset (used if prevalence not in config or CLI)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="BH-FDR threshold (overrides config; default 0.05)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Non-inferiority margin for three-criterion rule (overrides config; default 0.02)")
    parser.add_argument("--min-criteria", type=int, default=None, dest="min_criteria",
                        help="Criteria that must pass: 2=majority, 3=unanimous (overrides config; default 2)")
    parser.add_argument("--out", type=Path,
                        default=CEL_ROOT / "operations/incident-validation/analysis/out",
                        help="Output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Config: manifest → CLI override
    cfg = load_config(args.config) if args.config.exists() else {}
    prevalence = args.prevalence or cfg.get("prevalence")
    alpha       = args.alpha        or cfg.get("alpha",        0.05)
    delta       = args.delta        or cfg.get("delta",        0.02)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    min_criteria = args.min_criteria or cfg.get("min_criteria", 2)

    log.info("Config: prevalence=%s alpha=%.3f delta=%.3f n_bootstrap=%d min_criteria=%d",
             prevalence, alpha, delta, n_bootstrap, min_criteria)

    if prevalence is None:
        prevalence = estimate_prevalence(args.data)
    prevalence = float(prevalence)

    log.info("Loading %s", args.sat)
    df = pd.read_csv(args.sat)

    results, audits = compute_significance(df, prevalence, alpha, delta, n_bootstrap, min_criteria)

    out_csv = args.out / "purity_significance.csv"
    results.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    audit_path = args.out / "purity_significance_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audits, f, indent=2)
    log.info("Saved %s", audit_path)

    plot(results, prevalence, alpha, delta, args.out)

    print(f"\n=== Purity panel significance "
          f"(alpha={alpha}, prevalence={prevalence:.5f}, delta={delta:.4f}) ===")
    print(results[[
        "model", "panel_size", "test_auprc", "p_value", "p_adj",
        "significant", "pareto_min", "pareto_opt",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Pareto significance minima (smallest N beating null) ===")
    sig_rows = results[results["significant"]]
    if not sig_rows.empty:
        print(sig_rows.groupby("model")[["panel_size", "test_auprc", "p_adj"]]
              .first().to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Three-criterion Pareto optima (delta={delta:.4f}, min_criteria={min_criteria}) ===")
    opt_rows = results.groupby("model")[["pareto_opt", "test_auprc"]].first()
    print(opt_rows.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
