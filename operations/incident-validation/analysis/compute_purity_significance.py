#!/usr/bin/env python3
"""Bootstrap-based significance test for purity-ranked panel AUPRC.

For each (model, panel_size) in the purity ordering, tests whether
test AUPRC is significantly greater than the random-classifier baseline
(AUPRC_null = incident prevalence).

Method
------
SE is estimated from the 1000-sample bootstrap 95% CI already stored
in saturation_all_models.csv:

    SE  = (test_auprc_hi - test_auprc_lo) / 3.92
    z   = (test_auprc - prevalence) / SE
    p   = 1 - Phi(z)   [one-sided: better than chance]

BH-FDR is applied across panel sizes within each model.

Pareto minimum (significance) = smallest panel_size where p_adj < alpha.

Pareto optimum (tolerance) = smallest panel_size where
    test_auprc >= peak_auprc - tolerance
  i.e. the smallest panel that retains performance within a loss budget
  of `tolerance` AUPRC units relative to the best observed panel.

Outputs (under --out)
---------------------
    purity_significance.csv
    fig_purity_significance.{pdf,png}

Usage
-----
    python operations/incident-validation/analysis/compute_purity_significance.py
    python operations/incident-validation/analysis/compute_purity_significance.py \\
        --prevalence 0.00336 --alpha 0.05 --tolerance 0.01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CEL_ROOT = Path(__file__).resolve().parents[3]

SAT_PATH = CEL_ROOT / "operations/incident-validation/analysis/out/saturation_all_models.csv"
DATA_PATH = CEL_ROOT / "data/Celiac_dataset_proteomics_w_demo.parquet"


# ---------------------------------------------------------------------------
# Prevalence
# ---------------------------------------------------------------------------


def estimate_prevalence(data_path: Path) -> float:
    """Compute incident prevalence from the dataset."""
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
    # Enforce monotonicity (cummin from the right)
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return np.clip(adjusted, 0, 1)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_significance(
    df: pd.DataFrame,
    prevalence: float,
    alpha: float,
    tolerance: float,
) -> pd.DataFrame:
    purity = df[df["ordering"] == "purity"].copy()
    if purity.empty:
        raise ValueError("No 'purity' rows found in saturation_all_models.csv")

    purity["se"] = (purity["test_auprc_hi"] - purity["test_auprc_lo"]) / 3.92
    purity["z"] = (purity["test_auprc"] - prevalence) / purity["se"]
    purity["p_value"] = norm.sf(purity["z"])  # one-sided: AUPRC > prevalence

    results = []
    for model, grp in purity.groupby("model"):
        grp = grp.sort_values("panel_size").copy()
        grp["p_adj"] = bh_fdr(grp["p_value"].values)
        grp["significant"] = grp["p_adj"] < alpha

        # Tolerance-based Pareto: smallest N within `tolerance` of peak AUPRC
        peak_auprc = grp["test_auprc"].max()
        grp["peak_auprc"] = peak_auprc
        grp["within_tolerance"] = grp["test_auprc"] >= peak_auprc - tolerance

        results.append(grp)

    out = pd.concat(results, ignore_index=True)

    # Pareto minimum per model: smallest panel_size that is significant vs null
    pareto_sig = (
        out[out["significant"]]
        .groupby("model")["panel_size"]
        .min()
        .rename("pareto_min")
    )
    out = out.merge(pareto_sig, on="model", how="left")

    # Pareto optimum per model: smallest panel_size within tolerance of peak
    pareto_tol = (
        out[out["within_tolerance"]]
        .groupby("model")["panel_size"]
        .min()
        .rename("pareto_tol_min")
    )
    out = out.merge(pareto_tol, on="model", how="left")

    col_order = [
        "model", "panel_size", "n_features_used",
        "test_auprc", "test_auprc_lo", "test_auprc_hi",
        "se", "z", "p_value", "p_adj", "significant", "pareto_min",
        "peak_auprc", "within_tolerance", "pareto_tol_min",
    ]
    return out[col_order].sort_values(["model", "panel_size"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot(
    results: pd.DataFrame,
    prevalence: float,
    alpha: float,
    tolerance: float,
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
        pareto_tol_min = sub["pareto_tol_min"].iloc[0]
        peak_auprc = sub["peak_auprc"].iloc[0]
        sig = sub["significant"].values

        xs = sub["panel_size"].values
        ys = sub["test_auprc"].values
        lo = sub["test_auprc_lo"].values
        hi = sub["test_auprc_hi"].values

        # CI band
        ax.fill_between(xs, lo, hi, alpha=0.15, color="steelblue")

        # Points: significant vs not
        ax.plot(xs[~sig], ys[~sig], "o", color="gray", ms=6, zorder=3,
                label="p_adj ≥ {:.2f}".format(alpha))
        ax.plot(xs[sig], ys[sig], "o", color="steelblue", ms=7, zorder=4,
                label="p_adj < {:.2f}".format(alpha))
        ax.plot(xs, ys, "-", color="steelblue", lw=1.2, zorder=2)

        # Null baseline
        ax.axhline(prevalence, color="tomato", lw=1.2, ls="--",
                   label=f"null (prevalence={prevalence:.4f})")

        # Tolerance band: peak - tolerance
        ax.axhline(peak_auprc - tolerance, color="darkorange", lw=1.0, ls=":",
                   label=f"peak − {tolerance:.3f} ({peak_auprc - tolerance:.4f})")

        # Pareto minimum (significance vs null)
        if not np.isnan(pareto_min):
            ax.axvline(pareto_min, color="darkgreen", lw=1.5, ls=":",
                       label=f"pareto sig N={int(pareto_min)}")

        # Pareto optimum (tolerance-based)
        if not np.isnan(pareto_tol_min):
            ax.axvline(pareto_tol_min, color="purple", lw=1.5, ls="--",
                       label=f"pareto opt N={int(pareto_tol_min)} (tol={tolerance:.3f})")

        ax.set_xlabel("Panel size (N proteins)", fontsize=10)
        ax.set_ylabel("Test AUPRC (95% CI)", fontsize=10)
        ax.set_title(f"{model} — purity ordering", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle(
        f"Purity panel: AUPRC significance vs chance (BH-FDR) | tolerance={tolerance:.3f}",
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
    parser.add_argument(
        "--sat",
        type=Path,
        default=SAT_PATH,
        help="Path to saturation_all_models.csv",
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=None,
        help="Incident prevalence (null AUPRC). If omitted, computed from --data.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Parquet dataset path (used to estimate prevalence if not provided)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold (default: 0.05)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help=(
            "AUPRC loss budget for tolerance-based Pareto optimum: find smallest N "
            "where test_auprc >= peak_auprc - tolerance (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
        help="Output directory",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prevalence = args.prevalence
    if prevalence is None:
        prevalence = estimate_prevalence(args.data)

    log.info("Loading %s", args.sat)
    df = pd.read_csv(args.sat)

    results = compute_significance(df, prevalence, args.alpha, args.tolerance)

    out_path = args.out / "purity_significance.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved %s", out_path)

    plot(results, prevalence, args.alpha, args.tolerance, args.out)

    print(
        f"\n=== Purity panel significance "
        f"(alpha={args.alpha}, prevalence={prevalence:.5f}, tolerance={args.tolerance:.4f}) ==="
    )
    print(results[[
        "model", "panel_size", "test_auprc", "p_value", "p_adj",
        "significant", "pareto_min", "pareto_tol_min",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Pareto significance minima (smallest N beating null) ===")
    sig_pareto = (
        results[results["significant"]]
        .groupby("model")[["panel_size", "test_auprc", "p_adj"]]
        .first()
    )
    print(sig_pareto.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Pareto tolerance optima (smallest N within {args.tolerance:.4f} of peak) ===")
    tol_pareto = (
        results[results["within_tolerance"]]
        .groupby("model")[["panel_size", "test_auprc", "peak_auprc"]]
        .first()
    )
    print(tol_pareto.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
