#!/usr/bin/env python3
"""Optimal panel size selection via z-test against peak AUPRC.

For each (ordering, model), finds the smallest panel N that is
statistically non-inferior to the peak AUPRC:

    H0: AUPRC(peak) - AUPRC(N) <= 0   [N is non-inferior]
    H1: AUPRC(peak) - AUPRC(N) > 0    [N is inferior]

    SE_combined = sqrt(SE(N)^2 + SE(peak)^2)
    z = (AUPRC(peak) - AUPRC(N)) / SE_combined
    p = norm.sf(z)   [one-sided]

SE is recovered from the 1000-sample bootstrap 95% CI:
    SE = (test_auprc_hi - test_auprc_lo) / 3.92

Note: SE_combined is conservative (treats bootstraps as independent)
because all panel sizes share the same test set. The resulting
z-test is anti-conservative (easier to be non-inferior). Use
a stricter alpha (e.g., 0.01) if you want a more conservative
panel size.

BH-FDR is applied across panel sizes within each (model, ordering).

pareto_opt = smallest N where p_adj >= alpha (non-inferior to peak).

Outputs (under --out)
---------------------
    panel_size_selection.csv
    fig_panel_size_selection.{pdf,png}

Usage
-----
    python compute_panel_size_selection.py
    python compute_panel_size_selection.py --orderings purity rfe --alpha 0.01
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

ORDERING_STYLE = {
    "purity":    ("steelblue",  "s-",  "Purity"),
    "stability": ("darkorange", "o-",  "Stability"),
    "rfe":       ("seagreen",   "^--", "RFE"),
}


# ---------------------------------------------------------------------------
# BH-FDR
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
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
# Core: z-test vs peak
# ---------------------------------------------------------------------------

def compute_peak_noninferiority(
    df: pd.DataFrame,
    orderings: list[str],
    alpha: float,
) -> pd.DataFrame:
    results = []

    for ordering in orderings:
        sub = df[df["ordering"] == ordering].copy()
        if sub.empty:
            log.warning("No rows for ordering=%s, skipping", ordering)
            continue

        sub["se"] = (sub["test_auprc_hi"] - sub["test_auprc_lo"]) / 3.92

        for model, grp in sub.groupby("model"):
            grp = grp.sort_values("panel_size").copy()

            peak_idx = grp["test_auprc"].idxmax()
            peak_auprc = grp.loc[peak_idx, "test_auprc"]
            peak_se    = grp.loc[peak_idx, "se"]
            peak_n     = grp.loc[peak_idx, "panel_size"]

            se_combined = np.sqrt(grp["se"] ** 2 + peak_se ** 2)
            delta       = peak_auprc - grp["test_auprc"]
            z           = delta / se_combined

            p = norm.sf(z.values)
            # Peak itself: z=0 → p=0.5; set explicitly to avoid floating noise
            peak_mask = grp["panel_size"] == peak_n
            p[peak_mask.values] = 0.5

            grp["delta_from_peak"] = delta.values
            grp["z_vs_peak"]       = z.values
            grp["p_inferior"]      = p
            grp["p_adj"]           = bh_fdr(p)
            grp["noninferior"]     = grp["p_adj"] >= alpha
            grp["peak_n"]          = peak_n
            grp["peak_auprc"]      = peak_auprc

            noninf = grp[grp["noninferior"]]
            pareto_opt = int(noninf["panel_size"].min()) if not noninf.empty else np.nan
            grp["pareto_opt"] = pareto_opt

            log.info(
                "ordering=%-10s model=%-6s peak_n=%3d peak_auprc=%.4f pareto_opt=%s",
                ordering, model, peak_n, peak_auprc,
                str(pareto_opt) if not np.isnan(pareto_opt) else "none",
            )
            results.append(grp)

    if not results:
        raise ValueError("No data found for any requested ordering")

    out = pd.concat(results, ignore_index=True)
    col_order = [
        "ordering", "model", "panel_size", "n_features_used",
        "test_auprc", "test_auprc_lo", "test_auprc_hi",
        "se", "delta_from_peak", "z_vs_peak", "p_inferior", "p_adj",
        "noninferior", "peak_n", "peak_auprc", "pareto_opt",
    ]
    return out[[c for c in col_order if c in out.columns]].sort_values(
        ["ordering", "model", "panel_size"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(results: pd.DataFrame, alpha: float, out_dir: Path) -> None:
    models   = sorted(results["model"].unique())
    orderings = [o for o in ("purity", "stability", "rfe") if o in results["ordering"].unique()]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        for ordering in orderings:
            sub = results[
                (results["model"] == model) & (results["ordering"] == ordering)
            ].sort_values("panel_size")
            if sub.empty:
                continue

            color, marker, label = ORDERING_STYLE.get(ordering, ("gray", "o-", ordering))
            xs  = sub["panel_size"].values
            ys  = sub["test_auprc"].values
            lo  = sub["test_auprc_lo"].values
            hi  = sub["test_auprc_hi"].values
            opt = sub["pareto_opt"].iloc[0]
            pk  = sub["peak_n"].iloc[0]

            ax.fill_between(xs, lo, hi, alpha=0.10, color=color)
            ax.plot(xs, ys, marker, color=color, lw=1.4, ms=5,
                    label=f"{label} (peak={pk}, opt={opt})")
            if not np.isnan(opt):
                ax.axvline(opt, color=color, lw=1.5, ls=":",
                           alpha=0.85)

        ax.set_xlabel("Panel size (N proteins)", fontsize=10)
        ax.set_ylabel("Test AUPRC (95% CI)", fontsize=10)
        ax.set_title(model, fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle(
        f"Panel size selection: smallest N non-inferior to peak (BH-FDR α={alpha})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out_dir / f"fig_panel_size_selection.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved %s", p)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat", type=Path, default=SAT_PATH)
    parser.add_argument(
        "--orderings", nargs="+",
        default=["purity", "stability", "rfe"],
        choices=["purity", "stability", "rfe"],
    )
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="BH-FDR threshold for non-inferiority (default 0.05)")
    parser.add_argument(
        "--out", type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", args.sat)
    df = pd.read_csv(args.sat)

    results = compute_peak_noninferiority(df, args.orderings, args.alpha)

    out_csv = args.out / "panel_size_selection.csv"
    results.to_csv(out_csv, index=False)
    log.info("Saved %s", out_csv)

    plot(results, args.alpha, args.out)

    print(f"\n=== Panel size selection (z-test vs peak, BH-FDR α={args.alpha}) ===\n")
    summary = (
        results.groupby(["ordering", "model"])
        .apply(lambda g: pd.Series({
            "peak_n":     int(g["peak_n"].iloc[0]),
            "peak_auprc": g["peak_auprc"].iloc[0],
            "pareto_opt": g["pareto_opt"].iloc[0],
            "pareto_auprc": g.loc[g["panel_size"] == g["pareto_opt"].iloc[0], "test_auprc"].iloc[0]
                if not np.isnan(g["pareto_opt"].iloc[0]) else np.nan,
        }), include_groups=False)
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Full table ===")
    print(results[[
        "ordering", "model", "panel_size", "test_auprc",
        "delta_from_peak", "z_vs_peak", "p_adj", "noninferior", "pareto_opt",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
