#!/usr/bin/env python3
"""Compute prevalent-noise scores for protein panel optimization.

For each protein in the SVM panel, quantifies how much its coefficient shifts
when prevalent (post-diagnosis) cases are added to incident-only training.
High drift = protein capturing post-diagnosis biology, not pre-diagnostic risk.

Algorithm
---------
For each protein p, using best-weight scheme per model (L1→log, L2→none):
    μ_IO(p) = mean coef across 5 folds, incident_only + best_weight
    μ_IP(p) = mean coef across 5 folds, incident_prevalent + best_weight

    noise_score(p) = |μ_IP − μ_IO| / (|μ_IO| + ε)
    purity_score(p) = |μ_IO(p)| / (1 + noise_score(p))

Also computed pooled across all 4 weight schemes.
Combined purity score = mean of normalised L1 and L2 purity scores.

Outputs (under --out, default: operations/incident-validation/analysis/out/)
--------
    prevalent_noise_scores.csv
    fig_prevalent_noise.pdf / .png

Usage
-----
    python operations/incident-validation/analysis/compute_prevalent_noise_score.py
    python operations/incident-validation/analysis/compute_prevalent_noise_score.py \\
        --out results/incident-validation/noise_analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CEL_ROOT = Path(__file__).resolve().parents[3]

FOLD_COEF_PATHS = {
    "SVM_L1": CEL_ROOT / "results/incident-validation/lr/SVM_L1/fold_coefficients.csv",
    "SVM_L2": CEL_ROOT / "results/incident-validation/lr/SVM_L2/fold_coefficients.csv",
}
# Best weight scheme per model from incident-validation report
BEST_WEIGHT = {"SVM_L1": "log", "SVM_L2": "none"}

CORE_FEATURES_PATH = (
    CEL_ROOT / "operations/incident-validation/analysis/out/core_features.csv"
)
EPS = 1e-8


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def load_fold_coefs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["protein"] = df["protein"].str.replace("_resid$", "", regex=True).str.upper()
    return df


def strategy_means(df: pd.DataFrame, weight_scheme: str | None = None) -> pd.DataFrame:
    """Mean signed coefficient per (protein, strategy), optionally for one weight."""
    if weight_scheme is not None:
        df = df[df["weight_scheme"] == weight_scheme]
    return (
        df.groupby(["protein", "strategy"])["coefficient"]
        .mean()
        .unstack("strategy")
        .reset_index()
    )


def compute_scores(
    df: pd.DataFrame, best_weight: str
) -> pd.DataFrame:
    """Return per-protein scores for one model."""
    bw = strategy_means(df, best_weight)
    pooled = strategy_means(df, None)

    for label, sdf in [("best-weight", bw), ("pooled", pooled)]:
        missing = {"incident_only", "incident_prevalent", "prevalent_only"} - set(sdf.columns)
        if missing:
            raise ValueError(f"Missing strategies in {label}: {missing}")

    def ns(io, ip):
        return (ip - io).abs() / (io.abs() + EPS)

    def ps(io, n_score):
        return io.abs() / (1.0 + n_score)

    ns_bw = ns(bw["incident_only"], bw["incident_prevalent"])
    ns_pool = ns(pooled["incident_only"], pooled["incident_prevalent"])

    sign_io = np.sign(bw["incident_only"])
    sign_ip = np.sign(bw["incident_prevalent"])
    # Only call sign-flip if both are non-zero
    sign_flip = (sign_io != sign_ip) & (bw["incident_only"].abs() > EPS) & (bw["incident_prevalent"].abs() > EPS)

    return pd.DataFrame({
        "protein": bw["protein"].values,
        "io_coef": bw["incident_only"].values,
        "ip_coef": bw["incident_prevalent"].values,
        "po_coef": bw["prevalent_only"].values,
        "noise_score_bw": ns_bw.values,
        "purity_score_bw": ps(bw["incident_only"], ns_bw).values,
        "noise_score_pool": ns_pool.values,
        "purity_score_pool": ps(pooled["incident_only"], ns_pool).values,
        "sign_flip": sign_flip.values,
    })


def build_table(out_dir: Path) -> pd.DataFrame:
    model_scores = {}
    for model, path in FOLD_COEF_PATHS.items():
        log.info("Loading %s", path)
        df = load_fold_coefs(path)
        model_scores[model] = compute_scores(df, BEST_WEIGHT[model])

    merged = model_scores["SVM_L1"].rename(
        columns={c: f"L1_{c}" for c in model_scores["SVM_L1"].columns if c != "protein"}
    ).merge(
        model_scores["SVM_L2"].rename(
            columns={c: f"L2_{c}" for c in model_scores["SVM_L2"].columns if c != "protein"}
        ),
        on="protein",
        how="outer",
    )

    # Combined purity = mean of min-max normalised per-model scores
    for suffix in ("bw", "pool"):
        l1 = merged[f"L1_purity_score_{suffix}"]
        l2 = merged[f"L2_purity_score_{suffix}"]
        merged[f"combined_purity_{suffix}"] = (
            l1 / (l1.max() + EPS) + l2 / (l2.max() + EPS)
        ) / 2

    # Annotate core features
    if CORE_FEATURES_PATH.exists():
        core = set(pd.read_csv(CORE_FEATURES_PATH)["protein"].str.upper())
        merged["is_core"] = merged["protein"].isin(core)
    else:
        log.warning("core_features.csv not found; is_core set to False")
        merged["is_core"] = False

    merged = merged.sort_values("combined_purity_bw", ascending=False).reset_index(drop=True)
    merged.insert(0, "purity_rank", range(1, len(merged) + 1))

    out_path = out_dir / "prevalent_noise_scores.csv"
    merged.to_csv(out_path, index=False)
    log.info("Saved %s", out_path)
    return merged


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _scatter(axes[0], df, "L1_io_coef", "L1_noise_score_bw", "SVM L1 (log weight)")
    _scatter(axes[1], df, "L2_io_coef", "L2_noise_score_bw", "SVM L2 (no weight)")
    _purity_bars(axes[2], df)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_prevalent_noise.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        log.info("Saved %s", p)
    plt.close(fig)


def _scatter(ax, df, io_col, ns_col, title):
    io = df[io_col].abs()
    ns = df[ns_col]
    core = df["is_core"]
    flip = df.get("L1_sign_flip", pd.Series(False, index=df.index)) if "L1" in io_col else df.get("L2_sign_flip", pd.Series(False, index=df.index))

    ax.scatter(io[~core & ~flip], ns[~core & ~flip], alpha=0.35, s=18, color="steelblue", label="non-core")
    ax.scatter(io[flip], ns[flip], alpha=0.7, s=30, color="orange", zorder=4, label="sign flip")
    ax.scatter(io[core], ns[core], alpha=0.85, s=55, color="crimson", zorder=5, label="core (28)")

    for _, row in df[core].iterrows():
        ax.annotate(
            row["protein"],
            (abs(row[io_col]), row[ns_col]),
            fontsize=5.5,
            alpha=0.8,
            ha="left",
            va="bottom",
            xytext=(2, 2),
            textcoords="offset points",
        )

    ax.axvline(io.median(), color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(ns.median(), color="gray", lw=0.7, ls="--", alpha=0.5)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xm, ym = io.median(), ns.median()
    for txt, x, y, ha, va in [
        ("incident-pure\n(keep)", xm * 1.05, ym * 0.5, "left", "center"),
        ("prevalent-drift\n(scrutinize)", xm * 1.05, ym * 1.5, "left", "center"),
        ("weak signal", xm * 0.5, ym * 0.5, "right", "center"),
    ]:
        ax.text(x, y, txt, fontsize=6, color="gray", ha=ha, va=va, alpha=0.6)

    ax.set_xlabel("|coef| incident_only", fontsize=9)
    ax.set_ylabel("prevalent noise score", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def _purity_bars(ax, df):
    top = df.head(40).copy()
    colors = ["crimson" if c else "steelblue" for c in top["is_core"]]
    ax.barh(range(len(top)), top["combined_purity_bw"], color=colors, alpha=0.75)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["protein"], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("combined purity score (normalised)", fontsize=9)
    ax.set_title("Top 40 by incident purity rank", fontsize=10)
    ax.legend(
        handles=[
            mpatches.Patch(color="crimson", alpha=0.75, label="core (28)"),
            mpatches.Patch(color="steelblue", alpha=0.75, label="non-core"),
        ],
        fontsize=7,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
        help="Output directory",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = build_table(args.out)
    plot(df, args.out)

    # Quick summary to stdout
    print("\n=== Top 20 by incident purity rank ===")
    cols = ["purity_rank", "protein", "L1_io_coef", "L1_noise_score_bw",
            "L2_io_coef", "L2_noise_score_bw", "combined_purity_bw", "is_core"]
    print(df[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Core features ranked by purity (sign flips flagged) ===")
    core_df = df[df["is_core"]].copy()
    core_cols = ["purity_rank", "protein", "L1_io_coef", "L1_ip_coef",
                 "L1_noise_score_bw", "L1_sign_flip", "L2_noise_score_bw", "combined_purity_bw"]
    print(core_df[core_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
