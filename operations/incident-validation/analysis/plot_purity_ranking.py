#!/usr/bin/env python3
"""Dumbbell plot of incident-only vs incident+prevalent L1 coefficients.

For each protein (sorted by purity_rank), shows:
  - Blue dot:  |L1 coef| trained on incident-only (io)
  - Red dot:   |L1 coef| trained on incident+prevalent (ip)
  - Line:      drift between the two conditions (= noise signal)

Proteins at the top are high-purity (small drift, stable signal).
Proteins at the bottom are noisy (large drift, prevalent contamination).

Usage
-----
    python plot_purity_ranking.py
    python plot_purity_ranking.py --top 39 --out path/to/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

CEL_ROOT = Path(__file__).resolve().parents[3]
SCORES_PATH = CEL_ROOT / "operations/incident-validation/analysis/out/prevalent_noise_scores.csv"


def clean_label(name: str) -> str:
    return name.replace("_", " ").upper()


def plot_dumbbell(df: pd.DataFrame, out_dir: Path, top: int) -> None:
    df = df.sort_values("purity_rank").head(top).copy()
    df = df.sort_values("purity_rank", ascending=False)  # top rank at top of plot

    proteins = [clean_label(p) for p in df["protein"]]
    io  = df["L1_io_coef"].abs().values
    ip  = df["L1_ip_coef"].abs().values
    purity = df["combined_purity_bw"].values

    # Normalise purity to [0,1] for alpha
    p_norm = (purity - purity.min()) / (purity.max() - purity.min() + 1e-9)

    fig_h = max(6, top * 0.32)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for i, (io_val, ip_val, alpha) in enumerate(zip(io, ip, p_norm)):
        color = plt.cm.RdYlGn(0.15 + 0.7 * alpha)   # green = high purity, red = low
        ax.plot([io_val, ip_val], [i, i], color=color, lw=1.5, alpha=0.7, zorder=1)
        ax.scatter(io_val, i, color="#2166ac", s=40, zorder=3)
        ax.scatter(ip_val, i, color="#d6604d", s=40, zorder=3)

    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=7)
    ax.set_xlabel("|L1 coefficient|", fontsize=10)
    ax.set_title(
        f"Purity ranking: coefficient drift (incident-only vs incident+prevalent)\n"
        f"Top {top} proteins — green = high purity, red = high noise",
        fontsize=10,
    )
    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    io_patch  = mpatches.Patch(color="#2166ac", label="Incident-only |coef|")
    ip_patch  = mpatches.Patch(color="#d6604d", label="Incident+prevalent |coef|")
    ax.legend(handles=[io_patch, ip_patch], fontsize=8, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_purity_dumbbell.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def plot_scores(df: pd.DataFrame, out_dir: Path, top: int) -> None:
    """Bar chart of purity score and noise score side by side."""
    df = df.sort_values("purity_rank").head(top).copy()
    proteins = [clean_label(p) for p in df["protein"]]
    purity = df["combined_purity_bw"].values
    noise  = df["L1_noise_score_bw"].values

    x = np.arange(len(proteins))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, top * 0.45), 4))
    ax.bar(x - w/2, purity, width=w, label="Purity score", color="#2166ac", alpha=0.8)
    ax.bar(x + w/2, noise,  width=w, label="Noise score",  color="#d6604d", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(proteins, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Purity vs noise score — top {top} proteins (sorted by purity rank)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_purity_scores.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    parser.add_argument("--top", type=int, default=39, help="Top N proteins to show (default: all)")
    parser.add_argument(
        "--out", type=Path,
        default=CEL_ROOT / "operations/incident-validation/analysis/out",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores).sort_values("purity_rank")
    print(f"Loaded {len(df)} proteins")

    plot_dumbbell(df, args.out, args.top)
    plot_scores(df, args.out, args.top)


if __name__ == "__main__":
    main()
