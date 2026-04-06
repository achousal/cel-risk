#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "LR_EN": "Logistic (EN)",
    "LinSVM_cal": "Linear SVM",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "ENSEMBLE": "Ensemble",
}

MODEL_COLORS = {
    "LR_EN": "#4C78A8",
    "LinSVM_cal": "#1B9E77",
    "RF": "#D95F02",
    "XGBoost": "#7570B3",
    "ENSEMBLE": "#111111",
}

ORDER_COLORS = {
    "rra": "#E69F00",
    "importance": "#56B4E9",
    "pathway": "#009E73",
}

ORDER_LABELS = {
    "rra": "RRA",
    "importance": "Importance",
    "pathway": "Pathway",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cel-risk optimal-setup figures.")
    parser.add_argument("--results-dir", required=True, help="Path to cel-risk results directory.")
    parser.add_argument("--panel-dir", required=True, help="Path to panel-sweep/panels directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for figures.")
    return parser.parse_args()


def clean_marker(name: str) -> str:
    if name.endswith("_resid"):
        name = name[:-6]
    return name.upper()


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.read_csv(results_dir / "compiled_results_aggregated.csv")
    ens = pd.read_csv(results_dir / "compiled_results_ensemble.csv")
    return agg, ens


def fig01_svm_panel_tendency(agg: pd.DataFrame, out_dir: Path) -> None:
    svm = agg.loc[agg["model"] == "LinSVM_cal"].copy()
    svm = svm.sort_values(["order", "panel_size"])
    pathway = svm.loc[svm["order"] == "pathway"].copy()
    best_row = pathway.loc[pathway["summary_auroc_mean"].idxmax()]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(8, 10, color="#d9ead3", alpha=0.45, zorder=0)

    for order, subset in svm.groupby("order"):
        subset = subset.sort_values("panel_size")
        ax.plot(
            subset["panel_size"],
            subset["summary_auroc_mean"],
            color=ORDER_COLORS[order],
            linewidth=2.8 if order == "pathway" else 1.8,
            alpha=1.0 if order == "pathway" else 0.7,
            label=ORDER_LABELS[order],
        )
        ax.fill_between(
            subset["panel_size"],
            subset["summary_auroc_ci95_lo"],
            subset["summary_auroc_ci95_hi"],
            color=ORDER_COLORS[order],
            alpha=0.12 if order == "pathway" else 0.07,
        )

    ax.scatter(
        [best_row["panel_size"]],
        [best_row["summary_auroc_mean"]],
        s=70,
        color=ORDER_COLORS["pathway"],
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    ax.annotate(
        f"Best pathway SVM\np={int(best_row['panel_size'])}, AUROC={best_row['summary_auroc_mean']:.3f}",
        xy=(best_row["panel_size"], best_row["summary_auroc_mean"]),
        xytext=(best_row["panel_size"] + 1.3, best_row["summary_auroc_mean"] - 0.006),
        arrowprops={"arrowstyle": "-", "color": "#333333", "lw": 0.8},
        fontsize=9,
    )

    ax.set_title("Linear SVM Panel-Size Tendency", fontsize=14, weight="bold")
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Mean AUROC across seeds")
    ax.set_xticks([4, 7, 10, 15, 20, 25])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    sns.despine()
    save_both(fig, out_dir, "fig01_svm_panel_tendency")


def fig02_svm_marker_heatmap(panel_dir: Path, out_dir: Path) -> None:
    sizes = list(range(4, 26))
    size_to_markers: dict[int, list[str]] = {}
    marker_order: list[str] = []

    for size in sizes:
        path = panel_dir / f"pathway_{size}p.csv"
        markers = [clean_marker(line.strip()) for line in path.read_text().splitlines() if line.strip()]
        size_to_markers[size] = markers
        for marker in markers:
            if marker not in marker_order:
                marker_order.append(marker)

    matrix = pd.DataFrame(0, index=marker_order, columns=sizes, dtype=int)
    for size, markers in size_to_markers.items():
        matrix.loc[markers, size] = 1

    fig_h = max(5.5, 0.28 * len(marker_order) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette(["#F7F7F7", "#1B9E77"], as_cmap=True),
        cbar=False,
        linewidths=0.4,
        linecolor="#DDDDDD",
        ax=ax,
    )
    ax.set_title("Pathway Route Marker Inclusion Heatmap", fontsize=14, weight="bold")
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Marker")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=8)
    save_both(fig, out_dir, "fig02_pathway_marker_heatmap")


def fig03_all_models_comparison(agg: pd.DataFrame, ens: pd.DataFrame, out_dir: Path) -> None:
    base = agg.loc[agg["order"] == "pathway"].copy()
    models = ["LR_EN", "LinSVM_cal", "RF", "XGBoost"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, model in zip(axes, models):
        subset = base.loc[base["model"] == model].sort_values("panel_size")
        ax.plot(
            subset["panel_size"],
            subset["summary_auroc_mean"],
            color=MODEL_COLORS[model],
            linewidth=2.2,
        )
        ax.fill_between(
            subset["panel_size"],
            subset["summary_auroc_ci95_lo"],
            subset["summary_auroc_ci95_hi"],
            color=MODEL_COLORS[model],
            alpha=0.15,
        )
        ax.set_title(MODEL_LABELS[model], fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks([4, 7, 10, 15, 20, 25])

    ensemble = ens.loc[ens["order"] == "pathway"].sort_values("panel_size")
    inset = fig.add_axes([0.68, 0.12, 0.24, 0.18])
    inset.plot(
        ensemble["panel_size"],
        ensemble["pooled_test_auroc"],
        color=MODEL_COLORS["ENSEMBLE"],
        linewidth=2.0,
    )
    inset.set_title("Ensemble", fontsize=10, weight="bold")
    inset.set_xticks([4, 10, 20, 25])
    inset.tick_params(labelsize=8)
    inset.grid(axis="y", alpha=0.15)

    fig.suptitle("All-Model Panel-Size Comparison (Pathway Order)", fontsize=15, weight="bold")
    fig.supxlabel("Panel size")
    fig.supylabel("AUROC")
    sns.despine(fig=fig)
    save_both(fig, out_dir, "fig03_all_models_panel_size_comparison")


def fig04_route_map(panel_dir: Path, out_dir: Path) -> None:
    route_rows = []
    max_size = 25
    orders = ["rra", "importance", "pathway"]
    for order in orders:
        markers = [clean_marker(line.strip()) for line in (panel_dir / f"{order}_{max_size}p.csv").read_text().splitlines() if line.strip()]
        route_rows.append(markers)

    route_df = pd.DataFrame(route_rows, index=[ORDER_LABELS[o] for o in orders], columns=list(range(1, max_size + 1)))
    cat_map = {marker: idx for idx, marker in enumerate(pd.unique(route_df.to_numpy().ravel()), start=1)}
    value_df = route_df.replace(cat_map)

    fig, ax = plt.subplots(figsize=(16, 3.8))
    sns.heatmap(
        value_df,
        cmap=sns.color_palette("Spectral", as_cmap=True),
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=ax,
    )
    for r in range(route_df.shape[0]):
        for c in range(route_df.shape[1]):
            ax.text(c + 0.5, r + 0.5, route_df.iat[r, c], ha="center", va="center", fontsize=7, color="black")

    ax.set_title("Protein Addition Routes by Order", fontsize=14, weight="bold")
    ax.set_xlabel("Entry position")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    save_both(fig, out_dir, "fig04_order_route_map")


def fig05_model_size_heatmap(agg: pd.DataFrame, out_dir: Path) -> None:
    base = agg.pivot_table(
        index="model",
        columns="panel_size",
        values="pooled_test_auroc",
        aggfunc="mean",
    ).loc[["LR_EN", "LinSVM_cal", "RF", "XGBoost"]]
    base.index = [MODEL_LABELS[m] for m in base.index]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(base, cmap="YlGnBu", linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title("Model-by-Size AUROC Heatmap Across All Orders", fontsize=14, weight="bold")
    ax.set_xlabel("Panel size")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0)
    save_both(fig, out_dir, "fig05_model_by_size_heatmap")


def write_manifest(out_dir: Path) -> None:
    manifest = """Generated figures
- fig01_svm_panel_tendency
- fig02_pathway_marker_heatmap
- fig03_all_models_panel_size_comparison
- fig04_order_route_map
- fig05_model_by_size_heatmap
"""
    (out_dir / "MANIFEST.txt").write_text(manifest)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    panel_dir = Path(args.panel_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    agg, ens = load_data(results_dir)

    fig01_svm_panel_tendency(agg, out_dir)
    fig02_svm_marker_heatmap(panel_dir, out_dir)
    fig03_all_models_comparison(agg, ens, out_dir)
    fig04_route_map(panel_dir, out_dir)
    fig05_model_size_heatmap(agg, out_dir)
    write_manifest(out_dir)


if __name__ == "__main__":
    main()
