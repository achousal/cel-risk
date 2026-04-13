#!/usr/bin/env python3
"""Generate saturation-curve plots from ``saturation_results.csv`` artifacts.

Reads the panel-size sweep CSV produced by ``compute_saturation.py`` and draws:
  - Per-model saturation curve (CV AUPRC and test AUPRC vs. panel size)
  - Cross-model overlay if multiple CSVs are available

``ced_ml.plotting.panel_curve.plot_pareto_curve`` hardcodes AUROC axis labels
and annotations, so this script renders matplotlib directly rather than
shoe-horning AUPRC values into an AUROC-shaped API.

Usage (from project root):
    python operations/incident-validation/analysis/generate_saturation_plots_from_artifacts.py
    python operations/incident-validation/analysis/generate_saturation_plots_from_artifacts.py --models LR_EN

Inputs (searched per model, first hit wins):
    operations/incident-validation/analysis/out/saturation_results.csv  (LR_EN default)
    results/incident-validation/lr/<model>/saturation_results.csv

Outputs (under ``<out>/<model>/``):
    saturation.pdf
and if ``>=2`` models present:
    <out>/comparison/saturation_comparison.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CEL_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_saturation_plots")


DEFAULT_MODEL_IDS = ["LR_EN", "SVM_L1", "SVM_L2"]
MODEL_COLORS = {"LR_EN": "#4C78A8", "SVM_L1": "#E7298A", "SVM_L2": "#66A61E"}
REQUIRED_COLS = {"panel_size", "mean_cv_auprc"}


def default_out_dir() -> Path:
    return CEL_ROOT / "results" / "incident-validation" / "figures"


def find_saturation_csv(model_id: str) -> Path | None:
    candidates = [
        CEL_ROOT / "results" / "incident-validation" / "lr" / model_id / "saturation_results.csv",
        CEL_ROOT / "operations" / "incident-validation" / "analysis" / "out" / "saturation_results.csv"
        if model_id == "LR_EN"
        else None,
    ]
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


def load_saturation(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
    return df.sort_values("panel_size").reset_index(drop=True)


def plot_single_saturation(model_id: str, df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    color = MODEL_COLORS.get(model_id, "#4C78A8")
    sizes = df["panel_size"].to_numpy()
    cv_mean = df["mean_cv_auprc"].to_numpy()
    cv_std = df["std_cv_auprc"].to_numpy() if "std_cv_auprc" in df.columns else np.zeros_like(cv_mean)

    ax.plot(sizes, cv_mean, "o-", color=color, lw=2, label=f"{model_id} CV AUPRC")
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, color=color, alpha=0.2)

    if "test_auprc" in df.columns:
        test_mean = df["test_auprc"].to_numpy()
        ax.plot(sizes, test_mean, "s--", color="#E7298A", lw=1.5, label="Test AUPRC")
        if {"test_auprc_lo", "test_auprc_hi"}.issubset(df.columns):
            ax.fill_between(
                sizes,
                df["test_auprc_lo"].to_numpy(),
                df["test_auprc_hi"].to_numpy(),
                color="#E7298A",
                alpha=0.15,
            )

    ax.set_xlabel("Panel size (number of proteins)")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"{model_id} -- Saturation curve")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [%s] saturation -> %s", model_id, out_path)


def plot_overlay(model_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_id, df in model_dfs.items():
        color = MODEL_COLORS.get(model_id, None)
        sizes = df["panel_size"].to_numpy()
        cv_mean = df["mean_cv_auprc"].to_numpy()
        cv_std = df["std_cv_auprc"].to_numpy() if "std_cv_auprc" in df.columns else np.zeros_like(cv_mean)
        ax.plot(sizes, cv_mean, "o-", color=color, lw=2, label=model_id)
        ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, color=color, alpha=0.15)

    ax.set_xlabel("Panel size (number of proteins)")
    ax.set_ylabel("CV AUPRC")
    ax.set_title("Saturation curves (cross-model)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("  overlay saturation -> %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_IDS)
    parser.add_argument("--out", type=Path, default=default_out_dir())
    args = parser.parse_args()

    logger.info("CEL_ROOT: %s", CEL_ROOT)
    logger.info("Output root: %s", args.out)
    logger.info("Models: %s", args.models)

    loaded: dict[str, pd.DataFrame] = {}
    for model_id in args.models:
        csv = find_saturation_csv(model_id)
        if csv is None:
            logger.warning("Skipping %s: no saturation_results.csv found", model_id)
            continue
        try:
            loaded[model_id] = load_saturation(csv)
            logger.info("Loaded %s from %s (%d rows)", model_id, csv, len(loaded[model_id]))
        except Exception as exc:
            logger.warning("Skipping %s: %s", model_id, exc)

    if not loaded:
        logger.error("No saturation CSVs loaded; run compute_saturation.py first")
        return 1

    for model_id, df in loaded.items():
        out_path = args.out / model_id / "saturation.pdf"
        plot_single_saturation(model_id, df, out_path)

    if len(loaded) >= 2:
        plot_overlay(loaded, args.out / "comparison" / "saturation_comparison.pdf")

    n_files = len(loaded) + (1 if len(loaded) >= 2 else 0)
    logger.info("Done. %d saturation figures written under %s", n_files, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
