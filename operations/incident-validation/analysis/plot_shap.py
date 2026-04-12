#!/usr/bin/env python3
"""
plot_shap.py

Publication-ready SHAP figures for the incident validation analysis.

Reads pre-computed SHAP values from results directories (one per model) and
generates three figures:

  fig9_shap_beeswarm.png / .pdf  -- Three-panel manual beeswarm, top 20 features
  fig10_shap_bar.png / .pdf      -- Grouped bar, mean |SHAP|, top 15 features
  fig11_shap_dependence.png / .pdf -- Dependence plots, 5 core proteins, LR_EN

Usage:
  cd cel-risk
  python operations/incident-validation/analysis/plot_shap.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CEL_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "out"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_ROOT = CEL_ROOT / "results" / "incident-validation" / "lr"

MODEL_DIRS: dict[str, Path] = {
    "LR_EN": RESULTS_ROOT / "LR_EN",
    "SVM_L1": RESULTS_ROOT / "SVM_L1",
    "SVM_L2": RESULTS_ROOT / "SVM_L2",
}

MODEL_LABELS: dict[str, str] = {
    "LR_EN": "Logistic (EN)",
    "SVM_L1": "LinSVM (L1)",
    "SVM_L2": "LinSVM (L2)",
}

MODEL_COLORS: dict[str, str] = {
    "LR_EN": "#4C78A8",
    "SVM_L1": "#1B9E77",
    "SVM_L2": "#E7298A",
}

# Core proteins for dependence plots (fig11)
CORE_FEATURES = ["TGM2", "CKMT1A_CKMT1B", "MUC2", "CLEC4G", "NOS2"]

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def clean_name(col: str) -> str:
    """Strip '_resid' suffix and uppercase the protein name."""
    return col.replace("_resid", "").upper()


def load_model_data(model_id: str) -> dict:
    """
    Load SHAP values, expected value, and feature matrix for a model.

    Returns a dict with keys:
        shap_arr    -- np.ndarray (n_samples, n_features)
        X_arr       -- np.ndarray (n_samples, n_features), feature values
        feature_names -- list[str], display names (cleaned)
        raw_cols    -- list[str], original column names (with _resid suffix)
        expected_value -- float
    """
    model_dir = MODEL_DIRS[model_id]

    shap_path = model_dir / "shap_values.csv"
    xtest_path = model_dir / "X_test.csv"
    ev_path = model_dir / "shap_expected_value.txt"

    for p in (shap_path, xtest_path, ev_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    logger.info("[%s] Loading SHAP values from %s", model_id, shap_path)
    shap_df = pd.read_csv(shap_path)

    logger.info("[%s] Loading X_test from %s", model_id, xtest_path)
    xtest_df = pd.read_csv(xtest_path)

    expected_value = float(ev_path.read_text().strip())
    logger.info("[%s] Expected value: %.6f", model_id, expected_value)

    # Drop non-feature columns from X_test
    feature_cols = [c for c in xtest_df.columns if c not in ("eid", "y_true")]
    X_df = xtest_df[feature_cols]

    # Verify column alignment: SHAP and X_test must have the same feature order
    if list(shap_df.columns) != list(X_df.columns):
        raise ValueError(
            f"[{model_id}] Column mismatch between shap_values.csv and X_test.csv. "
            f"SHAP cols: {list(shap_df.columns)[:5]}... "
            f"X_test cols: {list(X_df.columns)[:5]}..."
        )

    shap_arr = shap_df.to_numpy(dtype=float)
    X_arr = X_df.to_numpy(dtype=float)

    raw_cols = list(shap_df.columns)
    feature_names = [clean_name(c) for c in raw_cols]

    logger.info(
        "[%s] Loaded: %d samples x %d features",
        model_id, shap_arr.shape[0], shap_arr.shape[1],
    )

    return {
        "shap_arr": shap_arr,
        "X_arr": X_arr,
        "feature_names": feature_names,
        "raw_cols": raw_cols,
        "expected_value": expected_value,
    }


def load_all_models() -> dict[str, dict]:
    """Load data for all three models."""
    data: dict[str, dict] = {}
    for model_id in MODEL_DIRS:
        data[model_id] = load_model_data(model_id)
    return data


# ---------------------------------------------------------------------------
# SHAP utility
# ---------------------------------------------------------------------------


def mean_abs_shap(shap_arr: np.ndarray) -> np.ndarray:
    """Return mean |SHAP| per feature (shape: n_features)."""
    return np.abs(shap_arr).mean(axis=0)


def top_k_indices(importance: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k features, sorted descending by importance."""
    return np.argsort(importance)[::-1][:k]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure as both PNG (300 DPI) and PDF."""
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Fig 9 -- SHAP beeswarm (manual implementation, 1x3 grid)
# ---------------------------------------------------------------------------


def _beeswarm_panel(
    ax: plt.Axes,
    shap_vals: np.ndarray,
    feature_vals: np.ndarray,
    feature_names: list[str],
    top_k: int = 20,
    title: str = "",
    color_map: str = "coolwarm",
    beeswarm_spread: float = 0.3,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Render a single beeswarm panel on `ax`.

    Features are ranked by mean |SHAP|; top_k are shown.
    Each row is a feature; x = SHAP value; y = feature rank + jitter.
    Points are colored by normalized feature value (low=blue, high=red).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    importance = mean_abs_shap(shap_vals)
    top_idx = top_k_indices(importance, top_k)  # descending: index 0 = most important

    cmap = plt.get_cmap(color_map)

    # Plot from bottom (least important) to top (most important) so that the
    # y-axis reads top-to-bottom with rank 1 at the top.
    n_features = len(top_idx)

    for rank, feat_idx in enumerate(reversed(top_idx)):
        # y position: rank 0 = bottom = least important
        y_center = rank

        s_vals = shap_vals[:, feat_idx]
        f_vals = feature_vals[:, feat_idx]

        # Normalize feature values to [0, 1] for coloring
        f_min, f_max = f_vals.min(), f_vals.max()
        if f_max > f_min:
            f_norm = (f_vals - f_min) / (f_max - f_min)
        else:
            f_norm = np.full_like(f_vals, 0.5)

        colors = cmap(f_norm)

        jitter = rng.uniform(-beeswarm_spread, beeswarm_spread, size=len(s_vals))
        y_vals = y_center + jitter

        ax.scatter(
            s_vals,
            y_vals,
            c=colors,
            s=4,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )

    # y-axis labels: bottom=least important, top=most important
    yticks = list(range(n_features))
    ylabels = [feature_names[i] for i in reversed(top_idx)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(-0.8, n_features - 0.2)


def fig9_shap_beeswarm(all_data: dict[str, dict]) -> None:
    """Three-panel manual beeswarm, one per model, top 20 features."""
    logger.info("Generating fig9_shap_beeswarm ...")

    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    rng = np.random.default_rng(42)

    for ax, (model_id, label) in zip(axes, MODEL_LABELS.items()):
        d = all_data[model_id]
        _beeswarm_panel(
            ax=ax,
            shap_vals=d["shap_arr"],
            feature_vals=d["X_arr"],
            feature_names=d["feature_names"],
            top_k=20,
            title=label,
            rng=rng,
        )

        # Colorbar hint on the rightmost panel only
        if ax is axes[-1]:
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Low", "High"])
            cbar.set_label("Feature value", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    caption = (
        "SHAP values on locked test set (N=8,805) | LinearExplainer | "
        "Top 20 by mean |SHAP|"
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=8, color="#444444")
    fig.tight_layout()
    _save(fig, "fig9_shap_beeswarm")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 10 -- grouped bar chart, mean |SHAP|, top 15 features
# ---------------------------------------------------------------------------


def fig10_shap_bar(all_data: dict[str, dict]) -> None:
    """Grouped bar chart of mean |SHAP| for top 15 features, 3 models side by side."""
    logger.info("Generating fig10_shap_bar ...")

    # Determine top-15 features ordered by LR_EN mean |SHAP|
    lr_data = all_data["LR_EN"]
    lr_importance = mean_abs_shap(lr_data["shap_arr"])
    top15_idx = top_k_indices(lr_importance, 15)  # descending; idx 0 = most important

    # Feature names from LR_EN (same protein panel across models by construction)
    feature_names = [lr_data["feature_names"][i] for i in top15_idx]

    # mean |SHAP| for each model at these feature positions
    # We need to map feature names back to each model's column order
    # (protein panels may differ across models; check and warn if so)
    model_ids = list(MODEL_LABELS.keys())
    importance_table: dict[str, list[float]] = {}

    for model_id in model_ids:
        d = all_data[model_id]
        model_importance = mean_abs_shap(d["shap_arr"])
        vals: list[float] = []
        for feat_idx in top15_idx:
            # Use the raw column name as the key to resolve cross-model
            raw_col = lr_data["raw_cols"][feat_idx]
            if raw_col in d["raw_cols"]:
                mi = d["raw_cols"].index(raw_col)
                vals.append(float(model_importance[mi]))
            else:
                logger.warning(
                    "Feature '%s' not found in %s panel; using 0.0", raw_col, model_id
                )
                vals.append(0.0)
        importance_table[model_id] = vals

    n_features = len(feature_names)
    n_models = len(model_ids)
    bar_width = 0.22
    group_spacing = 1.0
    x = np.arange(n_features) * group_spacing

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model_id in enumerate(model_ids):
        offset = (i - (n_models - 1) / 2) * bar_width
        vals = importance_table[model_id]
        # Plot with y=features (horizontal)
        bars = ax.barh(
            x + offset,
            vals,
            height=bar_width * 0.9,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
            alpha=0.85,
        )

    ax.set_yticks(x)
    ax.set_yticklabels(feature_names, fontsize=9)

    # Invert y so that rank 1 (LR_EN top feature) is at the top
    ax.invert_yaxis()

    ax.set_xlabel("Mean |SHAP| value", fontsize=10)
    ax.set_title("Mean |SHAP| feature importance", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="x", labelsize=8)

    caption = "Mean |SHAP| importance across 3 models | Locked test set (N=8,805)"
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="#444444")

    fig.tight_layout()
    _save(fig, "fig10_shap_bar")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 11 -- SHAP dependence plots, 5 core proteins, LR_EN
# ---------------------------------------------------------------------------


def _dependence_panel(
    ax: plt.Axes,
    shap_vals: np.ndarray,
    feature_vals: np.ndarray,
    feat_name: str,
    color_map: str = "coolwarm",
    lowess_frac: float = 0.3,
) -> None:
    """
    Render a single dependence plot: x=feature value, y=SHAP value.
    Points colored by (normalized) feature value. LOWESS trend overlay.
    """
    f_min, f_max = feature_vals.min(), feature_vals.max()
    if f_max > f_min:
        f_norm = (feature_vals - f_min) / (f_max - f_min)
    else:
        f_norm = np.full_like(feature_vals, 0.5)

    cmap = plt.get_cmap(color_map)
    colors = cmap(f_norm)

    ax.scatter(
        feature_vals,
        shap_vals,
        c=colors,
        s=6,
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )

    # LOWESS trend
    if len(feature_vals) > 20:
        sort_idx = np.argsort(feature_vals)
        x_sorted = feature_vals[sort_idx]
        y_sorted = shap_vals[sort_idx]
        smoothed = lowess(y_sorted, x_sorted, frac=lowess_frac, return_sorted=True)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color="black",
            linewidth=1.5,
            alpha=0.85,
            zorder=5,
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel(feat_name, fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title(feat_name, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def fig11_shap_dependence(all_data: dict[str, dict]) -> None:
    """Dependence plots for top 5 core proteins, LR_EN model."""
    logger.info("Generating fig11_shap_dependence ...")

    d = all_data["LR_EN"]
    shap_arr = d["shap_arr"]
    X_arr = d["X_arr"]
    feature_names = d["feature_names"]
    raw_cols = d["raw_cols"]

    # Resolve each core feature to its column index
    # CORE_FEATURES uses clean names (uppercase, no _resid)
    resolved: list[tuple[str, int]] = []
    for target in CORE_FEATURES:
        target_upper = target.upper()
        # Try clean name match first
        found = None
        for i, fn in enumerate(feature_names):
            if fn.upper() == target_upper:
                found = i
                break
        if found is None:
            # Try raw column
            for i, rc in enumerate(raw_cols):
                if clean_name(rc).upper() == target_upper:
                    found = i
                    break
        if found is None:
            logger.warning(
                "Core feature '%s' not found in LR_EN panel; skipping.", target
            )
        else:
            resolved.append((feature_names[found], found))
            logger.info("  Resolved '%s' -> col idx %d ('%s')", target, found, feature_names[found])

    n_panels = len(resolved)
    if n_panels == 0:
        logger.error("No core features resolved; skipping fig11.")
        return

    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes_flat = np.array(axes).flatten()

    for ax_idx, (feat_name, feat_col) in enumerate(resolved):
        ax = axes_flat[ax_idx]
        _dependence_panel(
            ax=ax,
            shap_vals=shap_arr[:, feat_col],
            feature_vals=X_arr[:, feat_col],
            feat_name=feat_name,
        )

    # Hide unused panels
    for ax_idx in range(n_panels, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    caption = (
        "SHAP dependence | LR_EN model | Locked test set | "
        "Feature values are age+sex residualized"
    )
    fig.text(0.5, -0.01, caption, ha="center", va="top", fontsize=8, color="#444444")
    fig.tight_layout()
    _save(fig, "fig11_shap_dependence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUT_DIR)

    # Load pre-computed SHAP artifacts for all 3 models
    all_data = load_all_models()

    # Generate figures
    fig9_shap_beeswarm(all_data)
    fig10_shap_bar(all_data)
    fig11_shap_dependence(all_data)

    logger.info("Done. Figures written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
