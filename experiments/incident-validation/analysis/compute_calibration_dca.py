#!/usr/bin/env python3
"""
compute_calibration_dca.py

Incident validation calibration and DCA analysis for 3 models:
  LR_EN  -- ElasticNet logistic regression
  SVM_L1 -- LinearSVC L1 (sparse)
  SVM_L2 -- LinearSVC L2 (dense)

Outputs (all written to analysis/out/):
  fig6_calibration.png / .pdf  -- Reliability diagram, LOESS smoothed, 3-model overlay
  fig7_dca.png / .pdf          -- Decision curve analysis, 3 models + baselines
  calibration_metrics.csv      -- ECE, ICI, Brier, intercept, slope, Spiegelhalter

Usage:
  cd cel-risk
  python experiments/optimal-setup/incident-validation/analysis/compute_calibration_dca.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

from ced_ml.metrics.dca import decision_curve_analysis
from ced_ml.models.calibration import (
    adaptive_expected_calibration_error,
    brier_score_decomposition,
    calibration_intercept_slope,
    expected_calibration_error,
    integrated_calibration_index,
    spiegelhalter_z_test,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logging.getLogger().handlers[0].stream = open(
    sys.stderr.fileno(), "w", buffering=1, closefd=False
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_ROOT = CEL_ROOT / "results" / "incident-validation" / "lr"
OUT_DIR = Path(__file__).resolve().parent / "out"

MODEL_DIRS = {
    "LR_EN": RESULTS_ROOT / "LR_EN",
    "SVM_L1": RESULTS_ROOT / "SVM_L1",
    "SVM_L2": RESULTS_ROOT / "SVM_L2",
}
MODEL_LABELS = {
    "LR_EN": "Logistic (EN)",
    "SVM_L1": "LinSVM (L1)",
    "SVM_L2": "LinSVM (L2)",
}
MODEL_COLORS = {
    "LR_EN": "#4C78A8",
    "SVM_L1": "#1B9E77",
    "SVM_L2": "#E7298A",
}

# Prevalence: 29/8805 incident cases in test set
PREVALENCE = 29 / 8805  # ~0.00329

# DCA threshold range: clinically relevant for ~1:300 prevalence
DCA_MIN_THRESHOLD = 0.001
DCA_MAX_THRESHOLD = 0.05
DCA_THRESHOLD_STEP = 0.0005

# Calibration curve x-axis cap: show up to 15% (covers 99th pct of predictions)
CALIB_X_MAX = 0.15

# Bins for raw reliability diagram (quantile-based)
N_CALIB_BINS = 12

# Minimum unique predictions for LOESS smoothing
_MIN_UNIQUE = 20

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_predictions() -> dict[str, pd.DataFrame]:
    """Load test_predictions.csv for all 3 models."""
    preds: dict[str, pd.DataFrame] = {}
    for key, model_dir in MODEL_DIRS.items():
        csv_path = model_dir / "test_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing test_predictions.csv: {csv_path}")
        df = pd.read_csv(csv_path)
        required = {"y_true", "y_prob"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")
        preds[key] = df
        n_pos = int(df["y_true"].sum())
        n_total = len(df)
        logger.info(
            "Loaded %s: n=%d, positives=%d, prevalence=%.4f",
            key,
            n_total,
            n_pos,
            n_pos / n_total,
        )
    return preds


# ---------------------------------------------------------------------------
# Calibration curve utilities
# ---------------------------------------------------------------------------


def _loess_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_points: np.ndarray,
) -> np.ndarray | None:
    """
    Fit a cubic spline smoother on (y_pred, y_true) and evaluate at eval_points.

    Uses the same approach as ced_ml._loess_calibration_errors: UnivariateSpline
    with s=len(y) for a smooth (non-interpolating) fit.

    Returns array of smoothed P(Y=1 | predicted=x) at eval_points, or None on failure.
    """
    n_unique = len(np.unique(y_pred))
    if n_unique < _MIN_UNIQUE:
        logger.warning("Too few unique prediction values (%d) for LOESS smoothing.", n_unique)
        return None

    sort_idx = np.argsort(y_pred)
    p_sorted = y_pred[sort_idx]
    y_sorted = y_true[sort_idx].astype(float)

    try:
        spline = UnivariateSpline(p_sorted, y_sorted, k=3, s=len(y_sorted), ext=3)
        curve = np.clip(spline(eval_points), 0.0, 1.0)
        return curve
    except Exception as exc:
        logger.warning("Spline fitting failed: %s", exc)
        return None


def _quantile_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = N_CALIB_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute quantile-based (adaptive) reliability diagram bins.

    Returns (bin_centers, observed_freq, bin_counts).
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_pred, quantiles)
    # Deduplicate edges (can occur when many predictions are identical)
    bin_edges = np.unique(bin_edges)

    centers, observed, counts = [], [], []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        centers.append(float(np.mean(y_pred[mask])))
        observed.append(float(np.mean(y_true[mask])))
        counts.append(n_bin)

    return np.array(centers), np.array(observed), np.array(counts)


# ---------------------------------------------------------------------------
# Figure 6 -- Calibration curves
# ---------------------------------------------------------------------------


def plot_calibration(
    preds: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Reliability diagram: LOESS smooth + quantile bins for all 3 models."""
    logger.info("Plotting calibration curves (fig6_calibration).")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

    # Perfect calibration diagonal
    diag_x = np.linspace(0, CALIB_X_MAX, 200)
    ax.plot(diag_x, diag_x, color="gray", linestyle="--", linewidth=1.2,
            label="Perfect calibration", zorder=1)

    eval_x = np.linspace(0.0, CALIB_X_MAX, 500)

    rug_offsets = {"LR_EN": -0.0010, "SVM_L1": -0.0020, "SVM_L2": -0.0030}

    for key in ("LR_EN", "SVM_L1", "SVM_L2"):
        df = preds[key]
        y = df["y_true"].to_numpy().astype(float)
        p = df["y_prob"].to_numpy().astype(float)
        color = MODEL_COLORS[key]
        label = MODEL_LABELS[key]

        # LOESS smooth curve
        smooth = _loess_calibration_curve(y, p, eval_x)
        if smooth is not None:
            ax.plot(eval_x, smooth, color=color, linewidth=2.0, label=label, zorder=3)
        else:
            logger.warning("LOESS failed for %s; falling back to bin means.", key)

        # Quantile-binned dots (shown faintly for context)
        bin_centers, bin_obs, bin_counts = _quantile_bins(y, p, n_bins=N_CALIB_BINS)
        # Filter to x-axis range
        in_range = bin_centers <= CALIB_X_MAX
        ax.scatter(
            bin_centers[in_range],
            bin_obs[in_range],
            color=color,
            s=28,
            alpha=0.55,
            zorder=4,
            linewidths=0,
        )

        # Rug for prediction distribution (events only, at bottom)
        event_probs = p[y == 1]
        event_probs_clip = event_probs[event_probs <= CALIB_X_MAX]
        y_rug = np.full_like(event_probs_clip, rug_offsets[key])
        ax.scatter(event_probs_clip, y_rug, marker="|", color=color,
                   s=30, alpha=0.7, linewidths=1.0, zorder=2)

    ax.set_xlim(0, CALIB_X_MAX)
    # Y-axis: show a bit above the highest observed rate
    all_obs = []
    for df in preds.values():
        y = df["y_true"].to_numpy().astype(float)
        p = df["y_prob"].to_numpy().astype(float)
        _, obs, _ = _quantile_bins(y, p, n_bins=N_CALIB_BINS)
        all_obs.extend(obs.tolist())
    y_max = max(max(all_obs) * 1.15, CALIB_X_MAX * 1.1)
    ax.set_ylim(-0.004, y_max)

    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed frequency", fontsize=12)
    ax.set_title("Calibration curves (incident validation)", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(False)

    legend = ax.legend(
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
        fontsize=10, loc="upper left",
    )
    legend.get_frame().set_linewidth(0.8)

    fig.text(
        0.01, -0.02,
        "Locked test set: 29 incident + 8,776 controls | LOESS smooth (k=3) | "
        "Quantile-binned scatter | Rug = individual predictions",
        fontsize=8, color="grey", ha="left", va="top",
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = out_dir / f"fig6_calibration.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 -- DCA
# ---------------------------------------------------------------------------


def plot_dca(
    preds: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Decision curve analysis for all 3 models + treat-all + treat-none."""
    logger.info("Plotting DCA (fig7_dca).")

    thresholds = np.arange(DCA_MIN_THRESHOLD, DCA_MAX_THRESHOLD + 1e-9, DCA_THRESHOLD_STEP)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    dca_results: dict[str, pd.DataFrame] = {}
    for key in ("LR_EN", "SVM_L1", "SVM_L2"):
        df = preds[key]
        y = df["y_true"].to_numpy().astype(int)
        p = df["y_prob"].to_numpy().astype(float)
        dca_df = decision_curve_analysis(
            y_true=y,
            y_pred_prob=p,
            thresholds=thresholds,
        )
        dca_results[key] = dca_df
        logger.info(
            "%s DCA: %d threshold points, NB range [%.5f, %.5f]",
            key,
            len(dca_df),
            dca_df["net_benefit_model"].min(),
            dca_df["net_benefit_model"].max(),
        )

    # Treat-all and treat-none from first model (same y_true for all)
    first_dca = next(iter(dca_results.values()))
    ax.plot(
        first_dca["threshold"],
        first_dca["net_benefit_all"],
        color="#888888",
        linewidth=1.5,
        linestyle=":",
        label="Treat all",
        zorder=2,
    )
    ax.axhline(
        0,
        color="#aaaaaa",
        linewidth=1.2,
        linestyle="-.",
        label="Treat none",
        zorder=2,
    )

    for key in ("LR_EN", "SVM_L1", "SVM_L2"):
        dca_df = dca_results[key]
        ax.plot(
            dca_df["threshold"],
            dca_df["net_benefit_model"],
            color=MODEL_COLORS[key],
            linewidth=2.0,
            label=MODEL_LABELS[key],
            zorder=3,
        )

    ax.set_xlim(DCA_MIN_THRESHOLD, DCA_MAX_THRESHOLD)

    # Y-axis: clip below a sensible floor to avoid distortion
    all_nb = np.concatenate(
        [dca_df["net_benefit_model"].to_numpy() for dca_df in dca_results.values()]
        + [first_dca["net_benefit_all"].to_numpy()]
    )
    y_min = max(float(np.nanmin(all_nb)) * 1.2, -PREVALENCE * 2)
    y_max = float(np.nanmax(all_nb)) * 1.2
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Threshold probability", fontsize=12)
    ax.set_ylabel("Net benefit", fontsize=12)
    ax.set_title("Decision curve analysis (incident validation)", fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(False)

    legend = ax.legend(
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
        fontsize=10, loc="upper right",
    )
    legend.get_frame().set_linewidth(0.8)

    fig.text(
        0.01, -0.02,
        f"Locked test set: 29 incident + 8,776 controls | Prevalence = {PREVALENCE:.4f} | "
        "Threshold range 0.001-0.05 | Vickers & Elkin (2006)",
        fontsize=8, color="grey", ha="left", va="top",
    )

    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = out_dir / f"fig7_dca.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out_path)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration metrics table
# ---------------------------------------------------------------------------


def compute_calibration_metrics(
    preds: dict[str, pd.DataFrame],
    out_dir: Path,
) -> pd.DataFrame:
    """Compute and save calibration_metrics.csv."""
    logger.info("Computing calibration metrics.")

    rows = []
    for key in ("LR_EN", "SVM_L1", "SVM_L2"):
        df = preds[key]
        y = df["y_true"].to_numpy().astype(int)
        p = df["y_prob"].to_numpy().astype(float)

        ece = expected_calibration_error(y, p, n_bins=10)
        aece = adaptive_expected_calibration_error(y, p)
        ici = integrated_calibration_index(y, p)
        brier = brier_score_decomposition(y, p)
        cal = calibration_intercept_slope(y, p)
        spieg = spiegelhalter_z_test(y, p)

        row = {
            "model": MODEL_LABELS[key],
            "model_key": key,
            "ece": ece,
            "adaptive_ece": aece,
            "ici": ici,
            "brier_score": brier.brier_score,
            "brier_reliability": brier.reliability,
            "brier_resolution": brier.resolution,
            "brier_uncertainty": brier.uncertainty,
            "calibration_intercept": cal.intercept,
            "calibration_slope": cal.slope,
            "spiegelhalter_z": spieg.z_statistic,
            "spiegelhalter_p": spieg.p_value,
        }
        rows.append(row)

        logger.info(
            "%s -- ECE=%.4f, ICI=%.4f, Brier=%.4f, intercept=%.3f, slope=%.3f, "
            "Spiegelhalter z=%.3f (p=%.3f)",
            key,
            ece,
            ici,
            brier.brier_score,
            cal.intercept,
            cal.slope,
            spieg.z_statistic if np.isfinite(spieg.z_statistic) else float("nan"),
            spieg.p_value if np.isfinite(spieg.p_value) else float("nan"),
        )

    metrics_df = pd.DataFrame(rows)

    csv_path = out_dir / "calibration_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Saved calibration metrics to %s", csv_path)

    return metrics_df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(metrics_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout via logging."""
    display_cols = [
        "model",
        "ece",
        "ici",
        "brier_score",
        "brier_reliability",
        "brier_resolution",
        "calibration_intercept",
        "calibration_slope",
        "spiegelhalter_z",
        "spiegelhalter_p",
    ]
    fmt = metrics_df[display_cols].copy()

    float_cols = display_cols[1:]
    for col in float_cols:
        fmt[col] = fmt[col].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "NA")

    col_widths = {col: max(len(col), fmt[col].astype(str).str.len().max()) for col in display_cols}

    header = "  ".join(col.ljust(col_widths[col]) for col in display_cols)
    sep = "  ".join("-" * col_widths[col] for col in display_cols)
    logger.info("Calibration metrics summary:")
    logger.info(header)
    logger.info(sep)
    for _, row in fmt.iterrows():
        line = "  ".join(str(row[col]).ljust(col_widths[col]) for col in display_cols)
        logger.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("CEL_ROOT: %s", CEL_ROOT)
    logger.info("Output directory: %s", OUT_DIR)
    logger.info("Prevalence: %.5f (%d / 8805)", PREVALENCE, round(PREVALENCE * 8805))

    preds = load_predictions()

    plot_calibration(preds, OUT_DIR)
    plot_dca(preds, OUT_DIR)
    metrics_df = compute_calibration_metrics(preds, OUT_DIR)
    print_summary(metrics_df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
