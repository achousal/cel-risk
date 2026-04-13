#!/usr/bin/env python3
"""Generate the full cel-risk Bucket 1 plot suite for completed incident-validation runs.

Reads per-model artifacts from ``results/incident-validation/lr/{model}/`` and
calls ``ced_ml.plotting.*`` directly to produce single-model plots plus cross-
model overlays. No retraining, no refitting — pure post-processing of existing
``test_predictions.csv`` and ``strategy_comparison.csv`` artifacts.

This is the "use ced_ml.plotting instead of reimplementing" thread from the
duplication audit, scoped to post-processing only (not a refactor of run_lr.py).

Usage (from project root):
    python operations/incident-validation/analysis/generate_plots_from_artifacts.py
    python operations/incident-validation/analysis/generate_plots_from_artifacts.py --models LR_EN SVM_L2
    python operations/incident-validation/analysis/generate_plots_from_artifacts.py --out results/incident-validation/figures

Outputs
-------
Per model (under ``<out>/<model>/``):
    roc.pdf, pr.pdf, calibration.pdf, risk_distribution.pdf, dca.pdf

Cross-model overlays (under ``<out>/comparison/``):
    roc_comparison.pdf, pr_comparison.pdf, calibration_comparison.pdf, dca_comparison.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve cel-risk root from this file's location
# operations/incident-validation/analysis/generate_plots_from_artifacts.py → parents[3] = project root
CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

# Import after sys.path is set so ced_ml is reachable from anywhere
from ced_ml.plotting.calibration import plot_calibration_curve
from ced_ml.plotting.comparison import (
    plot_calibration_comparison,
    plot_dca_comparison,
    plot_pr_comparison,
    plot_roc_comparison,
)
from ced_ml.plotting.dca import plot_dca_curve
from ced_ml.plotting.risk_dist import plot_risk_distribution
from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_plots")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DEFAULT_MODEL_IDS = ["LR_EN", "SVM_L1", "SVM_L2"]

MODEL_LABELS = {
    "LR_EN": "LR ElasticNet",
    "SVM_L1": "LinSVC L1 (calibrated)",
    "SVM_L2": "LinSVC L2 (calibrated)",
}


def default_model_dir(model_id: str) -> Path:
    return CEL_ROOT / "results" / "incident-validation" / "lr" / model_id


def default_out_dir() -> Path:
    return CEL_ROOT / "results" / "incident-validation" / "figures"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_predictions(model_dir: Path) -> pd.DataFrame:
    """Read ``test_predictions.csv`` and validate required columns.

    Expected columns: ``eid, y_true, y_prob, y_pred``. Returns the raw DataFrame.
    """
    p = model_dir / "test_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    required = {"y_true", "y_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    # Ensure numeric dtypes
    df["y_true"] = df["y_true"].astype(int)
    df["y_prob"] = df["y_prob"].astype(float)
    return df


def load_winning_combo(model_dir: Path) -> dict | None:
    """Read the best (strategy, weight_scheme, mean_auprc) row from strategy_comparison.csv.

    Returns ``None`` if the file is missing or malformed. The top row in this
    file is the best combo (run_lr.py sorts descending by mean_auprc before
    writing).
    """
    p = model_dir / "strategy_comparison.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if len(df) == 0:
            return None
        top = df.iloc[0].to_dict()
        return {
            "strategy": str(top.get("strategy", "?")),
            "weight_scheme": str(top.get("weight_scheme", "?")),
            "mean_auprc": float(top.get("mean_auprc", float("nan"))),
            "mean_auroc": float(top.get("mean_auroc", float("nan"))),
        }
    except Exception as exc:
        logger.warning("Failed to read %s: %s", p, exc)
        return None


def build_subtitle(model_id: str, combo: dict | None, n_incident: int, n_controls: int) -> str:
    """Compose a one-line subtitle summarizing the model + sample counts."""
    parts = [MODEL_LABELS.get(model_id, model_id)]
    if combo is not None:
        parts.append(f"{combo['strategy']} + {combo['weight_scheme']}")
        parts.append(f"CV AUPRC={combo['mean_auprc']:.3f}")
    parts.append(f"n={n_incident} incident / {n_controls} controls (locked test set)")
    return " | ".join(parts)


def build_meta_lines(model_id: str, combo: dict | None) -> list[str]:
    """Build the multi-line metadata footer passed to plotting functions."""
    lines = [f"Model: {MODEL_LABELS.get(model_id, model_id)} ({model_id})"]
    if combo is not None:
        lines.append(
            f"Winning combo: {combo['strategy']} + {combo['weight_scheme']} | "
            f"CV AUPRC={combo['mean_auprc']:.4f} | CV AUROC={combo['mean_auroc']:.4f}"
        )
    lines.append("Source: ced_ml.plotting via generate_plots_from_artifacts.py")
    return lines


# ---------------------------------------------------------------------------
# Single-model plot suite
# ---------------------------------------------------------------------------


def generate_single_model_plots(
    model_id: str,
    preds: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Produce ROC, PR, calibration (2x2), risk distribution, and DCA for one model.

    Returns a dict mapping plot kind to output path for everything that landed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = preds["y_true"].to_numpy()
    y_prob = preds["y_prob"].to_numpy()
    n_incident = int(y_true.sum())
    n_controls = int((y_true == 0).sum())

    combo = load_winning_combo(default_model_dir(model_id))
    subtitle = build_subtitle(model_id, combo, n_incident, n_controls)
    meta_lines = build_meta_lines(model_id, combo)
    title_prefix = MODEL_LABELS.get(model_id, model_id)

    written: dict[str, Path] = {}

    # ROC
    path = out_dir / "roc.pdf"
    logger.info("  [%s] ROC → %s", model_id, path)
    plot_roc_curve(
        y_true=y_true,
        y_pred=y_prob,
        out_path=path,
        title=f"{title_prefix} — ROC",
        subtitle=subtitle,
        meta_lines=meta_lines,
    )
    written["roc"] = path

    # PR
    path = out_dir / "pr.pdf"
    logger.info("  [%s] PR  → %s", model_id, path)
    plot_pr_curve(
        y_true=y_true,
        y_pred=y_prob,
        out_path=path,
        title=f"{title_prefix} — Precision-Recall",
        subtitle=subtitle,
        meta_lines=meta_lines,
    )
    written["pr"] = path

    # Calibration (2x2 panel)
    path = out_dir / "calibration.pdf"
    logger.info("  [%s] Cal → %s", model_id, path)
    plot_calibration_curve(
        y_true=y_true,
        y_pred=y_prob,
        out_path=path,
        title=f"{title_prefix} — Calibration",
        subtitle=subtitle,
        n_bins=10,
        meta_lines=meta_lines,
    )
    written["calibration"] = path

    # Risk distribution
    path = out_dir / "risk_distribution.pdf"
    logger.info("  [%s] Risk→ %s", model_id, path)
    plot_risk_distribution(
        y_true=y_true,
        scores=y_prob,
        out_path=path,
        title=f"{title_prefix} — Risk distribution",
        subtitle=subtitle,
        meta_lines=meta_lines,
    )
    written["risk_distribution"] = path

    # Decision curve analysis
    path = out_dir / "dca.pdf"
    logger.info("  [%s] DCA → %s", model_id, path)
    plot_dca_curve(
        y_true=y_true,
        y_pred=y_prob,
        out_path=str(path),
        title=f"{title_prefix} — Decision curve",
        subtitle=subtitle,
        max_pt=0.20,
        step=0.005,
        meta_lines=meta_lines,
    )
    written["dca"] = path

    return written


# ---------------------------------------------------------------------------
# Cross-model overlays
# ---------------------------------------------------------------------------


def generate_cross_model_overlays(
    all_preds: dict[str, pd.DataFrame],
    out_dir: Path,
) -> dict[str, Path]:
    """Produce 4 cross-model overlay figures via ced_ml.plotting.comparison."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the ModelCurveData dict expected by comparison.* functions
    models = {
        model_id: {
            "y_true": preds["y_true"].to_numpy(),
            "y_pred": preds["y_prob"].to_numpy(),
            "split_ids": None,
            "threshold_bundle": None,
        }
        for model_id, preds in all_preds.items()
    }

    # Shared subtitle across all overlays
    any_preds = next(iter(all_preds.values()))
    n_incident = int(any_preds["y_true"].sum())
    n_controls = int((any_preds["y_true"] == 0).sum())
    subtitle = (
        f"Incident validation, locked test set: "
        f"{n_incident} incident / {n_controls} controls"
    )
    meta_lines = [
        f"Models: {', '.join(sorted(all_preds.keys()))}",
        "Source: ced_ml.plotting.comparison via generate_plots_from_artifacts.py",
    ]

    written: dict[str, Path] = {}

    path = out_dir / "roc_comparison.pdf"
    logger.info("  overlay ROC → %s", path)
    plot_roc_comparison(
        models=models,
        out_path=path,
        title="Incident validation — ROC (cross-model)",
        subtitle=subtitle,
        meta_lines=meta_lines,
    )
    written["roc_comparison"] = path

    path = out_dir / "pr_comparison.pdf"
    logger.info("  overlay PR  → %s", path)
    plot_pr_comparison(
        models=models,
        out_path=path,
        title="Incident validation — Precision-Recall (cross-model)",
        subtitle=subtitle,
        meta_lines=meta_lines,
    )
    written["pr_comparison"] = path

    path = out_dir / "calibration_comparison.pdf"
    logger.info("  overlay Cal → %s", path)
    plot_calibration_comparison(
        models=models,
        out_path=path,
        title="Incident validation — Calibration (cross-model)",
        subtitle=subtitle,
        n_bins=10,
        meta_lines=meta_lines,
    )
    written["calibration_comparison"] = path

    path = out_dir / "dca_comparison.pdf"
    logger.info("  overlay DCA → %s", path)
    plot_dca_comparison(
        models=models,
        out_path=path,
        title="Incident validation — Decision curve (cross-model)",
        subtitle=subtitle,
        max_pt=0.20,
        step=0.005,
        meta_lines=meta_lines,
    )
    written["dca_comparison"] = path

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cel-risk plot suite for incident-validation runs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_IDS,
        help=f"Model IDs to process (default: {' '.join(DEFAULT_MODEL_IDS)})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out_dir(),
        help="Output directory root for figures (default: results/incident-validation/figures)",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=CEL_ROOT / "results" / "incident-validation" / "lr",
        help="Root directory containing per-model subdirs (default: results/incident-validation/lr)",
    )
    args = parser.parse_args()

    logger.info("CEL_ROOT: %s", CEL_ROOT)
    logger.info("Model root: %s", args.model_root)
    logger.info("Output root: %s", args.out)
    logger.info("Models requested: %s", args.models)

    # Load each requested model; skip with warning on failure
    all_preds: dict[str, pd.DataFrame] = {}
    for model_id in args.models:
        model_dir = args.model_root / model_id
        try:
            preds = load_predictions(model_dir)
            all_preds[model_id] = preds
            logger.info(
                "Loaded %s: %d rows (%d positive, %d negative)",
                model_id,
                len(preds),
                int(preds["y_true"].sum()),
                int((preds["y_true"] == 0).sum()),
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", model_id, exc)

    if not all_preds:
        logger.error("No models loaded; nothing to do")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # Single-model plot suite
    all_written: dict[str, dict[str, Path]] = {}
    for model_id, preds in all_preds.items():
        logger.info("Generating single-model plots for %s", model_id)
        model_out = args.out / model_id
        try:
            all_written[model_id] = generate_single_model_plots(model_id, preds, model_out)
        except Exception as exc:
            logger.error("Single-model plots failed for %s: %s", model_id, exc, exc_info=True)

    # Cross-model overlays (need ≥2 models)
    if len(all_preds) >= 2:
        logger.info("Generating cross-model overlays for %d models", len(all_preds))
        try:
            comparison_written = generate_cross_model_overlays(all_preds, args.out / "comparison")
            all_written["comparison"] = comparison_written
        except Exception as exc:
            logger.error("Cross-model overlays failed: %s", exc, exc_info=True)
    else:
        logger.info("Only one model loaded, skipping cross-model overlays")

    # Final summary
    n_files = sum(len(v) for v in all_written.values())
    logger.info("Done. %d figures written under %s", n_files, args.out)
    for section, files in all_written.items():
        logger.info("  %s:", section)
        for kind, path in files.items():
            logger.info("    %s → %s", kind, path.relative_to(CEL_ROOT))

    return 0


if __name__ == "__main__":
    sys.exit(main())
