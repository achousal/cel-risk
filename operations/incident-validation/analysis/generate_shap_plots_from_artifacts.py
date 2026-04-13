#!/usr/bin/env python3
"""Generate SHAP plots for incident-validation models from existing artifacts.

Reads per-model ``shap_values.csv`` + ``X_test.csv`` (produced by
``compute_shap_oof.py``) and calls ``ced_ml.plotting.shap_plots.*`` to render
beeswarm, bar-importance, and top-feature dependence plots. No refit, no
SHAP recomputation -- pure post-processing.

Usage (from project root):
    python operations/incident-validation/analysis/generate_shap_plots_from_artifacts.py
    python operations/incident-validation/analysis/generate_shap_plots_from_artifacts.py --models LR_EN
    python operations/incident-validation/analysis/generate_shap_plots_from_artifacts.py --top-dependence 5

Outputs (under ``<out>/<model>/shap/``):
    beeswarm.pdf
    bar_importance.pdf
    dependence_<rank>_<feature>.pdf   (one per top-K feature)

Prerequisites:
    ``compute_shap_oof.py`` must have been run first to produce
    ``shap_values.csv`` and ``X_test.csv`` in each model directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CEL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CEL_ROOT / "analysis" / "src"))

from ced_ml.plotting.shap_plots import (
    plot_bar_importance,
    plot_beeswarm,
    plot_dependence,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_shap_plots")


DEFAULT_MODEL_IDS = ["LR_EN", "SVM_L1", "SVM_L2"]
NON_FEATURE_COLS = {"eid", "y_true", "y_pred", "y_prob"}


def default_model_dir(model_id: str) -> Path:
    return CEL_ROOT / "results" / "incident-validation" / "lr" / model_id


def default_out_dir() -> Path:
    return CEL_ROOT / "results" / "incident-validation" / "figures"


def load_shap_artifacts(model_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (shap_values, X_test, feature_names) aligned on feature columns."""
    shap_path = model_dir / "shap_values.csv"
    xtest_path = model_dir / "X_test.csv"
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing {shap_path} (run compute_shap_oof.py first)")
    if not xtest_path.exists():
        raise FileNotFoundError(f"Missing {xtest_path} (run compute_shap_oof.py first)")

    shap_df = pd.read_csv(shap_path)
    xtest_df = pd.read_csv(xtest_path)

    feature_names = [c for c in shap_df.columns if c not in NON_FEATURE_COLS]
    xtest_features = [c for c in xtest_df.columns if c not in NON_FEATURE_COLS]
    if feature_names != xtest_features:
        common = [f for f in feature_names if f in xtest_features]
        if not common:
            raise ValueError(f"No shared features between {shap_path} and {xtest_path}")
        logger.warning(
            "Feature columns differ between shap_values and X_test; using %d shared",
            len(common),
        )
        feature_names = common

    shap_values = shap_df[feature_names].to_numpy(dtype=float)
    X_test = xtest_df[feature_names].to_numpy(dtype=float)
    return shap_values, X_test, feature_names


def generate_model_shap_plots(
    model_id: str,
    out_dir: Path,
    max_display: int,
    top_dependence: int,
) -> dict[str, Path]:
    model_dir = default_model_dir(model_id)
    shap_values, X_test, feature_names = load_shap_artifacts(model_dir)
    logger.info(
        "[%s] shap=%s X=%s features=%d",
        model_id, shap_values.shape, X_test.shape, len(feature_names),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    path = out_dir / "beeswarm.pdf"
    logger.info("  [%s] beeswarm -> %s", model_id, path)
    plot_beeswarm(
        shap_values=shap_values,
        X=X_test,
        feature_names=feature_names,
        max_display=max_display,
        outpath=path,
        shap_output_scale="raw",
    )
    written["beeswarm"] = path

    path = out_dir / "bar_importance.pdf"
    logger.info("  [%s] bar_importance -> %s", model_id, path)
    plot_bar_importance(
        shap_values=shap_values,
        feature_names=feature_names,
        max_display=max_display,
        outpath=path,
        shap_output_scale="raw",
    )
    written["bar_importance"] = path

    if top_dependence > 0:
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_dependence]
        for rank, idx in enumerate(order, start=1):
            fname = feature_names[idx]
            safe = fname.replace("/", "_").replace(" ", "_")
            path = out_dir / f"dependence_{rank:02d}_{safe}.pdf"
            logger.info("  [%s] dependence[%d] %s -> %s", model_id, rank, fname, path)
            try:
                plot_dependence(
                    shap_values=shap_values,
                    feature_idx=int(idx),
                    X=X_test,
                    feature_names=feature_names,
                    outpath=path,
                    shap_output_scale="raw",
                )
                written[f"dependence_{rank:02d}"] = path
            except Exception as exc:
                logger.warning("  dependence plot failed for %s: %s", fname, exc)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_IDS)
    parser.add_argument("--out", type=Path, default=default_out_dir())
    parser.add_argument("--max-display", type=int, default=20,
                        help="Max features shown in beeswarm/bar (default 20)")
    parser.add_argument("--top-dependence", type=int, default=5,
                        help="Number of top features for dependence plots (0 disables)")
    args = parser.parse_args()

    logger.info("CEL_ROOT: %s", CEL_ROOT)
    logger.info("Output root: %s", args.out)
    logger.info("Models: %s", args.models)

    all_written: dict[str, dict[str, Path]] = {}
    for model_id in args.models:
        try:
            model_out = args.out / model_id / "shap"
            all_written[model_id] = generate_model_shap_plots(
                model_id, model_out, args.max_display, args.top_dependence,
            )
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", model_id, exc)
        except Exception as exc:
            logger.error("Failed %s: %s", model_id, exc, exc_info=True)

    if not all_written:
        logger.error("No models processed; nothing to do")
        return 1

    n_files = sum(len(v) for v in all_written.values())
    logger.info("Done. %d SHAP figures written under %s", n_files, args.out)
    for model_id, files in all_written.items():
        logger.info("  %s:", model_id)
        for kind, path in files.items():
            logger.info("    %s -> %s", kind, path.relative_to(CEL_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
