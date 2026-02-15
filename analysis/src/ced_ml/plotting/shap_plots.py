"""SHAP visualization plots for CeD-ML pipeline.

Provides beeswarm, bar importance, waterfall, and dependence plots.
All plots are gated by output config flags in the plotting stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

if TYPE_CHECKING:
    import pandas as pd

    from ced_ml.features.shap_values import SHAPTestPayload

logger = logging.getLogger(__name__)


def _format_shap_ylabel(scale: str) -> str:
    """Generate scale-aware ylabel for SHAP plots."""
    labels = {
        "log_odds": "SHAP value (log-odds)",
        "margin": "SHAP value (margin)",
        "probability": "SHAP value (probability)",
        "raw": "SHAP value (raw, model-dependent)",
    }
    return labels.get(scale, f"SHAP value ({scale})")


def _require_shap():
    """Raise ImportError if shap is not installed."""
    try:
        import shap  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "SHAP is required for plotting. Install with: pip install -e 'analysis[shap]'"
        ) from err


def plot_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    outpath: Path | str | None = None,
    shap_output_scale: str = "raw",
) -> None:
    """Beeswarm plot showing feature impact distribution."""
    _require_shap()
    import shap

    explanation = shap.Explanation(
        values=shap_values,
        data=X,
        feature_names=feature_names,
    )

    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_ylabel(_format_shap_ylabel(shap_output_scale))

    if outpath is not None:
        fig = plt.gcf()
        fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
        logger.info("SHAP beeswarm plot saved: %s", outpath)
    plt.close("all")


def plot_bar_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
    outpath: Path | str | None = None,
    shap_output_scale: str = "raw",
) -> None:
    """Global bar plot of mean |SHAP| per feature."""
    _require_shap()
    import shap

    explanation = shap.Explanation(
        values=shap_values,
        feature_names=feature_names,
    )

    shap.plots.bar(explanation, max_display=max_display, show=False)
    ax = plt.gca()
    ax.set_xlabel(_format_shap_ylabel(shap_output_scale))

    if outpath is not None:
        fig = plt.gcf()
        fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
        logger.info("SHAP bar importance plot saved: %s", outpath)
    plt.close("all")


def plot_waterfall(
    shap_values: np.ndarray,
    expected_value: float,
    sample_idx: int,
    feature_names: list[str],
    outpath: Path | str | None = None,
    title: str | None = None,
    shap_output_scale: str = "raw",
) -> None:
    """Waterfall plot for a single sample showing feature contributions."""
    _require_shap()
    import shap

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        feature_names=feature_names,
    )

    shap.plots.waterfall(explanation, show=False)
    ax = plt.gca()
    ax.set_ylabel(_format_shap_ylabel(shap_output_scale))
    if title:
        plt.title(title, fontsize=12)

    if outpath is not None:
        fig = plt.gcf()
        fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
        logger.info("SHAP waterfall plot saved: %s", outpath)
    plt.close("all")


def plot_dependence(
    shap_values: np.ndarray,
    feature_idx: int,
    X: np.ndarray,
    feature_names: list[str],
    outpath: Path | str | None = None,
    shap_output_scale: str = "raw",
) -> None:
    """Dependence plot showing feature value vs SHAP value."""
    _require_shap()
    import shap

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        ax=ax,
    )
    ax.set_ylabel(_format_shap_ylabel(shap_output_scale))

    if outpath is not None:
        fig.savefig(str(outpath), dpi=300, bbox_inches="tight")
        logger.info("SHAP dependence plot saved: %s", outpath)
    plt.close("all")


def generate_all_shap_plots(
    test_payload: SHAPTestPayload,
    oof_shap_df: pd.DataFrame | None,
    threshold: float,
    config: Any,
    outdir: Path,
    test_preds_df: pd.DataFrame | None = None,
) -> None:
    """Orchestrator: generate all enabled SHAP plots -> outdir/shap/.

    Uses select_waterfall_samples() for waterfall plot sample selection.
    Gated on output config flags (plot_shap_summary, plot_shap_waterfall, etc.).
    """
    from ced_ml.features.shap_values import select_waterfall_samples

    shap_dir = outdir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    model = test_payload.model_name
    fmt = getattr(config.output, "plot_format", "png")

    # Resolve feature values for color axis: prefer transformed features over SHAP values
    feature_values = (
        test_payload.X_transformed
        if test_payload.X_transformed is not None
        else test_payload.values
    )

    # Bar + beeswarm (summary)
    if getattr(config.output, "plot_shap_summary", True):
        plot_bar_importance(
            test_payload.values,
            test_payload.feature_names,
            outpath=shap_dir / f"{model}__shap_bar.{fmt}",
            shap_output_scale=test_payload.shap_output_scale,
        )
        plot_beeswarm(
            test_payload.values,
            feature_values,
            test_payload.feature_names,
            outpath=shap_dir / f"{model}__shap_beeswarm.{fmt}",
            shap_output_scale=test_payload.shap_output_scale,
        )

    # Waterfall plots
    if getattr(config.output, "plot_shap_waterfall", True) and test_preds_df is not None:
        # Align predictions with SHAP samples (handle subsampling)
        if test_payload.sample_indices is not None:
            # SHAP was computed on a subsample
            aligned_preds_df = test_preds_df.iloc[test_payload.sample_indices].reset_index(
                drop=True
            )
        else:
            aligned_preds_df = test_preds_df

        # Use prevalence-adjusted probabilities if available (matches threshold computation)
        prob_col = "y_prob_adjusted" if "y_prob_adjusted" in aligned_preds_df.columns else "y_prob"
        y_pred_proba = aligned_preds_df[prob_col].values
        y_true = test_payload.y_true

        if y_true is not None:
            samples = select_waterfall_samples(y_pred_proba, y_true, threshold)
            shap_config = getattr(config.features, "shap", None)
            n_waterfall = getattr(shap_config, "n_waterfall_samples", 4) if shap_config else 4

            for info in samples[:n_waterfall]:
                idx = info["index"]
                cat = info["category"].split(" ")[0]  # TP, FP, FN, TN
                plot_waterfall(
                    test_payload.values,
                    test_payload.expected_value,
                    sample_idx=idx,
                    feature_names=test_payload.feature_names,
                    outpath=shap_dir / f"{model}__waterfall_{cat}_idx{idx}.{fmt}",
                    title=f"{model} - {info['category']} (p={info['pred_proba']:.3f})",
                    shap_output_scale=test_payload.shap_output_scale,
                )

    # Dependence plots (top 5 features)
    if getattr(config.output, "plot_shap_dependence", True):
        mean_abs = np.mean(np.abs(test_payload.values), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:5]

        for feat_idx in top_indices:
            feat_name = test_payload.feature_names[feat_idx]
            safe_name = feat_name.replace("/", "_").replace(" ", "_")
            plot_dependence(
                test_payload.values,
                feat_idx,
                feature_values,
                test_payload.feature_names,
                outpath=shap_dir / f"{model}__dependence_{safe_name}.{fmt}",
                shap_output_scale=test_payload.shap_output_scale,
            )

    logger.info("All SHAP plots saved to: %s", shap_dir)
