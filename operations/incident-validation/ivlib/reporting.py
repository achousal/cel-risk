"""Markdown summary reporting for the incident-validation pipeline."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_report(
    cfg,
    data: dict,
    features: dict,
    cv_results: "pd.DataFrame",
    final: dict,
    model_spec,
) -> str:
    lines = [
        f"# Incident Validation Report ({cfg.model})",
        "",
        "## Dataset",
        f"- Total samples: {len(data['df'])}",
        f"- Dev incident: {len(data['dev_incident_idx'])}",
        f"- Dev controls: {len(data['dev_control_idx'])}",
        f"- Prevalent (training only): {len(data['prevalent_idx'])}",
        f"- Test incident: {len(data['test_incident_idx'])}",
        f"- Test controls: {len(data['test_control_idx'])}",
        "",
        "## Feature Selection",
        f"- Method: Bootstrap stability ({cfg.screen_method} statistic)",
        f"- Resamples: {cfg.n_bootstrap}, top {cfg.bootstrap_top_k} per resample",
        f"- Stability threshold: {cfg.stability_threshold:.0%}",
        f"- Stable proteins (pre-prune): {len(features['stable_proteins'])}",
        f"- Correlation threshold: |r| > {cfg.corr_threshold}",
        f"- **Final panel: {len(features['pruned_proteins'])} proteins**",
        "",
        "## Strategy Comparison (5-fold CV, mean AUPRC)",
        "",
        final["summary"].to_markdown(index=False),
        "",
        f"**Best: {final['best_strategy']} + {final['best_weight']}** "
        f"(mean AUPRC = {final['summary'].iloc[0]['mean_auprc']:.4f})",
        "",
        "## Locked Test Set Results",
        "",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
        f"| **AUPRC** (primary) | {final['test_auprc']:.4f} "
        f"| [{final['auprc_ci'][0]:.4f}, {final['auprc_ci'][1]:.4f}] |",
        f"| AUROC | {final['test_auroc']:.4f} "
        f"| [{final['auroc_ci'][0]:.4f}, {final['auroc_ci'][1]:.4f}] |",
        f"| Sensitivity | {final['sensitivity']:.4f} | - |",
        f"| Specificity | {final['specificity']:.4f} | - |",
        f"| Precision | {final['precision']:.4f} | - |",
        "",
        f"Threshold: {final['threshold']:.4f} (Youden's J)",
        "",
        "## Final Model",
        f"- Model: {model_spec.display_name}",
        f"- Hyperparams: {model_spec.param_summary(final['best_params'])}",
        f"- Features with |coef| > 1e-8: "
        f"{len(final['nonzero_features'])}/{len(features['pruned_proteins'])}",
        "",
        "### Top Features (by |coefficient|)",
        "",
        final["nonzero_features"].head(20).to_markdown(index=False),
        "",
        "## Hyperparameters",
        f"- Optuna trials: {cfg.n_optuna_trials}",
        f"- Inner CV folds: {cfg.n_inner_folds}",
        f"- max_iter: {cfg.max_iter}",
        f"- Bootstrap CIs: {cfg.n_bootstrap_ci}",
        "",
    ]

    return "\n".join(lines)
