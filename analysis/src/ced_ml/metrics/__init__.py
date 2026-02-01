"""Metrics module for model evaluation."""

from ced_ml.metrics.bootstrap import (
    stratified_bootstrap_ci,
    stratified_bootstrap_diff_ci,
)
from ced_ml.metrics.dca import (
    compute_dca_summary,
    decision_curve_analysis,
    decision_curve_table,
    find_dca_zero_crossing,
    generate_dca_thresholds,
    net_benefit,
    net_benefit_treat_all,
    parse_dca_report_points,
    save_dca_results,
    threshold_dca_zero_crossing,
)
from ced_ml.metrics.discrimination import (
    alpha_sensitivity_at_specificity,
    auroc,
    compute_brier_score,
    compute_discrimination_metrics,
    compute_log_loss,
    prauc,
    youden_j,
)
from ced_ml.metrics.thresholds import (
    BinaryMetrics,
    ThresholdBundle,
    ThresholdMetrics,
    binary_metrics_at_threshold,
    choose_threshold_objective,
    compute_threshold_bundle,
    threshold_for_precision,
    threshold_for_specificity,
    threshold_from_controls,
    threshold_max_f1,
    threshold_max_fbeta,
    threshold_youden,
    top_risk_capture,
)

__all__ = [
    # Discrimination metrics
    "auroc",
    "prauc",
    "youden_j",
    "alpha_sensitivity_at_specificity",
    "compute_discrimination_metrics",
    "compute_brier_score",
    "compute_log_loss",
    # Threshold selection
    "threshold_max_f1",
    "threshold_max_fbeta",
    "threshold_youden",
    "threshold_for_specificity",
    "threshold_for_precision",
    "threshold_from_controls",
    "binary_metrics_at_threshold",
    "top_risk_capture",
    "choose_threshold_objective",
    # Threshold bundle (standardized interface for plotting)
    "BinaryMetrics",
    "ThresholdBundle",
    "ThresholdMetrics",
    "compute_threshold_bundle",
    # Decision Curve Analysis
    "decision_curve_analysis",
    "decision_curve_table",
    "net_benefit",
    "net_benefit_treat_all",
    "compute_dca_summary",
    "save_dca_results",
    "find_dca_zero_crossing",
    "generate_dca_thresholds",
    "parse_dca_report_points",
    "threshold_dca_zero_crossing",
    # Bootstrap confidence intervals
    "stratified_bootstrap_ci",
    "stratified_bootstrap_diff_ci",
]
