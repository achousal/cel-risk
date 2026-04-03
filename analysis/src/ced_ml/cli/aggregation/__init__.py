"""Aggregation submodules for split results processing."""

from ced_ml.cli.aggregation.aggregation import (
    compute_pooled_metrics_by_model,
    compute_pooled_threshold_metrics,
    compute_summary_stats,
    save_threshold_data,
)
from ced_ml.cli.aggregation.collection import (
    collect_best_hyperparams,
    collect_ensemble_hyperparams,
    collect_ensemble_predictions,
    collect_feature_reports,
    collect_metrics,
    collect_predictions,
)
from ced_ml.cli.aggregation.orchestrator import (
    aggregate_shap_metadata,
    aggregate_shap_values,
    build_aggregation_metadata,
    build_return_summary,
    compute_and_save_pooled_metrics,
    save_pooled_predictions,
    setup_aggregation_directories,
)
from ced_ml.cli.aggregation.plot_generator import (
    generate_aggregated_plots,
    generate_aggregated_shap_plots,
    generate_comparison_plots,
    generate_model_comparison_report,
)
from ced_ml.cli.aggregation.reporting import (
    aggregate_feature_reports,
    aggregate_feature_stability,
)
from ced_ml.cli.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)

__all__ = [
    # Discovery
    "discover_ensemble_dirs",
    "discover_split_dirs",
    # Collection
    "collect_best_hyperparams",
    "collect_ensemble_hyperparams",
    "collect_ensemble_predictions",
    "collect_feature_reports",
    "collect_metrics",
    "collect_predictions",
    # Aggregation
    "compute_pooled_metrics_by_model",
    "compute_pooled_threshold_metrics",
    "compute_summary_stats",
    "save_threshold_data",
    # Reporting
    "aggregate_feature_reports",
    "aggregate_feature_stability",
    # Plot Generation
    "generate_aggregated_plots",
    "generate_aggregated_shap_plots",
    "generate_comparison_plots",
    "generate_model_comparison_report",
    # Orchestration
    "aggregate_shap_metadata",
    "aggregate_shap_values",
    "build_aggregation_metadata",
    "build_return_summary",
    "compute_and_save_pooled_metrics",
    "save_pooled_predictions",
    "setup_aggregation_directories",
]
