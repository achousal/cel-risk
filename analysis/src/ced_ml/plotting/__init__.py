"""Plotting utilities for CeliacRiskML.

Publication-quality visualization for model evaluation:
- Risk distribution plots with threshold overlays
- ROC curves and calibration plots
- Decision curve analysis
- Learning curves
"""

from ced_ml.plotting.calibration import plot_calibration_curve
from ced_ml.plotting.comparison import (
    plot_calibration_comparison,
    plot_dca_comparison,
    plot_pr_comparison,
    plot_roc_comparison,
)
from ced_ml.plotting.dca import apply_plot_metadata, plot_dca, plot_dca_curve
from ced_ml.plotting.ensemble import (
    plot_aggregated_weights,
    plot_meta_learner_weights,
    plot_model_comparison,
)
from ced_ml.plotting.learning_curve import (
    aggregate_learning_curve_runs,
    compute_learning_curve,
    plot_learning_curve,
    plot_learning_curve_summary,
    save_learning_curve_csv,
)
from ced_ml.plotting.oof import plot_oof_combined
from ced_ml.plotting.risk_dist import (
    compute_distribution_stats,
    plot_risk_distribution,
)
from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve

__all__ = [
    # Comparison plots (multi-model overlay)
    "plot_roc_comparison",
    "plot_pr_comparison",
    "plot_calibration_comparison",
    "plot_dca_comparison",
    # DCA plotting
    "apply_plot_metadata",
    "plot_dca",
    "plot_dca_curve",
    # Risk distribution
    "compute_distribution_stats",
    "plot_risk_distribution",
    # Calibration
    "plot_calibration_curve",
    # ROC/PR curves
    "plot_pr_curve",
    "plot_roc_curve",
    # Learning curves
    "aggregate_learning_curve_runs",
    "compute_learning_curve",
    "plot_learning_curve",
    "plot_learning_curve_summary",
    "save_learning_curve_csv",
    # OOF combined plots
    "plot_oof_combined",
    # Ensemble-specific plots
    "plot_aggregated_weights",
    "plot_meta_learner_weights",
    "plot_model_comparison",
]
