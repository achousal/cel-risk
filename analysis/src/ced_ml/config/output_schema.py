"""Output and strictness configuration schemas for CeD-ML pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutputConfig(BaseModel):
    """Configuration for output file generation."""

    model_config = ConfigDict(extra="forbid")

    # Prediction artifacts
    save_train_preds: bool = False
    save_train_oof: bool = True
    save_val_preds: bool = True
    save_test_preds: bool = True

    # Calibration artifacts
    save_calibration: bool = True
    calib_bins: int = Field(default=10, ge=2)

    # Feature artifacts
    save_feature_importance: bool = True
    feature_reports: bool = True

    # Optuna artifacts
    save_optuna_study: bool = True
    save_optuna_trials: bool = True

    # Master plot controls
    save_plots: bool = True
    max_plot_splits: int = Field(default=0, ge=0)
    plot_format: str = "png"
    plot_dpi: int = 300

    # Individual plot type controls - Standard evaluation plots
    plot_roc: bool = True
    plot_pr: bool = True
    plot_calibration: bool = True
    plot_risk_distribution: bool = True
    plot_dca: bool = True

    # Training and optimization plots
    plot_learning_curve: bool = True
    plot_oof_combined: bool = True
    plot_optuna: bool = True
    optuna_plot_format: str = "html"  # "html" (HPC-safe) or "png"/"pdf" (requires Kaleido/Chrome)

    # Comparison plots (multi-model overlay)
    plot_model_comparison_curves: bool = True

    # Ensemble-specific plots
    plot_ensemble_weights: bool = True
    plot_ensemble_comparison: bool = True
    plot_base_correlations: bool = True

    # Aggregation-specific output controls
    save_pooled_preds: bool = True
    save_summary_csv: bool = True
    save_thresholds: bool = True
    save_individual: bool = False

    # Panel optimization output controls
    save_panel_csv: bool = True
    save_rfe_curve: bool = True
    save_stability_ranks: bool = True
    save_consensus_ranks: bool = True

    # SHAP output controls
    save_shap_importance: bool = True
    plot_shap_summary: bool = True
    plot_shap_waterfall: bool = True
    plot_shap_dependence: bool = True
    plot_shap_heatmap: bool = True


class StrictnessConfig(BaseModel):
    """Configuration for validation strictness."""

    level: Literal["off", "warn", "error"] = "warn"
    check_split_overlap: bool = True
    check_prevalent_in_eval: bool = True
    check_threshold_source: bool = True
    check_prevalence_adjustment: bool = True
    check_feature_leakage: bool = True
