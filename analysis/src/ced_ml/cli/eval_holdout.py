"""CLI implementation for holdout evaluation command."""

import logging

from ced_ml.evaluation import evaluate_holdout


def run_eval_holdout(
    infile: str,
    holdout_idx: str,
    model_artifact: str,
    outdir: str,
    scenario: str = None,
    compute_dca: bool = False,
    dca_threshold_min: float = None,
    dca_threshold_max: float = None,
    dca_threshold_step: float = None,
    dca_report_points: str = "",
    dca_use_target_prevalence: bool = False,
    save_preds: bool = False,
    toprisk_fracs: str = "0.01",
    target_prevalence: float = None,
    clinical_threshold_points: str = "",
    subgroup_min_n: int = 40,
):
    """
    Evaluate trained model on holdout set.

    Args:
        infile: Path to full dataset CSV
        holdout_idx: Path to holdout indices CSV
        model_artifact: Path to trained model artifact (.joblib)
        outdir: Output directory for results
        scenario: Override scenario (if not in artifact)
        compute_dca: Whether to compute decision curve analysis
        dca_threshold_min: Min threshold for DCA
        dca_threshold_max: Max threshold for DCA
        dca_threshold_step: Step size for DCA thresholds
        dca_report_points: Comma-separated thresholds to report
        dca_use_target_prevalence: Use prevalence-adjusted probs for DCA
        save_preds: Save individual predictions to CSV
        toprisk_fracs: Comma-separated top-risk fractions
        target_prevalence: Override target prevalence
        clinical_threshold_points: Comma-separated clinical thresholds
        subgroup_min_n: Minimum sample size for subgroup reporting
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting holdout evaluation")

    try:
        metrics = evaluate_holdout(
            infile=infile,
            holdout_idx_file=holdout_idx,
            model_artifact_path=model_artifact,
            outdir=outdir,
            scenario=scenario,
            compute_dca=compute_dca,
            dca_threshold_min=dca_threshold_min,
            dca_threshold_max=dca_threshold_max,
            dca_threshold_step=dca_threshold_step,
            dca_report_points=dca_report_points,
            dca_use_target_prevalence=dca_use_target_prevalence,
            save_preds=save_preds,
            toprisk_fracs=toprisk_fracs,
            target_prevalence=target_prevalence,
            clinical_threshold_points=clinical_threshold_points,
        )

        logger.info("Holdout evaluation complete")
        logger.info(f"AUROC: {metrics['AUROC_holdout']:.4f}")
        logger.info(f"Brier: {metrics['Brier_holdout']:.4f}")
        logger.info(f"Results saved to: {outdir}")

    except Exception as e:
        logger.error(f"Holdout evaluation failed: {e}")
        raise
