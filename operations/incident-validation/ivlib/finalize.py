"""Winner selection, exhaustive post-processing, and final refit + locked-test
evaluation for the incident-validation pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ced_ml.metrics.bootstrap import stratified_bootstrap_ci
from ced_ml.metrics.dca import decision_curve_table
from ced_ml.metrics.discrimination import auroc as auroc_fn
from ced_ml.metrics.discrimination import prauc
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    threshold_youden,
)

from .calibration import compute_calibration_metrics, fit_winner_calibrator
from .features import derive_core_panel, derive_per_strategy_panels, jaccard_overlap
from .strategies import get_training_indices
from .weights import compute_class_weight

logger = logging.getLogger(__name__)

auprc_fn = prauc


# ============================================================================
# Winner selection
# ============================================================================


def select_winner(cv_results: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """Pick best (strategy, weight) by mean AUPRC across folds; tie-break with
    bootstrap-paired Δ-AUPRC across folds (fallback to parsimony via
    n_nonzero_coefs if the 95% CI on Δ spans zero)."""
    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])
        .agg(
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_nonzero=("n_nonzero_coefs", "mean"),
        )
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
        .reset_index(drop=True)
    )

    top = summary.iloc[0]
    if len(summary) < 2:
        return str(top["strategy"]), str(top["weight_scheme"]), summary

    runner = summary.iloc[1]
    if (top["mean_auprc"] - runner["mean_auprc"]) > 0.02:
        return str(top["strategy"]), str(top["weight_scheme"]), summary

    a = cv_results[
        (cv_results["strategy"] == top["strategy"])
        & (cv_results["weight_scheme"] == top["weight_scheme"])
    ].sort_values("fold")["auprc"].to_numpy()
    b = cv_results[
        (cv_results["strategy"] == runner["strategy"])
        & (cv_results["weight_scheme"] == runner["weight_scheme"])
    ].sort_values("fold")["auprc"].to_numpy()
    if len(a) != len(b) or len(a) == 0:
        return str(top["strategy"]), str(top["weight_scheme"]), summary
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, len(a), size=len(a))
        deltas.append(float((a[idx] - b[idx]).mean()))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    if lo <= 0.0 <= hi:
        if runner["mean_nonzero"] < top["mean_nonzero"]:
            logger.info(
                "Tie-break (parsimony): picking %s+%s over %s+%s "
                "(Delta-AUPRC 95%% CI=[%.4f, %.4f])",
                runner["strategy"], runner["weight_scheme"],
                top["strategy"], top["weight_scheme"], lo, hi,
            )
            return str(runner["strategy"]), str(runner["weight_scheme"]), summary
    return str(top["strategy"]), str(top["weight_scheme"]), summary


# ============================================================================
# Final refit on dev + locked-test evaluation
# ============================================================================


def final_refit_and_test(
    cfg,
    data: dict,
    features: dict,
    cv_results: pd.DataFrame,
    model_spec,
    core_panel: list[str] | None = None,
    calibrator=None,
) -> dict:
    logger.info("=== Final Model Selection (%s) ===", cfg.model)

    agg_dict = {
        "mean_auprc": ("auprc", "mean"),
        "std_auprc": ("auprc", "std"),
        "mean_auroc": ("auroc", "mean"),
        "std_auroc": ("auroc", "std"),
    }

    summary = (
        cv_results.groupby(["strategy", "weight_scheme"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values("mean_auprc", ascending=False)
    )

    logger.info("\n%s", summary.to_string(index=False))

    best_row = summary.iloc[0]
    best_strategy = best_row["strategy"]
    best_weight = best_row["weight_scheme"]
    best_params = model_spec.aggregate_best_params(
        cv_results[
            (cv_results["strategy"] == best_strategy)
            & (cv_results["weight_scheme"] == best_weight)
        ]
    )

    logger.info(
        "\nBest: %s + %s (mean AUPRC=%.4f)",
        best_strategy, best_weight, best_row["mean_auprc"],
    )
    logger.info("  Median hyperparams: %s", model_spec.param_summary(best_params))

    full_panel = features["pruned_proteins"]
    protein_panel = list(core_panel) if core_panel is not None else full_panel
    if not protein_panel:
        raise ValueError("final_refit_and_test: protein panel is empty")
    if core_panel is not None:
        missing = [p for p in protein_panel if p not in full_panel]
        if missing:
            raise ValueError(
                f"core_panel contains proteins not in bootstrap panel: {missing[:5]}"
            )
        logger.info(
            "Final refit restricted to core panel: %d/%d proteins",
            len(protein_panel), len(full_panel),
        )

    df = data["df"]
    dev_incident_idx = data["dev_incident_idx"]
    dev_control_idx = data["dev_control_idx"]
    prevalent_idx = data["prevalent_idx"]

    train_idx, y_train = get_training_indices(
        best_strategy, dev_incident_idx, dev_control_idx, prevalent_idx,
    )

    X_train = df.loc[train_idx, protein_panel].to_numpy(dtype=float)
    cw = compute_class_weight(best_weight, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    final_model = model_spec.build(best_params, cw, cfg, cfg.split_seed)
    model_spec.fit(final_model, X_train_s, y_train, cw)

    coefs = model_spec.extract_coefs(final_model)
    logger.info(
        "Final model: %d/%d features with |coef| > 1e-8",
        int(np.sum(np.abs(coefs) > 1e-8)),
        len(protein_panel),
    )

    # === Locked test evaluation ===
    logger.info("=== Locked Test Evaluation ===")

    test_incident_idx = data["test_incident_idx"]
    test_control_idx = data["test_control_idx"]
    test_idx = np.concatenate([test_incident_idx, test_control_idx])
    y_test = np.concatenate([
        np.ones(len(test_incident_idx)),
        np.zeros(len(test_control_idx)),
    ]).astype(int)

    X_test = df.loc[test_idx, protein_panel].to_numpy(dtype=float)
    X_test_s = scaler.transform(X_test)
    y_prob_raw = final_model.predict_proba(X_test_s)[:, 1]
    if calibrator is not None:
        y_prob = calibrator.transform(y_prob_raw)
        logger.info(
            "Applied %s OOF calibrator to test predictions",
            getattr(calibrator, "method", "unknown"),
        )
    else:
        y_prob = y_prob_raw

    test_auprc = float(prauc(y_test, y_prob))
    test_auroc = float(auroc_fn(y_test, y_prob))

    best_threshold = float(threshold_youden(y_test, y_prob))
    bm = binary_metrics_at_threshold(y_test, y_prob, best_threshold)
    sensitivity = float(bm.sensitivity)
    specificity = float(bm.specificity)
    precision = float(bm.precision)

    logger.info("  AUPRC (primary): %.4f", test_auprc)
    logger.info("  AUROC:           %.4f", test_auroc)
    logger.info("  Threshold (Youden J): %.4f", best_threshold)
    logger.info("  Sensitivity:     %.4f", sensitivity)
    logger.info("  Specificity:     %.4f", specificity)
    logger.info("  Precision:       %.4f", precision)

    logger.info("  Computing %d bootstrap CIs...", cfg.n_bootstrap_ci)
    auprc_lo, auprc_hi = stratified_bootstrap_ci(
        y_test, y_prob, auprc_fn,
        n_boot=cfg.n_bootstrap_ci, seed=cfg.ci_seed,
    )
    auroc_lo, auroc_hi = stratified_bootstrap_ci(
        y_test, y_prob, auroc_fn,
        n_boot=cfg.n_bootstrap_ci, seed=cfg.ci_seed,
    )
    auprc_ci = (float(auprc_lo), float(auprc_hi))
    auroc_ci = (float(auroc_lo), float(auroc_hi))

    logger.info("  AUPRC 95%% CI: [%.4f, %.4f]", *auprc_ci)
    logger.info("  AUROC 95%% CI: [%.4f, %.4f]", *auroc_ci)

    coef_df = pd.DataFrame({
        "protein": protein_panel,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
        "stability_freq": [
            features["selection_freq"].get(p, 0.0) for p in protein_panel
        ],
    }).sort_values("abs_coef", ascending=False)

    nonzero_df = coef_df[coef_df["abs_coef"] > 1e-8].copy()

    return {
        "summary": summary,
        "best_strategy": best_strategy,
        "best_weight": best_weight,
        "best_params": best_params,
        "test_auprc": test_auprc,
        "test_auroc": test_auroc,
        "auprc_ci": auprc_ci,
        "auroc_ci": auroc_ci,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "threshold": best_threshold,
        "coef_df": coef_df,
        "nonzero_features": nonzero_df,
        "final_model": final_model,
        "scaler": scaler,
        "refit_panel": protein_panel,
        "test_y_prob": y_prob,
        "test_y_prob_raw": y_prob_raw,
        "test_y_true": y_test,
        "calibrator": calibrator,
    }


# ============================================================================
# Exhaustive post-processing
# ============================================================================


def run_exhaustive_post(
    cfg,
    data: dict,
    features: dict,
    cv_results: pd.DataFrame,
    fold_coefs: list[dict],
    oof_df: pd.DataFrame,
    model_spec,
    min_fold_stability: float = 4 / 5,
) -> dict:
    """Full exhaustive post-CV pipeline:
       1. Per-strategy panels with fold_stability + sign_consistency.
       2. Winner selection with paired-bootstrap tie-break.
       3. C(winner) core panel.
       4. OOFCalibrator (logistic_full) fitted on winner dev OOF.
       5. Final refit on C(winner), calibrated test eval, calibration metrics, DCA.
    """
    logger.info("=== Exhaustive Post-Processing ===")
    panel = features["pruned_proteins"]
    selection_freq = features["selection_freq"]
    n_folds = int(cfg.n_outer_folds)

    per_strategy = derive_per_strategy_panels(
        cv_results, fold_coefs, panel, selection_freq, n_folds,
    )
    for s, d in per_strategy.items():
        nz = int((d["fold_stability"] > 0).sum())
        core_like = int((
            (d["fold_stability"] >= min_fold_stability - 1e-12)
            & d["sign_consistent"].astype(bool)
        ).sum())
        logger.info(
            "  strategy=%s weight=%s  any-nonzero=%d  core-like(>=%.2f & signed)=%d",
            s, d["weight_scheme"].iloc[0] if len(d) else "?", nz,
            min_fold_stability, core_like,
        )

    best_strategy, best_weight, summary = select_winner(cv_results)
    logger.info("Winner: strategy=%s, weight=%s", best_strategy, best_weight)

    core_df = derive_core_panel(
        per_strategy, best_strategy,
        min_fold_stability=min_fold_stability,
        require_sign_consistent=True,
    )
    core_proteins = core_df["protein"].tolist()
    logger.info(
        "C(winner): %d proteins (>=%d/%d folds, sign-consistent)",
        len(core_proteins), int(round(min_fold_stability * n_folds)), n_folds,
    )
    if not core_proteins:
        logger.warning(
            "C(winner) empty with min_fold_stability=%.2f + sign_consistent; "
            "falling back to bootstrap panel.",
            min_fold_stability,
        )
        core_proteins = list(panel)

    calibrator, dev_oof_df = fit_winner_calibrator(
        oof_df, best_strategy, best_weight, method="logistic_full",
    )

    # Force winner through final_refit if tie-break overrode top-1.
    top_summary_row = summary.iloc[0]
    cv_for_refit = cv_results
    if (
        str(top_summary_row["strategy"]) != best_strategy
        or str(top_summary_row["weight_scheme"]) != best_weight
    ):
        cv_for_refit = cv_results[
            (cv_results["strategy"] == best_strategy)
            & (cv_results["weight_scheme"] == best_weight)
        ].copy()

    final = final_refit_and_test(
        cfg, data, features, cv_for_refit, model_spec,
        core_panel=core_proteins,
        calibrator=calibrator,
    )

    cal_metrics_cal = compute_calibration_metrics(
        final["test_y_true"], final["test_y_prob"],
    )
    cal_metrics_raw = compute_calibration_metrics(
        final["test_y_true"], final["test_y_prob_raw"],
    )

    scenario = f"{best_strategy}+{best_weight}"
    dca = decision_curve_table(
        scenario,
        final["test_y_true"],
        {"raw": final["test_y_prob_raw"], "calibrated": final["test_y_prob"]},
        max_pt=0.10,
        step=0.001,
    )

    core_sets: dict[str, set[str]] = {}
    for s, d in per_strategy.items():
        mask = (
            (d["fold_stability"] >= min_fold_stability - 1e-12)
            & d["sign_consistent"].astype(bool)
        )
        core_sets[s] = set(d.loc[mask, "protein"].tolist())
    jaccard_df = jaccard_overlap(core_sets) if core_sets else pd.DataFrame()

    return {
        "per_strategy_panels": per_strategy,
        "core_panel_df": core_df,
        "core_proteins": core_proteins,
        "winner_strategy": best_strategy,
        "winner_weight": best_weight,
        "summary": summary,
        "dev_oof_df": dev_oof_df,
        "calibrator": calibrator,
        "final": final,
        "cal_metrics_calibrated": cal_metrics_cal,
        "cal_metrics_raw": cal_metrics_raw,
        "dca": dca,
        "jaccard_df": jaccard_df,
        "min_fold_stability": min_fold_stability,
    }
