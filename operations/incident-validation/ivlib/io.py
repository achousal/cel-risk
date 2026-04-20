"""Artifact persistence for the incident-validation pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ced_ml.data.schema import ID_COL

from ivlib.reporting import build_report
from ivlib.strategies import STRATEGY_TAGS

logger = logging.getLogger(__name__)


def _panel_df(features: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "protein": features["pruned_proteins"],
        "stability_freq": [
            features["selection_freq"].get(p, 0.0)
            for p in features["pruned_proteins"]
        ],
    })


def _write_config(cfg, out: Path) -> None:
    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()
    }
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2))


def save_features(cfg, data: dict, features: dict) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _panel_df(features).to_csv(out / "feature_panel.csv", index=False)
    features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)
    pd.DataFrame(features["prune_map"]).to_csv(out / "corr_prune_map.csv", index=False)
    _write_config(cfg, out)
    logger.info("Feature selection artifacts saved to %s", out)


def load_features(cfg) -> dict:
    """Load pre-computed feature selection artifacts from output dir."""
    out = cfg.output_dir
    panel_df = pd.read_csv(out / "feature_panel.csv")
    pruned_proteins = panel_df["protein"].tolist()
    selection_freq = dict(zip(panel_df["protein"], panel_df["stability_freq"]))

    bootstrap_log = (
        pd.read_csv(out / "bootstrap_log.csv")
        if (out / "bootstrap_log.csv").exists()
        else pd.DataFrame()
    )
    prune_map = (
        pd.read_csv(out / "corr_prune_map.csv").to_dict("records")
        if (out / "corr_prune_map.csv").exists()
        else []
    )

    logger.info(
        "Loaded feature panel: %d proteins from %s",
        len(pruned_proteins),
        out / "feature_panel.csv",
    )

    return {
        "stable_proteins": pruned_proteins,
        "pruned_proteins": pruned_proteins,
        "selection_freq": selection_freq,
        "bootstrap_log": bootstrap_log,
        "prune_map": prune_map,
    }


def _build_test_predictions(data: dict, features: dict, final: dict) -> pd.DataFrame:
    """Assemble test_predictions.csv frame.

    Prefer final['test_y_prob'] (calibrated-if-available) over recomputation.
    Falls back to scaler + final_model recompute if probabilities absent.
    """
    test_incident_idx = data["test_incident_idx"]
    test_control_idx = data["test_control_idx"]
    test_idx = np.concatenate([test_incident_idx, test_control_idx])

    if "test_y_prob" in final and "test_y_true" in final:
        y_test = np.asarray(final["test_y_true"]).astype(int)
        y_prob = np.asarray(final["test_y_prob"])
    else:
        y_test = np.concatenate([
            np.ones(len(test_incident_idx)),
            np.zeros(len(test_control_idx)),
        ]).astype(int)
        X_test = data["df"].loc[test_idx, features["pruned_proteins"]].to_numpy(dtype=float)
        X_test_s = final["scaler"].transform(X_test)
        y_prob = final["final_model"].predict_proba(X_test_s)[:, 1]

    eids = (
        data["df"].loc[test_idx, ID_COL].values
        if ID_COL in data["df"].columns
        else test_idx
    )
    return pd.DataFrame({
        "eid": eids,
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": (y_prob >= final["threshold"]).astype(int),
    })


def save_results(
    cfg,
    data: dict,
    features: dict,
    cv_results: "pd.DataFrame",
    final: dict,
    fold_coefs: list[dict],
    model_spec,
    exhaustive: dict | None = None,
    oof_df: "pd.DataFrame | None" = None,
) -> None:
    """Save CV + final refit bundle (and exhaustive artifacts when supplied)."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cv_results.to_csv(out / "cv_results.csv", index=False)
    logger.info("Saved: %s", out / "cv_results.csv")

    protein_panel = features["pruned_proteins"]
    coef_rows = []
    for entry in fold_coefs:
        for i, p in enumerate(protein_panel):
            coef_rows.append({
                "fold": entry["fold"],
                "strategy": entry["strategy"],
                "weight_scheme": entry["weight_scheme"],
                "protein": p,
                "coefficient": entry["coefs"][i],
            })
    pd.DataFrame(coef_rows).to_csv(out / "fold_coefficients.csv", index=False)
    logger.info("Saved: %s", out / "fold_coefficients.csv")

    final["summary"].to_csv(out / "strategy_comparison.csv", index=False)
    logger.info("Saved: %s", out / "strategy_comparison.csv")

    _panel_df(features).to_csv(out / "feature_panel.csv", index=False)
    logger.info("Saved: %s", out / "feature_panel.csv")

    features["bootstrap_log"].to_csv(out / "bootstrap_log.csv", index=False)
    pd.DataFrame(features["prune_map"]).to_csv(out / "corr_prune_map.csv", index=False)

    final["coef_df"].to_csv(out / "feature_coefficients.csv", index=False)
    logger.info("Saved: %s", out / "feature_coefficients.csv")

    _build_test_predictions(data, features, final).to_csv(
        out / "test_predictions.csv", index=False
    )

    report = build_report(cfg, data, features, cv_results, final, model_spec)
    (out / "summary_report.md").write_text(report)
    logger.info("Saved: %s", out / "summary_report.md")

    _write_config(cfg, out)

    if exhaustive is not None:
        _save_exhaustive(out, final, exhaustive)

    logger.info("All outputs saved to %s", out)


def _save_exhaustive(out: Path, final: dict, exhaustive: dict) -> None:
    """Persist exhaustive post-processing artifacts."""
    for s, df in exhaustive["per_strategy_panels"].items():
        tag = STRATEGY_TAGS.get(s, s)
        df.to_csv(out / f"feature_panel_{tag}.csv", index=False)
        logger.info("Saved: %s", out / f"feature_panel_{tag}.csv")

    core_path = out / "core_panel_winner.csv"
    exhaustive["core_panel_df"].to_csv(core_path, index=False)
    logger.info("Saved: %s (n=%d)", core_path, len(exhaustive["core_panel_df"]))

    if not exhaustive["jaccard_df"].empty:
        exhaustive["jaccard_df"].to_csv(out / "strategy_panel_overlap.csv", index=False)
        logger.info("Saved: %s", out / "strategy_panel_overlap.csv")

    if not exhaustive["dev_oof_df"].empty:
        exhaustive["dev_oof_df"].to_csv(out / "dev_oof_predictions.csv", index=False)
        logger.info("Saved: %s", out / "dev_oof_predictions.csv")

    # ced_ml decision_curve_table long-format: scenario,threshold,model,net_benefit.
    # Persist verbatim; downstream plotting handles reshape.
    exhaustive["dca"].to_csv(out / "dca.csv", index=False)
    logger.info("Saved: %s", out / "dca.csv")

    cal_payload = {
        "winner_strategy": exhaustive["winner_strategy"],
        "winner_weight": exhaustive["winner_weight"],
        "core_panel_size": len(exhaustive["core_proteins"]),
        "min_fold_stability": exhaustive["min_fold_stability"],
        "calibrator_method": (
            getattr(exhaustive["calibrator"], "method", None)
            if exhaustive["calibrator"] is not None else None
        ),
        "calibrated": exhaustive["cal_metrics_calibrated"],
        "raw": exhaustive["cal_metrics_raw"],
    }
    (out / "calibration_metrics.json").write_text(json.dumps(cal_payload, indent=2))
    logger.info("Saved: %s", out / "calibration_metrics.json")

    final_metrics = {
        "winner_strategy": exhaustive["winner_strategy"],
        "winner_weight": exhaustive["winner_weight"],
        "best_params": final["best_params"],
        "refit_panel_size": len(final.get("refit_panel", [])),
        "test_auprc": float(final["test_auprc"]),
        "test_auroc": float(final["test_auroc"]),
        "auprc_ci": [float(final["auprc_ci"][0]), float(final["auprc_ci"][1])],
        "auroc_ci": [float(final["auroc_ci"][0]), float(final["auroc_ci"][1])],
        "sensitivity": float(final["sensitivity"]),
        "specificity": float(final["specificity"]),
        "precision": float(final["precision"]),
        "threshold": float(final["threshold"]),
        "calibration": cal_payload,
    }
    (out / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    logger.info("Saved: %s", out / "final_metrics.json")

    try:
        import joblib
        if exhaustive["calibrator"] is not None:
            joblib.dump(exhaustive["calibrator"], out / "calibrator.pkl")
            logger.info("Saved: %s", out / "calibrator.pkl")
        joblib.dump(final["final_model"], out / "final_model.pkl")
        joblib.dump(final["scaler"], out / "scaler.pkl")
        logger.info("Saved: final_model.pkl, scaler.pkl")
    except Exception as exc:
        logger.warning("joblib persistence failed: %s", exc)
