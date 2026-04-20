#!/usr/bin/env python3
"""Thin CLI runner for the incident-validation pipeline. Logic lives in ivlib."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_OP_ROOT = Path(__file__).resolve().parent.parent
if str(_OP_ROOT) not in sys.path:
    sys.path.insert(0, str(_OP_ROOT))

import pandas as pd  # noqa: E402

from ivlib import (  # noqa: E402
    MODEL_OUTPUT_DIRS,
    VALID_MODELS,
    get_model_spec,
    load_and_split,
    load_features,
    run_cv,
    run_exhaustive_post,
    run_feature_selection,
    save_features,
    save_results,
)

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Pipeline configuration (shared across all models)."""
    data_path: Path = Path("data/Celiac_dataset_proteomics_w_demo.parquet")
    output_dir: Path = Path("results/incident-validation/lr/LR_EN")
    model: str = "LR_EN"
    test_frac: float = 0.20
    n_outer_folds: int = 5
    split_seed: int = 42
    n_bootstrap: int = 100
    bootstrap_top_k: int = 200
    stability_threshold: float = 0.70
    screen_method: str = "wald"
    corr_threshold: float = 0.85
    corr_method: str = "spearman"
    n_optuna_trials: int = 50
    n_inner_folds: int = 3
    max_iter: int = 2_000
    solver: str = "saga"
    calibration_cv: int = 5
    n_bootstrap_ci: int = 2_000
    ci_seed: int = 99
    strategies: list[str] = field(default_factory=lambda: [
        "incident_only", "incident_prevalent", "prevalent_only"])
    weight_schemes: list[str] = field(default_factory=lambda: [
        "none", "balanced", "sqrt", "log"])
    smoke: bool = False

    def __post_init__(self) -> None:
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        if self.smoke:
            self.n_bootstrap = 10
            self.bootstrap_top_k = 50
            self.n_optuna_trials = 5
            self.n_bootstrap_ci = 100
            self.n_outer_folds = 3
            self.max_iter = 500
            # Smoke exercises wiring, not the full factorial. Keep 2 strategies
            # (so winner selection runs) and 1 weight scheme.
            self.strategies = ["incident_only", "incident_prevalent"]
            self.weight_schemes = ["none"]


def _phase_cv(cfg: Config, model_spec: Any, strategy: str, weight_scheme: str) -> None:
    cfg.strategies = [strategy]
    cfg.weight_schemes = [weight_scheme]
    data = load_and_split(cfg)
    features = load_features(cfg)
    cv_results, fold_coefs, oof_df = run_cv(cfg, data, features, model_spec)

    tag = f"{strategy}_{weight_scheme}"
    combo_dir = cfg.output_dir / "combos"
    combo_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(combo_dir / f"cv_{tag}.csv", index=False)

    panel = features["pruned_proteins"]
    coef_rows = [
        {"fold": e["fold"], "strategy": e["strategy"],
         "weight_scheme": e["weight_scheme"],
         "protein": p, "coefficient": e["coefs"][i]}
        for e in fold_coefs for i, p in enumerate(panel)
    ]
    pd.DataFrame(coef_rows).to_csv(combo_dir / f"coefs_{tag}.csv", index=False)
    oof_df.to_csv(combo_dir / f"oof_{tag}.csv", index=False)
    logger.info("Saved combo results to %s", combo_dir)


def _phase_aggregate(cfg: Config, model_spec: Any) -> None:
    combo_dir = cfg.output_dir / "combos"
    if not combo_dir.exists():
        raise FileNotFoundError(f"No combo results at {combo_dir}")
    cv_parts = sorted(combo_dir.glob("cv_*.csv"))
    coef_parts = sorted(combo_dir.glob("coefs_*.csv"))
    oof_parts = sorted(combo_dir.glob("oof_*.csv"))
    if not cv_parts:
        raise FileNotFoundError(f"No cv_*.csv files in {combo_dir}")

    cv_results = pd.concat([pd.read_csv(f) for f in cv_parts], ignore_index=True)
    logger.info("Merged %d combo files -> %d CV rows", len(cv_parts), len(cv_results))
    fold_coefs_df = pd.concat([pd.read_csv(f) for f in coef_parts], ignore_index=True)
    if oof_parts:
        oof_df = pd.concat([pd.read_csv(f) for f in oof_parts], ignore_index=True)
    else:
        logger.warning("No oof_*.csv files; OOF calibration will be skipped.")
        oof_df = pd.DataFrame(columns=[
            "fold", "strategy", "weight_scheme", "df_idx", "y_true", "y_prob"])

    data = load_and_split(cfg)
    features = load_features(cfg)
    panel = features["pruned_proteins"]

    fold_coefs_list = []
    for (fold, strat, wt), grp in fold_coefs_df.groupby(
        ["fold", "strategy", "weight_scheme"]
    ):
        grp_sorted = grp.set_index("protein").loc[panel]
        fold_coefs_list.append({
            "fold": fold, "strategy": strat, "weight_scheme": wt,
            "coefs": grp_sorted["coefficient"].values,
        })

    exhaustive = run_exhaustive_post(
        cfg, data, features, cv_results, fold_coefs_list, oof_df, model_spec,
    )
    save_results(
        cfg, data, features, cv_results, exhaustive["final"],
        fold_coefs_list, model_spec, exhaustive=exhaustive, oof_df=oof_df,
    )


def _phase_all(cfg: Config, model_spec: Any) -> None:
    data = load_and_split(cfg)
    features = run_feature_selection(cfg, data)
    cv_results, fold_coefs, oof_df = run_cv(cfg, data, features, model_spec)
    exhaustive = run_exhaustive_post(
        cfg, data, features, cv_results, fold_coefs, oof_df, model_spec,
    )
    save_results(
        cfg, data, features, cv_results, exhaustive["final"],
        fold_coefs, model_spec, exhaustive=exhaustive, oof_df=oof_df,
    )


def _phase_features(cfg: Config, model_spec: Any) -> None:
    data = load_and_split(cfg)
    features = run_feature_selection(cfg, data)
    save_features(cfg, data, features)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incident-validation pipeline runner (thin CLI over ivlib)."
    )
    p.add_argument("--model", required=True, choices=list(VALID_MODELS))
    p.add_argument("--phase", default="all",
                   choices=["all", "features", "cv", "aggregate"])
    p.add_argument("--strategy", default=None)
    p.add_argument("--weight-scheme", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.phase == "cv" and (not args.strategy or not args.weight_scheme):
        p.error("--phase=cv requires --strategy and --weight-scheme")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S", force=True,
    )
    root_handlers = logging.getLogger().handlers
    if root_handlers:
        root_handlers[0].stream = open(
            sys.stderr.fileno(), "w", buffering=1, closefd=False,
        )

    args = _parse_args()
    cfg = Config(smoke=args.smoke, model=args.model)
    cfg.output_dir = (Path(args.output_dir) if args.output_dir
                      else Path(MODEL_OUTPUT_DIRS[args.model]))
    if args.data_path:
        cfg.data_path = Path(args.data_path)
    model_spec = get_model_spec(args.model)

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INCIDENT VALIDATION PIPELINE -- %s (phase=%s)",
                args.model, args.phase)
    logger.info("=" * 60)
    if cfg.smoke:
        logger.info("*** SMOKE TEST MODE ***")

    dispatch = {
        "all": lambda: _phase_all(cfg, model_spec),
        "features": lambda: _phase_features(cfg, model_spec),
        "cv": lambda: _phase_cv(cfg, model_spec, args.strategy, args.weight_scheme),
        "aggregate": lambda: _phase_aggregate(cfg, model_spec),
    }
    dispatch[args.phase]()

    elapsed = time.time() - t0
    logger.info("Phase '%s' complete in %.1f minutes", args.phase, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
