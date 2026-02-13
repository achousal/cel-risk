"""CLI implementation for permutation testing of model significance.

This module provides the `ced permutation-test` command for testing whether
trained models generalize above chance level via label permutation testing.

USAGE MODES:
    1. Local parallel mode: --run-id <id> --model <model> [--n-jobs N]
    2. Run-based (all models): --run-id <id>

WORKFLOW:
    1. Train base models across splits (ced train)
    2. Test significance:
       - Local: ced permutation-test --run-id <RUN_ID> --model LR_EN --n-jobs 4
       - HPC (orchestrator): submit via ced run-pipeline --hpc

OUTPUT STRUCTURE:
    results/run_{RUN_ID}/{MODEL}/significance/
        permutation_test_results.csv     # Summary metrics per fold
        null_distribution.csv            # Full null distributions
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.significance.aggregation import (
    pool_null_distribution,
)
from ced_ml.significance.permutation_test import (
    aggregate_permutation_results,
    run_permutation_test,
    save_null_distributions,
)


def find_trained_model_path(
    run_id: str,
    model: str,
    split_seed: int = 0,
) -> Path:
    """Find trained model artifact path for a given run, model, and split.

    Args:
        run_id: Run ID (e.g., "20260127_115115")
        model: Model name (e.g., "LR_EN")
        split_seed: Split seed (default: 0)

    Returns:
        Path to trained model joblib file

    Raises:
        FileNotFoundError: If model artifact not found
    """
    from ced_ml.cli.discovery import get_run_path

    run_path = get_run_path(run_id)
    model_path = (
        run_path
        / model
        / "splits"
        / f"split_seed{split_seed}"
        / "core"
        / f"{model}__final_model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run: {run_id}, Model: {model}, Split: {split_seed}\n"
            f"Ensure model has been trained via 'ced train' before running permutation tests."
        )

    return model_path


def _aggregate_existing_results(
    *,
    sig_dir: Path,
    run_path: Path,
    model_name: str,
    run_id: str,
    logger: logging.Logger,
) -> bool:
    """Aggregate per-seed null distribution CSVs for a single model.

    Looks for null_distribution_seed*.csv files produced by the full-command
    HPC mode (one job per seed, all permutations in one job).

    Returns True if aggregation was performed, False if no results found.
    """
    null_csv_files = sorted(sig_dir.glob("null_distribution_seed*.csv"))
    if not null_csv_files:
        return False

    logger.info(
        f"Found {len(null_csv_files)} per-seed null distribution CSVs "
        f"- aggregating across seeds"
    )
    dfs = []
    for csv_file in null_csv_files:
        try:
            df_seed = pd.read_csv(csv_file)
            dfs.append(df_seed)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")

    if not dfs:
        return False

    df_combined = pd.concat(dfs, ignore_index=True)

    val_metrics_file = run_path / model_name / "aggregated" / "metrics" / "pooled_val_metrics.csv"
    if val_metrics_file.exists():
        val_df = pd.read_csv(val_metrics_file)
        if "auroc" in val_df.columns:
            df_combined["observed_auroc"] = float(val_df["auroc"].iloc[0])
    else:
        logger.warning(
            f"Observed AUROC file not found: {val_metrics_file}\n"
            f"Run 'ced aggregate-splits --run-id {run_id}' first to "
            f"produce pooled_val_metrics.csv."
        )

    result = pool_null_distribution(df_combined, model=model_name, alpha=0.05)

    agg_df = pd.DataFrame(
        [
            {
                "model": result.model,
                "observed_auroc": result.observed_auroc,
                "empirical_p_value": result.empirical_p_value,
                "n_seeds": result.n_seeds,
                "n_perms_total": result.n_perms_total,
                "significant": result.significant,
                "alpha": result.alpha,
                **result.summary_stats(),
            }
        ]
    )
    agg_path = sig_dir / "aggregated_significance.csv"
    agg_df.to_csv(agg_path, index=False)

    print(f"\n{'='*60}")
    print(f"Aggregated Permutation Results: {model_name}")
    print(f"{'='*60}")
    if result.n_seeds > 1:
        print(f"Seeds aggregated: {result.n_seeds}")
    print(f"Permutations aggregated: {result.n_perms_total}")
    print(f"Observed AUROC: {result.observed_auroc:.4f}")
    print(f"Empirical p-value: {result.empirical_p_value:.4f}")
    print(f"Significant (alpha=0.05): {result.significant}")
    print(f"Saved to: {agg_path}")
    print(f"{'='*60}\n")

    return True


def run_permutation_test_cli(
    run_id: str | None = None,
    model: str | None = None,
    split_seeds: list[int] | None = None,
    n_perms: int = 200,
    metric: str = "auroc",
    n_jobs: int = 1,
    outdir: str | None = None,
    random_state: int = 42,
    log_level: int | None = None,
    aggregate_only: bool = False,
) -> None:
    """Run permutation test for trained model(s) across one or more split seeds.

    Tests null hypothesis that model performance is no better than chance
    by comparing observed AUROC against null distribution from label permutations.

    Args:
        run_id: Run ID to test (required for auto-discovery)
        model: Specific model to test (default: all models with results)
        split_seeds: List of split seeds to test (default: [0])
        n_perms: Number of permutations per seed (default: 200)
        metric: Metric to use (default: 'auroc', only AUROC supported per ADR-007)
        n_jobs: Parallel jobs for in-process parallelization (default: 1)
        outdir: Output directory (default: {run_dir}/{model}/significance/)
        random_state: Random seed for reproducibility
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        aggregate_only: If True, only aggregate existing per-seed CSVs into
            a single aggregated_significance.csv (do not run permutations).
            Used after HPC per-seed jobs complete.

    Raises:
        FileNotFoundError: If model artifacts or data not found
        ValueError: If invalid parameters provided
    """
    if split_seeds is None:
        split_seeds = [0]
    if log_level is None:
        log_level = logging.INFO

    from ced_ml.utils.logging import setup_command_logger

    if metric != "auroc":
        raise ValueError(f"Only metric='auroc' is supported (per ADR-007). Got: {metric}")

    if not run_id:
        raise ValueError("--run-id is required for auto-discovery")

    from ced_ml.cli.discovery import discover_models_for_run, get_run_path

    run_path = get_run_path(run_id)

    if model:
        models_to_test = [model]
    else:
        models_to_test = discover_models_for_run(run_id, skip_ensemble=True)
        if not models_to_test:
            raise FileNotFoundError(f"No base models found for run {run_id} (ENSEMBLE excluded)")

    logger = setup_command_logger(
        command="permutation-test",
        log_level=log_level,
        outdir=run_path,
        run_id=run_id,
        model=model or "all",
        split_seed=split_seeds[0],
        logger_name=f"ced_ml.permutation_test.{run_id}",
    )

    logger.info(
        f"Starting permutation test: run_id={run_id}, models={models_to_test}, "
        f"split_seeds={split_seeds}, n_perms={n_perms}, n_jobs={n_jobs}"
    )

    for model_name in models_to_test:
        logger.info(f"\n{'='*60}\nTesting model: {model_name}\n{'='*60}")

        sig_dir = run_path / model_name / "significance"

        # --- Aggregate-only mode ---
        # Only entered when explicitly requested via --aggregate-only.
        # Pools existing per-seed null distribution CSVs into a single
        # aggregated_significance.csv with a model-level p-value.
        if aggregate_only:
            if not sig_dir.exists():
                logger.warning(
                    f"No significance directory for {model_name} -- nothing to aggregate"
                )
                continue

            aggregated = _aggregate_existing_results(
                sig_dir=sig_dir,
                run_path=run_path,
                model_name=model_name,
                run_id=run_id,
                logger=logger,
            )
            if not aggregated:
                logger.warning(f"No permutation results found to aggregate for {model_name}")
            continue

        # Load first seed's model bundle to get column metadata and scenario
        first_model_path = find_trained_model_path(run_id, model_name, split_seeds[0])
        first_bundle = joblib.load(first_model_path)
        if not isinstance(first_bundle, dict):
            raise ValueError("Model bundle must be a dictionary")

        resolved_cols = first_bundle.get("resolved_columns", {})
        scenario = first_bundle.get("scenario", "IncidentPlusPrevalent")

        protein_cols = resolved_cols.get("protein_cols", [])
        cat_cols = resolved_cols.get("categorical_metadata", [])
        meta_num_cols = resolved_cols.get("numeric_metadata", [])

        if not protein_cols:
            raise ValueError("Model bundle missing protein_cols")

        logger.info(f"Model metadata: {len(protein_cols)} proteins, scenario={scenario}")

        # Auto-detect data file and split directory from run metadata
        metadata_file = run_path / "run_metadata.json"
        if not metadata_file.exists():
            split_dirs = list((run_path / model_name).glob("splits/split_seed*"))
            if split_dirs:
                metadata_file = split_dirs[0] / "run_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Could not find run_metadata.json in {run_path}. "
                f"Cannot auto-detect input file and split directory."
            )

        import json

        with open(metadata_file) as f:
            metadata = json.load(f)

        infile = metadata.get("infile")
        split_dir = metadata.get("split_dir") or metadata.get("splits_dir")
        if not infile or not split_dir:
            model_meta = metadata.get("models", {}).get(model_name, {})
            infile = infile or model_meta.get("infile")
            split_dir = split_dir or model_meta.get("split_dir")

        if not infile or not split_dir:
            raise ValueError(
                f"Could not auto-detect infile and split_dir from run metadata. "
                f"Metadata: {metadata_file}"
            )

        logger.info(f"Input file: {infile}")
        logger.info(f"Split dir: {split_dir}")

        # Load data once per model (shared across seeds)
        logger.info("Loading data...")
        df_raw = read_proteomics_file(infile, validate=True)

        df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
        logger.info(f"Filtered: {filter_stats['n_in']:,} -> {filter_stats['n_out']:,} rows")

        feature_cols = protein_cols + cat_cols + meta_num_cols
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing[:10]}...")
            feature_cols = [c for c in feature_cols if c in df.columns]

        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in data")

        positive_label = get_positive_label(scenario)
        y_all = (df[TARGET_COL] == positive_label).astype(int).values
        X_all = df[feature_cols]

        split_path = Path(split_dir)

        # Determine output directory
        if outdir is None:
            seed_outdir = run_path / model_name / "significance"
        else:
            seed_outdir = Path(outdir)
        seed_outdir.mkdir(parents=True, exist_ok=True)

        # -- Per-seed loop --
        for split_seed in split_seeds:
            logger.info(f"\n{'-'*40}\n" f"  Split seed: {split_seed}\n" f"{'-'*40}")

            # Seed-level idempotency: skip if THIS seed's output already exists
            existing_null_csv = seed_outdir / f"null_distribution_seed{split_seed}.csv"
            if existing_null_csv.exists():
                logger.info(
                    f"Seed {split_seed} already completed: {existing_null_csv.name} "
                    f"exists -- skipping"
                )
                continue

            # Load model bundle for this seed
            model_path = find_trained_model_path(run_id, model_name, split_seed)
            logger.info(f"Loading model from: {model_path}")

            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict):
                raise ValueError("Model bundle must be a dictionary")

            pipeline = bundle.get("model")
            if pipeline is None:
                raise ValueError(f"Model bundle missing 'model' key: {model_path}")

            # Load split indices for this seed
            train_file = split_path / f"train_idx_{scenario}_seed{split_seed}.csv"
            val_file = split_path / f"val_idx_{scenario}_seed{split_seed}.csv"

            if not train_file.exists() or not val_file.exists():
                raise FileNotFoundError(
                    f"Split files not found: {train_file}, {val_file}\n"
                    f"Run 'ced save-splits' to generate splits with scenario={scenario}"
                )

            train_idx = pd.read_csv(train_file).squeeze().values
            val_idx = pd.read_csv(val_file).squeeze().values

            logger.info(f"Train: {len(train_idx)} samples, {y_all[train_idx].sum()} cases")
            logger.info(f"Val: {len(val_idx)} samples, {y_all[val_idx].sum()} cases")

            # Local parallel mode: run full permutation test
            logger.info(f"Running {n_perms} permutations with {n_jobs} jobs")

            result = run_permutation_test(
                pipeline=pipeline,
                X=X_all,
                y=y_all,
                train_idx=train_idx,
                test_idx=val_idx,
                model_name=model_name,
                split_seed=split_seed,
                outer_fold=0,
                n_perms=n_perms,
                n_jobs=n_jobs,
                random_state=random_state,
            )

            results = [result]

            summary_df = aggregate_permutation_results(results)
            summary_path = seed_outdir / f"permutation_test_results_seed{split_seed}.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Saved summary to {summary_path}")

            null_path = seed_outdir / f"null_distribution_seed{split_seed}.csv"
            save_null_distributions(results, str(null_path))

            print(f"\n{'='*60}")
            print(f"Permutation Test Results: {model_name} (seed {split_seed})")
            print(f"{'='*60}")
            print(f"Observed AUROC: {result.observed_auroc:.4f}")
            print(f"p-value: {result.p_value:.4f}")
            print("Null distribution:")
            print(f"  Mean: {np.mean(result.null_aurocs):.4f}")
            print(f"  Std: {np.std(result.null_aurocs):.4f}")
            print(f"  Min: {np.min(result.null_aurocs):.4f}")
            print(f"  Max: {np.max(result.null_aurocs):.4f}")
            print(f"  Median: {np.median(result.null_aurocs):.4f}")
            print("\nInterpretation:")
            if result.p_value < 0.05:
                print(
                    f"  Strong evidence that model generalizes above chance "
                    f"(p={result.p_value:.4f} < 0.05)"
                )
            elif result.p_value < 0.10:
                print(f"  Marginal evidence of generalization (p={result.p_value:.4f} < 0.10)")
            else:
                print(
                    f"  No evidence of generalization above chance "
                    f"(p={result.p_value:.4f} >= 0.10)"
                )
            print(f"\nResults saved to: {seed_outdir}")
            print(f"  - {summary_path.name}")
            print(f"  - {null_path.name}")
            print(f"{'='*60}\n")

        # Auto-aggregate across seeds when running multiple seeds locally
        # (not needed in aggregate_only mode -- that handles aggregation above)
        if len(split_seeds) > 1 and not aggregate_only:
            logger.info("Auto-aggregating results across seeds...")
            _aggregate_existing_results(
                sig_dir=seed_outdir,
                run_path=run_path,
                model_name=model_name,
                run_id=run_id,
                logger=logger,
            )

    logger.info("Permutation testing completed successfully")
