"""CLI implementation for cross-model consensus panel generation.

This module provides the `ced consensus-panel` command for creating consensus
protein panels from multiple models via Robust Rank Aggregation.

USAGE:
    ced consensus-panel --run-id 20260127_115115

WORKFLOW:
    1. Train base models (ced train --model LR_EN/RF/XGBoost/LinSVM_cal)
    2. Aggregate results (ced aggregate-splits)
    3. (Optional) Run panel optimization (ced optimize-panel)
    4. Generate consensus: ced consensus-panel --run-id <RUN_ID>
    5. Validate: ced train --fixed-panel results/consensus_panel/.../final_panel.txt --split-seed 10

OUTPUT:
    results/consensus_panel/run_<RUN_ID>/
        final_panel.txt          # One protein per line (for --fixed-panel)
        final_panel.csv          # Panel with uncertainty metrics
        consensus_ranking.csv    # All proteins with RRA scores and uncertainty
        uncertainty_summary.csv  # Focused uncertainty report for final panel
        per_model_rankings.csv   # Per-model composite rankings
        correlation_clusters.csv # Cluster assignments
        consensus_metadata.json  # Run parameters, statistics, and uncertainty summary

UNCERTAINTY METRICS:
    - n_models_present: Number of models with this protein (cross-model agreement)
    - agreement_strength: Fraction of models agreeing (0-1)
    - rank_std: Standard deviation of ranks across models
    - rank_cv: Coefficient of variation (std/mean) - lower = more stable ranking
"""

import logging
from pathlib import Path

import pandas as pd

from ced_ml.cli.discovery import (
    auto_detect_data_paths,
    discover_models_with_aggregated_results,
)
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.features.consensus import (
    ConsensusResult,
    build_consensus_panel,
    save_consensus_results,
)
from ced_ml.features.corr_prune import (
    build_correlation_graph,
    compute_correlation_matrix,
    find_connected_components,
)
from ced_ml.features.drop_column import (
    aggregate_drop_column_results,
    compute_drop_column_importance,
)

logger = logging.getLogger(__name__)


def _extract_bundle_metadata(model_path: Path) -> dict:
    """Extract only metadata keys from a model bundle without full deserialization.

    When the training and loading environments differ (e.g. pandas version
    mismatch causing StringDtype pickle errors), the sklearn pipeline inside
    the bundle cannot be unpickled.  This function uses pickle's Unpickler
    with a permissive class loader so that unresolvable objects are replaced
    by stubs, allowing the top-level dict and its lightweight values
    (``resolved_columns``, ``scenario``) to be recovered.
    """
    import io
    import pickle

    class _StubUnpickler(pickle.Unpickler):
        """Unpickler that replaces unresolvable classes with a placeholder."""

        def find_class(self, module: str, name: str):
            try:
                return super().find_class(module, name)
            except Exception:
                # Return a no-op callable that absorbs any constructor args
                return lambda *a, **kw: None

    with open(model_path, "rb") as fh:
        raw = fh.read()

    bundle = _StubUnpickler(io.BytesIO(raw)).load()
    if not isinstance(bundle, dict):
        raise ValueError(f"Metadata-only extraction failed: expected dict, got {type(bundle)}")
    return bundle


def load_aggregated_significance(run_dir: Path) -> pd.DataFrame | None:
    """Load aggregated significance results for all models in a run.

    Args:
        run_dir: Path to run directory (results/run_{ID}/)

    Returns:
        DataFrame with columns [model, empirical_p_value, significant, ...] or None if not found.
    """
    sig_files = list(run_dir.glob("*/significance/aggregated_significance.csv"))
    if not sig_files:
        return None

    dfs = []
    for f in sig_files:
        try:
            df = pd.read_csv(f)
            if "model" not in df.columns:
                model_name = f.parent.parent.name
                df["model"] = model_name
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def load_model_stability(
    aggregated_dir: Path,
    stability_threshold: float = 0.0,
) -> pd.DataFrame:
    """Load feature stability summary from aggregated results.

    Args:
        aggregated_dir: Path to model's aggregated directory.
        stability_threshold: Minimum selection fraction (0 = load all).

    Returns:
        DataFrame with columns [protein, selection_fraction, ...].

    Raises:
        FileNotFoundError: If stability file not found.
    """
    stability_file = aggregated_dir / "panels" / "feature_stability_summary.csv"

    if not stability_file.exists():
        raise FileNotFoundError(
            f"Feature stability file not found: {stability_file}\nRun 'ced aggregate-splits' first."
        )

    df = pd.read_csv(stability_file)

    # Clean protein names (may have extra quotes)
    if "protein" in df.columns:
        df["protein"] = df["protein"].str.strip('"')

    # Filter by threshold if requested
    if stability_threshold > 0 and "selection_fraction" in df.columns:
        df = df[df["selection_fraction"] >= stability_threshold].copy()

    return df


def load_model_oof_importance(
    aggregated_dir: Path,
) -> pd.DataFrame | None:
    """Load OOF grouped importance from aggregated results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.

    Returns:
        DataFrame with OOF importance (columns: feature, mean_importance, etc.), or None.
    """
    # Check for aggregated OOF importance file
    oof_file = aggregated_dir / "importance" / "aggregated_oof_importance.csv"

    if not oof_file.exists():
        # Try alternative location
        oof_file = aggregated_dir / "importance" / "importance_aggregated.csv"

    if not oof_file.exists():
        return None

    try:
        df = pd.read_csv(oof_file)
        # Standardize column names
        if "feature" not in df.columns and "protein" in df.columns:
            df = df.rename(columns={"protein": "feature"})
        if "importance" in df.columns and "mean_importance" not in df.columns:
            df = df.rename(columns={"importance": "mean_importance"})
        return df
    except Exception as e:
        logger.warning(f"Failed to load OOF importance from {oof_file}: {e}")
        return None


def load_model_essentiality(
    aggregated_dir: Path,
    threshold: str = "95pct",
) -> pd.DataFrame | None:
    """Load drop-column essentiality from optimize-panel results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.
        threshold: Panel threshold to load (e.g., "95pct", "99pct").

    Returns:
        DataFrame with essentiality data (columns: cluster_id, mean_delta_auroc, etc.), or None.
    """
    # Check for essentiality file
    ess_file = (
        aggregated_dir / "optimize_panel" / "essentiality" / f"panel_{threshold}_essentiality.csv"
    )

    if not ess_file.exists():
        # Try alternative: single drop_column_validation.csv
        ess_file = aggregated_dir / "optimize_panel" / "drop_column_validation.csv"

    if not ess_file.exists():
        return None

    try:
        df = pd.read_csv(ess_file)
        return df
    except Exception as e:
        logger.warning(f"Failed to load essentiality from {ess_file}: {e}")
        return None


def run_consensus_panel(
    run_id: str,
    infile: str | None = None,
    split_dir: str | None = None,
    stability_threshold: float = 0.90,
    corr_threshold: float = 0.85,
    target_size: int = 25,
    rra_method: str = "geometric_mean",
    outdir: str | None = None,
    log_level: int | None = None,
    require_significance: bool = False,
    significance_alpha: float = 0.05,
    min_significant_models: int = 2,
    oof_weight: float = 0.6,
    essentiality_weight: float = 0.3,
    stability_weight: float = 0.1,
    run_essentiality: bool = True,
    essentiality_corr_threshold: float = 0.75,
    essentiality_include_brier: bool = False,
) -> ConsensusResult:
    """Run consensus panel generation from multiple models.

    Uses composite ranking per ADR-004: OOF importance (primary) +
    Essentiality (secondary) + Stability (tie-break).

    Args:
        run_id: Run ID to process.
        infile: Input data file (auto-detected if None).
        split_dir: Split directory (auto-detected if None).
        stability_threshold: Minimum selection fraction for stable proteins.
        corr_threshold: Correlation threshold for clustering.
        target_size: Target panel size.
        rra_method: RRA aggregation method.
        outdir: Output directory (default: results/consensus_panel/run_<RUN_ID>).
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)
        require_significance: Whether to filter models by permutation test significance.
        significance_alpha: P-value threshold for significance filtering.
        min_significant_models: Minimum number of significant models required.
        oof_weight: Weight for OOF importance (default 0.6).
        essentiality_weight: Weight for essentiality (default 0.3).
        stability_weight: Weight for stability (default 0.1).
        run_essentiality: Whether to run drop-column essentiality validation (default True).
        essentiality_corr_threshold: Correlation threshold for clustering in drop-column (default 0.75).
        essentiality_include_brier: Whether to include Brier score delta in output (default False).

    Returns:
        ConsensusResult with final panel and intermediate data.

    Raises:
        FileNotFoundError: If required files not found.
        ValueError: If insufficient models or proteins.
    """
    # Setup logging
    from ced_ml.utils.logging import setup_command_logger

    if log_level is None:
        log_level = logging.INFO

    # Auto-file-logging
    logger = setup_command_logger(
        command="consensus-panel",
        log_level=log_level,
        outdir=outdir or "results",
        run_id=run_id,
        logger_name=f"ced_ml.consensus_panel.{run_id}",
    )
    logger.info(f"Consensus panel generation started for run_id={run_id}")

    # Determine results root (CED_RESULTS_DIR env var for testability)
    import os

    from ced_ml.utils.paths import get_project_root

    results_dir_env = os.environ.get("CED_RESULTS_DIR")
    if results_dir_env:
        results_dir = Path(results_dir_env)
    else:
        results_dir = get_project_root() / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Discover models with aggregated results
    logger.info("Discovering models with aggregated stability results...")
    model_dirs = discover_models_with_aggregated_results(
        run_id=run_id,
        results_dir=results_dir,
        skip_ensemble=True,
    )

    logger.info(f"Found {len(model_dirs)} models: {list(model_dirs.keys())}")

    # Significance filtering
    if require_significance:
        run_path = results_dir / f"run_{run_id}"
        sig_df = load_aggregated_significance(run_path)

        if sig_df is not None:
            # Filter to significant models only
            sig_models = sig_df[sig_df["empirical_p_value"] < significance_alpha]["model"].tolist()

            original_count = len(model_dirs)
            model_dirs = {m: p for m, p in model_dirs.items() if m in sig_models}

            if len(model_dirs) < original_count:
                skipped = original_count - len(model_dirs)
                logger.info(
                    f"Significance filtering: {skipped} model(s) excluded "
                    f"(p >= {significance_alpha})"
                )

            if len(model_dirs) < min_significant_models:
                raise ValueError(
                    f"Only {len(model_dirs)} significant model(s) found "
                    f"(need {min_significant_models}). "
                    f"Significant models: {list(model_dirs.keys())}\n"
                    f"Run permutation tests first: ced permutation-test --run-id {run_id}"
                )

            logger.info(
                f"Proceeding with {len(model_dirs)} significant model(s): {list(model_dirs.keys())}"
            )
        else:
            logger.warning(
                f"No significance data found for run {run_id}. "
                f"Run permutation tests first: ced permutation-test --run-id {run_id}\n"
                f"Proceeding without significance filtering."
            )

    # Auto-detect data paths if not provided
    if not infile or not split_dir:
        auto_infile, auto_split_dir = auto_detect_data_paths(run_id, results_dir)
        if not infile:
            infile = auto_infile
        if not split_dir:
            split_dir = auto_split_dir

    if not infile:
        raise ValueError(
            f"Could not auto-detect input file for run_id={run_id}.\n"
            f"Searched in: {results_dir}\n"
            f"Found models: {list(model_dirs.keys())}\n"
            f"Please provide --infile explicitly."
        )
    if not split_dir:
        raise ValueError(
            f"Could not auto-detect split directory for run_id={run_id}.\n"
            f"Searched in: {results_dir}\n"
            f"Found models: {list(model_dirs.keys())}\n"
            f"Please provide --split-dir explicitly."
        )

    logger.info(f"Input file: {infile}")
    logger.info(f"Split directory: {split_dir}")

    # Load stability data for each model
    logger.info("Loading ranking data from each model...")
    model_stability = {}
    model_oof_importance = {}
    model_essentiality = {}

    for model_name, aggregated_dir in model_dirs.items():
        # Load stability (always needed)
        stability_df = load_model_stability(aggregated_dir, stability_threshold=0.0)
        model_stability[model_name] = stability_df

        n_stable = (stability_df["selection_fraction"] >= stability_threshold).sum()

        # Load OOF importance and essentiality
        has_oof = False
        has_essentiality = False

        oof_df = load_model_oof_importance(aggregated_dir)
        if oof_df is not None:
            model_oof_importance[model_name] = oof_df
            has_oof = True

        ess_df = load_model_essentiality(aggregated_dir)
        if ess_df is not None:
            model_essentiality[model_name] = ess_df
            has_essentiality = True

        logger.info(
            f"  {model_name}: {len(stability_df)} total proteins, "
            f"{n_stable} stable (>={stability_threshold}), "
            f"OOF={'yes' if has_oof else 'no'}, essentiality={'yes' if has_essentiality else 'no'}"
        )

    # Load training data for correlation computation
    logger.info(f"Loading training data from {infile}...")
    df_raw = read_proteomics_file(infile, validate=True)

    # Get metadata columns from first available model
    first_model = next(iter(model_dirs.keys()))
    first_aggregated = model_dirs[first_model]
    run_dir = first_aggregated.parent

    # Find representative split for metadata
    split_dirs = sorted(run_dir.glob("splits/split_seed*"))
    if not split_dirs:
        split_dirs = sorted(run_dir.glob("split_seed*"))
    if not split_dirs:
        raise FileNotFoundError(f"No split directories found in {run_dir}")

    representative_split = split_dirs[0]
    representative_seed = int(representative_split.name.replace("split_seed", ""))

    # Load model bundle for metadata (only need resolved_columns + scenario)
    model_path = representative_split / "core" / f"{first_model}__final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    import joblib

    try:
        bundle = joblib.load(model_path)
    except TypeError as exc:
        # Pickle deserialization can fail when pandas versions differ between
        # the environment that trained the model and the one loading it
        # (e.g. StringDtype signature changes).  We only need two lightweight
        # metadata keys, so attempt a selective unpickle.
        logger.warning(f"Full bundle load failed ({exc}); attempting metadata-only extraction")
        bundle = _extract_bundle_metadata(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentPlusPrevalent")
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    # Apply row filters
    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} -> {filter_stats['n_out']:,} rows")

    # Load train indices
    split_path = Path(split_dir)
    train_file = split_path / f"train_idx_{scenario}_seed{representative_seed}.csv"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Train indices not found: {train_file}\n"
            f"Expected scenario-specific format: train_idx_{{scenario}}_seed{{N}}.csv\n"
            f"Scenario: {scenario}, Seed: {representative_seed}\n"
            f"Run 'ced save-splits' to generate splits with the modern format."
        )

    train_idx = pd.read_csv(train_file).squeeze().values
    df_train = df.iloc[train_idx].copy()

    logger.info(f"Training data: {len(df_train)} samples")

    # Build consensus panel
    logger.info("Building consensus panel...")
    result = build_consensus_panel(
        model_stability=model_stability,
        df_train=df_train,
        stability_threshold=stability_threshold,
        corr_threshold=corr_threshold,
        target_size=target_size,
        rra_method=rra_method,
        model_oof_importance=model_oof_importance,
        model_essentiality=model_essentiality,
        oof_weight=oof_weight,
        essentiality_weight=essentiality_weight,
        stability_weight=stability_weight,
    )

    # Save results
    if outdir is None:
        outdir = results_dir / f"run_{run_id}" / "consensus"
    else:
        outdir = Path(outdir)

    _paths = save_consensus_results(result, outdir)  # noqa: F841

    # --- Drop-column essentiality validation on consensus panel ---
    essentiality_summary = None
    if run_essentiality and len(result.final_panel) > 0:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("Running drop-column essentiality validation on consensus panel...")
            logger.info("=" * 60)

            # Create binary target vector
            if TARGET_COL not in df.columns:
                raise ValueError(f"Target column '{TARGET_COL}' not found in data")

            positive_label = get_positive_label(scenario)
            y_all = (df[TARGET_COL] == positive_label).astype(int).values
            logger.info(f"Target vector: {y_all.sum()} positive cases out of {len(y_all)} samples")

            # Create essentiality output directory
            essentiality_dir = outdir / "essentiality"
            essentiality_dir.mkdir(parents=True, exist_ok=True)

            # Get consensus panel features
            panel_features = [p for p in result.final_panel if p in df.columns]
            logger.info(f"Validating {len(panel_features)} panel features")

            # Build correlation clusters for the consensus panel
            X_corr = df_train[panel_features]
            corr_matrix = compute_correlation_matrix(X_corr, panel_features, method="spearman")
            adj_graph = build_correlation_graph(corr_matrix, threshold=essentiality_corr_threshold)
            clusters = find_connected_components(adj_graph)
            logger.info(f"Found {len(clusters)} correlation clusters for essentiality validation")

            # Run drop-column across all available splits
            drop_column_results_per_fold = []
            for split_dir_path in split_dirs:
                seed = int(split_dir_path.name.replace("split_seed", ""))

                # Try to find a model for this seed (use first available model)
                model_path = split_dir_path / "core" / f"{first_model}__final_model.joblib"

                if not model_path.exists():
                    logger.warning(f"Seed {seed}: model not found at {model_path}, skipping")
                    continue

                # Load split indices
                split_path = Path(split_dir)
                train_file = split_path / f"train_idx_{scenario}_seed{seed}.csv"
                val_file = split_path / f"val_idx_{scenario}_seed{seed}.csv"

                if not train_file.exists() or not val_file.exists():
                    logger.warning(f"Seed {seed}: split indices not found, skipping")
                    continue

                train_idx = pd.read_csv(train_file).squeeze().values
                val_idx = pd.read_csv(val_file).squeeze().values

                # Load model pipeline
                seed_bundle = joblib.load(model_path)
                pipeline = seed_bundle.get("model")

                if pipeline is None:
                    logger.warning(f"Seed {seed}: model bundle missing 'model' key, skipping")
                    continue

                X_train_seed = df.iloc[train_idx][panel_features]
                y_train_seed = y_all[train_idx]
                X_val_seed = df.iloc[val_idx][panel_features]
                y_val_seed = y_all[val_idx]

                fold_results = compute_drop_column_importance(
                    estimator=pipeline,
                    X_train=X_train_seed,
                    y_train=y_train_seed,
                    X_val=X_val_seed,
                    y_val=y_val_seed,
                    feature_clusters=clusters,
                    random_state=seed,
                )
                drop_column_results_per_fold.append(fold_results)
                logger.info(f"  Seed {seed}: completed drop-column validation")

            if drop_column_results_per_fold:
                # Aggregate results across folds
                drop_column_df = aggregate_drop_column_results(drop_column_results_per_fold)

                # Save essentiality results
                ess_path = essentiality_dir / "consensus_panel_essentiality.csv"
                drop_column_df.to_csv(ess_path, index=False)
                logger.info(f"Saved essentiality results to {ess_path}")

                # Create summary
                essentiality_summary = {
                    "panel_size": len(panel_features),
                    "n_clusters": len(clusters),
                    "n_folds_validated": len(drop_column_results_per_fold),
                    "mean_delta_auroc": float(drop_column_df["mean_delta_auroc"].mean()),
                    "max_delta_auroc": float(drop_column_df["mean_delta_auroc"].max()),
                    "top_cluster_id": str(drop_column_df.iloc[0]["cluster_id"]),
                    "top_cluster_delta": float(drop_column_df.iloc[0]["mean_delta_auroc"]),
                }

                # Save summary
                import json

                summary_path = essentiality_dir / "essentiality_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(essentiality_summary, f, indent=2)
                logger.info(f"Saved essentiality summary to {summary_path}")

            else:
                logger.warning("No folds completed drop-column validation")

        except Exception as e:
            logger.warning(f"Drop-column essentiality validation failed: {e}", exc_info=True)
            logger.info("Continuing without essentiality validation results")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Consensus Panel Generation Complete")
    print(f"{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(model_dirs.keys())}")
    print("\nParameters:")
    print(
        f"  Composite weights: OOF={oof_weight}, Ess={essentiality_weight}, Stab={stability_weight}"
    )
    print(f"  Stability threshold: {stability_threshold}")
    print(f"  Correlation threshold: {corr_threshold}")
    print(f"  Target size: {target_size}")
    print(f"  RRA method: {rra_method}")

    print("\nResults:")
    print(f"  Total proteins across models: {len(result.consensus_ranking)}")
    print(f"  Clusters after correlation pruning: {result.metadata['results']['n_clusters']}")
    print(f"  Final panel size: {len(result.final_panel)}")

    if require_significance:
        print("\nSignificance filtering:")
        print(f"  Required alpha: {significance_alpha}")
        print(f"  Minimum models: {min_significant_models}")
        print(f"  Significant models used: {list(model_dirs.keys())}")

    print("\nTop 10 proteins in consensus panel:")
    for i, protein in enumerate(result.final_panel[:10], 1):
        protein_row = result.consensus_ranking[result.consensus_ranking["protein"] == protein]
        score = protein_row["consensus_score"].iloc[0]
        n_models = protein_row["n_models_present"].iloc[0]
        agreement = protein_row["agreement_strength"].iloc[0]
        print(
            f"  {i:2d}. {protein} (score: {score:.4f}, "
            f"{n_models}/{len(model_dirs)} models, agreement: {agreement:.2f})"
        )

    # Print uncertainty summary
    if "uncertainty" in result.metadata:
        unc = result.metadata["uncertainty"]
        print("\nUncertainty Summary:")
        print(f"  Mean agreement strength: {unc['mean_agreement_strength']:.2f}")
        print(f"  Min agreement strength: {unc['min_agreement_strength']:.2f}")
        print(f"  Mean rank CV: {unc['mean_rank_cv']:.3f}")
        print(f"  Max rank CV: {unc['max_rank_cv']:.3f}")
        print(
            f"  Proteins in all models: {unc['proteins_in_all_models']}/{len(result.final_panel)}"
        )
        majority = unc["proteins_in_majority_models"]
        print(f"  Proteins in majority: {majority}/{len(result.final_panel)}")

    # Print essentiality summary if available
    if essentiality_summary:
        print("\nDrop-Column Essentiality Validation:")
        print(f"  Clusters validated: {essentiality_summary['n_clusters']}")
        print(f"  Folds validated: {essentiality_summary['n_folds_validated']}")
        print(f"  Mean delta AUROC: {essentiality_summary['mean_delta_auroc']:+.4f}")
        print(f"  Max delta AUROC: {essentiality_summary['max_delta_auroc']:+.4f}")
        print(
            f"  Top cluster: {essentiality_summary['top_cluster_id']} "
            f"(delta={essentiality_summary['top_cluster_delta']:+.4f})"
        )

    print(f"\nOutput saved to: {outdir}")
    print("  - final_panel.txt (for --fixed-panel)")
    print("  - final_panel.csv (with uncertainty metrics)")
    print("  - consensus_ranking.csv (all proteins with uncertainty)")
    print("  - uncertainty_summary.csv (focused uncertainty report)")
    print("  - per_model_rankings.csv")
    print("  - correlation_clusters.csv")
    print("  - consensus_metadata.json")
    if essentiality_summary:
        print("  - essentiality/consensus_panel_essentiality.csv")
        print("  - essentiality/essentiality_summary.json")

    print("\nNext step: Validate with new split seed:")
    print(f"  ced train --model LR_EN --fixed-panel {outdir}/final_panel.txt --split-seed 10")
    print(f"{'=' * 60}\n")

    logger.info("Consensus panel generation completed successfully")

    return result
