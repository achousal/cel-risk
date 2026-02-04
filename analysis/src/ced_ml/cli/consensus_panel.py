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
        final_panel.csv          # Panel with uncertainty metrics (n_models_present, agreement_strength, rank_cv)
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
from ced_ml.features.consensus import (
    ConsensusResult,
    build_consensus_panel,
    save_consensus_results,
)


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
            f"Feature stability file not found: {stability_file}\n"
            f"Run 'ced aggregate-splits' first."
        )

    df = pd.read_csv(stability_file)

    # Clean protein names (may have extra quotes)
    if "protein" in df.columns:
        df["protein"] = df["protein"].str.strip('"')

    # Filter by threshold if requested
    if stability_threshold > 0 and "selection_fraction" in df.columns:
        df = df[df["selection_fraction"] >= stability_threshold].copy()

    return df


def load_model_rfe_ranking(
    aggregated_dir: Path,
) -> dict[str, int] | None:
    """Load RFE feature ranking from aggregated results (if available).

    Args:
        aggregated_dir: Path to model's aggregated directory.

    Returns:
        Dict mapping protein -> elimination_order, or None if not available.

    Raises:
        FileNotFoundError: If feature_ranking_aggregated.csv exists but is malformed.
    """
    rfe_file = aggregated_dir / "optimize_panel" / "feature_ranking_aggregated.csv"

    if not rfe_file.exists():
        return None

    try:
        df = pd.read_csv(rfe_file)
        if "protein" not in df.columns or "elimination_order" not in df.columns:
            raise FileNotFoundError(
                f"Malformed RFE ranking file: {rfe_file}\n"
                f"Required columns: 'protein', 'elimination_order'\n"
                f"Found columns: {list(df.columns)}\n"
                f"Run 'ced optimize-panel' to generate aggregated RFE results."
            )
        return dict(zip(df["protein"], df["elimination_order"], strict=False))
    except pd.errors.EmptyDataError as e:
        raise FileNotFoundError(
            f"Empty RFE ranking file: {rfe_file}\n"
            f"Run 'ced optimize-panel' to generate aggregated RFE results."
        ) from e
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load RFE ranking file: {rfe_file}\n"
            f"Error: {e}\n"
            f"Run 'ced optimize-panel' to generate aggregated RFE results."
        ) from e


def run_consensus_panel(
    run_id: str,
    infile: str | None = None,
    split_dir: str | None = None,
    stability_threshold: float = 0.90,
    corr_threshold: float = 0.85,
    target_size: int = 25,
    rfe_weight: float = 0.5,
    rra_method: str = "geometric_mean",
    outdir: str | None = None,
    log_level: int | None = None,
) -> ConsensusResult:
    """Run consensus panel generation from multiple models.

    Args:
        run_id: Run ID to process.
        infile: Input data file (auto-detected if None).
        split_dir: Split directory (auto-detected if None).
        stability_threshold: Minimum selection fraction for stable proteins.
        corr_threshold: Correlation threshold for clustering.
        target_size: Target panel size.
        rfe_weight: Weight for RFE vs stability (0-1).
        rra_method: RRA aggregation method.
        outdir: Output directory (default: results/consensus_panel/run_<RUN_ID>).
        log_level: Logging level constant (logging.DEBUG, logging.INFO, etc.)

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
    logger.info("Loading stability data from each model...")
    model_stability = {}
    model_rfe_rankings = {}

    for model_name, aggregated_dir in model_dirs.items():
        # Load stability
        stability_df = load_model_stability(aggregated_dir, stability_threshold=0.0)
        model_stability[model_name] = stability_df

        # Load RFE ranking (optional)
        rfe_ranking = load_model_rfe_ranking(aggregated_dir)
        model_rfe_rankings[model_name] = rfe_ranking

        n_stable = (stability_df["selection_fraction"] >= stability_threshold).sum()
        has_rfe = rfe_ranking is not None

        logger.info(
            f"  {model_name}: {len(stability_df)} total proteins, "
            f"{n_stable} stable (>={stability_threshold}), RFE={'yes' if has_rfe else 'no'}"
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

    # Load model bundle for metadata
    model_path = representative_split / "core" / f"{first_model}__final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    import joblib

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary")

    resolved_cols = bundle.get("resolved_columns", {})
    scenario = bundle.get("scenario", "IncidentOnly")
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
        model_rfe_rankings=model_rfe_rankings,
        df_train=df_train,
        stability_threshold=stability_threshold,
        corr_threshold=corr_threshold,
        target_size=target_size,
        rfe_weight=rfe_weight,
        rra_method=rra_method,
    )

    # Save results
    if outdir is None:
        outdir = results_dir / f"run_{run_id}" / "consensus"
    else:
        outdir = Path(outdir)

    _paths = save_consensus_results(result, outdir)  # noqa: F841

    # Print summary
    print(f"\n{'='*60}")
    print("Consensus Panel Generation Complete")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Models: {', '.join(model_dirs.keys())}")
    print("\nParameters:")
    print(f"  Stability threshold: {stability_threshold}")
    print(f"  Correlation threshold: {corr_threshold}")
    print(f"  Target size: {target_size}")
    print(f"  RFE weight: {rfe_weight}")
    print(f"  RRA method: {rra_method}")

    print("\nResults:")
    print(f"  Total proteins across models: {len(result.consensus_ranking)}")
    print(f"  Clusters after correlation pruning: {result.metadata['results']['n_clusters']}")
    print(f"  Final panel size: {len(result.final_panel)}")

    print("\nTop 10 proteins in consensus panel:")
    for i, protein in enumerate(result.final_panel[:10], 1):
        protein_row = result.consensus_ranking[result.consensus_ranking["protein"] == protein]
        score = protein_row["consensus_score"].iloc[0]
        n_models = protein_row["n_models_present"].iloc[0]
        agreement = protein_row["agreement_strength"].iloc[0]
        print(
            f"  {i:2d}. {protein} (score: {score:.4f}, {n_models}/{len(model_dirs)} models, agreement: {agreement:.2f})"
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
        print(
            f"  Proteins in majority: {unc['proteins_in_majority_models']}/{len(result.final_panel)}"
        )

    print(f"\nOutput saved to: {outdir}")
    print("  - final_panel.txt (for --fixed-panel)")
    print("  - final_panel.csv (with uncertainty metrics)")
    print("  - consensus_ranking.csv (all proteins with uncertainty)")
    print("  - uncertainty_summary.csv (focused uncertainty report)")
    print("  - per_model_rankings.csv")
    print("  - correlation_clusters.csv")
    print("  - consensus_metadata.json")

    print("\nNext step: Validate with new split seed:")
    print(f"  ced train --model LR_EN --fixed-panel {outdir}/final_panel.txt --split-seed 10")
    print(f"{'='*60}\n")

    logger.info("Consensus panel generation completed successfully")

    return result
