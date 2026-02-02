#!/usr/bin/env python3
"""
Complete factorial experiment runner (chains tuning, experiment, and analysis).

This script orchestrates the full 2x2x2 factorial experiment workflow:
1. Tune baseline hyperparameters (if needed)
2. Run the full factorial experiment
3. Analyze results and generate visualizations

Usage:
    # Full workflow with tuning
    python run_factorial_complete.py \\
        --data-path ../../../data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path ../../../data/fixed_panel.csv \\
        --output-dir ../../../results/factorial_2x2x2 \\
        --n-seeds 10

    # Skip tuning (use existing hyperparams)
    python run_factorial_complete.py \\
        --data-path ../../../data/Celiac_dataset_proteomics_w_demo.parquet \\
        --panel-path ../../../data/fixed_panel.csv \\
        --output-dir ../../../results/factorial_2x2x2 \\
        --hyperparams-path ../../../results/factorial_2x2x2/frozen_hyperparams.yaml \\
        --n-seeds 10
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> None:
    """Run a subprocess command and handle errors."""
    logger.info("Starting: %s", description)
    logger.info("Command: %s", " ".join(str(c) for c in cmd))

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error("%s failed with return code %d", description, result.returncode)
        sys.exit(result.returncode)

    logger.info("Completed: %s", description)


def main():
    parser = argparse.ArgumentParser(
        description="Complete 2x2x2 factorial experiment (tune + run + analyze)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to proteomics parquet file",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        required=True,
        help="Path to fixed_panel.csv (one protein per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/factorial_2x2x2"),
        help="Output directory (default: results/factorial_2x2x2)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of seeds for experiment (default: 10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LR_EN", "XGBoost"],
        help="Models to run (default: LR_EN XGBoost)",
    )
    parser.add_argument(
        "--hyperparams-path",
        type=Path,
        default=None,
        help="Path to frozen_hyperparams.yaml (if exists, skip tuning)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning (requires --hyperparams-path)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top features for Jaccard overlap analysis (default: 15)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.data_path.exists():
        logger.error("Data file not found: %s", args.data_path)
        sys.exit(1)

    if not args.panel_path.exists():
        logger.error("Panel file not found: %s", args.panel_path)
        sys.exit(1)

    if args.skip_tuning and args.hyperparams_path is None:
        logger.error("--skip-tuning requires --hyperparams-path")
        sys.exit(1)

    # Determine hyperparams path
    if args.hyperparams_path is None:
        args.hyperparams_path = args.output_dir / "frozen_hyperparams.yaml"

    # Get path to run and analyze scripts
    script_dir = Path(__file__).parent
    run_script = script_dir / "run_factorial_2x2x2.py"
    analyze_script = script_dir / "analyze_factorial_2x2x2.py"

    if not run_script.exists():
        logger.error("Run script not found: %s", run_script)
        sys.exit(1)

    if not analyze_script.exists():
        logger.error("Analyze script not found: %s", analyze_script)
        sys.exit(1)

    # Step 1: Tune baseline (if needed)
    if not args.skip_tuning and not args.hyperparams_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 1: Tuning baseline hyperparameters")
        logger.info("=" * 60)

        tune_cmd = [
            sys.executable,
            str(run_script),
            "--data-path",
            str(args.data_path),
            "--panel-path",
            str(args.panel_path),
            "--output-dir",
            str(args.output_dir),
            "--models",
            *args.models,
            "--tune-baseline",
        ]

        run_command(tune_cmd, "Hyperparameter tuning")
    else:
        logger.info(
            "Skipping hyperparameter tuning (using existing: %s)", args.hyperparams_path
        )

    # Verify hyperparams exist
    if not args.hyperparams_path.exists():
        logger.error("Hyperparameters file not found: %s", args.hyperparams_path)
        sys.exit(1)

    # Step 2: Run experiment
    logger.info("=" * 60)
    logger.info("STEP 2: Running factorial experiment")
    logger.info("=" * 60)

    experiment_cmd = [
        sys.executable,
        str(run_script),
        "--data-path",
        str(args.data_path),
        "--panel-path",
        str(args.panel_path),
        "--output-dir",
        str(args.output_dir),
        "--n-seeds",
        str(args.n_seeds),
        "--models",
        *args.models,
        "--hyperparams-path",
        str(args.hyperparams_path),
    ]

    run_command(experiment_cmd, "Factorial experiment")

    # Verify results exist
    results_path = args.output_dir / "factorial_results.csv"
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        sys.exit(1)

    # Step 3: Analyze results
    logger.info("=" * 60)
    logger.info("STEP 3: Analyzing results and generating visualizations")
    logger.info("=" * 60)

    analyze_cmd = [
        sys.executable,
        str(analyze_script),
        "--results",
        str(results_path),
        "--top-k",
        str(args.top_k),
        "--output-dir",
        str(args.output_dir / "analysis"),
    ]

    # Add feature importances path if it exists
    fi_path = args.output_dir / "feature_importances.csv"
    if fi_path.exists():
        analyze_cmd.extend(["--feature-importances", str(fi_path)])

    run_command(analyze_cmd, "Results analysis")

    # Summary
    logger.info("=" * 60)
    logger.info("COMPLETE: All steps finished successfully")
    logger.info("=" * 60)
    logger.info("Results directory: %s", args.output_dir)
    logger.info("Analysis directory: %s", args.output_dir / "analysis")
    logger.info("")
    logger.info("Key outputs:")
    logger.info("  - Hyperparameters: %s", args.hyperparams_path)
    logger.info("  - Results: %s", results_path)
    logger.info("  - Feature importances: %s", fi_path)
    logger.info(
        "  - Analysis CSVs: %s/{main_effects,interactions,feature_jaccard}.csv",
        args.output_dir / "analysis",
    )
    logger.info("  - Visualizations: %s/*.png", args.output_dir / "analysis")
    logger.info("  - Summaries: %s/summary_*.md", args.output_dir / "analysis")


if __name__ == "__main__":
    main()
