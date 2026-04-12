#!/usr/bin/env python3
"""Extract top-K best hyperparameters from scout studies for warm-starting.

Reads Optuna JournalStorage files from the scout run, extracts the
best trial params per model, and writes a JSON file that config_gen.py
can inject into the main factorial cells via warm_start_params.

Usage:
    python extract_scout_params.py \
        --storage-dir /path/to/optuna/ \
        --output scout_top_params.json \
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_scout_params(
    storage_dir: Path,
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """Extract top-K best params per model from scout studies.

    Parameters
    ----------
    storage_dir : Path
        Directory containing .optuna.journal files.
    top_k : int
        Number of best params to extract per model.

    Returns
    -------
    dict[str, list[dict]]
        Mapping model name -> list of param dicts (best first).
    """
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    results: dict[str, list[dict]] = {}

    journal_files = sorted(storage_dir.glob("*.optuna.journal"))
    if not journal_files:
        logger.warning("No .optuna.journal files found in %s", storage_dir)
        return results

    for journal_path in journal_files:
        try:
            lock_obj = optuna.storages.JournalFileOpenLock(str(journal_path))
            storage = JournalStorage(
                JournalFileStorage(str(journal_path), lock_obj=lock_obj)
            )
        except Exception as e:
            logger.warning("Failed to open %s: %s", journal_path, e)
            continue

        summaries = optuna.study.get_all_study_summaries(storage)
        for summary in summaries:
            try:
                study = optuna.load_study(
                    study_name=summary.study_name, storage=storage
                )
            except Exception as e:
                logger.warning("Failed to load study '%s': %s", summary.study_name, e)
                continue

            model = study.user_attrs.get("model")
            if not model:
                continue

            # Skip if we already have enough for this model
            if model in results and len(results[model]) >= top_k:
                continue

            # Get Pareto-optimal trials sorted by AUROC (first objective, descending)
            try:
                best_trials = sorted(
                    study.best_trials,
                    key=lambda t: t.values[0],
                    reverse=True,
                )
            except Exception:
                # Single-objective fallback
                completed = [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                best_trials = sorted(
                    completed, key=lambda t: t.value, reverse=True
                )

            for trial in best_trials[:top_k]:
                if model not in results:
                    results[model] = []
                if len(results[model]) < top_k:
                    results[model].append(trial.params)

    for model, params in results.items():
        logger.info("Model %s: extracted %d param sets", model, len(params))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract scout params for warm-starting"
    )
    parser.add_argument(
        "--storage-dir",
        required=True,
        type=Path,
        help="Directory with .optuna.journal files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best params per model (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    results = extract_scout_params(args.storage_dir, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Wrote %d models to %s", len(results), args.output)


if __name__ == "__main__":
    main()
