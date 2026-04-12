#!/usr/bin/env python3
"""Live monitoring of factorial experiment progress via Optuna storage.

Queries JournalStorage files for study summaries and prints a progress
report. Run from the login node during SLURM execution.

Usage:
    python monitor_factorial.py --storage-dir /path/to/optuna/
    python monitor_factorial.py --storage-dir /path/to/optuna/ --detail
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def monitor(storage_dir: Path, detail: bool = False) -> None:
    """Print progress report from Optuna storage files.

    Parameters
    ----------
    storage_dir : Path
        Directory containing .optuna.journal files.
    detail : bool
        If True, print per-study details.
    """
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    journal_files = sorted(storage_dir.glob("*.optuna.journal"))
    if not journal_files:
        print(f"No .optuna.journal files found in {storage_dir}")
        return

    total_studies = 0
    total_started = 0
    total_trials = 0

    for journal_path in journal_files:
        try:
            lock_obj = optuna.storages.JournalFileOpenLock(str(journal_path))
            storage = JournalStorage(
                JournalFileStorage(str(journal_path), lock_obj=lock_obj)
            )
            summaries = optuna.study.get_all_study_summaries(storage)
        except Exception as e:
            print(f"  {journal_path.name}: ERROR - {e}")
            continue

        recipe = journal_path.stem.replace(".optuna", "")
        n_total = len(summaries)
        n_started = sum(1 for s in summaries if s.n_trials > 0)
        n_trials = sum(s.n_trials for s in summaries)

        total_studies += n_total
        total_started += n_started
        total_trials += n_trials

        file_size_mb = journal_path.stat().st_size / (1024 * 1024)
        print(
            f"  {recipe}: {n_started}/{n_total} studies started, "
            f"{n_trials} total trials, {file_size_mb:.1f} MB"
        )

        if detail and summaries:
            # Group by model from study names
            models: dict[str, int] = {}
            for s in summaries:
                parts = s.study_name.split("__")
                if len(parts) >= 2:
                    cell_part = parts[1] if len(parts) > 1 else "unknown"
                    model = cell_part.split("_")[0]
                    models[model] = models.get(model, 0) + s.n_trials
            for model, trials in sorted(models.items()):
                print(f"    {model}: {trials} trials")

    print(f"\nTotal: {total_started}/{total_studies} studies started, {total_trials} trials")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor factorial experiment progress"
    )
    parser.add_argument(
        "--storage-dir",
        required=True,
        type=Path,
        help="Directory with .optuna.journal files",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show per-model detail",
    )
    args = parser.parse_args()

    # Suppress Optuna logging noise
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.WARNING)

    monitor(args.storage_dir, args.detail)


if __name__ == "__main__":
    import optuna  # noqa: F811

    main()
