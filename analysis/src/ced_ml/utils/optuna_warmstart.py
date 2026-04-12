"""Optuna warm-start helpers.

Extracts top-K best hyperparameters from prior Optuna studies (typically a
"scout" run with fewer seeds/trials) so they can be enqueued via
``study.enqueue_trial()`` to seed a subsequent main run.

Used by the CellML factorial workflow today, but the API is generic — any
Optuna ``JournalStorage``-backed study set works.

Example
-------
>>> from pathlib import Path
>>> from ced_ml.utils.optuna_warmstart import extract_top_params
>>> top = extract_top_params(Path("/path/to/optuna/"), top_k=5)
>>> top["LR_EN"]   # list of param dicts, best first
[{'C': 0.12, 'l1_ratio': 0.4}, ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_top_params(
    storage_dir: Path,
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """Extract top-K best params per model from Optuna journal studies.

    Walks every ``*.optuna.journal`` file under ``storage_dir`` and, for each
    study whose ``user_attrs["model"]`` is set, collects the params of the
    top-K trials sorted by the first objective value (descending). Multi-
    objective studies use Pareto-optimal trials; single-objective studies fall
    back to all completed trials.

    Parameters
    ----------
    storage_dir
        Directory containing ``*.optuna.journal`` files.
    top_k
        Number of best params to extract per model.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from model name (read from ``study.user_attrs["model"]``) to a
        list of param dicts, best first. Empty dict if no journals are found.
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
            storage = JournalStorage(JournalFileStorage(str(journal_path), lock_obj=lock_obj))
        except Exception as e:
            logger.warning("Failed to open %s: %s", journal_path, e)
            continue

        summaries = optuna.study.get_all_study_summaries(storage)
        for summary in summaries:
            try:
                study = optuna.load_study(study_name=summary.study_name, storage=storage)
            except Exception as e:
                logger.warning("Failed to load study '%s': %s", summary.study_name, e)
                continue

            model = study.user_attrs.get("model")
            if not model:
                continue

            if model in results and len(results[model]) >= top_k:
                continue

            try:
                best_trials = sorted(
                    study.best_trials,
                    key=lambda t: t.values[0],
                    reverse=True,
                )
            except Exception:
                completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                best_trials = sorted(completed, key=lambda t: t.value, reverse=True)

            for trial in best_trials[:top_k]:
                if model not in results:
                    results[model] = []
                if len(results[model]) < top_k:
                    results[model].append(trial.params)

    for model, params in results.items():
        logger.info("Model %s: extracted %d param sets", model, len(params))

    return results
