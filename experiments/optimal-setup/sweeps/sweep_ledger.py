"""Append-only sweep iteration ledger.

Tracks every iteration of every sweep: parameters, job IDs, metrics,
and decisions. The ledger is the traceability layer -- never edited
retroactively, only appended.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_FIELDS = [
    "sweep_id",
    "iteration",
    "timestamp",
    "params_json",
    "job_id",
    "status",
    "metric_value",
    "delta_baseline",
    "delta_previous",
    "running_best",
    "decision",
    "notes",
]


@dataclass
class LedgerEntry:
    """One row in the sweep ledger."""

    sweep_id: str
    iteration: int
    timestamp: str
    params_json: str
    job_id: str | None
    status: str
    metric_value: float | None
    delta_baseline: float | None
    delta_previous: float | None
    running_best: float | None
    decision: str  # keep | discard | terminate | error
    notes: str


class SweepLedger:
    """Append-only CSV ledger for one sweep."""

    def __init__(self, ledger_dir: Path, sweep_id: str):
        self._path = ledger_dir / f"{sweep_id}_ledger.csv"
        self._sweep_id = sweep_id
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Create file with header if it doesn't exist
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
                writer.writeheader()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: LedgerEntry) -> None:
        """Append one entry to the ledger."""
        row = {
            "sweep_id": entry.sweep_id,
            "iteration": entry.iteration,
            "timestamp": entry.timestamp,
            "params_json": entry.params_json,
            "job_id": entry.job_id or "",
            "status": entry.status,
            "metric_value": f"{entry.metric_value:.6f}" if entry.metric_value is not None else "",
            "delta_baseline": f"{entry.delta_baseline:.6f}" if entry.delta_baseline is not None else "",
            "delta_previous": f"{entry.delta_previous:.6f}" if entry.delta_previous is not None else "",
            "running_best": f"{entry.running_best:.6f}" if entry.running_best is not None else "",
            "decision": entry.decision,
            "notes": entry.notes,
        }
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
            writer.writerow(row)
        logger.debug("Ledger append: sweep=%s iter=%d decision=%s", entry.sweep_id, entry.iteration, entry.decision)

    def read_history(self) -> list[dict]:
        """Read all ledger entries as dicts."""
        if not self._path.exists():
            return []
        with open(self._path, newline="") as f:
            return list(csv.DictReader(f))

    def last_iteration(self) -> int:
        """Return the highest iteration number, or 0 if empty."""
        history = self.read_history()
        if not history:
            return 0
        return max(int(row["iteration"]) for row in history)

    def consecutive_no_improve(self) -> int:
        """Count consecutive non-improving iterations from the end."""
        history = self.read_history()
        count = 0
        for row in reversed(history):
            if row["decision"] == "discard":
                count += 1
            else:
                break
        return count

    def running_best(self) -> float | None:
        """Return the current running best metric value."""
        history = self.read_history()
        if not history:
            return None
        bests = [float(row["running_best"]) for row in history if row["running_best"]]
        return bests[-1] if bests else None

    @staticmethod
    def make_entry(
        sweep_id: str,
        iteration: int,
        params: dict,
        job_id: str | None = None,
        status: str = "pending",
        metric_value: float | None = None,
        delta_baseline: float | None = None,
        delta_previous: float | None = None,
        running_best: float | None = None,
        decision: str = "pending",
        notes: str = "",
    ) -> LedgerEntry:
        """Convenience factory for a ledger entry."""
        return LedgerEntry(
            sweep_id=sweep_id,
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            params_json=json.dumps(params, sort_keys=True),
            job_id=job_id,
            status=status,
            metric_value=metric_value,
            delta_baseline=delta_baseline,
            delta_previous=delta_previous,
            running_best=running_best,
            decision=decision,
            notes=notes,
        )
