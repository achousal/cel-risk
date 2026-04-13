"""CSV-backed experiment registry.

One CSV at experiments/_registry.csv. fcntl.flock around every write to
survive concurrent ced cellml invocations. Reads are lock-free (the CSV
is append-mostly and small enough to rewrite on update).
"""

from __future__ import annotations

import csv
import datetime as _dt
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ced_ml.cellml.schema import ExperimentSpec

REGISTRY_COLUMNS = [
    "name",
    "status",
    "created",
    "spec_path",
    "cells",
    "recipes",
    "submitted_at",
    "job_id",
    "last_status_check",
    "best_prauc",
    "notes",
]


def _default_registry_path() -> Path:
    return Path("experiments") / "_registry.csv"


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive lock on a sentinel file adjacent to the registry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_all(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in rows:
            # Only write known columns; drop extras silently.
            writer.writerow({k: row.get(k, "") for k in REGISTRY_COLUMNS})
    tmp.replace(path)


def register(
    spec: ExperimentSpec,
    spec_path: Path,
    *,
    registry_path: Path | None = None,
    cells: int | None = None,
    recipes: int | None = None,
) -> None:
    """Insert or upsert an experiment entry."""
    path = registry_path or _default_registry_path()
    now = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with _locked(path):
        rows = _read_all(path)
        existing = next((r for r in rows if r["name"] == spec.name), None)
        entry: dict[str, Any] = {
            "name": spec.name,
            "status": "registered",
            "created": now,
            "spec_path": str(spec_path),
            "cells": str(cells or ""),
            "recipes": str(recipes or len(spec.panels)),
            "submitted_at": "",
            "job_id": "",
            "last_status_check": "",
            "best_prauc": "",
            "notes": "",
        }
        if existing is not None:
            existing.update(entry)
        else:
            rows.append(entry)
        _write_all(path, rows)


def update_status(
    name: str,
    *,
    registry_path: Path | None = None,
    **fields: Any,
) -> None:
    """Patch fields on a registered experiment."""
    path = registry_path or _default_registry_path()
    with _locked(path):
        rows = _read_all(path)
        for row in rows:
            if row["name"] == name:
                for k, v in fields.items():
                    if k not in REGISTRY_COLUMNS:
                        raise KeyError(f"unknown registry column: {k}")
                    row[k] = "" if v is None else str(v)
                row["last_status_check"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                break
        else:
            raise KeyError(f"experiment '{name}' not in registry")
        _write_all(path, rows)


def get(name: str, *, registry_path: Path | None = None) -> dict[str, str]:
    """Return one registry row (dict). Raises KeyError if missing."""
    path = registry_path or _default_registry_path()
    for row in _read_all(path):
        if row["name"] == name:
            return row
    raise KeyError(f"experiment '{name}' not in registry")


def list_all(*, registry_path: Path | None = None) -> list[dict[str, str]]:
    """Return all registry rows."""
    path = registry_path or _default_registry_path()
    return _read_all(path)
