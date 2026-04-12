"""Dataset fingerprinting for calibration cache keys.

A fingerprint is a stable short hash over dataset identity features that
matter for hyperparameter-tuning validity: row count, label prevalence,
column signature, and the explicit seed set. Two datasets with the same
fingerprint are interchangeable for the purpose of Optuna calibration
caching.

Fingerprints deliberately do NOT hash full cell values -- that would be
slow and too brittle (a single re-export would invalidate every cache).
They hash structural identity, not bitwise identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".parquet", ".pkl", ".pickle", ".csv"}


def resolve_binary_labels(
    labels: pd.Series,
    scenario: str | None,
) -> tuple[pd.Series, np.ndarray]:
    """Map a label column to a binary 0/1 array under a scenario.

    For string label columns (e.g. "Controls"/"Incident"/"Prevalent"),
    the cel-risk canonical scenario definitions determine which rows
    are kept and which class is positive. For already-numeric columns,
    this is a pass-through.

    Returns
    -------
    (kept_mask, y)
        kept_mask indexes which rows survive scenario filtering.
        y is the 0/1 integer label array for those rows.
    """
    # Fast path: already numeric-like
    if pd.api.types.is_numeric_dtype(labels) or pd.api.types.is_bool_dtype(labels):
        mask = labels.notna()
        y = labels[mask].astype(int).to_numpy()
        return mask, y

    # String labels require a scenario to disambiguate positive class
    if scenario is None:
        raise ValueError(
            "Label column is non-numeric but no scenario was provided. "
            "Set spec.scenario (e.g. 'IncidentOnly') to map string labels to 0/1."
        )

    # Import lazily to avoid a circular dep during schema import
    from ced_ml.data.schema import SCENARIO_DEFINITIONS

    if scenario not in SCENARIO_DEFINITIONS:
        raise ValueError(f"Unknown scenario '{scenario}'. Valid: {sorted(SCENARIO_DEFINITIONS)}")
    defn = SCENARIO_DEFINITIONS[scenario]
    keep_labels = set(defn["labels"])
    positive_label = defn["positive_label"]

    mask = labels.isin(keep_labels)
    filtered = labels[mask]
    y = (filtered == positive_label).astype(int).to_numpy()
    return mask, y


def _load_any(path: Path) -> pd.DataFrame:
    """Load a dataframe from parquet/pickle/csv based on suffix."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported dataset suffix '{suffix}' for fingerprinting. "
        f"Expected one of {sorted(SUPPORTED_SUFFIXES)}."
    )


def dataset_fingerprint(
    data_path: Path,
    label_col: str,
    seeds: list[int] | None = None,
    extra: dict | None = None,
    scenario: str | None = None,
) -> str:
    """Compute a stable 16-char fingerprint for a dataset.

    Parameters
    ----------
    data_path
        Path to parquet/pickle/csv dataset.
    label_col
        Column name holding the binary label. Required: prevalence is
        load-bearing for calibration validity.
    seeds
        Explicit seed set used by the parent sweep. Different seed sets
        produce different fingerprints even on the same underlying file.
    extra
        Optional additional keys to fold into the fingerprint (e.g. a
        recipe ID, a split strategy name). Sorted before hashing.

    Returns
    -------
    16-character hex digest (first 16 chars of sha256).

    Raises
    ------
    FileNotFoundError
        If data_path does not exist. No silent fallback.
    ValueError
        If label_col is absent or the file suffix is unsupported.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = _load_any(data_path)

    if label_col not in df.columns:
        raise ValueError(
            f"label_col '{label_col}' not found in {data_path.name}. "
            f"Cannot fingerprint without label prevalence."
        )

    _, y = resolve_binary_labels(df[label_col], scenario)
    n_rows = int(len(y))
    n_positive = int(y.sum())
    prevalence = round(n_positive / n_rows, 6) if n_rows else 0.0

    columns_sorted = sorted(df.columns.astype(str).tolist())
    columns_digest = hashlib.sha256(json.dumps(columns_sorted).encode("utf-8")).hexdigest()[:12]

    payload = {
        "n_rows": n_rows,
        "n_positive": n_positive,
        "prevalence": prevalence,
        "n_columns": len(columns_sorted),
        "columns_digest": columns_digest,
        "label_col": label_col,
        "scenario": scenario or "",
        "seeds": sorted(seeds) if seeds else [],
        "extra": {k: extra[k] for k in sorted(extra)} if extra else {},
    }

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    logger.debug(
        "Fingerprint %s for %s: n=%d prev=%.4f cols=%d seeds=%s",
        digest,
        data_path.name,
        n_rows,
        prevalence,
        len(columns_sorted),
        payload["seeds"],
    )
    return digest


def space_hash(parameter_space: dict) -> str:
    """Compute a stable hash over a sweep parameter space.

    Two parameter spaces with the same hash are interchangeable for the
    purpose of calibration reuse. Parameter NAMES, types, and bounds are
    hashed; descriptions and comments are not.
    """
    canonical: dict = {}
    for name in sorted(parameter_space):
        pdef = parameter_space[name]
        # Accept either a pydantic model or a plain dict
        if hasattr(pdef, "model_dump"):
            raw = pdef.model_dump()
        else:
            raw = dict(pdef)
        canonical[name] = {
            "type": str(raw.get("type", "")),
            "values": sorted(raw["values"]) if raw.get("values") else None,
            "low": raw.get("low"),
            "high": raw.get("high"),
            "step": raw.get("step"),
        }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
