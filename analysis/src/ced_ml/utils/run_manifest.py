"""Run-level metadata manifest helpers.

This module manages ``run_metadata.json`` as a shared run manifest.
It uses non-destructive merge semantics so concurrent/independent stages
do not overwrite existing entries.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ced_ml.utils.serialization import save_json

logger = logging.getLogger(__name__)


def build_model_manifest_entry(
    *,
    scenario: str | None = None,
    infile: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> dict[str, str]:
    """Build a stable model-level manifest entry."""
    entry: dict[str, str] = {}
    if scenario:
        entry["scenario"] = str(scenario)
    if infile is not None:
        entry["infile"] = str(infile)
    if split_dir is not None:
        entry["split_dir"] = str(split_dir)
    return entry


def ensure_run_manifest(
    *,
    run_level_dir: str | Path,
    run_id: str,
    infile: str | Path,
    split_dir: str | Path,
    model_entries: Mapping[str, Mapping[str, Any]] | None = None,
    hpc_config: str | Path | None = None,
) -> tuple[Path, bool]:
    """Create/update ``run_metadata.json`` with non-destructive semantics.

    Rules:
    - Create file when missing.
    - Set top-level fields only when missing.
    - Add model entries only when missing.
    - Never overwrite existing model entry fields.

    Returns:
        Tuple of (manifest_path, changed).
    """
    run_level_path = Path(run_level_dir)
    manifest_path = run_level_path / "run_metadata.json"
    run_level_path.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    metadata = loaded
                else:
                    logger.warning(
                        "Ignoring non-dict run metadata at %s (type=%s)",
                        manifest_path,
                        type(loaded).__name__,
                    )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s (%s); rebuilding manifest", manifest_path, exc)

    changed = False

    if "run_id" not in metadata:
        metadata["run_id"] = run_id
        changed = True
    if "infile" not in metadata:
        metadata["infile"] = str(infile)
        changed = True
    if "split_dir" not in metadata:
        metadata["split_dir"] = str(split_dir)
        changed = True
    if hpc_config is not None and "hpc_config" not in metadata:
        metadata["hpc_config"] = str(hpc_config)
        changed = True

    models_obj = metadata.get("models")
    if not isinstance(models_obj, dict):
        models_obj = {}
        metadata["models"] = models_obj
        changed = True

    if model_entries:
        for model_name, model_entry in model_entries.items():
            if model_name not in models_obj:
                models_obj[model_name] = dict(model_entry)
                changed = True

    if changed or not manifest_path.exists():
        save_json(metadata, manifest_path, indent=2)

    return manifest_path, changed
