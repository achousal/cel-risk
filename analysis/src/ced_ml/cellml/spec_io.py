"""Load ExperimentSpec from YAML, optionally with dotted-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ced_ml.cellml.schema import ExperimentSpec


def _apply_override(data: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply one ``a.b.c=value`` override to a nested dict, in place.

    Values are parsed as YAML scalars so integers / floats / bools land
    with the correct type (e.g. ``optuna.n_trials=50`` -> int 50).
    """
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_spec(
    path: Path,
    overrides: list[str] | None = None,
) -> ExperimentSpec:
    """Load a YAML spec, apply ``key=value`` overrides, validate.

    Parameters
    ----------
    path : Path
    overrides : list[str]
        Items like ``optuna.n_trials=50`` or ``name=test_run``.
    """
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = copy.deepcopy(data)
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must be key=value, got: {item}")
            key, value = item.split("=", 1)
            _apply_override(data, key.strip(), value.strip())
    return ExperimentSpec.model_validate(data)
