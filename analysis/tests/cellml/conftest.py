"""Shared fixtures for cellml tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ced_ml.cellml.schema import ExperimentSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATES = REPO_ROOT / "experiments" / "templates"


@pytest.fixture
def minimal_spec() -> ExperimentSpec:
    data = yaml.safe_load((TEMPLATES / "minimal.yaml").read_text())
    return ExperimentSpec.model_validate(data)


@pytest.fixture
def svm_spec() -> ExperimentSpec:
    data = yaml.safe_load((TEMPLATES / "svm.yaml").read_text())
    return ExperimentSpec.model_validate(data)


@pytest.fixture
def full_spec() -> ExperimentSpec:
    data = yaml.safe_load((TEMPLATES / "full.yaml").read_text())
    return ExperimentSpec.model_validate(data)
