"""Schema parsing + failure modes."""

from __future__ import annotations

import pytest
import yaml

from ced_ml.cellml.schema import ExperimentSpec


def test_minimal_roundtrip(minimal_spec):
    # Dumping to dict and back should produce an equivalent model.
    dumped = minimal_spec.model_dump(mode="json")
    roundtripped = ExperimentSpec.model_validate(dumped)
    assert roundtripped.name == minimal_spec.name
    assert len(roundtripped.panels) == len(minimal_spec.panels)


def test_svm_template_parses(svm_spec):
    assert len(svm_spec.panels) == 1
    assert "LinSVM_cal" in svm_spec.axes.model


def test_full_template_parses(full_spec):
    assert len(full_spec.panels) == 2
    assert full_spec.panels[0].source == "derived"
    assert full_spec.panels[1].source == "fixed_csv"


def test_derived_panel_requires_trunk_id():
    with pytest.raises(ValueError, match="trunk_id"):
        ExperimentSpec.model_validate(
            {
                "name": "bad",
                "base_configs": {
                    "training": "t.yaml",
                    "pipeline": "p.yaml",
                    "splits": "s.yaml",
                },
                "trunks": [],
                "panels": [{"id": "P", "source": "derived"}],
                "axes": {
                    "model": ["LR_EN"],
                    "calibration": ["logistic_intercept"],
                    "weighting": ["none"],
                    "downsampling": [1.0],
                },
            }
        )


def test_fixed_csv_requires_csv_path():
    with pytest.raises(ValueError, match="csv"):
        ExperimentSpec.model_validate(
            {
                "name": "bad",
                "base_configs": {
                    "training": "t.yaml",
                    "pipeline": "p.yaml",
                    "splits": "s.yaml",
                },
                "trunks": [],
                "panels": [{"id": "P", "source": "fixed_csv"}],
                "axes": {
                    "model": ["LR_EN"],
                    "calibration": ["logistic_intercept"],
                    "weighting": ["none"],
                    "downsampling": [1.0],
                },
            }
        )


def test_unknown_trunk_reference_rejected():
    with pytest.raises(ValueError, match="references trunk"):
        ExperimentSpec.model_validate(
            {
                "name": "bad",
                "base_configs": {
                    "training": "t.yaml",
                    "pipeline": "p.yaml",
                    "splits": "s.yaml",
                },
                "trunks": [
                    {
                        "id": "T1",
                        "proteins_csv": "nowhere.csv",
                    }
                ],
                "panels": [
                    {
                        "id": "P",
                        "source": "derived",
                        "trunk_id": "T_DOES_NOT_EXIST",
                        "ordering": {"type": "consensus_score_descending"},
                        "size_rule": {"type": "significance_count"},
                    }
                ],
                "axes": {
                    "model": ["LR_EN"],
                    "calibration": ["logistic_intercept"],
                    "weighting": ["none"],
                    "downsampling": [1.0],
                },
            }
        )


def test_yaml_roundtrip_via_dict(minimal_spec):
    payload = minimal_spec.model_dump(mode="json")
    text = yaml.safe_dump(payload)
    reparsed = ExperimentSpec.model_validate(yaml.safe_load(text))
    assert reparsed.name == minimal_spec.name
