"""Registry round-trip tests."""

from __future__ import annotations

import pytest

from ced_ml.cellml.registry import REGISTRY_COLUMNS, get, list_all, register, update_status
from ced_ml.cellml.schema import AxesSpec, BaseConfigs, ExperimentSpec, PanelSpec


def _spec(name: str) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        base_configs=BaseConfigs(training="t.yaml", pipeline="p.yaml", splits="s.yaml"),
        trunks=[],
        panels=[PanelSpec(id="P", source="fixed_csv", csv="panel.csv")],
        axes=AxesSpec(
            model=["LR_EN"],
            calibration=["logistic_intercept"],
            weighting=["none"],
            downsampling=[1.0],
        ),
    )


def test_register_and_get(tmp_path):
    reg = tmp_path / "_registry.csv"
    register(_spec("exp_a"), spec_path=tmp_path / "a.yaml", registry_path=reg, cells=10)
    row = get("exp_a", registry_path=reg)
    assert row["name"] == "exp_a"
    assert row["status"] == "registered"
    assert row["cells"] == "10"


def test_update_status(tmp_path):
    reg = tmp_path / "_registry.csv"
    register(_spec("exp_b"), spec_path=tmp_path / "b.yaml", registry_path=reg)
    update_status("exp_b", registry_path=reg, status="submitted", job_id="12345")
    row = get("exp_b", registry_path=reg)
    assert row["status"] == "submitted"
    assert row["job_id"] == "12345"
    assert row["last_status_check"]  # non-empty


def test_list_all(tmp_path):
    reg = tmp_path / "_registry.csv"
    register(_spec("e1"), spec_path=tmp_path / "e1.yaml", registry_path=reg)
    register(_spec("e2"), spec_path=tmp_path / "e2.yaml", registry_path=reg)
    rows = list_all(registry_path=reg)
    assert {r["name"] for r in rows} == {"e1", "e2"}
    assert all(set(r.keys()) == set(REGISTRY_COLUMNS) for r in rows)


def test_update_unknown_column_raises(tmp_path):
    reg = tmp_path / "_registry.csv"
    register(_spec("e"), spec_path=tmp_path / "e.yaml", registry_path=reg)
    with pytest.raises(KeyError):
        update_status("e", registry_path=reg, not_a_column="x")


def test_register_twice_upserts(tmp_path):
    reg = tmp_path / "_registry.csv"
    register(_spec("exp_c"), spec_path=tmp_path / "c.yaml", registry_path=reg, cells=5)
    register(_spec("exp_c"), spec_path=tmp_path / "c.yaml", registry_path=reg, cells=7)
    rows = list_all(registry_path=reg)
    matching = [r for r in rows if r["name"] == "exp_c"]
    assert len(matching) == 1
    assert matching[0]["cells"] == "7"
