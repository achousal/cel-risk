"""Cell expansion tests."""

from __future__ import annotations

import pytest

from ced_ml.cellml.plan import cell_count, expand_cells
from ced_ml.cellml.schema import (
    AxesSpec,
    BaseConfigs,
    ExperimentSpec,
    PanelSpec,
    TrunkSpec,
)
from ced_ml.recipes.schema import (
    OrderingRule,
    OrderingType,
    SizeRule,
    SizeRuleType,
    TrainingStrategy,
)


def _make_spec(
    *,
    panels=None,
    model=("LR_EN",),
    cal=("logistic_intercept",),
    wt=("none",),
    ds=(1.0,),
    scenario=(),
    fs=("fixed_panel",),
    ctrl=(5,),
) -> ExperimentSpec:
    panels = panels or [
        PanelSpec(
            id="P1",
            source="derived",
            trunk_id="T1",
            ordering=OrderingRule(type=OrderingType.consensus_score_descending),
            size_rule=SizeRule(type=SizeRuleType.significance_count),
        )
    ]
    return ExperimentSpec(
        name="t",
        base_configs=BaseConfigs(training="t.yaml", pipeline="p.yaml", splits="s.yaml"),
        trunks=[TrunkSpec(id="T1", proteins_csv="dummy.csv")],
        panels=panels,
        axes=AxesSpec(
            model=list(model),
            calibration=list(cal),
            weighting=list(wt),
            downsampling=list(ds),
            scenario=list(scenario),
            feature_selection=list(fs),
            control_ratio=list(ctrl),
        ),
    )


@pytest.mark.parametrize(
    "model,cal,wt,ds,expected",
    [
        (("A",), ("c1",), ("w1",), (1.0,), 1),
        (("A", "B"), ("c1", "c2"), ("w1",), (1.0,), 4),
        (("A", "B"), ("c1",), ("w1", "w2"), (1.0, 2.0), 8),
        (("A",) * 3, ("c1", "c2"), ("w",), (1.0, 2.0), 12),
    ],
)
def test_cartesian_count(model, cal, wt, ds, expected):
    spec = _make_spec(model=model, cal=cal, wt=wt, ds=ds)
    assert cell_count(spec) == expected


def test_pinned_model_collapses_axis():
    panels = [
        PanelSpec(
            id="P_pinned",
            source="derived",
            trunk_id="T1",
            pinned_model="RF",
            ordering=OrderingRule(type=OrderingType.consensus_score_descending),
            size_rule=SizeRule(type=SizeRuleType.significance_count),
        )
    ]
    spec = _make_spec(panels=panels, model=("A", "B", "C", "D", "E"))
    cells = expand_cells(spec)
    assert len(cells) == 1
    assert cells[0]["model"] == "RF"


def test_scenario_axis_adds_suffix():
    scenarios = [
        TrainingStrategy(
            name="IncidentOnly",
            scenarios=["IncidentOnly"],
            prevalent_train_only=True,
        ),
        TrainingStrategy(
            name="IncPrev_0.5",
            scenarios=["IncidentPlusPrevalent"],
            prevalent_train_only=True,
            prevalent_train_frac=0.5,
        ),
    ]
    spec = _make_spec(scenario=scenarios)
    cells = expand_cells(spec)
    assert len(cells) == 2
    names = {c["cell_name"] for c in cells}
    assert any(n.endswith("_IncidentOnly") for n in names)
    assert any(n.endswith("_IncPrev_0.5") for n in names)


def test_empty_scenario_preserves_legacy_name():
    spec = _make_spec()
    cell = expand_cells(spec)[0]
    assert cell["scenario"] is None
    # Legacy naming: {model}_{cal}_{wt}_ds{ratio}
    assert cell["cell_name"] == "LR_EN_logistic_intercept_none_ds1"


def test_template_minimal_cell_count(minimal_spec):
    assert cell_count(minimal_spec) == 20
