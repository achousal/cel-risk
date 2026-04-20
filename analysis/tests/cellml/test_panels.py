"""Panel resolution tests — fixed_csv + reference NotImplementedError."""

from __future__ import annotations

import pytest

from ced_ml.cellml.panels import _resolve_fixed_csv, _resolve_reference
from ced_ml.cellml.schema import PanelSpec


def test_fixed_csv_copies_file(tmp_path):
    src = tmp_path / "src_panel.csv"
    src.write_text("protein\nAAA\nBBB\n")
    panel = PanelSpec(id="P1", source="fixed_csv", csv=src)
    out_dir = tmp_path / "panels"
    dest = _resolve_fixed_csv(panel, out_dir)
    assert dest.exists()
    assert dest.read_text() == "protein\nAAA\nBBB\n"
    assert dest == out_dir / "P1" / "panel.csv"


def test_fixed_csv_missing_source_raises(tmp_path):
    panel = PanelSpec(id="P1", source="fixed_csv", csv=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        _resolve_fixed_csv(panel, tmp_path / "out")


def test_reference_source_not_implemented(tmp_path):
    panel = PanelSpec(
        id="P1",
        source="reference",
        experiment="prior_exp",
        extract="best_prauc",
    )
    with pytest.raises(NotImplementedError):
        _resolve_reference(panel, tmp_path / "out")
