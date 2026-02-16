"""Tests for panel optimization helper utilities."""

import pandas as pd
import pytest

from ced_ml.cli.panel_optimization_helpers import load_and_filter_data
from ced_ml.data.schema import CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL, TARGET_COL


def test_load_and_filter_data_applies_scenario_filter(monkeypatch):
    """Scenario filtering should align rows with split label space."""
    df_raw = pd.DataFrame(
        {
            TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
            "prot_a_resid": [0.1, 0.2, 0.3],
            "age": [40.0, 41.0, 42.0],
            "BMI": [24.0, 25.0, 26.0],
        }
    )

    def _fake_read_proteomics_file(_infile, validate=True):  # noqa: ARG001
        return df_raw

    def _fake_apply_row_filters(df, meta_num_cols=None):  # noqa: ARG001
        return df.copy(), {"n_in": len(df), "n_out": len(df)}

    monkeypatch.setattr(
        "ced_ml.cli.panel_optimization_helpers.read_proteomics_file",
        _fake_read_proteomics_file,
    )
    monkeypatch.setattr(
        "ced_ml.cli.panel_optimization_helpers.apply_row_filters",
        _fake_apply_row_filters,
    )

    df, feature_cols, protein_cols = load_and_filter_data(
        infile="dummy.parquet",
        feature_cols=["prot_a_resid", "age", "BMI"],
        protein_cols=["prot_a_resid"],
        meta_num_cols=["age", "BMI"],
        scenario="IncidentOnly",
    )

    assert set(df[TARGET_COL].unique()) == {CONTROL_LABEL, INCIDENT_LABEL}
    assert len(df) == 2
    assert feature_cols == ["prot_a_resid", "age", "BMI"]
    assert protein_cols == ["prot_a_resid"]


def test_load_and_filter_data_invalid_scenario_raises(monkeypatch):
    """Unknown scenario names should fail fast."""
    df_raw = pd.DataFrame(
        {
            TARGET_COL: [CONTROL_LABEL, INCIDENT_LABEL],
            "prot_a_resid": [0.1, 0.2],
            "age": [40.0, 41.0],
            "BMI": [24.0, 25.0],
        }
    )

    def _fake_read_proteomics_file(_infile, validate=True):  # noqa: ARG001
        return df_raw

    def _fake_apply_row_filters(df, meta_num_cols=None):  # noqa: ARG001
        return df.copy(), {"n_in": len(df), "n_out": len(df)}

    monkeypatch.setattr(
        "ced_ml.cli.panel_optimization_helpers.read_proteomics_file",
        _fake_read_proteomics_file,
    )
    monkeypatch.setattr(
        "ced_ml.cli.panel_optimization_helpers.apply_row_filters",
        _fake_apply_row_filters,
    )

    with pytest.raises(ValueError, match="Unknown scenario"):
        load_and_filter_data(
            infile="dummy.parquet",
            feature_cols=["prot_a_resid", "age", "BMI"],
            protein_cols=["prot_a_resid"],
            meta_num_cols=["age", "BMI"],
            scenario="NotARealScenario",
        )
