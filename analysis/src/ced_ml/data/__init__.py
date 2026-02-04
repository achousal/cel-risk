"""Data handling and schema definitions."""

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io_helpers import (
    normalize_protein_names,
    read_feature_report,
    read_metrics,
    read_predictions,
    save_feature_report,
    save_metrics,
    save_predictions,
)
from ced_ml.data.persistence import (
    check_split_files_exist,
    save_holdout_indices,
    save_holdout_metadata,
    save_split_indices,
    save_split_metadata,
    validate_existing_splits,
    validate_split_indices,
)
from ced_ml.data.schema import (
    CAT_COLS,
    CED_DATE_COL,
    CONTROL_LABEL,
    ID_COL,
    INCIDENT_LABEL,
    META_NUM_COLS,
    PREVALENT_LABEL,
    SCENARIO_DEFINITIONS,
    TARGET_COL,
    get_positive_label,
    get_protein_columns,
    get_scenario_labels,
)

__all__ = [
    # Schema
    "ID_COL",
    "TARGET_COL",
    "CED_DATE_COL",
    "META_NUM_COLS",
    "CAT_COLS",
    "CONTROL_LABEL",
    "INCIDENT_LABEL",
    "PREVALENT_LABEL",
    "SCENARIO_DEFINITIONS",
    "get_protein_columns",
    "get_scenario_labels",
    "get_positive_label",
    # Persistence
    "validate_split_indices",
    "validate_existing_splits",
    "check_split_files_exist",
    "save_split_indices",
    "save_holdout_indices",
    "save_split_metadata",
    "save_holdout_metadata",
    # Filters
    "apply_row_filters",
    # I/O Helpers
    "read_predictions",
    "save_predictions",
    "read_feature_report",
    "save_feature_report",
    "read_metrics",
    "save_metrics",
    "normalize_protein_names",
]
