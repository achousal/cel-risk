"""
Data schema definitions and constants.

Defines column names, labels, and data structures used throughout the pipeline.
Matches exactly with current implementation for behavioral equivalence.
"""

from enum import Enum

# ============================================================================
# Column Names
# ============================================================================

# Identifier column
ID_COL = "eid"

# Target column
TARGET_COL = "CeD_comparison"

# Date column for temporal splits
CED_DATE_COL = "CeD_date"

# Demographic columns (numeric)
META_NUM_COLS = ["age", "BMI"]

# Demographic columns (categorical)
CAT_COLS = ["sex", "Genetic ethnic grouping"]

# All demographic columns
META_COLS = META_NUM_COLS + CAT_COLS

# ============================================================================
# Class Labels
# ============================================================================

CONTROL_LABEL = "Controls"
INCIDENT_LABEL = "Incident"
PREVALENT_LABEL = "Prevalent"

# Valid labels
VALID_LABELS = [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL]

# Case labels (non-control)
CASE_LABELS = [INCIDENT_LABEL, PREVALENT_LABEL]

# ============================================================================
# Scenario Definitions
# ============================================================================

SCENARIO_DEFINITIONS = {
    "IncidentOnly": {
        "labels": [CONTROL_LABEL, INCIDENT_LABEL],
        "description": "Controls + Incident (prospective prediction, recommended)",
        "positive_label": INCIDENT_LABEL,
    },
    "PrevalentOnly": {
        "labels": [CONTROL_LABEL, PREVALENT_LABEL],
        "description": "Controls + Prevalent (biomarker discovery only, NOT for prediction)",
        "positive_label": PREVALENT_LABEL,
    },
    "IncidentPlusPrevalent": {
        "labels": [CONTROL_LABEL, INCIDENT_LABEL, PREVALENT_LABEL],
        "description": "Controls + Incident + Prevalent",
        "positive_label": INCIDENT_LABEL,  # Incident is the primary target
    },
}

# ============================================================================
# Column Patterns
# ============================================================================

# Protein columns end with "_resid"
PROTEIN_SUFFIX = "_resid"

# Transformed column patterns (after preprocessing)
TRANSFORMED_PREFIX = "transformed_"

# ============================================================================
# Data Quality Constants
# ============================================================================

# Expected number of proteins (2920 in current dataset)
EXPECTED_N_PROTEINS = 2920

# Missing value handling
MISSING_CATEGORY = "Missing"  # For categorical columns with missing values

# ============================================================================
# Split Names
# ============================================================================

SPLIT_TRAIN = "TRAIN"
SPLIT_VAL = "VAL"
SPLIT_TEST = "TEST"
SPLIT_HOLDOUT = "HOLDOUT"

VALID_SPLITS = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_HOLDOUT]

# ============================================================================
# Model Names
# ============================================================================


class ModelName(str, Enum):
    """Valid model identifiers."""

    LR = "LR"
    LR_L1 = "LR_L1"
    LR_L2 = "LR_L2"
    LR_EN = "LR_EN"
    LinSVM = "LinSVM"
    LinSVM_cal = "LinSVM_cal"
    LinSVM_L1_cal = "LinSVM_L1_cal"
    SVM_rbf = "SVM_rbf"
    SVM_rbf_cal = "SVM_rbf_cal"
    RF = "RF"
    XGBoost = "XGBoost"


# Valid model identifiers (derived from enum)
VALID_MODELS = [m.value for m in ModelName]

# Model display names
MODEL_DISPLAY_NAMES = {
    ModelName.LR: "Logistic Regression",
    ModelName.LR_L1: "Logistic Regression (L1)",
    ModelName.LR_L2: "Logistic Regression (L2)",
    ModelName.LR_EN: "Logistic Regression (ElasticNet)",
    ModelName.LinSVM: "Linear SVM",
    ModelName.LinSVM_cal: "Linear SVM (L2, calibrated)",
    ModelName.LinSVM_L1_cal: "Linear SVM (L1, calibrated)",
    ModelName.SVM_rbf: "SVM (RBF kernel)",
    ModelName.SVM_rbf_cal: "SVM (RBF, calibrated)",
    ModelName.RF: "Random Forest",
    ModelName.XGBoost: "XGBoost",
}

# ============================================================================
# Metric Names
# ============================================================================

# Primary metrics
METRIC_BRIER = "brier_score"
METRIC_AUROC = "auroc"
METRIC_PRAUC = "prauc"

# Calibration metrics
METRIC_CALIB_SLOPE = "calibration_slope"
METRIC_CALIB_INTERCEPT = "calibration_intercept"
METRIC_ECE = "expected_calibration_error"

# Threshold-dependent metrics
METRIC_SENSITIVITY = "sensitivity"
METRIC_SPECIFICITY = "specificity"
METRIC_PPV = "ppv"
METRIC_NPV = "npv"
METRIC_F1 = "f1_score"

# All standard metrics
STANDARD_METRICS = [
    METRIC_BRIER,
    METRIC_AUROC,
    METRIC_PRAUC,
    METRIC_CALIB_SLOPE,
    METRIC_CALIB_INTERCEPT,
    METRIC_ECE,
    METRIC_SENSITIVITY,
    METRIC_SPECIFICITY,
    METRIC_PPV,
    METRIC_NPV,
    METRIC_F1,
]

# ============================================================================
# Output File Names
# ============================================================================

# Core outputs
FINAL_MODEL_FILE = "final_model.joblib"
VAL_METRICS_FILE = "val_metrics.csv"
TEST_METRICS_FILE = "test_metrics.csv"
CONFIG_FILE = "config.yaml"
METADATA_FILE = "run_metadata.json"

# Prediction files
TRAIN_PREDS_FILE = "train_preds.csv"
VAL_PREDS_FILE = "val_preds.csv"
TEST_PREDS_FILE = "test_preds.csv"

# Feature reports
STABLE_PANEL_FILE = "stable_panel.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"

# Calibration
CALIBRATION_CURVE_FILE = "calibration_curve.csv"

# DCA
DCA_CURVE_FILE = "dca_curve.csv"

# ============================================================================
# Directory Structure
# ============================================================================

# Standard output directories
DIR_CORE = "core"
DIR_PREDS = "preds"
DIR_DIAGNOSTICS = "diagnostics"
DIR_REPORTS = "reports"

# Subdirectories
SUBDIR_CALIBRATION = "calibration"
SUBDIR_DCA = "dca"
SUBDIR_LEARNING_CURVE = "learning_curve"
SUBDIR_STABLE_PANEL = "stable_panel"
SUBDIR_FEATURE_IMPORTANCE = "feature_importance"

# ============================================================================
# Helper Functions
# ============================================================================


def get_protein_columns(df_columns: list[str]) -> list[str]:
    """Extract protein column names from DataFrame columns."""
    return [col for col in df_columns if col.endswith(PROTEIN_SUFFIX)]


def get_scenario_labels(scenario: str) -> list[str]:
    """Get valid labels for a scenario."""
    if scenario not in SCENARIO_DEFINITIONS:
        raise ValueError(
            f"Unknown scenario: {scenario}. Valid: {list(SCENARIO_DEFINITIONS.keys())}"
        )
    return SCENARIO_DEFINITIONS[scenario]["labels"]


def get_positive_label(scenario: str) -> str:
    """Get positive class label for a scenario."""
    if scenario not in SCENARIO_DEFINITIONS:
        raise ValueError(
            f"Unknown scenario: {scenario}. Valid: {list(SCENARIO_DEFINITIONS.keys())}"
        )
    return SCENARIO_DEFINITIONS[scenario]["positive_label"]
