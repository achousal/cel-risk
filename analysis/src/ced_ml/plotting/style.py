"""Centralized plotting style constants.

All plotting modules should import style values from here
rather than using hardcoded literals.
"""

import logging

logger = logging.getLogger(__name__)

# -- Figure defaults --
DPI = 150
PAD_INCHES = 0.1
BBOX_INCHES = "tight"

# -- Grid --
GRID_ALPHA = 0.3

# -- Font sizes --
FONT_TITLE = 13
FONT_LABEL = 11
FONT_LEGEND = 9
FONT_TICK = 10

# -- Line widths --
LW_PRIMARY = 2.0
LW_REFERENCE = 1.0

# -- Marker sizes --
MARKER_SIZE_SMALL = 30
MARKER_SIZE_LARGE = 60

# -- Default figure sizes (width, height) --
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_TALL = (8, 10)

# -- Default figure sizes (additional) --
FIGSIZE_ROC = (6.5, 6)
FIGSIZE_DCA = (10, 6)
FIGSIZE_CALIBRATION = (14, 10)

# -- Color palette --
COLOR_PRIMARY = "steelblue"
COLOR_SECONDARY = "darkorange"
COLOR_TERTIARY = "firebrick"
COLOR_POSITIVE = "crimson"
COLOR_NEGATIVE = "steelblue"
COLOR_REFERENCE = "gray"
COLOR_EDGE = "darkblue"
COLOR_FILL_ALPHA = 0.2

# -- Confidence band alpha values --
ALPHA_CI = 0.15
ALPHA_SD = 0.30

# -- Plot element alpha values --
ALPHA_SCATTER = 0.7  # Scatter plot markers
ALPHA_LINE = 0.6  # Line plots connecting points
ALPHA_REFERENCE = 0.7  # Reference lines (perfect calibration, ideal)
ALPHA_RECALIBRATION = 0.8  # Recalibration line (slightly higher for emphasis)
ALPHA_LEGEND_MARKER = 0.7  # Legend marker alpha (unified)

# -- Ensemble / bar chart palette --
COLOR_BAR_POSITIVE = "#2a9d8f"
COLOR_BAR_NEGATIVE = "#e76f51"
COLOR_BAR_PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

# -- Panel curve / Pareto plot colors --
COLOR_PARETO_MAIN = "#2563eb"
COLOR_PARETO_CV = "#64748b"
COLOR_THRESHOLD_GREEN = "#10b981"
COLOR_THRESHOLD_AMBER = "#f59e0b"
COLOR_THRESHOLD_RED = "#ef4444"

# -- Line width (secondary) --
LW_SECONDARY = 1.5


# -- Model comparison palette --
# Maps canonical model names to colorblind-friendly colors
MODEL_COLORS: dict[str, str] = {
    "LR_EN": "#264653",  # dark teal
    "RF": "#2a9d8f",  # teal
    "LinSVM_cal": "#e9c46a",  # gold
    "XGBoost": "#f4a261",  # orange
    "ENSEMBLE": "#e76f51",  # coral (drawn last, on top)
}
MODEL_COLOR_FALLBACK = "#6c757d"  # gray for unknown models

# -- Comparison plot style --
ALPHA_CI_COMPARISON = 0.08  # Lower alpha for multi-model CI bands
LW_COMPARISON = 1.8  # Slightly thinner lines for multi-model overlay
LW_COMPARISON_ENSEMBLE = 2.5  # Thicker line for ensemble emphasis
MARKER_SIZE_COMPARISON = 60  # Smaller threshold markers for comparison
FIGSIZE_COMPARISON = (8, 6.5)  # Standard comparison figure size
FIGSIZE_COMPARISON_DCA = (11, 6.5)  # Wider for DCA


def get_model_color(model_name: str) -> str:
    """Get color for a model name, with fallback for unknown models."""
    return MODEL_COLORS.get(model_name, MODEL_COLOR_FALLBACK)


def configure_backend() -> None:
    """Set non-interactive backend. Call at module level in plotting modules."""
    import matplotlib

    try:
        matplotlib.use("Agg")
    except ImportError:
        logger.debug("Could not set matplotlib backend to Agg")
