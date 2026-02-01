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

# -- Ensemble / bar chart palette  --
COLOR_BAR_POSITIVE = "lightseagreen"
COLOR_BAR_NEGATIVE = "tomato"
COLOR_BAR_PALETTE = ["darkslategray", "lightseagreen", "burlywood", "sandybrown", "tomato"]

# -- Panel curve / Pareto plot colors --
COLOR_PARETO_MAIN = "royalblue"
COLOR_PARETO_CV = "slategray"
COLOR_THRESHOLD_GREEN = "mediumseagreen"
COLOR_THRESHOLD_AMBER = "orange"
COLOR_THRESHOLD_RED = "tomato"

# -- Line width (secondary) --
LW_SECONDARY = 1.5


def configure_backend() -> None:
    """Set non-interactive backend. Call at module level in plotting modules."""
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        logger.debug("Could not set matplotlib backend to Agg")
