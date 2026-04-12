#!/usr/bin/env Rscript
# _theme_factorial.R -- Extended theme for factorial analysis.
# Sources the base _theme.R and adds factorial-specific palettes,
# dynamic recipe colors, and data loading helpers.
#
# Usage: source("_theme_factorial.R") at the top of each analysis script.

# ── Source base theme ────────────────────────────────────────────────────────
# Provides: theme_cel(), MODEL_COLORS, MODEL_LABELS, ORDER_COLORS,
#           ORDER_LABELS, save_fig(), clean_protein(), CEL_ROOT, RESULTS_DIR
ANALYSIS_DIR <- normalizePath(file.path("..", ".."), mustWork = FALSE)
BASE_THEME   <- file.path(ANALYSIS_DIR, "_gen1_archive", "narrative", "_theme.R")
if (file.exists(BASE_THEME)) {
  source(BASE_THEME)
} else {
  # Fallback: define essentials inline if base theme not found
  suppressPackageStartupMessages({
    library(dplyr); library(tidyr); library(ggplot2)
    library(stringr); library(scales)
  })
  CEL_ROOT    <- normalizePath(file.path("..", "..", "..", ".."), mustWork = FALSE)
  RESULTS_DIR <- file.path(CEL_ROOT, "results")

  theme_cel <- function(base_size = 11) {
    theme_bw(base_size = base_size) +
      theme(
        panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        strip.background   = element_rect(fill = "grey93", color = NA),
        strip.text         = element_text(face = "bold", size = base_size),
        legend.position    = "bottom",
        plot.title         = element_text(face = "bold", size = base_size + 2),
        plot.subtitle      = element_text(color = "grey40", size = base_size),
        plot.caption       = element_text(color = "grey50", size = base_size - 2, hjust = 0)
      )
  }

  save_fig <- function(p, name, width = 8, height = 6, dpi = 300) {
    dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(file.path(OUT_DIR, paste0(name, ".pdf")), p, width = width, height = height)
    ggsave(file.path(OUT_DIR, paste0(name, ".png")), p, width = width, height = height, dpi = dpi)
    invisible(p)
  }

  MODEL_COLORS <- c(LR_EN = "#4C78A8", LinSVM_cal = "#1B9E77", RF = "#D95F02", XGBoost = "#7570B3")
  MODEL_LABELS <- c(LR_EN = "Logistic (EN)", LinSVM_cal = "Linear SVM", RF = "Random Forest", XGBoost = "XGBoost")
}

# ── Output directories ──────────────────────────────────────────────────────
# Set relative to calling script's location, not hardcoded
FACTORIAL_OUT_DIR <- file.path(dirname(sys.frame(1)$ofile %||% "."), "figures")
FACTORIAL_TBL_DIR <- file.path(dirname(sys.frame(1)$ofile %||% "."), "tables")
dir.create(FACTORIAL_OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FACTORIAL_TBL_DIR, recursive = TRUE, showWarnings = FALSE)

# Override OUT_DIR for save_fig() to use factorial output
OUT_DIR <- FACTORIAL_OUT_DIR

# ── Factorial factor palettes (fixed -- these are design choices) ────────────
CALIBRATION_COLORS <- c(
  logistic_intercept = "#2166AC",
  beta               = "#B2182B",
  isotonic           = "#4DAF4A"
)
CALIBRATION_LABELS <- c(
  logistic_intercept = "Logistic",
  beta               = "Beta",
  isotonic           = "Isotonic"
)

WEIGHTING_COLORS <- c(
  log  = "#E66101",
  sqrt = "#FDB863",
  none = "#5E4FA2"
)
WEIGHTING_LABELS <- c(
  log  = "Log weights",
  sqrt = "Sqrt weights",
  none = "Unweighted"
)

DOWNSAMPLING_COLORS <- c(
  `1`   = "#1B7837",
  `2`   = "#A6DBA0",
  `5`   = "#762A83"
)
DOWNSAMPLING_LABELS <- c(
  `1`   = "1:1 (none)",
  `2`   = "2:1",
  `5`   = "5:1"
)

# ── Dynamic recipe palette ──────────────────────────────────────────────────
# Generated from data -- works on any dataset with any recipe set.
generate_recipe_colors <- function(recipe_ids) {
  n <- length(recipe_ids)
  if (n == 0) return(character(0))
  colors <- hue_pal()(n)
  names(colors) <- sort(recipe_ids)
  colors
}

# ── Data loading helper ─────────────────────────────────────────────────────
load_factorial <- function(path = NULL) {
  # Auto-discover path if not provided
  if (is.null(path)) {
    candidates <- c(
      file.path(RESULTS_DIR, "factorial_compiled.csv"),
      file.path(CEL_ROOT, "results", "factorial_compiled.csv")
    )
    path <- Find(file.exists, candidates)
    if (is.null(path)) stop("Could not find factorial_compiled.csv. Provide path explicitly.")
  }

  df <- read.csv(path, stringsAsFactors = FALSE)

  # Auto-factor the guaranteed factorial columns (if present)
  factor_cols <- c("recipe_id", "factorial_model", "factorial_calibration",
                   "factorial_weighting", "factorial_downsampling")
  for (col in intersect(factor_cols, names(df))) {
    df[[col]] <- factor(df[[col]])
  }

  # Generate dynamic recipe colors
  if ("recipe_id" %in% names(df)) {
    RECIPE_COLORS <<- generate_recipe_colors(levels(df$recipe_id))
  }

  message(sprintf(
    "Loaded factorial data: %d rows, %d columns, %d recipes",
    nrow(df), ncol(df),
    if ("recipe_id" %in% names(df)) nlevels(df$recipe_id) else 0
  ))

  df
}

# ── Factorial-specific helpers ──────────────────────────────────────────────
factorial_save_fig <- function(p, name, width = 10, height = 7, dpi = 300) {
  save_fig(p, name, width = width, height = height, dpi = dpi)
}

factorial_save_table <- function(df, name) {
  dir.create(FACTORIAL_TBL_DIR, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(FACTORIAL_TBL_DIR, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE)
  message(sprintf("Saved table: %s (%d rows)", path, nrow(df)))
  invisible(path)
}
