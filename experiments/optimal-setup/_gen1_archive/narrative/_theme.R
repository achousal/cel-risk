#!/usr/bin/env Rscript
# _theme.R -- shared theme, palettes, and helpers for narrative figures.
# Source this at the top of each figure script.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(scales)
})

# ── Paths ────────────────────────────────────────────────────────────────────
CEL_ROOT    <- normalizePath("../../..", mustWork = TRUE)  # cel-risk root
RESULTS_DIR <- file.path(CEL_ROOT, "results")
PANELS_DIR  <- file.path(CEL_ROOT, "experiments/optimal-setup/panel-sweep/panels")
OUT_DIR     <- file.path(CEL_ROOT, "experiments/optimal-setup/narrative/out")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Palettes ─────────────────────────────────────────────────────────────────
MODEL_COLORS <- c(
  LR_EN      = "#4C78A8",
  LinSVM_cal = "#1B9E77",
  RF         = "#D95F02",
  XGBoost    = "#7570B3"
)

MODEL_LABELS <- c(
  LR_EN      = "Logistic (EN)",
  LinSVM_cal = "Linear SVM",
  RF         = "Random Forest",
  XGBoost    = "XGBoost"
)

ORDER_COLORS <- c(
  rra        = "#E69F00",
  importance = "#56B4E9",
  pathway    = "#009E73"
)

ORDER_LABELS <- c(
  rra        = "RRA",
  importance = "Importance",
  pathway    = "Pathway"
)

# ── Theme ────────────────────────────────────────────────────────────────────
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
      plot.caption       = element_text(color = "grey50", size = base_size - 2,
                                        hjust = 0)
    )
}

# ── Helpers ──────────────────────────────────────────────────────────────────
clean_protein <- function(x) str_remove(x, "_resid$") |> toupper()

save_fig <- function(p, stem, width, height) {
  ggsave(file.path(OUT_DIR, paste0(stem, ".pdf")), p,
         width = width, height = height)
  ggsave(file.path(OUT_DIR, paste0(stem, ".png")), p,
         width = width, height = height, dpi = 300)
  message("  -> ", file.path(OUT_DIR, paste0(stem, ".pdf")))
}
