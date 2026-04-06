#!/usr/bin/env Rscript
# 01_panel_curve.R
#
# Panel performance curve: AUROC vs panel size for LR_EN and RF.
# Horizontal reference line = full model AUROC, shaded noninferiority margin.
# Point annotations for accepted/rejected decisions.
#
# Usage:
#   Rscript experiments/optimal-setup/claude/viz/01_panel_curve.R
#
# Outputs: optimal-setup/claude/viz/out/fig01_panel_curve.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
  library(scales)
})

# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR <- "results/run_20260317_131842"
OUT_DIR <- "experiments/optimal-setup/claude/viz/out"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

MODELS <- c("LR_EN", "RF")

MODEL_LABELS <- c(
  LR_EN = "Logistic (EN)",
  RF    = "Random Forest"
)

MODEL_COLORS <- c(
  LR_EN = "#4878CF",
  RF    = "#D65F5F"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_cel <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor  = element_blank(),
      strip.background  = element_rect(fill = "grey92", color = NA),
      legend.position   = "bottom",
      plot.title        = element_text(face = "bold", size = 12),
      plot.subtitle     = element_text(color = "grey40", size = 10)
    )
}

# ── Load data ─────────────────────────────────────────────────────────────────
load_selection <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "optimal_panel_selection.json")
  sel <- fromJSON(path)

  decisions <- sel$decisions %>%
    mutate(
      model      = model,
      mean_auroc = stability$mean_auroc,
      std_auroc  = stability$std_auroc,
      cv         = stability$cv,
      stable     = stability$stable
    )

  list(
    decisions      = decisions,
    full_auroc     = sel$full_model_auroc,
    selected_size  = sel$selected_size,
    delta          = sel$delta_used
  )
}

all_data <- lapply(MODELS, load_selection)
names(all_data) <- MODELS

decisions <- bind_rows(lapply(all_data, `[[`, "decisions"))

full_aurocs <- data.frame(
  model      = MODELS,
  full_auroc = sapply(all_data, `[[`, "full_auroc")
)

delta <- all_data[[1]]$delta  # same for both

# ── Build annotations ─────────────────────────────────────────────────────────
decisions <- decisions %>%
  mutate(
    status = case_when(
      accepted ~ "Accepted",
      !stable  ~ "Unstable",
      TRUE     ~ "Non-inferior fail"
    ),
    status = factor(status, levels = c("Accepted", "Non-inferior fail", "Unstable"))
  )

# ── Fig 1a: AUROC vs panel size, both models ─────────────────────────────────
message("Fig 1a: Panel curve (both models) …")

p1a <- ggplot(decisions, aes(x = size, y = mean_auroc, color = model)) +
  # Noninferiority bands (per model)
  geom_rect(
    data = full_aurocs,
    aes(xmin = 0, xmax = Inf,
        ymin = full_auroc - delta, ymax = full_auroc,
        fill = model),
    inherit.aes = FALSE, alpha = 0.08
  ) +
  # Full model reference lines

  geom_hline(
    data = full_aurocs,
    aes(yintercept = full_auroc, color = model),
    linetype = "dashed", linewidth = 0.5
  ) +
  # Ribbon for SD
  geom_ribbon(
    aes(ymin = mean_auroc - std_auroc, ymax = mean_auroc + std_auroc,
        fill = model),
    alpha = 0.15, color = NA
  ) +
  # Line + points
  geom_line(linewidth = 0.8) +
  geom_point(aes(shape = status), size = 2.5, stroke = 0.6) +
  # Scales
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS) +
  scale_fill_manual(values = MODEL_COLORS, labels = MODEL_LABELS, guide = "none") +
  scale_shape_manual(
    values = c("Accepted" = 16, "Non-inferior fail" = 17, "Unstable" = 4),
    name   = "Decision"
  ) +
  scale_x_continuous(breaks = sort(unique(decisions$size))) +
  coord_cartesian(ylim = c(0.80, 0.94)) +
  labs(
    title    = "Panel Optimization: AUROC vs Panel Size",
    subtitle = sprintf("RFE with re-tuning, 30 seeds. Dashed = full model. Shaded = delta=%.2f margin.", delta),
    x = "Panel size (number of proteins)",
    y = "Mean AUROC (validation)",
    color = "Model"
  ) +
  theme_cel() +
  guides(color = guide_legend(order = 1), shape = guide_legend(order = 2))

ggsave(file.path(OUT_DIR, "fig01a_panel_curve_both.pdf"), p1a,
       width = 9, height = 5.5)

# ── Fig 1b: Multi-metric facet (AUROC, PR-AUC, Brier, Sens@95Spec) ──────────
message("Fig 1b: Panel curve (multi-metric) …")

load_metrics <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "metrics_summary_aggregated.csv")
  read.csv(path, stringsAsFactors = FALSE) %>%
    mutate(model = model)
}

metrics <- bind_rows(lapply(MODELS, load_metrics))

metrics_long <- metrics %>%
  select(model, size, auroc_val, prauc_val, brier_val, sens_at_95spec_val) %>%
  pivot_longer(cols = c(auroc_val, prauc_val, brier_val, sens_at_95spec_val),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric = recode(metric,
      auroc_val          = "AUROC",
      prauc_val          = "PR-AUC",
      brier_val          = "Brier Score",
      sens_at_95spec_val = "Sens @ 95% Spec"
    ),
    metric = factor(metric, levels = c("AUROC", "PR-AUC", "Brier Score", "Sens @ 95% Spec"))
  )

p1b <- ggplot(metrics_long, aes(x = size, y = value, color = model)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  facet_wrap(~ metric, scales = "free_y", ncol = 2) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS) +
  labs(
    title    = "Panel Optimization: Multi-Metric View",
    subtitle = "Validation metrics across panel sizes (30 seeds)",
    x = "Panel size",
    y = "Metric value",
    color = "Model"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig01b_panel_curve_multimetric.pdf"), p1b,
       width = 9, height = 7)

message("Done: fig01a, fig01b -> ", OUT_DIR)
