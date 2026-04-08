#!/usr/bin/env Rscript
# 04_rank_agreement.R
#
# Cross-model rank agreement: heatmap of per-model rank for top proteins,
# plus bump chart showing rank trajectories across models.
#
# Usage:
#   Rscript experiments/optimal-setup/claude/viz/04_rank_agreement.R
#
# Outputs: optimal-setup/claude/viz/out/fig04_rank_agreement.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
})

# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR <- "results/run_20260317_131842"
OUT_DIR <- "experiments/optimal-setup/claude/viz/out"
TOP_N   <- 25
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

MODEL_LABELS <- c(
  LR_EN      = "Logistic\n(EN)",
  LinSVM_cal = "Linear\nSVM",
  RF         = "Random\nForest",
  XGBoost    = "XGBoost"
)

MODEL_COLORS <- c(
  LR_EN      = "#4878CF",
  LinSVM_cal = "#6ACC65",
  RF         = "#D65F5F",
  XGBoost    = "#B47CC7"
)

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

# ── Load consensus ranking ────────────────────────────────────────────────────
consensus <- read.csv(
  file.path(RUN_DIR, "consensus", "consensus_ranking.csv"),
  stringsAsFactors = FALSE
)

top <- consensus %>%
  arrange(consensus_rank) %>%
  slice_head(n = TOP_N) %>%
  mutate(protein_clean = str_remove(protein, "_resid$"))

# ── Fig 4a: Rank heatmap across models ────────────────────────────────────────
message("Fig 4a: Rank heatmap …")

rank_data <- top %>%
  select(protein, protein_clean, consensus_rank,
         LR_EN_rank, LinSVM_cal_rank, RF_rank, XGBoost_rank) %>%
  pivot_longer(
    cols      = ends_with("_rank"),
    names_to  = "model",
    values_to = "rank"
  ) %>%
  mutate(
    model = str_remove(model, "_rank$"),
    model_label = MODEL_LABELS[model],
    model_label = factor(model_label, levels = MODEL_LABELS),
    protein_clean = factor(protein_clean,
                           levels = rev(str_remove(top$protein, "_resid$")))
  )

p4a <- ggplot(rank_data, aes(x = model_label, y = protein_clean, fill = rank)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = round(rank)), size = 2.5, color = "white") +
  scale_fill_viridis_c(
    option    = "magma",
    direction = -1,
    name      = "Rank",
    limits    = c(1, max(rank_data$rank, na.rm = TRUE))
  ) +
  labs(
    title    = "Cross-Model Feature Ranking Agreement",
    subtitle = sprintf("Top %d proteins by consensus score. Lower rank = more important.", TOP_N),
    x = "", y = ""
  ) +
  theme_cel() +
  theme(
    axis.text.y   = element_text(size = 7, family = "mono"),
    axis.text.x   = element_text(size = 9),
    panel.grid     = element_blank()
  )

ggsave(file.path(OUT_DIR, "fig04a_rank_heatmap.pdf"), p4a,
       width = 7, height = 9)

# ── Fig 4b: Bump chart ───────────────────────────────────────────────────────
message("Fig 4b: Bump chart …")

# Use only top 15 for readability in bump chart
bump_data <- rank_data %>%
  filter(protein %in% top$protein[1:15]) %>%
  mutate(
    protein_clean = factor(protein_clean,
                           levels = str_remove(top$protein[1:15], "_resid$"))
  )

p4b <- ggplot(bump_data, aes(x = model_label, y = rank, group = protein_clean,
                              color = protein_clean)) +
  geom_line(linewidth = 0.6, alpha = 0.7) +
  geom_point(size = 2.5) +
  scale_y_reverse(breaks = seq(1, max(bump_data$rank, na.rm = TRUE), by = 5)) +
  scale_color_manual(
    values = scales::hue_pal()(15),
    name   = "Protein"
  ) +
  labs(
    title    = "Feature Rank Trajectories Across Models",
    subtitle = "Top 15 consensus proteins. Lower = more important.",
    x = "", y = "Rank"
  ) +
  theme_cel() +
  guides(color = guide_legend(ncol = 3, override.aes = list(linewidth = 1.5)))

ggsave(file.path(OUT_DIR, "fig04b_bump_chart.pdf"), p4b,
       width = 9, height = 7)

# ── Fig 4c: Rank CV bar chart (disagreement metric) ──────────────────────────
message("Fig 4c: Rank disagreement …")

disagree <- top %>%
  select(protein_clean, rank_cv, consensus_rank) %>%
  mutate(protein_clean = factor(protein_clean,
                                levels = rev(protein_clean)))

p4c <- ggplot(disagree, aes(x = protein_clean, y = rank_cv)) +
  geom_col(aes(fill = rank_cv), show.legend = FALSE) +
  scale_fill_gradient(low = "#66c2a5", high = "#fc8d62") +
  coord_flip() +
  labs(
    title    = "Cross-Model Rank Disagreement",
    subtitle = "CV of rank across 4 models. Higher = more disagreement.",
    x = "", y = "Rank CV"
  ) +
  theme_cel() +
  theme(axis.text.y = element_text(size = 7, family = "mono"))

ggsave(file.path(OUT_DIR, "fig04c_rank_disagreement.pdf"), p4c,
       width = 7, height = 8)

message("Done: fig04a, fig04b, fig04c -> ", OUT_DIR)
