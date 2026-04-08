#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
})

RUN_DIR <- "results/run_20260317_131842"
OUT_DIR <- "experiments/optimal-setup/codex_viz/cviz/out"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TOP_N <- 12
MODEL_LABELS <- c(
  LR_EN = "Logistic\n(EN)",
  LinSVM_cal = "Linear\nSVM",
  RF = "Random\nForest",
  XGBoost = "XGBoost"
)

theme_cel <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = NA),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(color = "grey40", size = 10)
    )
}

consensus <- read.csv(
  file.path(RUN_DIR, "consensus", "consensus_ranking.csv"),
  stringsAsFactors = FALSE
)

top <- consensus %>%
  arrange(consensus_rank) %>%
  slice_head(n = TOP_N) %>%
  mutate(protein_clean = str_remove(protein, "_resid$"))

rank_data <- top %>%
  select(protein, protein_clean, consensus_rank, LR_EN_rank, LinSVM_cal_rank, RF_rank, XGBoost_rank) %>%
  pivot_longer(cols = ends_with("_rank"), names_to = "model", values_to = "rank") %>%
  mutate(
    model = str_remove(model, "_rank$"),
    model_label = factor(MODEL_LABELS[model], levels = MODEL_LABELS),
    protein_clean = factor(protein_clean, levels = rev(top$protein_clean))
  )

p <- ggplot(rank_data, aes(x = model_label, y = protein_clean, fill = rank)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = round(rank)), size = 2.7, color = "white") +
  scale_fill_viridis_c(
    option = "magma",
    direction = -1,
    name = "Rank",
    limits = c(1, max(rank_data$rank, na.rm = TRUE))
  ) +
  labs(
    title = "Cross-Model Agreement on Top Consensus Proteins",
    subtitle = "Lower rank means greater feature importance within a model.",
    x = "",
    y = ""
  ) +
  theme_cel() +
  theme(
    axis.text.y = element_text(size = 8, family = "mono"),
    panel.grid = element_blank()
  )

ggsave(file.path(OUT_DIR, "fig04_cross_model_agreement.pdf"), p, width = 7, height = 6.5, create.dir = TRUE)
