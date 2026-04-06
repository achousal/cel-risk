#!/usr/bin/env Rscript
# 03_feature_persistence.R
#
# Feature persistence heatmap: rows = top proteins (by consensus rank),
# columns = panel sizes. Shows which proteins are "core" vs "passengers".
# Split by model to show agreement.
#
# Usage:
#   Rscript experiments/optimal-setup/claude/viz/03_feature_persistence.R
#
# Outputs: optimal-setup/claude/viz/out/fig03_feature_persistence.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
  library(stringr)
  library(scales)
})

# ── Config ────────────────────────────────────────────────────────────────────
RUN_DIR  <- "results/run_20260317_131842"
OUT_DIR  <- "experiments/optimal-setup/claude/viz/out"
TOP_N    <- 30  # proteins to show
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

MODELS <- c("LR_EN", "RF")

MODEL_LABELS <- c(
  LR_EN = "Logistic (EN)",
  RF    = "Random Forest"
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

# ── Load consensus ranking for ordering ───────────────────────────────────────
consensus <- read.csv(
  file.path(RUN_DIR, "consensus", "consensus_ranking.csv"),
  stringsAsFactors = FALSE
)

top_proteins <- consensus %>%
  arrange(consensus_rank) %>%
  slice_head(n = TOP_N) %>%
  mutate(protein_clean = str_remove(protein, "_resid$")) %>%
  pull(protein)

top_labels <- str_remove(top_proteins, "_resid$")

# ── Load panel curve data (protein lists per size) ────────────────────────────
load_panels <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "panel_curve_aggregated.csv")
  df <- read.csv(path, stringsAsFactors = FALSE)

  # Parse protein lists
  rows <- lapply(seq_len(nrow(df)), function(i) {
    proteins <- fromJSON(df$proteins[i])
    data.frame(
      model   = model,
      size    = df$size[i],
      protein = proteins,
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

panels <- bind_rows(lapply(MODELS, load_panels))

# ── Build presence matrix ─────────────────────────────────────────────────────
presence <- panels %>%
  filter(protein %in% top_proteins) %>%
  mutate(
    present       = TRUE,
    protein_clean = str_remove(protein, "_resid$"),
    protein       = factor(protein, levels = rev(top_proteins)),
    model_label   = MODEL_LABELS[model]
  )

# Full grid for missing = absent
full_grid <- expand.grid(
  protein     = top_proteins,
  size        = sort(unique(panels$size)),
  model_label = MODEL_LABELS,
  stringsAsFactors = FALSE
) %>%
  mutate(protein = factor(protein, levels = rev(top_proteins)))

presence_full <- full_grid %>%
  left_join(
    presence %>% select(protein, size, model_label, present),
    by = c("protein", "size", "model_label")
  ) %>%
  mutate(present = ifelse(is.na(present), FALSE, TRUE))

# ── Fig 3a: Feature persistence heatmap (split by model) ─────────────────────
message("Fig 3a: Feature persistence heatmap …")

p3a <- ggplot(presence_full,
              aes(x = factor(size), y = protein, fill = present)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_wrap(~ model_label) +
  scale_fill_manual(
    values = c(`TRUE` = "#2166ac", `FALSE` = "#f7f7f7"),
    labels = c(`TRUE` = "In panel", `FALSE` = "Excluded"),
    name   = ""
  ) +
  scale_y_discrete(labels = function(x) str_remove(x, "_resid$")) +
  labs(
    title    = "Feature Persistence Across Panel Sizes",
    subtitle = sprintf("Top %d proteins by consensus rank. Ordered top (rank 1) to bottom.", TOP_N),
    x = "Panel size",
    y = ""
  ) +
  theme_cel() +
  theme(
    axis.text.y  = element_text(size = 7, family = "mono"),
    axis.text.x  = element_text(size = 8, angle = 45, hjust = 1),
    panel.grid    = element_blank()
  )

ggsave(file.path(OUT_DIR, "fig03a_feature_persistence.pdf"), p3a,
       width = 12, height = 9)

# ── Fig 3b: Persistence score bar chart ───────────────────────────────────────
message("Fig 3b: Persistence score …")

# Count how many panel sizes each protein appears in, per model
n_sizes <- panels %>%
  filter(protein %in% top_proteins) %>%
  group_by(protein) %>%
  summarize(
    n_appearances = n(),
    max_possible  = length(unique(panels$size)) * length(MODELS),
    persistence   = n_appearances / max_possible,
    .groups = "drop"
  ) %>%
  mutate(
    protein_clean = str_remove(protein, "_resid$"),
    protein = factor(protein, levels = top_proteins)
  ) %>%
  arrange(desc(persistence))

p3b <- ggplot(n_sizes, aes(x = reorder(protein_clean, persistence),
                           y = persistence)) +
  geom_col(fill = "#2166ac", alpha = 0.8) +
  geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50") +
  coord_flip() +
  scale_y_continuous(labels = percent_format(), limits = c(0, 1.05)) +
  labs(
    title    = "Feature Persistence Score",
    subtitle = "Fraction of (model x panel-size) combinations where protein is present",
    x = "", y = "Persistence"
  ) +
  theme_cel() +
  theme(axis.text.y = element_text(size = 7, family = "mono"))

ggsave(file.path(OUT_DIR, "fig03b_persistence_score.pdf"), p3b,
       width = 7, height = 8)

message("Done: fig03a, fig03b -> ", OUT_DIR)
