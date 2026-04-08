#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(stringr)
})

RUN_DIR <- "results/run_20260317_131842"
OUT_DIR <- "experiments/optimal-setup/codex_viz/cviz/out"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PANEL_DIR <- "experiments/optimal-setup/panel-sweep/panels"
MILESTONE_SIZES <- c(4, 7, 8, 10, 19)
TOP_N <- 15

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

consensus <- read.csv(file.path(RUN_DIR, "consensus", "consensus_ranking.csv"), stringsAsFactors = FALSE)

top_proteins <- consensus %>%
  arrange(consensus_rank) %>%
  slice_head(n = TOP_N) %>%
  pull(protein)

load_pathway_panel <- function(size) {
  path <- file.path(PANEL_DIR, sprintf("pathway_%sp.csv", size))
  proteins <- readLines(path, warn = FALSE)
  data.frame(
    size = size,
    protein = proteins,
    stringsAsFactors = FALSE
  )
}

panels <- bind_rows(lapply(MILESTONE_SIZES, load_pathway_panel))

presence <- expand.grid(
  protein = top_proteins,
  size = MILESTONE_SIZES,
  stringsAsFactors = FALSE
) %>%
  left_join(
    panels %>% mutate(present = TRUE),
    by = c("protein", "size")
  ) %>%
  mutate(
    present = ifelse(is.na(present), FALSE, TRUE),
    protein_clean = str_remove(protein, "_resid$"),
    protein = factor(protein, levels = rev(top_proteins))
  )

p <- ggplot(presence, aes(x = factor(size), y = protein, fill = present)) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_manual(
    values = c(`TRUE` = "#2166ac", `FALSE` = "#f7f7f7"),
    labels = c(`TRUE` = "In panel", `FALSE` = "Excluded"),
    name = ""
  ) +
  scale_y_discrete(labels = function(x) str_remove(x, "_resid$")) +
  labs(
    title = "Core-to-Extension Persistence at Milestone Panel Sizes",
    subtitle = "Pathway-order milestones emphasize the 4-protein core, 7-protein consensus panel, operating range, and accepted extension.",
    x = "Panel size milestone",
    y = ""
  ) +
  theme_cel() +
  theme(
    axis.text.y = element_text(size = 7, family = "mono"),
    panel.grid = element_blank()
  )

ggsave(file.path(OUT_DIR, "fig03_core_extension_persistence.pdf"), p, width = 7, height = 8.5, create.dir = TRUE)
