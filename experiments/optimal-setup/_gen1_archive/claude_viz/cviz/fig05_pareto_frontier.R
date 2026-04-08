#!/usr/bin/env Rscript
# fig05_pareto_frontier.R
#
# AUROC vs Brier scatter for all 264 base-model configs.
# Pareto-optimal points highlighted and labeled.
# Point size = panel size, color = model.
#
# Data: results/compiled_results_aggregated.csv

source("_theme.R")

# ── Load ─────────────────────────────────────────────────────────────────────
agg <- read.csv(
  file.path(RESULTS_DIR, "compiled_results_aggregated.csv"),
  stringsAsFactors = FALSE
)

base <- agg %>%
  filter(model != "ENSEMBLE") %>%
  select(order, panel_size, model,
         auroc = pooled_test_auroc,
         brier = pooled_test_brier_score) %>%
  mutate(model_label = MODEL_LABELS[model])

# ── Compute Pareto front ────────────────────────────────────────────────────
# Maximize AUROC, minimize Brier
compute_pareto <- function(df) {
  df <- df %>% arrange(desc(auroc))
  pareto <- logical(nrow(df))
  min_brier <- Inf
  for (i in seq_len(nrow(df))) {
    if (df$brier[i] < min_brier) {
      pareto[i] <- TRUE
      min_brier <- df$brier[i]
    }
  }
  df$pareto <- pareto
  df
}

base <- compute_pareto(base)
front <- base %>% filter(pareto) %>% arrange(desc(auroc))

# ── Fig 5: Pareto frontier ─────────────────────────────────────────────────
message("Fig 5: Pareto frontier ...")

p5 <- ggplot(base, aes(x = brier, y = auroc)) +
  # All points
  geom_point(aes(color = model_label, size = panel_size),
             alpha = 0.5, shape = 16) +
  # Pareto front line
  geom_step(data = front, direction = "vh",
            color = "grey30", linewidth = 0.5, linetype = "dashed") +
  # Pareto points highlighted
  geom_point(data = front, aes(color = model_label),
             size = 4, shape = 21, stroke = 0.8, fill = NA) +
  # Labels for Pareto points
  ggrepel::geom_text_repel(
    data = front,
    aes(label = sprintf("%s\np=%d, %s",
                        ORDER_LABELS[order], panel_size, model_label)),
    size = 2.5, color = "grey20",
    nudge_x = 0.002, segment.size = 0.3, segment.color = "grey60",
    max.overlaps = 20, lineheight = 0.85
  ) +
  # Recommended config annotation
  geom_point(
    data = base %>% filter(order == "pathway", model == "LinSVM_cal",
                           panel_size == 10),
    shape = 23, size = 5, fill = "#1B9E77", color = "black", stroke = 0.8
  ) +
  scale_color_manual(values = setNames(MODEL_COLORS, MODEL_LABELS),
                     name = "Model") +
  scale_size_continuous(range = c(1, 5), name = "Panel size",
                        breaks = c(4, 10, 15, 20, 25)) +
  labs(
    title    = "Multi-Objective Trade-off: Discrimination vs Calibration",
    subtitle = "Each point = one (order x model x panel-size) config. Diamond = recommended.",
    x = "Brier score (lower = better calibration)",
    y = "AUROC (higher = better discrimination)",
    caption  = "Dashed line = Pareto frontier. 264 base-model configurations."
  ) +
  theme_cel() +
  guides(
    color = guide_legend(order = 1, override.aes = list(size = 3, alpha = 1)),
    size  = guide_legend(order = 2)
  )

save_fig(p5, "fig05_pareto_frontier", width = 10, height = 7)
message("Done: fig05")
