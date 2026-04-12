#!/usr/bin/env Rscript
# fig02_pareto_all_models.R
#
# AUROC vs Brier scatter for all 264 base-model configs (collapsed across orders
# since fig01 shows ordering doesn't matter much).
# Color = model, size = panel size, Pareto front highlighted.
# Purpose: identify SVM as the optimal model on the discrimination-calibration
# frontier.

source("_theme.R")
library(ggrepel)

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

# ── Compute Pareto front (maximize AUROC, minimize Brier) ───────────────────
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

# Count SVM on Pareto front
n_pareto     <- sum(base$pareto)
n_svm_pareto <- sum(base$pareto & base$model == "LinSVM_cal")

# ── Plot ─────────────────────────────────────────────────────────────────────
message("Fig 2: Pareto frontier ...")

p2 <- ggplot(base, aes(x = brier, y = auroc)) +
  # All points
  geom_point(aes(color = model_label, size = panel_size),
             alpha = 0.45, shape = 16) +
  # Pareto front line
  geom_step(data = front, direction = "vh",
            color = "grey30", linewidth = 0.5, linetype = "dashed") +
  # Pareto points highlighted
  geom_point(data = front, aes(color = model_label),
             size = 4, shape = 21, stroke = 0.8, fill = NA) +
  # Labels for Pareto points
  geom_text_repel(
    data = front,
    aes(label = sprintf("%s, p=%d", model_label, panel_size)),
    size = 2.5, color = "grey20",
    nudge_x = 0.002, segment.size = 0.3, segment.color = "grey60",
    max.overlaps = 20, lineheight = 0.85
  ) +
  scale_color_manual(
    values = setNames(MODEL_COLORS, MODEL_LABELS),
    name = "Model"
  ) +
  scale_size_continuous(
    range = c(1.5, 5), name = "Panel size",
    breaks = c(4, 10, 15, 20, 25)
  ) +
  labs(
    title    = "Multi-Objective Trade-off: Discrimination vs Calibration",
    subtitle = sprintf(
      "Each point = one (order x model x panel-size) config. %d/%d Pareto-optimal points are Linear SVM.",
      n_svm_pareto, n_pareto
    ),
    x = "Brier score (lower = better calibration)",
    y = "AUROC (higher = better discrimination)",
    caption = sprintf(
      "Dashed line = Pareto frontier. %d base-model configurations.", nrow(base)
    )
  ) +
  theme_cel() +
  guides(
    color = guide_legend(order = 1, override.aes = list(size = 3, alpha = 1)),
    size  = guide_legend(order = 2)
  )

save_fig(p2, "fig02_pareto_all_models", width = 10, height = 7)
message("Done: fig02")
