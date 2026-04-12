#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
})

RUN_DIR <- "results/run_20260317_131842"
OUT_DIR <- "experiments/optimal-setup/codex_viz/cviz/out"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

MODELS <- c("LR_EN", "RF")
MODEL_LABELS <- c(LR_EN = "Logistic (EN)", RF = "Random Forest")
MODEL_COLORS <- c(LR_EN = "#4878CF", RF = "#D65F5F")

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

load_selection <- function(model) {
  path <- file.path(
    RUN_DIR, model, "aggregated", "optimize_panel", "optimal_panel_selection.json"
  )
  sel <- fromJSON(path)

  sel$decisions %>%
    mutate(
      model = model,
      full_auroc = sel$full_model_auroc,
      delta_margin = sel$delta_used,
      mean_auroc = stability$mean_auroc,
      std_auroc = stability$std_auroc,
      cv = stability$cv,
      stable = stability$stable,
      ni_ci_up = if (!is.null(noninferiority)) noninferiority$ci_upper else NA_real_,
      status = case_when(
        accepted ~ "Accepted",
        !stable ~ "Unstable",
        TRUE ~ "Non-inferior fail"
      )
    ) %>%
    select(model, size, full_auroc, delta_margin, mean_auroc, std_auroc, cv, stable, ni_ci_up, status)
}

decisions <- bind_rows(lapply(MODELS, load_selection)) %>%
  mutate(
    model_label = MODEL_LABELS[model],
    status = factor(status, levels = c("Accepted", "Non-inferior fail", "Unstable"))
  )

reference <- decisions %>%
  distinct(model, model_label, full_auroc, delta_margin)

p <- ggplot(decisions, aes(x = size, y = mean_auroc, color = model)) +
  geom_rect(
    data = reference,
    aes(
      xmin = -Inf, xmax = Inf,
      ymin = full_auroc - delta_margin, ymax = full_auroc,
      fill = model
    ),
    inherit.aes = FALSE,
    alpha = 0.08
  ) +
  geom_hline(
    data = reference,
    aes(yintercept = full_auroc, color = model),
    linetype = "dashed",
    linewidth = 0.5
  ) +
  geom_ribbon(
    aes(ymin = mean_auroc - std_auroc, ymax = mean_auroc + std_auroc, fill = model),
    alpha = 0.15,
    color = NA
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(shape = status), size = 2.5, stroke = 0.6) +
  facet_wrap(~ model_label, ncol = 1) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS) +
  scale_fill_manual(values = MODEL_COLORS, guide = "none") +
  scale_shape_manual(
    values = c("Accepted" = 16, "Non-inferior fail" = 17, "Unstable" = 4),
    name = "Decision"
  ) +
  scale_x_continuous(breaks = sort(unique(decisions$size))) +
  labs(
    title = "Decision Frontier for Panel Optimization",
    subtitle = "Dashed line = full model AUROC. Shaded band = non-inferiority margin. Points encode acceptance status.",
    x = "Panel size",
    y = "Mean AUROC across seeds",
    color = "Model"
  ) +
  theme_cel()

ggsave(
  file.path(OUT_DIR, "fig01_decision_frontier.pdf"),
  p, width = 8.5, height = 8, create.dir = TRUE
)
