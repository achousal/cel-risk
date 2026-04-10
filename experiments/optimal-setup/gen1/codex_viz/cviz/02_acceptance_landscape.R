#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
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

load_decisions <- function(model) {
  path <- file.path(
    RUN_DIR, model, "aggregated", "optimize_panel", "optimal_panel_selection.json"
  )
  sel <- fromJSON(path)
  d <- sel$decisions

  d %>%
    mutate(
      model = model,
      mean_auroc = stability$mean_auroc,
      cv = stability$cv,
      stable = stability$stable,
      ni_delta = if (!is.null(noninferiority)) noninferiority$delta_estimate else NA_real_,
      ni_ci_up = if (!is.null(noninferiority)) noninferiority$ci_upper else NA_real_,
      ni_pass = if (!is.null(noninferiority)) noninferiority$rejected else FALSE,
      status = case_when(
        accepted ~ "Accepted",
        !stable ~ "Unstable",
        TRUE ~ "Non-inferior fail"
      )
    ) %>%
    select(model, size, accepted, cv, stable, ni_delta, ni_ci_up, ni_pass, status)
}

decisions <- bind_rows(lapply(MODELS, load_decisions))

criteria_long <- decisions %>%
  mutate(
    model_label = MODEL_LABELS[model],
    panel_label = paste0("k=", size),
    `Stability\n(CV < 0.05)` = stable,
    `Non-inferiority\n(CI upper < 0.02)` = ni_pass,
    Overall = accepted
  ) %>%
  select(
    model_label, size, panel_label,
    `Stability\n(CV < 0.05)`, `Non-inferiority\n(CI upper < 0.02)`, Overall
  ) %>%
  pivot_longer(
    cols = c(`Stability\n(CV < 0.05)`, `Non-inferiority\n(CI upper < 0.02)`, Overall),
    names_to = "criterion",
    values_to = "pass"
  ) %>%
  mutate(
    criterion = factor(
      criterion,
      levels = c("Stability\n(CV < 0.05)", "Non-inferiority\n(CI upper < 0.02)", "Overall")
    )
  )

p1 <- ggplot(criteria_long, aes(x = criterion, y = reorder(panel_label, size), fill = pass)) +
  geom_tile(color = "white", linewidth = 0.8) +
  facet_wrap(~ model_label) +
  scale_fill_manual(
    values = c(`TRUE` = "#66c2a5", `FALSE` = "#fc8d62"),
    labels = c(`TRUE` = "Pass", `FALSE` = "Fail"),
    name = ""
  ) +
  labs(
    title = "Acceptance Landscape",
    subtitle = "Each panel size is evaluated by stability, non-inferiority, and overall acceptance.",
    x = "",
    y = "Panel size"
  ) +
  theme_cel() +
  theme(panel.grid = element_blank())

ggsave(file.path(OUT_DIR, "fig02a_acceptance_landscape.pdf"), p1, width = 8, height = 6, create.dir = TRUE)

tradeoff <- decisions %>%
  mutate(
    model_label = MODEL_LABELS[model],
    status = factor(status, levels = c("Accepted", "Non-inferior fail", "Unstable"))
  ) %>%
  filter(!is.na(ni_delta))

p2 <- ggplot(tradeoff, aes(x = cv, y = ni_delta)) +
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "grey60") +
  geom_hline(yintercept = 0.02, linetype = "dashed", color = "grey60") +
  geom_point(aes(color = model, shape = status), size = 3, alpha = 0.8) +
  geom_text(aes(label = size), size = 2.5, vjust = -1, color = "grey30") +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = "Model") +
  scale_shape_manual(
    values = c("Accepted" = 16, "Non-inferior fail" = 17, "Unstable" = 4),
    name = "Decision"
  ) +
  labs(
    title = "Stability vs Performance Tradeoff",
    subtitle = "Dashed lines show the acceptance thresholds for stability and AUROC loss.",
    x = "CV of AUROC across seeds",
    y = "Delta from full-model AUROC"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig02b_stability_tradeoff.pdf"), p2, width = 8, height = 6, create.dir = TRUE)
