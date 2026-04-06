#!/usr/bin/env Rscript
# 02_decision_heatmap.R
#
# Decision landscape heatmap: rows = panel sizes, columns = decision criteria.
# Shows WHY each candidate panel was accepted or rejected.
#
# Usage:
#   Rscript experiments/optimal-setup/claude/viz/02_decision_heatmap.R
#
# Outputs: optimal-setup/claude/viz/out/fig02_decision_heatmap.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(jsonlite)
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
load_decisions <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "optimal_panel_selection.json")
  sel <- fromJSON(path)
  decisions <- sel$decisions

  # noninferiority may have NAs for unstable panels
  ni <- decisions$noninferiority
  decisions %>%
    mutate(
      model      = model,
      mean_auroc = stability$mean_auroc,
      cv         = stability$cv,
      stable     = stability$stable,
      ni_delta   = if (!is.null(ni)) ni$delta_estimate else NA_real_,
      ni_ci_up   = if (!is.null(ni)) ni$ci_upper       else NA_real_,
      ni_pval    = if (!is.null(ni)) ni$p_value         else NA_real_,
      ni_pass    = if (!is.null(ni)) ni$rejected         else FALSE
    ) %>%
    select(model, size, accepted, mean_auroc, cv, stable, ni_delta, ni_ci_up, ni_pval, ni_pass)
}

decisions <- bind_rows(lapply(MODELS, load_decisions))

# ── Build criteria matrix ─────────────────────────────────────────────────────
# Each criterion: pass (TRUE) or fail (FALSE)
criteria <- decisions %>%
  mutate(
    `Stability\n(CV < 0.05)` = stable,
    `Non-inferiority\n(CI upper < 0.02)` = ni_pass,
    `Overall` = accepted,
    panel_label = paste0("k=", size)
  )

# Long format for heatmap
criteria_long <- criteria %>%
  select(model, size, panel_label,
         `Stability\n(CV < 0.05)`,
         `Non-inferiority\n(CI upper < 0.02)`,
         `Overall`) %>%
  pivot_longer(
    cols = c(`Stability\n(CV < 0.05)`, `Non-inferiority\n(CI upper < 0.02)`, `Overall`),
    names_to  = "criterion",
    values_to = "pass"
  ) %>%
  mutate(
    criterion = factor(criterion, levels = c(
      "Stability\n(CV < 0.05)",
      "Non-inferiority\n(CI upper < 0.02)",
      "Overall"
    )),
    model_label = MODEL_LABELS[model]
  )

# ── Fig 2a: Pass/fail tile heatmap ───────────────────────────────────────────
message("Fig 2a: Decision pass/fail heatmap …")

p2a <- ggplot(criteria_long,
              aes(x = criterion, y = reorder(panel_label, size), fill = pass)) +
  geom_tile(color = "white", linewidth = 0.8) +
  facet_wrap(~ model_label) +
  scale_fill_manual(
    values = c(`TRUE` = "#66c2a5", `FALSE` = "#fc8d62"),
    labels = c(`TRUE` = "Pass", `FALSE` = "Fail"),
    name   = ""
  ) +
  labs(
    title    = "Panel Selection Decision Landscape",
    subtitle = "Pass/fail for each gate at each panel size",
    x = "", y = "Panel size"
  ) +
  theme_cel() +
  theme(
    axis.text.x  = element_text(size = 9),
    panel.grid    = element_blank()
  )

ggsave(file.path(OUT_DIR, "fig02a_decision_passfail.pdf"), p2a,
       width = 8, height = 6)

# ── Fig 2b: Continuous criteria values ────────────────────────────────────────
message("Fig 2b: Decision criteria continuous …")

cont_data <- decisions %>%
  mutate(model_label = MODEL_LABELS[model]) %>%
  select(model_label, size, cv, ni_delta, ni_ci_up) %>%
  pivot_longer(cols = c(cv, ni_delta, ni_ci_up),
               names_to = "criterion", values_to = "value") %>%
  mutate(
    criterion = recode(criterion,
      cv        = "CV of AUROC",
      ni_delta  = "Delta (AUROC drop)",
      ni_ci_up  = "CI upper bound"
    ),
    criterion = factor(criterion, levels = c("CV of AUROC", "Delta (AUROC drop)", "CI upper bound"))
  )

# Threshold lines
thresholds <- data.frame(
  criterion = factor(c("CV of AUROC", "Delta (AUROC drop)", "CI upper bound"),
                     levels = c("CV of AUROC", "Delta (AUROC drop)", "CI upper bound")),
  threshold = c(0.05, 0.02, 0.02)
)

p2b <- ggplot(cont_data, aes(x = size, y = value, color = model_label)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  geom_hline(data = thresholds, aes(yintercept = threshold),
             linetype = "dashed", color = "grey40", linewidth = 0.4) +
  facet_wrap(~ criterion, scales = "free_y", ncol = 1) +
  scale_color_manual(
    values = c("Logistic (EN)" = "#4878CF", "Random Forest" = "#D65F5F"),
    name = "Model"
  ) +
  labs(
    title    = "Decision Criteria vs Panel Size",
    subtitle = "Dashed line = acceptance threshold",
    x = "Panel size",
    y = "Value"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig02b_decision_continuous.pdf"), p2b,
       width = 7, height = 8)

message("Done: fig02a, fig02b -> ", OUT_DIR)
