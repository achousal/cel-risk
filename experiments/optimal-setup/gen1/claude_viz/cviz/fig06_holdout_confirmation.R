#!/usr/bin/env Rscript
# fig06_holdout_confirmation.R
#
# Forest plot: 10-protein panel vs 4-protein core on holdout data.
# LinSVM_cal only. Shows AUROC and PR-AUC with 95% CI.
# The final gate: does the locked setup hold?
#
# Data: results/run_phase3_holdout/LinSVM_cal/aggregated/metrics/
#       results/run_phase3_holdout_4protein/LinSVM_cal/aggregated/metrics/

source("_theme.R")

# ── Load summary metrics ─────────────────────────────────────────────────────
read_summary <- function(run_dir) {
  read.csv(
    file.path(RESULTS_DIR, run_dir, "LinSVM_cal", "aggregated", "metrics",
              "test_metrics_summary.csv"),
    stringsAsFactors = FALSE
  )
}

full_panel <- read_summary("run_phase3_holdout")
core_panel <- read_summary("run_phase3_holdout_4protein")

# ── Assemble data ────────────────────────────────────────────────────────────
metrics <- c("auroc", "prauc")
metric_labels <- c(auroc = "AUROC", prauc = "PR-AUC")

build_row <- function(df, panel_name, metric) {
  data.frame(
    panel   = panel_name,
    metric  = metric_labels[metric],
    mean    = df[[paste0(metric, "_mean")]],
    lo      = df[[paste0(metric, "_ci95_lo")]],
    hi      = df[[paste0(metric, "_ci95_hi")]],
    stringsAsFactors = FALSE
  )
}

forest_data <- bind_rows(
  build_row(full_panel, "10-protein panel", "auroc"),
  build_row(full_panel, "10-protein panel", "prauc"),
  build_row(core_panel, "4-protein core", "auroc"),
  build_row(core_panel, "4-protein core", "prauc")
) %>%
  mutate(
    panel  = factor(panel, levels = c("4-protein core", "10-protein panel")),
    metric = factor(metric, levels = c("AUROC", "PR-AUC"))
  )

# ── Fig 6: Holdout forest plot ──────────────────────────────────────────────
message("Fig 6: Holdout confirmation ...")

p6 <- ggplot(forest_data, aes(x = mean, y = panel, color = panel)) +
  geom_pointrange(aes(xmin = lo, xmax = hi),
                  linewidth = 0.7, size = 2.5) +
  geom_text(aes(label = sprintf("%.3f [%.3f, %.3f]", mean, lo, hi)),
            vjust = -1.2, size = 3, color = "grey25", show.legend = FALSE) +
  facet_wrap(~ metric, scales = "free_x", ncol = 2) +
  scale_color_manual(
    values = c("10-protein panel" = "#1B9E77", "4-protein core" = "#E69F00"),
    guide = "none"
  ) +
  labs(
    title    = "Holdout Confirmation: 10-Protein Panel vs 4-Protein Core",
    subtitle = "LinSVM_cal, 10 holdout splits. Point = mean, whiskers = 95% CI.",
    x = "Metric value (holdout test set)",
    y = "",
    caption  = "Source: Phase 3 holdout evaluation (seeds 200-209, n=150 per split)"
  ) +
  theme_cel() +
  theme(
    panel.grid.major.y = element_blank(),
    strip.text         = element_text(size = 12)
  )

save_fig(p6, "fig06_holdout_confirmation", width = 10, height = 4.5)
message("Done: fig06")
