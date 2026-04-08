#!/usr/bin/env Rscript
# fig01_order_heatmaps.R
#
# Three side-by-side heatmaps (one per feature ordering strategy):
#   Model (y) x Panel size (x), fill = AUROC.
# Purpose: settle the ordering question -- show performance is stable across
# orders, then pick one and move on.
#
# Annotation: BH-significant proteins (top 4) marked at panel_size = 4.

source("_theme.R")
library(patchwork)

# ── Load ─────────────────────────────────────────────────────────────────────
agg <- read.csv(
  file.path(RESULTS_DIR, "compiled_results_aggregated.csv"),
  stringsAsFactors = FALSE
)

base <- agg %>%
  filter(model != "ENSEMBLE") %>%
  select(order, panel_size, model, auroc = pooled_test_auroc) %>%
  mutate(model_label = MODEL_LABELS[model])

# ── Per-order heatmap builder ────────────────────────────────────────────────
make_heatmap <- function(df, ord, panel_label) {
  dat <- df %>%
    filter(order == ord) %>%
    mutate(
      model_label = factor(model_label,
                           levels = rev(MODEL_LABELS[c("LR_EN", "LinSVM_cal",
                                                        "RF", "XGBoost")]))
    )

  # Best cell per column
  best <- dat %>%
    group_by(panel_size) %>%
    slice_max(auroc, n = 1) %>%
    ungroup()

  # Global best for this order
  global_best <- dat %>% slice_max(auroc, n = 1)

  ggplot(dat, aes(x = factor(panel_size), y = model_label, fill = auroc)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_tile(data = best, color = "grey30", linewidth = 0.5, fill = NA) +
    geom_tile(data = global_best, color = "black", linewidth = 0.9, fill = NA) +
    geom_text(aes(label = sprintf("%.3f", auroc)),
              size = 2.0, color = "grey20") +
    # Mark the BH-significant panel size = 4
    geom_vline(xintercept = which(levels(factor(dat$panel_size)) == "4"),
               linetype = "dashed", color = "red", alpha = 0.5) +
    scale_fill_gradient2(
      low = "#D95F02", mid = "#FFFFBF", high = "#1B9E77",
      midpoint = median(dat$auroc),
      limits = range(df$auroc),
      name = "AUROC"
    ) +
    labs(
      title = panel_label,
      x = "Panel size", y = ""
    ) +
    theme_cel(base_size = 9) +
    theme(
      panel.grid       = element_blank(),
      axis.text.x      = element_text(size = 6, angle = 45, hjust = 1),
      legend.key.height = unit(0.6, "cm"),
      legend.position   = "none"
    )
}

# ── Build three heatmaps ────────────────────────────────────────────────────
p_rra  <- make_heatmap(base, "rra",        "A. RRA Order")
p_imp  <- make_heatmap(base, "importance", "B. Importance Order")
p_path <- make_heatmap(base, "pathway",    "C. Pathway Order")

# Shared legend from one panel
legend_plot <- base %>%
  filter(order == "rra") %>%
  mutate(model_label = factor(model_label,
                              levels = rev(MODEL_LABELS[c("LR_EN", "LinSVM_cal",
                                                           "RF", "XGBoost")]))) %>%
  ggplot(aes(x = factor(panel_size), y = model_label, fill = auroc)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "#D95F02", mid = "#FFFFBF", high = "#1B9E77",
    midpoint = median(base$auroc),
    limits = range(base$auroc),
    name = "AUROC"
  ) +
  theme_cel() +
  theme(legend.position = "bottom")

shared_legend <- cowplot::get_legend(legend_plot)

# ── Compose ──────────────────────────────────────────────────────────────────
message("Fig 1: Order heatmaps ...")

# Use patchwork for layout
p_combined <- (p_rra | p_imp | p_path) +
  plot_annotation(
    title    = "Model x Panel Size Performance Across Feature Ordering Strategies",
    subtitle = "Red dashed line = BH-significant 4-protein core. Black border = global best per order. Grey border = best model per panel size.",
    caption  = "264 configurations (4 models x 22 panel sizes x 3 orders). AUROC from pooled test predictions across 10 seeds.",
    theme = theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "grey40", size = 9),
      plot.caption  = element_text(color = "grey50", size = 8, hjust = 0)
    )
  )

# Add shared legend at bottom
if (requireNamespace("cowplot", quietly = TRUE)) {
  p_final <- cowplot::plot_grid(
    p_combined, shared_legend,
    ncol = 1, rel_heights = c(1, 0.06)
  )
} else {
  p_final <- p_combined
}

save_fig(p_final, "fig01_order_heatmaps", width = 16, height = 6)
message("Done: fig01")
