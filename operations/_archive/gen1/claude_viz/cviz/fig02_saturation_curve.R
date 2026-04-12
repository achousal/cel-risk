#!/usr/bin/env Rscript
# fig02_saturation_curve.R
#
# AUROC vs panel size for LinSVM_cal across 3 ordering strategies.
# Highlights the 8-10 protein sweet spot and the p=7 dip.
# CI ribbon from per-seed summary stats.
#
# Data: results/compiled_results_aggregated.csv

source("_theme.R")

# ── Load ─────────────────────────────────────────────────────────────────────
agg <- read.csv(
  file.path(RESULTS_DIR, "compiled_results_aggregated.csv"),
  stringsAsFactors = FALSE
)

svm <- agg %>%
  filter(model == "LinSVM_cal") %>%
  select(order, panel_size,
         auroc      = summary_auroc_mean,
         auroc_lo   = summary_auroc_ci95_lo,
         auroc_hi   = summary_auroc_ci95_hi,
         brier      = pooled_test_brier_score) %>%
  mutate(
    order_label = ORDER_LABELS[order],
    order_label = factor(order_label, levels = c("Pathway", "Importance", "RRA"))
  )

# Best point (pathway p=10)
best <- svm %>% filter(order == "pathway") %>%
  filter(auroc == max(auroc))

# ── Fig 2: Saturation curve ─────────────────────────────────────────────────
message("Fig 2: Saturation curve ...")

# Importance and RRA share identical panels from p>=8, so their lines overlap.
# Use linetype + slight vertical dodge to keep all three visible.
dodge_amt <- 0.0004
svm <- svm %>%
  mutate(
    auroc_dodge = case_when(
      order_label == "Importance" ~ auroc + dodge_amt,
      order_label == "RRA"        ~ auroc - dodge_amt,
      TRUE                        ~ auroc
    ),
    auroc_lo_dodge = case_when(
      order_label == "Importance" ~ auroc_lo + dodge_amt,
      order_label == "RRA"        ~ auroc_lo - dodge_amt,
      TRUE                        ~ auroc_lo
    ),
    auroc_hi_dodge = case_when(
      order_label == "Importance" ~ auroc_hi + dodge_amt,
      order_label == "RRA"        ~ auroc_hi - dodge_amt,
      TRUE                        ~ auroc_hi
    )
  )

# Recompute best after dodge (use undodged value)
best <- svm %>% filter(order == "pathway") %>%
  filter(auroc == max(auroc))

p2 <- ggplot(svm, aes(x = panel_size, y = auroc_dodge, color = order_label,
                       fill = order_label, linetype = order_label)) +
  # Sweet-spot zone
  annotate("rect", xmin = 8, xmax = 10, ymin = -Inf, ymax = Inf,
           fill = "#d9ead3", alpha = 0.45) +
  annotate("text", x = 9, y = min(svm$auroc_lo, na.rm = TRUE) + 0.001,
           label = "Sweet spot", color = "#3c763d", size = 3,
           fontface = "italic") +
  # CI ribbon (pathway only — the others are identical from p>=8)
  geom_ribbon(
    data = svm %>% filter(order_label == "Pathway"),
    aes(ymin = auroc_lo, ymax = auroc_hi),
    alpha = 0.12, color = NA
  ) +
  # Lines
  geom_line(aes(linewidth = order_label)) +
  geom_point(aes(shape = order_label), size = 2) +
  # Best point annotation
  geom_point(data = best, aes(x = panel_size, y = auroc),
             size = 3.5, shape = 21, stroke = 0.7,
             fill = ORDER_COLORS["pathway"], color = "black",
             inherit.aes = FALSE, show.legend = FALSE) +
  annotate("text",
           x = best$panel_size + 1.5,
           y = best$auroc - 0.002,
           label = sprintf("p=%d\nAUROC=%.3f", best$panel_size, best$auroc),
           size = 3, color = "grey25", hjust = 0) +
  # Note about convergence
  annotate("text", x = 20, y = min(svm$auroc_lo, na.rm = TRUE) + 0.001,
           label = "Importance & RRA share identical\npanels from p >= 8",
           size = 2.5, color = "grey50", fontface = "italic") +
  # Scales
  scale_color_manual(values = setNames(ORDER_COLORS, ORDER_LABELS),
                     name = "Ordering") +
  scale_fill_manual(values = setNames(ORDER_COLORS, ORDER_LABELS),
                    guide = "none") +
  scale_linetype_manual(values = c("Pathway" = "solid",
                                    "Importance" = "dashed",
                                    "RRA" = "dotted"),
                         name = "Ordering") +
  scale_shape_manual(values = c("Pathway" = 16, "Importance" = 17, "RRA" = 15),
                      name = "Ordering") +
  scale_linewidth_manual(values = c("Pathway" = 1.2, "Importance" = 0.7,
                                     "RRA" = 0.7),
                          guide = "none") +
  scale_x_continuous(breaks = c(4, 7, 10, 15, 20, 25)) +
  labs(
    title    = "Panel-Size Saturation: Linear SVM",
    subtitle = "Mean AUROC across 10 seeds (50 Optuna trials). Ribbon = 95% CI (Pathway).",
    x = "Panel size (number of proteins)",
    y = "Mean AUROC (test)",
    caption  = "Source: Panel saturation sweep (264 configs)"
  ) +
  theme_cel() +
  theme(panel.grid.major.x = element_line(color = "grey92"))

save_fig(p2, "fig02_saturation_curve", width = 9, height = 5.5)
message("Done: fig02")
