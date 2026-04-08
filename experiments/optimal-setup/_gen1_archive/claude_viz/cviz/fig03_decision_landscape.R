#!/usr/bin/env Rscript
# fig03_decision_landscape.R
#
# Two-panel figure:
#   A) ANOVA variance decomposition — % of AUROC variance by factor
#   B) Model x panel-size heatmap (pathway order) — which combos are best
#
# Data: compiled_results_aggregated.csv

source("_theme.R")
library(patchwork)

# ── Load ─────────────────────────────────────────────────────────────────────
agg <- read.csv(
  file.path(RESULTS_DIR, "compiled_results_aggregated.csv"),
  stringsAsFactors = FALSE
)

base <- agg %>% filter(model != "ENSEMBLE")

# ── Panel A: Variance decomposition ─────────────────────────────────────────
# Pre-computed from sweep-analysis.md ANOVA (saturated design, descriptive %)
anova_data <- data.frame(
  factor = c("Model", "Panel size", "Panel x Order",
             "3-way", "Panel x Model", "Order", "Model x Order"),
  pct    = c(32.0, 22.1, 15.5, 14.1, 9.8, 5.1, 1.3),
  stringsAsFactors = FALSE
) %>%
  mutate(
    factor    = factor(factor, levels = rev(factor)),
    load_bearing = pct > 5,
    label     = paste0(pct, "%")
  )

pa <- ggplot(anova_data, aes(x = pct, y = factor, fill = load_bearing)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = label), hjust = -0.15, size = 3.2, color = "grey25") +
  geom_vline(xintercept = 5, linetype = "dotted", color = "grey60") +
  scale_fill_manual(
    values = c(`TRUE` = "#1B9E77", `FALSE` = "#D9D9D9"),
    labels = c(`TRUE` = ">5% (load-bearing)", `FALSE` = "<5% (negligible)"),
    name = ""
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = "A. What Matters Most?",
    subtitle = "ANOVA variance decomposition of AUROC",
    x = "% of total variance", y = ""
  ) +
  theme_cel() +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position    = c(0.75, 0.2)
  )

# ── Panel B: Model x size heatmap (pathway order) ───────────────────────────
pathway <- base %>%
  filter(order == "pathway") %>%
  select(model, panel_size, auroc = pooled_test_auroc) %>%
  mutate(model_label = MODEL_LABELS[model])

# Grand mean for delta encoding
grand_mean <- mean(pathway$auroc)

heatmap_data <- pathway %>%
  mutate(
    delta_mAUROC = (auroc - grand_mean) * 1000,
    model_label  = factor(model_label, levels = rev(MODEL_LABELS[c("LR_EN", "LinSVM_cal", "RF", "XGBoost")]))
  )

# Best model per panel size (for border)
best_per_size <- heatmap_data %>%
  group_by(panel_size) %>%
  slice_max(delta_mAUROC, n = 1) %>%
  ungroup()

pb <- ggplot(heatmap_data, aes(x = factor(panel_size), y = model_label,
                                fill = delta_mAUROC)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_tile(data = best_per_size, color = "black", linewidth = 0.6,
            fill = NA) +
  geom_text(aes(label = sprintf("%+.0f", delta_mAUROC)),
            size = 2.2, color = "grey20") +
  scale_fill_gradient2(
    low = "#D95F02", mid = "white", high = "#1B9E77",
    midpoint = 0, name = "Delta\n(mAUROC)"
  ) +
  labs(
    title    = "B. Model x Panel Size (Pathway Order)",
    subtitle = sprintf("Delta from grand mean AUROC (%.3f). Black border = best model.", grand_mean),
    x = "Panel size", y = ""
  ) +
  theme_cel() +
  theme(
    panel.grid       = element_blank(),
    axis.text.x      = element_text(size = 7, angle = 45, hjust = 1),
    legend.key.height = unit(0.8, "cm")
  )

# ── Compose ──────────────────────────────────────────────────────────────────
message("Fig 3: Decision landscape ...")
p3 <- pa / pb + plot_layout(heights = c(1, 1.2))
save_fig(p3, "fig03_decision_landscape", width = 10, height = 10)
message("Done: fig03")
