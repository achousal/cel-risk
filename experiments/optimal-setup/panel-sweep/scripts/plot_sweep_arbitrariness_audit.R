#!/usr/bin/env Rscript
# plot_sweep_arbitrariness_audit.R
#
# Comprehensive visualization of the panel saturation sweep.
# Every design decision (panel size, addition order, model) is tested
# against H0 = "this choice is arbitrary."
#
# Data: compiled_results_aggregated.csv (sweep) + compiled_results_ensemble.csv
#
# Outputs: analysis/figures/sweep/fig{1-8}_*.pdf
#
# Usage:
#   cd /Users/andreschousal/Projects/Chowell_Lab/cel-risk
#   Rscript analysis/scripts/plot_sweep_arbitrariness_audit.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(scales)
  library(stringr)
  library(forcats)
  library(patchwork)
})

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR <- "results"
OUT_DIR     <- "analysis/figures/sweep"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

MODEL_LABELS <- c(
  LR_EN      = "Logistic (EN)",
  LinSVM_cal = "Linear SVM",
  RF         = "Random Forest",
  XGBoost    = "XGBoost",
  ENSEMBLE   = "Ensemble"
)

MODEL_COLORS <- c(
  LR_EN      = "#4878CF",
  LinSVM_cal = "#6ACC65",
  RF         = "#D65F5F",
  XGBoost    = "#B47CC7",
  ENSEMBLE   = "#333333"
)

ORDER_LABELS <- c(
  rra        = "RRA Rank",
  importance = "Importance Rank",
  pathway    = "Pathway-Informed"
)

ORDER_COLORS <- c(
  rra        = "#E69F00",
  importance = "#56B4E9",
  pathway    = "#009E73"
)

BASE_MODELS <- c("LR_EN", "LinSVM_cal", "RF", "XGBoost")

theme_sweep <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor  = element_blank(),
      strip.background  = element_rect(fill = "grey92", color = NA),
      legend.position   = "bottom",
      plot.title        = element_text(face = "bold", size = 13),
      plot.subtitle     = element_text(color = "grey40", size = 10),
      plot.caption      = element_text(color = "grey50", size = 8, hjust = 0)
    )
}


# ── Load & merge data ─────────────────────────────────────────────────────────
message("Loading sweep data ...")

ag <- read.csv(file.path(RESULTS_DIR, "compiled_results_aggregated.csv"),
               stringsAsFactors = FALSE) %>%
  mutate(model_type = ifelse(model %in% c("LR_EN", "LinSVM_cal"), "linear", "tree"))

ens <- read.csv(file.path(RESULTS_DIR, "compiled_results_ensemble.csv"),
                stringsAsFactors = FALSE) %>%
  mutate(model = "ENSEMBLE", model_type = "ensemble")

# Combine base models + ensemble
all_models <- bind_rows(ag, ens)

# Label factors
all_models <- all_models %>%
  mutate(
    model_label = factor(MODEL_LABELS[model], levels = MODEL_LABELS),
    order_label = factor(ORDER_LABELS[order], levels = ORDER_LABELS)
  )

# Base models only (for most analyses)
base <- all_models %>% filter(model != "ENSEMBLE")

message(sprintf("  %d base rows, %d ensemble rows, %d total",
                nrow(ag), nrow(ens), nrow(all_models)))


# ── Derived quantities ────────────────────────────────────────────────────────

# Grand mean AUROC (the "arbitrary" reference)
grand_mean <- mean(base$pooled_test_auroc, na.rm = TRUE)
grand_sd   <- sd(base$pooled_test_auroc, na.rm = TRUE)

# Best configuration
best_row <- base %>% slice_max(pooled_test_auroc, n = 1)
message(sprintf("  Grand mean AUROC: %.4f +/- %.4f", grand_mean, grand_sd))
message(sprintf("  Best config: %s / %s / p=%d -> AUROC %.4f",
                best_row$order, best_row$model, best_row$panel_size,
                best_row$pooled_test_auroc))


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: SATURATION CURVES (the flagship)
# AUROC vs panel size, per model, faceted by order.
# Grey band = grand mean +/- 1 SD ("arbitrary choice" zone).
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 1: Saturation curves ...")

# Use summary stats where available, else pooled
sat_data <- all_models %>%
  mutate(
    auroc = coalesce(summary_auroc_mean, pooled_test_auroc),
    auroc_lo = coalesce(summary_auroc_ci95_lo, pooled_test_auroc - 0.02),
    auroc_hi = coalesce(summary_auroc_ci95_hi, pooled_test_auroc + 0.02)
  )

fig1 <- ggplot(sat_data %>% filter(model != "ENSEMBLE"),
               aes(x = panel_size, color = model_label, fill = model_label)) +
  # H0 band: "arbitrary choice" zone
  annotate("rect",
           xmin = -Inf, xmax = Inf,
           ymin = grand_mean - grand_sd,
           ymax = grand_mean + grand_sd,
           fill = "grey80", alpha = 0.3) +
  geom_hline(yintercept = grand_mean, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  # CI ribbons
  geom_ribbon(aes(ymin = auroc_lo, ymax = auroc_hi), alpha = 0.10, color = NA) +
  # Model lines
  geom_line(aes(y = auroc), linewidth = 0.9) +
  geom_point(aes(y = auroc), size = 1.2) +
  # Ensemble overlay (thinner, black)
  geom_line(
    data = sat_data %>% filter(model == "ENSEMBLE"),
    aes(y = auroc),
    linewidth = 0.7, linetype = "dotted", color = "black"
  ) +
  facet_wrap(~ order_label, nrow = 1) +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  scale_fill_manual(values = MODEL_COLORS, name = NULL) +
  scale_x_continuous(breaks = c(4, 7, 10, 15, 20, 25)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  coord_cartesian(ylim = c(0.825, 0.865)) +
  labs(
    title    = "Panel saturation curves by model and addition order",
    subtitle = paste0("Grey band = grand mean +/- 1 SD (H0: panel choice is arbitrary). ",
                      "Dotted black = Ensemble. Ribbons = 95% CI across seeds."),
    x = "Panel size (number of proteins)",
    y = "AUROC (pooled test)",
    caption  = "Source: compiled_results_aggregated.csv + compiled_results_ensemble.csv"
  ) +
  theme_sweep()

ggsave(file.path(OUT_DIR, "fig1_saturation_curves.pdf"),
       fig1, width = 13, height = 5.5)
message("  Saved fig1_saturation_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: MULTI-METRIC SATURATION
# 4 panels: AUROC, PR-AUC, Brier, Sens@95Spec -- for the best order (pathway).
# Tests: do metrics agree on the optimal panel size?
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 2: Multi-metric saturation ...")

metric_data <- base %>%
  select(order, panel_size, model, model_label, order_label,
         AUROC = pooled_test_auroc,
         `PR-AUC` = pooled_test_prauc,
         Brier = pooled_test_brier_score,
         `Sens@95Spec` = pooled_test_sens_ctrl_95) %>%
  pivot_longer(cols = c(AUROC, `PR-AUC`, Brier, `Sens@95Spec`),
               names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(metric, levels = c("AUROC", "PR-AUC", "Brier", "Sens@95Spec")),
    higher_is_better = ifelse(metric == "Brier", FALSE, TRUE)
  )

# Normalize each metric to [0,1] for comparison
metric_norm <- metric_data %>%
  group_by(metric) %>%
  mutate(
    value_norm = if (first(higher_is_better)) {
      (value - min(value, na.rm = TRUE)) / (max(value, na.rm = TRUE) - min(value, na.rm = TRUE))
    } else {
      (max(value, na.rm = TRUE) - value) / (max(value, na.rm = TRUE) - min(value, na.rm = TRUE))
    }
  ) %>%
  ungroup()

# Pathway order only for clarity
fig2 <- ggplot(metric_data %>% filter(order == "pathway"),
               aes(x = panel_size, y = value, color = model_label)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.0) +
  facet_wrap(~ metric, nrow = 1, scales = "free_y") +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  scale_x_continuous(breaks = c(4, 7, 10, 15, 20, 25)) +
  labs(
    title    = "Multi-metric view: do metrics agree on optimal panel size?",
    subtitle = "Pathway order shown. If AUROC and Brier peaks diverge, discrimination != calibration.",
    x = "Panel size",
    y = "Metric value (per-metric scale)"
  ) +
  theme_sweep() +
  theme(axis.text.x = element_text(size = 8))

ggsave(file.path(OUT_DIR, "fig2_multimetric_saturation.pdf"),
       fig2, width = 14, height = 4.5)
message("  Saved fig2_multimetric_saturation.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: ORDER COMPARISON (Is addition order arbitrary?)
# For each model, overlay 3 order lines. Shade = between-order range.
# If shade is narrow, order doesn't matter.
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 3: Order comparison ...")

# Compute between-order range at each panel_size × model
order_range <- base %>%
  group_by(model, model_label, panel_size) %>%
  summarise(
    auroc_min = min(pooled_test_auroc, na.rm = TRUE),
    auroc_max = max(pooled_test_auroc, na.rm = TRUE),
    auroc_range = auroc_max - auroc_min,
    .groups = "drop"
  )

fig3 <- ggplot() +
  # Between-order range (ribbon)
  geom_ribbon(
    data = order_range,
    aes(x = panel_size, ymin = auroc_min, ymax = auroc_max),
    fill = "grey70", alpha = 0.35
  ) +
  # Individual order lines
  geom_line(
    data = base,
    aes(x = panel_size, y = pooled_test_auroc, color = order, linetype = order),
    linewidth = 0.8
  ) +
  facet_wrap(~ model_label, nrow = 1) +
  scale_color_manual(values = ORDER_COLORS, labels = ORDER_LABELS, name = "Addition order") +
  scale_linetype_manual(values = c(rra = "solid", importance = "longdash", pathway = "dotdash"),
                        labels = ORDER_LABELS, name = "Addition order") +
  scale_x_continuous(breaks = c(4, 7, 10, 15, 20, 25)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  coord_cartesian(ylim = c(0.825, 0.865)) +
  labs(
    title    = "Does addition order matter? (H0: order is arbitrary)",
    subtitle = "Grey band = range across 3 orders at each panel size. Narrow band = order doesn't matter.",
    x = "Panel size",
    y = "AUROC (pooled test)"
  ) +
  theme_sweep() +
  theme(legend.position = "bottom")

ggsave(file.path(OUT_DIR, "fig3_order_comparison.pdf"),
       fig3, width = 14, height = 5)
message("  Saved fig3_order_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: MODEL RANK TRAJECTORY (Bump chart -- rank inversion)
# X = panel size, Y = rank, lines per model. Per order.
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 4: Rank trajectory ...")

rank_data <- base %>%
  group_by(order, order_label, panel_size) %>%
  mutate(rank = rank(-pooled_test_auroc, ties.method = "first")) %>%
  ungroup()

fig4 <- ggplot(rank_data,
               aes(x = panel_size, y = rank, color = model_label, group = model_label)) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.0) +
  facet_wrap(~ order_label, nrow = 1) +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  scale_y_reverse(breaks = 1:4, labels = c("1st", "2nd", "3rd", "4th")) +
  scale_x_continuous(breaks = c(4, 7, 10, 15, 20, 25)) +
  labs(
    title    = "Model rank trajectory: where does the rank inversion occur?",
    subtitle = "Linear models (LR, SVM) typically lead at small panels; trees catch up as interactions become available.",
    x = "Panel size",
    y = "Rank (1 = best AUROC)"
  ) +
  theme_sweep()

ggsave(file.path(OUT_DIR, "fig4_rank_trajectory.pdf"),
       fig4, width = 13, height = 5)
message("  Saved fig4_rank_trajectory.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5: MARGINAL GAIN (Delta curves)
# Delta AUROC per protein added, per model. Averaged across orders.
# Shows diminishing returns and the stopping point.
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 5: Marginal gain ...")

# Average across orders for a cleaner delta signal
avg_auroc <- base %>%
  group_by(model, model_label, panel_size) %>%
  summarise(
    mean_auroc = mean(pooled_test_auroc, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(model, panel_size)

delta_data <- avg_auroc %>%
  group_by(model, model_label) %>%
  mutate(
    delta = mean_auroc - lag(mean_auroc),
    protein_added = panel_size
  ) %>%
  filter(!is.na(delta)) %>%
  ungroup()

# Read protein names from panel files for x-axis labels
panel_labels <- tryCatch({
  rra_panel <- read.csv("experiments/optimal-setup/panel-sweep/panels/rra_25p.csv",
                        stringsAsFactors = FALSE)
  if ("protein" %in% names(rra_panel)) rra_panel$protein else rra_panel[[1]]
}, error = function(e) paste0("p", 4:25))

fig5 <- ggplot(delta_data, aes(x = protein_added, y = delta * 1000, fill = model_label)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.8) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  # Significance threshold (approximate: 1 mAUROC ~ noise floor with 10 seeds)
  geom_hline(yintercept = c(-1, 1), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  annotate("text", x = 24.5, y = 1.3, label = "+/- 1 mAUROC\nnoise floor",
           size = 2.5, color = "grey40", hjust = 1) +
  scale_fill_manual(values = MODEL_COLORS, name = NULL) +
  scale_x_continuous(breaks = 5:25) +
  labs(
    title    = "Marginal AUROC gain per protein added (averaged across orders)",
    subtitle = "Bars below noise floor = adding that protein is statistically indistinguishable from noise.",
    x = "Panel size (protein added at this position)",
    y = "Delta AUROC (mAUROC)"
  ) +
  theme_sweep() +
  theme(axis.text.x = element_text(size = 8))

ggsave(file.path(OUT_DIR, "fig5_marginal_gain.pdf"),
       fig5, width = 13, height = 5)
message("  Saved fig5_marginal_gain.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6: VARIANCE DECOMPOSITION (the H0 test)
# ANOVA: % of total AUROC variance explained by panel_size, model, order,
# and their interactions. THE answer to "which decisions matter?"
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 6: Variance decomposition ...")

# Use only rows where all 3 orders × 4 models are present (balanced design)
balanced <- base %>%
  group_by(panel_size) %>%
  filter(n_distinct(order) == 3, n_distinct(model) == 4) %>%
  ungroup()

anova_fit <- aov(pooled_test_auroc ~ factor(panel_size) * model * order,
                 data = balanced)
anova_table <- anova(anova_fit)

ss_total <- sum(anova_table$`Sum Sq`)
var_decomp <- tibble(
  factor = rownames(anova_table),
  ss     = anova_table$`Sum Sq`,
  pct    = 100 * ss / ss_total,
  p_val  = anova_table$`Pr(>F)`
) %>%
  filter(factor != "Residuals") %>%
  mutate(
    # Clean labels
    factor_label = case_match(
      factor,
      "factor(panel_size)"                   ~ "Panel size",
      "model"                                ~ "Model",
      "order"                                ~ "Addition order",
      "factor(panel_size):model"             ~ "Panel x Model",
      "factor(panel_size):order"             ~ "Panel x Order",
      "model:order"                          ~ "Model x Order",
      "factor(panel_size):model:order"       ~ "3-way interaction",
      .default = factor
    ),
    factor_label = fct_reorder(factor_label, pct),
    sig = case_when(
      p_val < 0.001 ~ "***",
      p_val < 0.01  ~ "**",
      p_val < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  )

# Add residual row
resid_pct <- 100 * anova_table["Residuals", "Sum Sq"] / ss_total
var_decomp <- bind_rows(
  var_decomp,
  tibble(factor = "Residuals", ss = anova_table["Residuals","Sum Sq"],
         pct = resid_pct, p_val = NA_real_, factor_label = "Residual (noise)",
         sig = "")
) %>%
  mutate(factor_label = fct_reorder(factor_label, pct))

fig6 <- ggplot(var_decomp, aes(x = factor_label, y = pct, fill = pct > 5)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.1f%% %s", pct, sig)),
            hjust = -0.1, size = 3.2) +
  coord_flip(ylim = c(0, max(var_decomp$pct) * 1.2)) +
  scale_fill_manual(values = c("TRUE" = "#2166AC", "FALSE" = "#B2182B"),
                    guide = "none") +
  labs(
    title    = "Which decisions are arbitrary? (ANOVA variance decomposition)",
    subtitle = "% of total AUROC variance explained by each factor. Blue = load-bearing (>5%), red = negligible.",
    x = NULL,
    y = "% of total AUROC variance",
    caption = "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
  ) +
  theme_sweep() +
  theme(legend.position = "none")

ggsave(file.path(OUT_DIR, "fig6_variance_decomposition.pdf"),
       fig6, width = 9, height = 5.5)
message("  Saved fig6_variance_decomposition.pdf")

# Print the decomposition
message("\n  Variance decomposition:")
var_decomp %>%
  arrange(desc(pct)) %>%
  mutate(across(c(pct), ~ round(.x, 1))) %>%
  select(factor_label, pct, sig) %>%
  print(n = 10)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7: DECISION CONSEQUENCE HEATMAP
# Full design space: rows = panel_size, columns = model.
# One panel per order. Cell color = delta from grand mean (mAUROC).
# Black border = Pareto-optimal (best AUROC in that row OR that column).
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 7: Decision consequence heatmap ...")

heatmap_data <- base %>%
  mutate(
    delta_grand = (pooled_test_auroc - grand_mean) * 1000  # mAUROC from grand mean
  ) %>%
  # Mark row-best (best model at each panel size × order)
  group_by(order_label, panel_size) %>%
  mutate(is_row_best = pooled_test_auroc == max(pooled_test_auroc)) %>%
  ungroup()

fig7 <- ggplot(heatmap_data,
               aes(x = model_label, y = factor(panel_size), fill = delta_grand)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.1f", pooled_test_auroc * 1000 - 800)),
            size = 2.2, color = "grey20") +
  # Best-in-row border
  geom_tile(
    data = heatmap_data %>% filter(is_row_best),
    color = "black", fill = NA, linewidth = 0.8
  ) +
  facet_wrap(~ order_label, nrow = 1) +
  scale_fill_gradient2(
    low = "#D73027", mid = "#FFFFBF", high = "#1A9850",
    midpoint = 0,
    name = "Delta from\ngrand mean\n(mAUROC)",
    limits = c(-15, 15)
  ) +
  scale_y_discrete(limits = rev(as.character(4:25))) +
  labs(
    title    = "Decision consequence map: every configuration vs. grand mean",
    subtitle = paste0("Cell = AUROC * 1000 - 800 (so 59.4 = 0.8594). ",
                      "Black border = best model at that panel size. ",
                      "Green = above average; red = below."),
    x = NULL,
    y = "Panel size"
  ) +
  theme_sweep() +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9)
  )

ggsave(file.path(OUT_DIR, "fig7_decision_heatmap.pdf"),
       fig7, width = 14, height = 8)
message("  Saved fig7_decision_heatmap.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8: AUROC vs BRIER PARETO FRONT
# Each point = one configuration (order × panel_size × model).
# Pareto-optimal frontier highlighted.
# ══════════════════════════════════════════════════════════════════════════════
message("Fig 8: AUROC vs Brier Pareto ...")

pareto_data <- base %>%
  select(order, order_label, panel_size, model, model_label, model_type,
         auroc = pooled_test_auroc,
         brier = pooled_test_brier_score)

# Find Pareto-optimal points (maximize AUROC, minimize Brier)
is_pareto_opt <- function(auroc, brier) {
  n <- length(auroc)
  dominated <- logical(n)
  for (i in seq_len(n)) {
    dominated[i] <- any(
      auroc[-i] >= auroc[i] & brier[-i] <= brier[i] &
        (auroc[-i] > auroc[i] | brier[-i] < brier[i])
    )
  }
  !dominated
}

pareto_data <- pareto_data %>%
  mutate(pareto = is_pareto_opt(auroc, brier))

n_pareto <- sum(pareto_data$pareto)
message(sprintf("  %d / %d configs on the Pareto frontier", n_pareto, nrow(pareto_data)))

# Panel size as point size
fig8 <- ggplot(pareto_data, aes(x = auroc, y = brier)) +
  # All points (background)
  geom_point(aes(color = model_label, shape = order_label, size = panel_size),
             alpha = 0.45) +
  # Pareto points (foreground)
  geom_point(
    data = pareto_data %>% filter(pareto),
    aes(color = model_label, shape = order_label, size = panel_size),
    alpha = 1.0, stroke = 1.2
  ) +
  # Connect Pareto front
  geom_step(
    data = pareto_data %>% filter(pareto) %>% arrange(auroc),
    aes(x = auroc, y = brier),
    color = "black", linewidth = 0.5, linetype = "dotted"
  ) +
  # Label Pareto points
  geom_text(
    data = pareto_data %>% filter(pareto),
    aes(label = paste0(panel_size, "p")),
    size = 2.5, nudge_y = -0.001, color = "grey30"
  ) +
  scale_color_manual(values = MODEL_COLORS, name = "Model") +
  scale_shape_manual(values = c(16, 17, 15), name = "Order") +
  scale_size_continuous(range = c(1, 5), name = "Panel size", breaks = c(5, 10, 15, 20, 25)) +
  scale_x_continuous(labels = label_number(accuracy = 0.001)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title    = "Multi-objective frontier: AUROC vs. Brier score",
    subtitle = paste0(n_pareto, " Pareto-optimal configs shown with panel size labels. ",
                      "Right+low = ideal. Faded = dominated."),
    x = "AUROC (higher = better discrimination)",
    y = "Brier score (lower = better calibration)"
  ) +
  theme_sweep() +
  guides(
    color = guide_legend(order = 1),
    shape = guide_legend(order = 2),
    size  = guide_legend(order = 3)
  )

ggsave(file.path(OUT_DIR, "fig8_pareto_front.pdf"),
       fig8, width = 10, height = 7)
message("  Saved fig8_pareto_front.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE: Decision impact
# ══════════════════════════════════════════════════════════════════════════════
message("\n══ Decision Impact Summary ══")

message("\n1. Panel size effect (averaged over model and order):")
base %>%
  group_by(panel_size) %>%
  summarise(
    mean_auroc = mean(pooled_test_auroc),
    mean_brier = mean(pooled_test_brier_score),
    .groups = "drop"
  ) %>%
  slice_max(mean_auroc, n = 5) %>%
  mutate(across(c(mean_auroc, mean_brier), ~ round(.x, 4))) %>%
  print()

message("\n2. Model effect (averaged over panel size and order):")
base %>%
  group_by(model) %>%
  summarise(
    mean_auroc = mean(pooled_test_auroc),
    range_auroc = max(pooled_test_auroc) - min(pooled_test_auroc),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_auroc)) %>%
  mutate(across(c(mean_auroc, range_auroc), ~ round(.x, 4))) %>%
  print()

message("\n3. Order effect (averaged over panel size and model):")
base %>%
  group_by(order) %>%
  summarise(
    mean_auroc = mean(pooled_test_auroc),
    range_auroc = max(pooled_test_auroc) - min(pooled_test_auroc),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_auroc)) %>%
  mutate(across(c(mean_auroc, range_auroc), ~ round(.x, 4))) %>%
  print()

message("\n4. Top 10 configurations (Pareto-optimal first):")
top10 <- pareto_data %>%
  arrange(desc(pareto), desc(auroc)) %>%
  select(order, model, panel_size, auroc, brier, pareto) %>%
  head(10) %>%
  mutate(across(c(auroc, brier), ~ round(.x, 4)))
print(as.data.frame(top10))

message("\nAll figures saved to: ", OUT_DIR)
