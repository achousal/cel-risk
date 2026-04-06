#!/usr/bin/env Rscript
# 05_seed_distributions.R
#
# Seed-level AUROC distributions: violin + jitter for each panel size,
# with full model reference distribution.
#
# Usage:
#   Rscript experiments/optimal-setup/claude/viz/05_seed_distributions.R
#
# Outputs: optimal-setup/claude/viz/out/fig05_seed_distributions.pdf

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

MODEL_COLORS <- c(
  LR_EN = "#4878CF",
  RF    = "#D65F5F"
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

# ── Load per-seed AUROC from selection JSON ───────────────────────────────────
load_seed_data <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "optimal_panel_selection.json")
  sel <- fromJSON(path)

  # Full model seed distribution
  full_seeds <- data.frame(
    model = model,
    size  = "Full",
    auroc = sel$full_auroc_by_seed,
    stringsAsFactors = FALSE
  )

  # Per-panel-size distributions from decisions
  d <- sel$decisions
  panel_info <- data.frame(
    model      = model,
    size       = as.character(d$size),
    mean_auroc = d$stability$mean_auroc,
    std_auroc  = d$stability$std_auroc,
    stringsAsFactors = FALSE
  )

  list(full_seeds = full_seeds, panel_info = panel_info)
}

all_seed_data <- lapply(MODELS, load_seed_data)
names(all_seed_data) <- MODELS

# ── Fig 5a: Full model seed distributions (violin + jitter) ──────────────────
message("Fig 5a: Full model seed distributions …")

full_seeds <- bind_rows(lapply(all_seed_data, `[[`, "full_seeds")) %>%
  mutate(model_label = MODEL_LABELS[model])

p5a <- ggplot(full_seeds, aes(x = model_label, y = auroc, fill = model)) +
  geom_violin(alpha = 0.3, color = NA) +
  geom_jitter(aes(color = model), width = 0.15, size = 1.5, alpha = 0.7) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.5) +
  scale_fill_manual(values = MODEL_COLORS, guide = "none") +
  scale_color_manual(values = MODEL_COLORS, guide = "none") +
  labs(
    title    = "Full Model AUROC Distribution Across Seeds",
    subtitle = "30 seeds per model. Box = IQR, whiskers = 1.5x IQR.",
    x = "", y = "AUROC (validation)"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig05a_full_seed_dist.pdf"), p5a,
       width = 5, height = 5)

# ── Fig 5b: Mean +/- SD across panel sizes (forest-plot style) ───────────────
message("Fig 5b: Panel size forest plot …")

panel_info <- bind_rows(lapply(all_seed_data, `[[`, "panel_info")) %>%
  mutate(
    model_label = MODEL_LABELS[model],
    size_num    = as.numeric(size)
  )

# Add full model stats
full_stats <- full_seeds %>%
  group_by(model, model_label) %>%
  summarize(
    mean_auroc = mean(auroc),
    std_auroc  = sd(auroc),
    .groups = "drop"
  ) %>%
  mutate(size = "Full", size_num = max(panel_info$size_num) + 30)

forest_data <- bind_rows(
  panel_info %>% select(model, model_label, size, size_num, mean_auroc, std_auroc),
  full_stats
) %>%
  mutate(
    lo = mean_auroc - std_auroc,
    hi = mean_auroc + std_auroc
  )

# Dodge for overlapping models
p5b <- ggplot(forest_data,
              aes(x = reorder(size, size_num), y = mean_auroc, color = model)) +
  geom_pointrange(
    aes(ymin = lo, ymax = hi),
    position = position_dodge(width = 0.5),
    size = 0.4, linewidth = 0.5
  ) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = "Model") +
  labs(
    title    = "AUROC Distribution Summary by Panel Size",
    subtitle = "Mean +/- 1 SD across 30 seeds. 'Full' = all features.",
    x = "Panel size",
    y = "AUROC (validation)"
  ) +
  theme_cel() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(OUT_DIR, "fig05b_panel_forest.pdf"), p5b,
       width = 9, height = 5.5)

# ── Fig 5c: Stability tradeoff scatter ────────────────────────────────────────
message("Fig 5c: Stability vs noninferiority scatter …")

load_tradeoff <- function(model) {
  path <- file.path(RUN_DIR, model, "aggregated", "optimize_panel",
                    "optimal_panel_selection.json")
  sel <- fromJSON(path)
  d <- sel$decisions

  ni <- d$noninferiority
  d %>%
    mutate(
      model  = model,
      cv     = stability$cv,
      delta  = if (!is.null(ni)) ni$delta_estimate else NA_real_,
      stable = stability$stable,
      status = case_when(
        accepted         ~ "Accepted",
        !stable          ~ "Unstable",
        TRUE             ~ "Non-inferior fail"
      )
    ) %>%
    select(model, size, cv, delta, status, accepted)
}

tradeoff <- bind_rows(lapply(MODELS, load_tradeoff)) %>%
  mutate(
    model_label = MODEL_LABELS[model],
    status = factor(status, levels = c("Accepted", "Non-inferior fail", "Unstable"))
  )

p5c <- ggplot(tradeoff, aes(x = cv, y = delta)) +
  # Quadrant lines

  geom_vline(xintercept = 0.05, linetype = "dashed", color = "grey60") +
  geom_hline(yintercept = 0.02, linetype = "dashed", color = "grey60") +
  # Points

  geom_point(aes(color = model, shape = status), size = 3, alpha = 0.8) +
  # Labels
  geom_text(aes(label = size), size = 2.5, vjust = -1, color = "grey30") +
  # Quadrant annotations
  annotate("text", x = 0.032, y = 0.008, label = "Accept\nzone",
           color = "#66c2a5", size = 3, fontface = "bold") +
  annotate("text", x = 0.057, y = 0.008, label = "Unstable",
           color = "#fc8d62", size = 3, fontface = "italic") +
  annotate("text", x = 0.032, y = 0.038, label = "Too much\nAUROC loss",
           color = "#fc8d62", size = 3, fontface = "italic") +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS, name = "Model") +
  scale_shape_manual(
    values = c("Accepted" = 16, "Non-inferior fail" = 17, "Unstable" = 4),
    name = "Decision"
  ) +
  labs(
    title    = "Stability vs Performance Tradeoff",
    subtitle = "Each point = one panel size. Dashed = acceptance thresholds.",
    x = "CV of AUROC across seeds (stability)",
    y = "Delta (AUROC drop from full model)"
  ) +
  theme_cel()

ggsave(file.path(OUT_DIR, "fig05c_stability_tradeoff.pdf"), p5c,
       width = 8, height = 6)

message("Done: fig05a, fig05b, fig05c -> ", OUT_DIR)
