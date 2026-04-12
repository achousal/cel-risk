#!/usr/bin/env Rscript
# fig01_recipe_screen.R -- Recipe landscape: 3-panel heatmap triptych
#
# Panels: A) Mean AUROC, B) Mean AUPRC, C) Mean Reliability
# All by recipe x model, averaged over calibration x weighting x downsampling.

source(file.path("..", "_theme_factorial.R"))
library(patchwork)

# ── Load data ────────────────────────────────────────────────────────────────
df <- load_factorial()

# ── Aggregate: recipe x model summaries ──────────────────────────────────────
recipe_model_summary <- df %>%
  group_by(recipe_id, factorial_model) %>%
  summarise(
    auroc_mean       = mean(summary_auroc_mean, na.rm = TRUE),
    prauc_mean       = mean(summary_prauc_mean, na.rm = TRUE),
    reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
    n_cells          = n(),
    .groups = "drop"
  )

# ── Row ordering: recipes sorted by mean AUROC descending ────────────────────
recipe_order <- recipe_model_summary %>%
  group_by(recipe_id) %>%
  summarise(grand_auroc = mean(auroc_mean), .groups = "drop") %>%
  arrange(grand_auroc) %>%
  pull(recipe_id)

recipe_model_summary <- recipe_model_summary %>%
  mutate(recipe_id = factor(recipe_id, levels = recipe_order))

# ── Detect shared (R*) vs model-specific (MS_*) boundary ────────────────────
is_shared <- grepl("^R[0-9]", levels(recipe_model_summary$recipe_id))
shared_indices <- which(is_shared)
ms_indices     <- which(!is_shared)

# hline position: between the last shared and first MS (or vice versa)
# In factor-level space, positions are integers; hline goes at 0.5 boundary
boundary_y <- NULL
if (length(shared_indices) > 0 && length(ms_indices) > 0) {
  # Find the boundary in ordered factor levels
  all_levels <- levels(recipe_model_summary$recipe_id)
  is_shared_ordered <- grepl("^R[0-9]", all_levels)
  # Walk levels and find transitions
  for (i in seq_len(length(all_levels) - 1)) {
    if (is_shared_ordered[i] != is_shared_ordered[i + 1]) {
      boundary_y <- i + 0.5
      break
    }
  }
}

# ── Helper: build a heatmap panel ────────────────────────────────────────────
make_heatmap <- function(data, fill_col, fill_name, panel_label,
                         viridis_option = "inferno", viridis_direction = -1) {
  p <- ggplot(data, aes(x = factorial_model, y = recipe_id)) +
    geom_tile(aes(fill = .data[[fill_col]]),
              color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", .data[[fill_col]])),
              size = 2.5, color = "white") +
    scale_fill_viridis_c(option = viridis_option, direction = viridis_direction,
                         name = fill_name) +
    labs(title = panel_label, x = NULL, y = NULL) +
    theme_cel() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "right"
    )

  # Add boundary line if detected
  if (!is.null(boundary_y)) {
    p <- p + geom_hline(yintercept = boundary_y, linetype = "dashed",
                        color = "grey30", linewidth = 0.6)
  }

  p
}

# ── Build panels ─────────────────────────────────────────────────────────────
pa <- make_heatmap(recipe_model_summary, "auroc_mean", "AUROC",
                   "A. Mean AUROC by Recipe x Model")

pb <- make_heatmap(recipe_model_summary, "prauc_mean", "AUPRC",
                   "B. Mean AUPRC by Recipe x Model")

pc <- make_heatmap(recipe_model_summary, "reliability_mean", "Reliability",
                   "C. Mean Reliability by Recipe x Model",
                   viridis_direction = 1)  # lower is better

# ── Compose ──────────────────────────────────────────────────────────────────
p <- (pa | pb | pc) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Factorial Landscape: Recipe x Model Performance",
    subtitle = "Averaged over calibration x weighting x downsampling cells",
    caption = "Rows ordered by mean AUROC descending. Horizontal line separates shared-panel (R*) from model-specific (MS_*) recipes."
  )

factorial_save_fig(p, "fig01_recipe_screen", width = 16, height = 7)

# ── Save summary table ──────────────────────────────────────────────────────
factorial_save_table(recipe_model_summary, "fig01_recipe_model_summary")

# ── Agent-friendly summary ───────────────────────────────────────────────────
n_recipes <- nlevels(recipe_model_summary$recipe_id)
n_models  <- n_distinct(recipe_model_summary$factorial_model)
top_recipe <- tail(recipe_order, 1)

message(sprintf(
  "fig01_recipe_screen: Plotted %d recipes x %d models across 3 metrics (AUROC, AUPRC, Reliability). Top recipe by AUROC: %s. Boundary line: %s.",
  n_recipes, n_models, top_recipe,
  if (!is.null(boundary_y)) sprintf("y = %.1f", boundary_y) else "not detected"
))
