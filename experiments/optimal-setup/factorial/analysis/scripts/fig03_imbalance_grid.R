#!/usr/bin/env Rscript
# fig03_imbalance_grid.R -- V3: Joint weighting x downsampling utility heatmap + CI forest
#
# Panels: A) Normalized utility heatmap (3x3), B) Bootstrap CI forest (9 combos)
# Locked to best recipe x model from V1/V2 validation CSVs, or discovered from data.

source(file.path("..", "_theme_factorial.R"))
library(patchwork)

set.seed(42)

# ── Load data ────────────────────────────────────────────────────────────────
df <- load_factorial()

# ── Resolve locked recipe + model from validation CSVs ─────────────────────
v1_files <- Sys.glob(file.path(RESULTS_DIR, "*v1_summary*"))
v2_files <- Sys.glob(file.path(RESULTS_DIR, "*v2_pareto*"))

# Also check the factorial tables directory
v1_files <- c(v1_files, Sys.glob(file.path(FACTORIAL_TBL_DIR, "*v1*summary*")))
v2_files <- c(v2_files, Sys.glob(file.path(FACTORIAL_TBL_DIR, "*v2*pareto*")))

locked_recipe <- NULL
locked_model  <- NULL

if (length(v1_files) > 0) {
  v1 <- tryCatch(read.csv(v1_files[1], stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(v1)) {
    # Pick the recipe with highest mean AUROC across models
    if ("recipe_id" %in% names(v1) && "auroc_mean" %in% names(v1)) {
      best_recipe <- v1 %>%
        group_by(recipe_id) %>%
        summarise(grand_auroc = mean(auroc_mean, na.rm = TRUE), .groups = "drop") %>%
        slice_max(grand_auroc, n = 1)
      locked_recipe <- as.character(best_recipe$recipe_id[1])
    }
  }
}

if (length(v2_files) > 0) {
  v2 <- tryCatch(read.csv(v2_files[1], stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(v2)) {
    # Pick the non-dominated model with highest AUROC (or best Pareto model)
    if ("factorial_model" %in% names(v2) && "auroc_mean" %in% names(v2)) {
      pareto_col <- intersect(c("pareto_label", "dominated"), names(v2))
      if (length(pareto_col) > 0 && pareto_col[1] == "pareto_label") {
        v2_front <- v2 %>% filter(pareto_label == "Non-dominated")
      } else if (length(pareto_col) > 0 && pareto_col[1] == "dominated") {
        v2_front <- v2 %>% filter(!dominated)
      } else {
        v2_front <- v2
      }
      if (nrow(v2_front) > 0) {
        locked_model <- as.character(v2_front$factorial_model[v2_front$auroc_mean == max(v2_front$auroc_mean)][1])
      }
    }
  }
}

# Fallback: discover from raw data
if (is.null(locked_recipe)) {
  recipe_rank <- df %>%
    group_by(recipe_id) %>%
    summarise(auroc = mean(summary_auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    slice_max(auroc, n = 1)
  locked_recipe <- as.character(recipe_rank$recipe_id[1])
  message(sprintf("No V1 CSV found; using best recipe from data: %s", locked_recipe))
}

if (is.null(locked_model)) {
  model_rank <- df %>%
    filter(recipe_id == locked_recipe) %>%
    group_by(factorial_model) %>%
    summarise(auroc = mean(summary_auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    slice_max(auroc, n = 1)
  locked_model <- as.character(model_rank$factorial_model[1])
  message(sprintf("No V2 CSV found; using best model from data: %s", locked_model))
}

message(sprintf("Locked: recipe=%s, model=%s", locked_recipe, locked_model))

# ── Filter to locked recipe x model ────────────────────────────────────────
df_filtered <- df %>%
  filter(recipe_id == locked_recipe, factorial_model == locked_model)

if (nrow(df_filtered) == 0) {
  stop(sprintf("No data for recipe=%s, model=%s. Check factor levels.", locked_recipe, locked_model))
}

# ── Discover factor levels from data ───────────────────────────────────────
wt_levels <- sort(unique(as.character(df_filtered$factorial_weighting)))
ds_levels <- sort(unique(as.numeric(as.character(df_filtered$factorial_downsampling))))

# ── Panel A: Normalized utility heatmap ────────────────────────────────────
imb <- df_filtered %>%
  group_by(factorial_weighting, factorial_downsampling) %>%
  summarise(
    auprc_mean       = mean(summary_prauc_mean, na.rm = TRUE),
    reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
    .groups = "drop"
  )

# Normalize within group
auprc_range <- max(imb$auprc_mean) - min(imb$auprc_mean)
rel_range   <- max(imb$reliability_mean) - min(imb$reliability_mean)

imb <- imb %>%
  mutate(
    auprc_norm = (auprc_mean - min(auprc_mean)) / max(1e-10, auprc_range),
    rel_norm   = (max(reliability_mean) - reliability_mean) / max(1e-10, rel_range),
    utility    = 0.5 * auprc_norm + 0.5 * rel_norm
  )

best_idx       <- which.max(imb$utility)
imb$is_best    <- seq_len(nrow(imb)) == best_idx
imb$is_parsimony <- as.character(imb$factorial_weighting) == "none" &
  as.numeric(as.character(imb$factorial_downsampling)) == max(ds_levels)

# ── Panel A: Bootstrap CIs for utility (reused in Panel B) ─────────────────
# Compute cell-level raw metrics for bootstrap
cell_data <- df_filtered %>%
  mutate(
    wt_ds = paste0(factorial_weighting, " / ", factorial_downsampling)
  )

combos <- unique(cell_data$wt_ds)
n_boot <- 1000

# Pre-compute global min/max for normalization stability
global_stats <- cell_data %>%
  group_by(factorial_weighting, factorial_downsampling) %>%
  summarise(
    auprc_mean = mean(summary_prauc_mean, na.rm = TRUE),
    rel_mean   = mean(summary_brier_reliability_mean, na.rm = TRUE),
    .groups = "drop"
  )

boot_results <- lapply(seq_len(n_boot), function(b) {
  # Resample within each cell
  boot_sample <- cell_data %>%
    group_by(factorial_weighting, factorial_downsampling) %>%
    slice_sample(prop = 1, replace = TRUE) %>%
    summarise(
      auprc_mean = mean(summary_prauc_mean, na.rm = TRUE),
      rel_mean   = mean(summary_brier_reliability_mean, na.rm = TRUE),
      .groups = "drop"
    )

  # Normalize within this bootstrap replicate
  auprc_rng <- max(boot_sample$auprc_mean) - min(boot_sample$auprc_mean)
  rel_rng   <- max(boot_sample$rel_mean) - min(boot_sample$rel_mean)

  boot_sample %>%
    mutate(
      auprc_norm = (auprc_mean - min(auprc_mean)) / max(1e-10, auprc_rng),
      rel_norm   = (max(rel_mean) - rel_mean) / max(1e-10, rel_rng),
      utility    = 0.5 * auprc_norm + 0.5 * rel_norm,
      boot_id    = b
    )
})

boot_df <- bind_rows(boot_results)

# Summarise bootstrap CIs per cell
boot_ci <- boot_df %>%
  mutate(combo = paste0(factorial_weighting, " / ", factorial_downsampling)) %>%
  group_by(combo, factorial_weighting, factorial_downsampling) %>%
  summarise(
    utility_mean = mean(utility),
    ci_lo        = quantile(utility, 0.025),
    ci_hi        = quantile(utility, 0.975),
    .groups = "drop"
  )

# Merge point estimates into boot_ci for consistency
boot_ci <- boot_ci %>%
  left_join(
    imb %>%
      mutate(combo = paste0(factorial_weighting, " / ", factorial_downsampling)) %>%
      select(combo, utility, is_best, is_parsimony),
    by = "combo"
  )

# ── Build Panel A ──────────────────────────────────────────────────────────
pa <- ggplot(imb, aes(x = factor(factorial_downsampling), y = factorial_weighting, fill = utility)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_tile(data = imb[imb$is_best, ], color = "black", linewidth = 1.5, fill = NA) +
  geom_text(aes(label = sprintf("%.3f", utility)), size = 4, fontface = "bold") +
  geom_text(data = imb[imb$is_parsimony, ], aes(label = "\u2217"),
            vjust = -0.5, size = 6, color = "black") +
  scale_fill_viridis_c(option = "mako", name = "Utility") +
  labs(
    title = "A. Normalized Utility",
    x = "Downsampling ratio",
    y = "Weighting"
  ) +
  theme_cel() +
  theme(legend.position = "right")

# ── Build Panel B: CI forest plot ──────────────────────────────────────────
# Identify parsimony utility for reference line
parsimony_row <- boot_ci %>%
  filter(is_parsimony)
parsimony_utility <- if (nrow(parsimony_row) > 0) parsimony_row$utility[1] else NA_real_

pb <- ggplot(boot_ci, aes(x = utility, y = reorder(combo, utility))) +
  geom_pointrange(aes(xmin = ci_lo, xmax = ci_hi), size = 0.8, linewidth = 0.6) +
  {
    if (!is.na(parsimony_utility)) {
      geom_vline(xintercept = parsimony_utility, linetype = "dashed", color = "grey50")
    }
  } +
  geom_point(data = boot_ci[boot_ci$is_best, ],
             aes(x = utility, y = reorder(combo, utility)),
             color = "black", shape = 18, size = 4) +
  labs(
    title = "B. Utility Bootstrap CIs",
    x = "Normalized utility",
    y = NULL
  ) +
  theme_cel()

# ── Compose ──────────────────────────────────────────────────────────────────
# Determine parsimony default label from data
parsimony_wt <- "none"
parsimony_ds <- max(ds_levels)

p <- (pa | pb) +
  plot_layout(widths = c(1, 1.2)) +
  plot_annotation(
    title = "V3: Imbalance Handling \u2014 Joint Weighting \u00d7 Downsampling",
    subtitle = sprintf(
      "Locked: recipe=%s, model=%s | * = parsimony default (%s, %s)",
      locked_recipe, locked_model, parsimony_wt, parsimony_ds
    ),
    caption = "Utility = 0.5 \u00d7 AUPRC_norm + 0.5 \u00d7 Reliability_norm. Black border = best. Dashed line = parsimony default. CIs from 1000 bootstrap replicates."
  )

factorial_save_fig(p, "fig03_imbalance_grid", width = 14, height = 6)

# ── Save summary tables ────────────────────────────────────────────────────
factorial_save_table(
  imb %>% select(factorial_weighting, factorial_downsampling,
                 auprc_mean, reliability_mean, auprc_norm, rel_norm, utility, is_best, is_parsimony),
  "fig03_imbalance_utility"
)
factorial_save_table(
  boot_ci %>% select(combo, utility, ci_lo, ci_hi, is_best, is_parsimony),
  "fig03_imbalance_bootstrap_ci"
)

# ── Agent-friendly summary ─────────────────────────────────────────────────
best_combo <- boot_ci %>% filter(is_best)
message(sprintf(
  "fig03_imbalance_grid: %d weighting x %d downsampling combos. Best utility=%.3f (%s). Parsimony default utility=%.3f.",
  length(wt_levels), length(ds_levels),
  best_combo$utility[1], best_combo$combo[1],
  if (!is.na(parsimony_utility)) parsimony_utility else NA_real_
))
