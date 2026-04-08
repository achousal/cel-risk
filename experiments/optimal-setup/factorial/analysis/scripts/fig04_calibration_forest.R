#!/usr/bin/env Rscript
# fig04_calibration_forest.R -- V4: Calibration method comparison as paired forest plots
#
# Panels: A) Brier Reliability forest, B) AUROC forest
# Locked to best recipe x model x imbalance from V1-V3 validation CSVs, or discovered from data.

source(file.path("..", "_theme_factorial.R"))
library(patchwork)

# ── Load data ────────────────────────────────────────────────────────────────
df <- load_factorial()

# ── Resolve locked recipe + model + imbalance from validation CSVs ─────────
v1_files <- Sys.glob(file.path(RESULTS_DIR, "*v1_summary*"))
v2_files <- Sys.glob(file.path(RESULTS_DIR, "*v2_pareto*"))
v3_files <- Sys.glob(file.path(RESULTS_DIR, "*v3_imbalance*"))

# Also check factorial tables directory
v1_files <- c(v1_files, Sys.glob(file.path(FACTORIAL_TBL_DIR, "*v1*summary*")))
v2_files <- c(v2_files, Sys.glob(file.path(FACTORIAL_TBL_DIR, "*v2*pareto*")))
v3_files <- c(v3_files, Sys.glob(file.path(FACTORIAL_TBL_DIR, "*v3*imbalance*")))

locked_recipe <- NULL
locked_model  <- NULL
locked_wt     <- NULL
locked_ds     <- NULL

# Try V1: locked recipe
if (length(v1_files) > 0) {
  v1 <- tryCatch(read.csv(v1_files[1], stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(v1) && "recipe_id" %in% names(v1) && "auroc_mean" %in% names(v1)) {
    best_recipe <- v1 %>%
      group_by(recipe_id) %>%
      summarise(grand_auroc = mean(auroc_mean, na.rm = TRUE), .groups = "drop") %>%
      slice_max(grand_auroc, n = 1)
    locked_recipe <- as.character(best_recipe$recipe_id[1])
  }
}

# Try V2: locked model
if (length(v2_files) > 0) {
  v2 <- tryCatch(read.csv(v2_files[1], stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(v2) && "factorial_model" %in% names(v2) && "auroc_mean" %in% names(v2)) {
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

# Try V3: locked weighting + downsampling
if (length(v3_files) > 0) {
  v3 <- tryCatch(read.csv(v3_files[1], stringsAsFactors = FALSE), error = function(e) NULL)
  if (!is.null(v3)) {
    # Look for the best or flagged row
    best_col <- intersect(c("is_best", "utility"), names(v3))
    if ("is_best" %in% names(v3)) {
      best_row <- v3 %>% filter(is_best == TRUE)
    } else if ("utility" %in% names(v3)) {
      best_row <- v3 %>% slice_max(utility, n = 1)
    } else {
      best_row <- data.frame()
    }
    if (nrow(best_row) > 0) {
      if ("factorial_weighting" %in% names(best_row))
        locked_wt <- as.character(best_row$factorial_weighting[1])
      if ("factorial_downsampling" %in% names(best_row))
        locked_ds <- as.character(best_row$factorial_downsampling[1])
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

if (is.null(locked_wt)) {
  # Use parsimony default: "none"
  wt_levels <- unique(as.character(df$factorial_weighting))
  locked_wt <- if ("none" %in% wt_levels) "none" else wt_levels[1]
  message(sprintf("No V3 CSV found; using default weighting: %s", locked_wt))
}

if (is.null(locked_ds)) {
  # Use parsimony default: largest downsampling ratio (least downsampling)
  ds_levels <- sort(unique(as.numeric(as.character(df$factorial_downsampling))), decreasing = TRUE)
  locked_ds <- as.character(ds_levels[1])
  message(sprintf("No V3 CSV found; using default downsampling: %s", locked_ds))
}

message(sprintf("Locked: recipe=%s, model=%s, wt=%s, ds=%s",
                locked_recipe, locked_model, locked_wt, locked_ds))

# ── Filter to locked dimensions ────────────────────────────────────────────
df_filtered <- df %>%
  filter(
    recipe_id             == locked_recipe,
    factorial_model       == locked_model,
    as.character(factorial_weighting)    == locked_wt,
    as.character(factorial_downsampling) == locked_ds
  )

if (nrow(df_filtered) == 0) {
  stop(sprintf(
    "No data for recipe=%s, model=%s, wt=%s, ds=%s. Check factor levels.",
    locked_recipe, locked_model, locked_wt, locked_ds
  ))
}

# ── Aggregate per calibration method ───────────────────────────────────────
cal_levels <- sort(unique(as.character(df_filtered$factorial_calibration)))
n_per_cal  <- df_filtered %>%
  group_by(factorial_calibration) %>%
  summarise(n = n(), .groups = "drop")

cal <- df_filtered %>%
  group_by(factorial_calibration) %>%
  summarise(
    reliability_mean = mean(summary_brier_reliability_mean, na.rm = TRUE),
    reliability_se   = sd(summary_brier_reliability_mean, na.rm = TRUE) / sqrt(n()),
    auroc_mean       = mean(summary_auroc_mean, na.rm = TRUE),
    auroc_se         = sd(summary_auroc_mean, na.rm = TRUE) / sqrt(n()),
    n_cells          = n(),
    .groups = "drop"
  ) %>%
  mutate(
    rel_ci_lo   = reliability_mean - 1.96 * reliability_se,
    rel_ci_hi   = reliability_mean + 1.96 * reliability_se,
    auroc_ci_lo = auroc_mean - 1.96 * auroc_se,
    auroc_ci_hi = auroc_mean + 1.96 * auroc_se
  )

# Assign complexity for parsimony ordering (discover from data)
# Known complexity hierarchy: logistic_intercept < beta < isotonic
known_complexity <- c(logistic_intercept = 1, logistic = 1, beta = 2, isotonic = 3)
cal <- cal %>%
  mutate(
    complexity = ifelse(
      as.character(factorial_calibration) %in% names(known_complexity),
      known_complexity[as.character(factorial_calibration)],
      # Unknown methods: assign by alphabetical order after known ones
      max(known_complexity, na.rm = TRUE) + as.integer(factor(factorial_calibration))
    )
  )

# Order by parsimony: simplest on top (lowest complexity = top of y-axis)
cal <- cal %>%
  mutate(factorial_calibration = reorder(factorial_calibration, complexity))

# ── Calibration colors: use palette from theme, extend dynamically ─────────
cal_color_map <- CALIBRATION_COLORS
# Add any calibration levels not in the predefined palette
missing_levels <- setdiff(as.character(cal$factorial_calibration), names(cal_color_map))
if (length(missing_levels) > 0) {
  extra_colors <- scales::hue_pal()(length(missing_levels))
  names(extra_colors) <- missing_levels
  cal_color_map <- c(cal_color_map, extra_colors)
}

# ── Panel A: Reliability forest ────────────────────────────────────────────
pa <- ggplot(cal, aes(x = reliability_mean,
                       y = reorder(factorial_calibration, -complexity),
                       color = factorial_calibration)) +
  geom_pointrange(aes(xmin = rel_ci_lo, xmax = rel_ci_hi),
                  size = 1.2, linewidth = 0.8) +
  scale_color_manual(values = cal_color_map, guide = "none") +
  scale_x_reverse() +
  annotate("text",
           x = max(cal$rel_ci_hi), y = 0.5,
           label = "\u2190 better", hjust = 1,
           fontface = "italic", color = "grey50", size = 3) +
  labs(
    title = "A. Brier Reliability",
    x = "Reliability (lower is better)",
    y = NULL
  ) +
  theme_cel()

# ── Panel B: AUROC forest ─────────────────────────────────────────────────
pb <- ggplot(cal, aes(x = auroc_mean,
                       y = reorder(factorial_calibration, -complexity),
                       color = factorial_calibration)) +
  geom_pointrange(aes(xmin = auroc_ci_lo, xmax = auroc_ci_hi),
                  size = 1.2, linewidth = 0.8) +
  scale_color_manual(values = cal_color_map, guide = "none") +
  annotate("text",
           x = max(cal$auroc_ci_hi), y = 0.5,
           label = "better \u2192", hjust = 0,
           fontface = "italic", color = "grey50", size = 3) +
  labs(
    title = "B. AUROC",
    x = "AUROC (higher is better)",
    y = NULL
  ) +
  theme_cel()

# ── Compose ──────────────────────────────────────────────────────────────────
p <- (pa | pb) +
  plot_annotation(
    title = "V4: Calibration Method Selection",
    subtitle = sprintf(
      "Locked: recipe=%s, model=%s, wt=%s, ds=%s | Ordered by parsimony (simplest on top)",
      locked_recipe, locked_model, locked_wt, locked_ds
    ),
    caption = "95% CIs. CI overlap \u2192 prefer simpler (top). Complexity: logistic_intercept < beta < isotonic."
  )

factorial_save_fig(p, "fig04_calibration_forest", width = 14, height = 6)

# ── Save summary table ────────────────────────────────────────────────────
factorial_save_table(
  cal %>% select(factorial_calibration, complexity, n_cells,
                 reliability_mean, reliability_se, rel_ci_lo, rel_ci_hi,
                 auroc_mean, auroc_se, auroc_ci_lo, auroc_ci_hi),
  "fig04_calibration_summary"
)

# ── Agent-friendly summary ─────────────────────────────────────────────────
best_rel <- cal %>% slice_min(reliability_mean, n = 1)
best_auc <- cal %>% slice_max(auroc_mean, n = 1)

message(sprintf(
  "fig04_calibration_forest: %d calibration methods. Best reliability: %s (%.4f). Best AUROC: %s (%.4f). n per method: %s.",
  nrow(cal),
  as.character(best_rel$factorial_calibration[1]), best_rel$reliability_mean[1],
  as.character(best_auc$factorial_calibration[1]), best_auc$auroc_mean[1],
  paste(sprintf("%s=%d", cal$factorial_calibration, cal$n_cells), collapse = ", ")
))
