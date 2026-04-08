#!/usr/bin/env Rscript
# fig07_cascade.R -- V1-V6 Decision Cascade Summary
#
# The entire validation tree in one figure. Shows how 1,566+ cells were
# narrowed to one locked configuration through 6 validation levels.
#
# Layout: Single panel, 16x9". 6 horizontal rows (one per validation level)
# built entirely with ggplot2 geoms on a coordinate grid.

source(file.path("..", "_theme_factorial.R"))

# ── Load factorial data ─────────────────────────────────────────────────────
df <- load_factorial()
n_total_cells <- nrow(df)

# ── Read validation CSVs (graceful if missing) ──────────────────────────────
read_validation <- function(suffix) {
  # Search in RESULTS_DIR, FACTORIAL_TBL_DIR, and tables/ sibling
  patterns <- c(
    file.path(RESULTS_DIR, paste0("*", suffix, "*.csv")),
    file.path(FACTORIAL_TBL_DIR, paste0("*", suffix, "*.csv")),
    file.path(dirname(FACTORIAL_TBL_DIR), "tables", paste0("*", suffix, "*.csv"))
  )
  for (pat in patterns) {
    files <- Sys.glob(pat)
    if (length(files) > 0) {
      out <- tryCatch(read.csv(files[1], stringsAsFactors = FALSE), error = function(e) NULL)
      if (!is.null(out)) {
        message(sprintf("  Loaded %s: %s (%d rows)", suffix, files[1], nrow(out)))
        return(out)
      }
    }
  }
  NULL
}

message("Loading validation CSVs...")
v1 <- read_validation("v1_summary")
v2 <- read_validation("v2_pareto")
v3 <- read_validation("v3_imbalance")
v4 <- read_validation("v4_calibration")
v5 <- read_validation("v5_confirmation")
v6 <- read_validation("v6_ensemble")

# ── Extract locked values from validation CSVs or fallback to data ──────────

# V1: Recipe
if (!is.null(v1) && "recommended" %in% names(v1)) {
  v1_winners <- v1 %>% filter(recommended) %>% pull(recipe_id) %>% unique()
  v1_all <- unique(v1$recipe_id)
  v1_aurocs <- v1 %>%
    group_by(recipe_id) %>%
    summarise(auroc = mean(auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(auroc))
} else if (!is.null(v1) && "best" %in% names(v1)) {
  v1_winners <- v1 %>% filter(best) %>% pull(recipe_id) %>% unique()
  v1_all <- unique(v1$recipe_id)
  v1_aurocs <- v1 %>%
    group_by(recipe_id) %>%
    summarise(auroc = mean(auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(auroc))
} else {
  # Fallback: discover from raw data
  v1_aurocs <- df %>%
    group_by(recipe_id) %>%
    summarise(auroc = mean(summary_auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(auroc))
  v1_all <- unique(v1_aurocs$recipe_id)
  v1_winners <- as.character(v1_aurocs$recipe_id[1])
}
locked_recipe <- v1_winners[1]
n_recipes <- length(v1_all)

# V2: Model
if (!is.null(v2) && "selected" %in% names(v2)) {
  v2_models <- unique(v2$factorial_model)
  v2_selected <- v2 %>% filter(selected)
  v2_nondom <- v2 %>%
    filter(if ("robust_nondominated" %in% names(.)) robust_nondominated else !robust_dominated)
  locked_model <- as.character(v2_selected$factorial_model[1])
} else {

  v2_models <- unique(as.character(df$factorial_model))
  model_rank <- df %>%
    filter(recipe_id == locked_recipe) %>%
    group_by(factorial_model) %>%
    summarise(auroc = mean(summary_auroc_mean, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(auroc))
  locked_model <- as.character(model_rank$factorial_model[1])
  v2_selected <- NULL
  v2_nondom <- NULL
}
n_models <- length(v2_models)

# V3: Imbalance (weighting x downsampling)
wt_levels <- sort(unique(as.character(df$factorial_weighting)))
ds_levels <- sort(unique(as.character(df$factorial_downsampling)))
n_imbalance <- length(wt_levels) * length(ds_levels)

if (!is.null(v3) && "selected" %in% names(v3)) {
  v3_winner <- v3 %>% filter(selected) %>% head(1)
  locked_wt <- as.character(v3_winner$factorial_weighting)
  locked_ds <- as.character(v3_winner$factorial_downsampling)
} else {
  locked_wt <- "none"
  locked_ds <- as.character(max(as.numeric(ds_levels)))
}

# V4: Calibration
cal_levels <- sort(unique(as.character(df$factorial_calibration)))
n_calibrations <- length(cal_levels)

if (!is.null(v4) && "selected" %in% names(v4)) {
  v4_winner <- v4 %>% filter(selected) %>% head(1)
  locked_cal <- as.character(v4_winner$factorial_calibration)
} else {
  locked_cal <- cal_levels[1]
}

# V5: Confirmation
if (!is.null(v5) && "status" %in% names(v5)) {
  v5_status <- v5$status[1]
} else {
  v5_status <- "pending"
}

# V6: Ensemble
if (!is.null(v6) && "status" %in% names(v6)) {
  v6_status <- v6$status[1]
} else {
  v6_status <- "pending"
}

# ── Row layout ──────────────────────────────────────────────────────────────
rows <- data.frame(
  level = c("V1", "V2", "V3", "V4", "V5", "V6"),
  label = c("Recipe", "Model", "Imbalance", "Calibration", "Confirmation", "Ensemble"),
  y_center = c(6, 5, 4, 3, 2, 1),
  row_height = 0.7,
  stringsAsFactors = FALSE
)

# Candidate count text
rows$candidates <- c(
  sprintf("%d candidates", n_recipes),
  sprintf("%d candidates", n_models),
  sprintf("%d combinations", n_imbalance),
  sprintf("%d methods", n_calibrations),
  "",
  ""
)

# Locked value text
rows$locked_value <- c(
  locked_recipe,
  locked_model,
  sprintf("wt=%s, ds=%s", locked_wt, locked_ds),
  if (!is.null(v4) && locked_cal %in% names(CALIBRATION_LABELS)) CALIBRATION_LABELS[locked_cal] else locked_cal,
  switch(v5_status,
    "confirmed" = "Confirmed",
    "confirmation_drop" = "Drop > 1 SE",
    "confirmation_pending" = "Pending",
    "pending" = "Pending",
    v5_status
  ),
  switch(v6_status,
    "single_model_preferred" = "Single model",
    "ensemble_recommended" = "Ensemble flagged",
    "skipped" = "Skipped",
    "pending" = "Pending",
    v6_status
  )
)

# Track which rows have data
rows$has_data <- c(
  !is.null(v1) || n_recipes > 0,
  !is.null(v2) || n_models > 0,
  !is.null(v3) || n_imbalance > 0,
  !is.null(v4) || n_calibrations > 0,
  !is.null(v5),
  !is.null(v6)
)

# ── Build canvas ────────────────────────────────────────────────────────────
p <- ggplot() +
  xlim(-0.5, 16.5) + ylim(-0.2, 7.5) +
  theme_void() +
  theme(plot.margin = margin(15, 15, 15, 15))

# 1. Row backgrounds
p <- p + geom_rect(
  data = rows,
  aes(xmin = 0.2, xmax = 15.8,
      ymin = y_center - 0.35, ymax = y_center + 0.35),
  fill = "grey97", color = "grey85", linewidth = 0.3
)

# 2. Level labels (left column)
p <- p + geom_text(
  data = rows,
  aes(x = 0.5, y = y_center + 0.08,
      label = paste0(level, ": ", label)),
  hjust = 0, fontface = "bold", size = 4, color = "grey20"
)

# 3. Candidate counts (below label)
p <- p + geom_text(
  data = rows %>% filter(candidates != ""),
  aes(x = 0.5, y = y_center - 0.18, label = candidates),
  hjust = 0, size = 2.8, color = "grey50"
)

# 4. Locked values (right column) -- pending in gray italic, locked in blue bold
for (i in seq_len(nrow(rows))) {
  is_pending <- grepl("Pending|pending", rows$locked_value[i])
  p <- p + annotate(
    "text", x = 15.5, y = rows$y_center[i],
    label = rows$locked_value[i],
    hjust = 1,
    fontface = if (is_pending) "italic" else "bold",
    size = 3.5,
    color = if (is_pending) "grey60" else "#2166AC"
  )
}

# 5. Arrows between rows
for (i in 1:5) {
  p <- p + geom_segment(
    x = 8, xend = 8,
    y = rows$y_center[i] - 0.35, yend = rows$y_center[i + 1] + 0.35,
    arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
    color = "grey60", linewidth = 0.4
  )
}

# ── Mini-visualizations (center column, x = 3.5 to 11.5) ───────────────────

# -- V1: Recipe bar chart --
if (nrow(v1_aurocs) > 0) {
  # Show top recipes as horizontal colored bars
  n_show <- min(nrow(v1_aurocs), 12)
  v1_plot <- v1_aurocs[1:n_show, ]
  v1_plot$rank <- seq_len(n_show)

  # Normalize bar widths to fit in center column
  auroc_min <- min(v1_plot$auroc) - 0.005
  auroc_range <- max(v1_plot$auroc) - auroc_min
  v1_plot$bar_width <- (v1_plot$auroc - auroc_min) / auroc_range * 7.5

  # Assign recipe colors
  if (exists("RECIPE_COLORS") && length(RECIPE_COLORS) > 0) {
    v1_plot$fill <- RECIPE_COLORS[as.character(v1_plot$recipe_id)]
    v1_plot$fill[is.na(v1_plot$fill)] <- "grey70"
  } else {
    v1_plot$fill <- hue_pal()(n_show)
  }

  # Y-distribute within the row
  y_top <- rows$y_center[1] + 0.28
  y_bot <- rows$y_center[1] - 0.28
  bar_h <- (y_top - y_bot) / n_show
  v1_plot$y_mid <- y_top - (v1_plot$rank - 0.5) * bar_h

  for (j in seq_len(n_show)) {
    is_winner <- as.character(v1_plot$recipe_id[j]) == locked_recipe
    p <- p + annotate(
      "rect",
      xmin = 3.5, xmax = 3.5 + v1_plot$bar_width[j],
      ymin = v1_plot$y_mid[j] - bar_h * 0.4,
      ymax = v1_plot$y_mid[j] + bar_h * 0.4,
      fill = v1_plot$fill[j],
      color = if (is_winner) "black" else NA,
      linewidth = if (is_winner) 0.8 else 0,
      alpha = if (is_winner) 1.0 else 0.6
    )
  }
} else {
  p <- p + annotate("text", x = 7.5, y = rows$y_center[1],
                     label = "Pending", color = "grey60", fontface = "italic", size = 3.5)
}

# -- V2: Model dots --
if (n_models > 0) {
  model_names <- sort(v2_models)
  n_m <- length(model_names)
  x_positions <- seq(4.5, 10.5, length.out = n_m)

  model_dot_df <- data.frame(
    model = model_names,
    x = x_positions,
    y = rows$y_center[2],
    stringsAsFactors = FALSE
  )

  # Color from palette
  model_dot_df$fill <- MODEL_COLORS[model_dot_df$model]
  model_dot_df$fill[is.na(model_dot_df$fill)] <- "grey70"

  # Determine non-dominated status
  if (!is.null(v2_nondom)) {
    nd_models <- unique(as.character(v2_nondom$factorial_model))
  } else {
    nd_models <- locked_model
  }
  model_dot_df$is_nd <- model_dot_df$model %in% nd_models

  # Non-dominated: larger ring
  for (j in seq_len(nrow(model_dot_df))) {
    if (model_dot_df$is_nd[j]) {
      p <- p + annotate("point", x = model_dot_df$x[j], y = model_dot_df$y[j],
                         size = 6, shape = 21, fill = NA,
                         color = model_dot_df$fill[j], stroke = 1.2)
    }
    p <- p + annotate("point", x = model_dot_df$x[j], y = model_dot_df$y[j],
                       size = 3.5, color = model_dot_df$fill[j],
                       alpha = if (model_dot_df$is_nd[j]) 1.0 else 0.3)
    # Short model label below
    short_label <- if (model_dot_df$model[j] %in% names(MODEL_LABELS)) {
      MODEL_LABELS[model_dot_df$model[j]]
    } else {
      model_dot_df$model[j]
    }
    p <- p + annotate("text", x = model_dot_df$x[j],
                       y = model_dot_df$y[j] - 0.22,
                       label = short_label, size = 2.2, color = "grey40")
  }
} else {
  p <- p + annotate("text", x = 7.5, y = rows$y_center[2],
                     label = "Pending", color = "grey60", fontface = "italic", size = 3.5)
}

# -- V3: Imbalance mini-grid (weighting x downsampling) --
if (length(wt_levels) > 0 && length(ds_levels) > 0) {
  n_wt <- length(wt_levels)
  n_ds <- length(ds_levels)

  # Grid dimensions within center column
  grid_x_start <- 5.5
  grid_x_end   <- 9.5
  cell_w <- (grid_x_end - grid_x_start) / n_ds
  cell_h <- 0.56 / n_wt

  y_row3 <- rows$y_center[3]

  grid_cells <- expand.grid(wt = wt_levels, ds = ds_levels, stringsAsFactors = FALSE)
  grid_cells$wt_idx <- match(grid_cells$wt, wt_levels)
  grid_cells$ds_idx <- match(grid_cells$ds, ds_levels)
  grid_cells$x_mid <- grid_x_start + (grid_cells$ds_idx - 0.5) * cell_w
  grid_cells$y_mid <- y_row3 + 0.28 - (grid_cells$wt_idx - 0.5) * cell_h
  grid_cells$is_best <- grid_cells$wt == locked_wt & grid_cells$ds == locked_ds
  grid_cells$is_parsimony <- grid_cells$wt == "none" &
    grid_cells$ds == as.character(max(as.numeric(ds_levels)))

  for (j in seq_len(nrow(grid_cells))) {
    p <- p + annotate(
      "rect",
      xmin = grid_cells$x_mid[j] - cell_w * 0.4,
      xmax = grid_cells$x_mid[j] + cell_w * 0.4,
      ymin = grid_cells$y_mid[j] - cell_h * 0.4,
      ymax = grid_cells$y_mid[j] + cell_h * 0.4,
      fill = if (grid_cells$is_best[j]) "#2166AC" else "grey80",
      color = if (grid_cells$is_best[j]) "black" else "grey70",
      linewidth = if (grid_cells$is_best[j]) 0.6 else 0.3
    )
    if (grid_cells$is_parsimony[j] && !grid_cells$is_best[j]) {
      p <- p + annotate("text", x = grid_cells$x_mid[j], y = grid_cells$y_mid[j],
                         label = "*", size = 3.5, fontface = "bold")
    }
  }

  # Axis labels for grid
  for (k in seq_along(ds_levels)) {
    p <- p + annotate("text",
                       x = grid_x_start + (k - 0.5) * cell_w,
                       y = y_row3 - 0.32,
                       label = paste0("ds=", ds_levels[k]),
                       size = 2, color = "grey50")
  }
  for (k in seq_along(wt_levels)) {
    short_wt <- if (wt_levels[k] %in% names(WEIGHTING_LABELS)) {
      sub(" weights", "", WEIGHTING_LABELS[wt_levels[k]])
    } else {
      wt_levels[k]
    }
    p <- p + annotate("text",
                       x = grid_x_start - 0.25,
                       y = y_row3 + 0.28 - (k - 0.5) * cell_h,
                       label = short_wt, size = 2, color = "grey50", hjust = 1)
  }
} else {
  p <- p + annotate("text", x = 7.5, y = rows$y_center[3],
                     label = "Pending", color = "grey60", fontface = "italic", size = 3.5)
}

# -- V4: Calibration mini-bars --
if (n_calibrations > 0) {
  cal_x_start <- 4.5
  cal_x_end <- 10.5
  bar_w <- (cal_x_end - cal_x_start) / n_calibrations
  y_row4 <- rows$y_center[4]

  for (k in seq_along(cal_levels)) {
    cal_name <- cal_levels[k]
    is_selected <- cal_name == locked_cal
    fill_col <- if (cal_name %in% names(CALIBRATION_COLORS)) {
      CALIBRATION_COLORS[cal_name]
    } else {
      "grey70"
    }

    p <- p + annotate(
      "rect",
      xmin = cal_x_start + (k - 1) * bar_w + 0.1,
      xmax = cal_x_start + k * bar_w - 0.1,
      ymin = y_row4 - 0.2,
      ymax = y_row4 + 0.2,
      fill = fill_col,
      alpha = if (is_selected) 1.0 else 0.25,
      color = if (is_selected) "black" else NA,
      linewidth = if (is_selected) 0.6 else 0
    )

    short_cal <- if (cal_name %in% names(CALIBRATION_LABELS)) {
      CALIBRATION_LABELS[cal_name]
    } else {
      cal_name
    }
    p <- p + annotate("text",
                       x = cal_x_start + (k - 0.5) * bar_w,
                       y = y_row4 - 0.28,
                       label = short_cal, size = 2.2, color = "grey40")
  }
} else {
  p <- p + annotate("text", x = 7.5, y = rows$y_center[4],
                     label = "Pending", color = "grey60", fontface = "italic", size = 3.5)
}

# -- V5: Confirmation status symbol --
v5_symbol <- switch(v5_status,
  "confirmed"            = "\u2713",
  "confirmation_drop"    = "!",
  "confirmation_pending" = "?",
  "pending"              = "?",
  "?"
)
v5_color <- switch(v5_status,
  "confirmed"            = "#1B7837",
  "confirmation_drop"    = "#B2182B",
  "confirmation_pending" = "grey60",
  "pending"              = "grey60",
  "grey60"
)

p <- p + annotate("text", x = 7.5, y = rows$y_center[5],
                   label = v5_symbol, size = 10, fontface = "bold", color = v5_color)

# Add supplementary text
v5_detail <- switch(v5_status,
  "confirmed"            = "Winner generalizes to held-out seeds",
  "confirmation_drop"    = "AUROC drops > 1 SE on held-out seeds",
  "confirmation_pending" = "Awaiting per-seed data",
  "pending"              = "Awaiting validation",
  ""
)
if (nchar(v5_detail) > 0) {
  p <- p + annotate("text", x = 9.5, y = rows$y_center[5],
                     label = v5_detail, size = 2.5, color = "grey50", hjust = 0)
}

# -- V6: Ensemble status symbols --
v6_single_color <- switch(v6_status,
  "single_model_preferred" = "black",
  "ensemble_recommended"   = "grey60",
  "skipped"                = "grey60",
  "pending"                = "grey60",
  "grey60"
)
v6_ens_color <- switch(v6_status,
  "ensemble_recommended"   = "black",
  "single_model_preferred" = "grey60",
  "skipped"                = "grey60",
  "pending"                = "grey60",
  "grey60"
)

# Circle = single model, diamond = ensemble
p <- p + annotate("point", x = 6.5, y = rows$y_center[6],
                   shape = 16, size = 5, color = v6_single_color)
p <- p + annotate("text", x = 6.5, y = rows$y_center[6] - 0.22,
                   label = "Single", size = 2.2, color = "grey40")
p <- p + annotate("point", x = 8.5, y = rows$y_center[6],
                   shape = 18, size = 6, color = v6_ens_color)
p <- p + annotate("text", x = 8.5, y = rows$y_center[6] - 0.22,
                   label = "Ensemble", size = 2.2, color = "grey40")

# Ensemble gain annotation if available
if (!is.null(v6) && "auroc_gain" %in% names(v6)) {
  gain_text <- sprintf("Gain: %.4f", v6$auroc_gain[1])
  p <- p + annotate("text", x = 10.5, y = rows$y_center[6],
                     label = gain_text, size = 2.5, color = "grey50")
}

# ── Title block ─────────────────────────────────────────────────────────────
p <- p + labs(
  title = sprintf("Validation Cascade: %s Cells \u2192 Locked Configuration",
                  format(n_total_cells, big.mark = ","))
) +
  theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5))

# ── Bottom summary block ───────────────────────────────────────────────────
locked_text <- sprintf(
  "Recipe: %s | Model: %s | Weighting: %s | Downsampling: %s | Calibration: %s",
  locked_recipe, locked_model, locked_wt, locked_ds, locked_cal
)

p <- p + annotate(
  "label", x = 8, y = 0.15,
  label = locked_text,
  fill = "#F0F7FF", color = "#2166AC",
  fontface = "bold", size = 3.5,
  label.padding = unit(0.5, "lines")
)

# ── Save ────────────────────────────────────────────────────────────────────
factorial_save_fig(p, "fig07_cascade", width = 16, height = 9)

# ── Agent-friendly summary ──────────────────────────────────────────────────
message(sprintf(
  paste0(
    "fig07_cascade: %s total cells across %d recipes x %d models. ",
    "Locked: recipe=%s, model=%s, wt=%s, ds=%s, cal=%s. ",
    "V5=%s, V6=%s. ",
    "CSVs found: V1=%s V2=%s V3=%s V4=%s V5=%s V6=%s."
  ),
  format(n_total_cells, big.mark = ","), n_recipes, n_models,
  locked_recipe, locked_model, locked_wt, locked_ds, locked_cal,
  v5_status, v6_status,
  !is.null(v1), !is.null(v2), !is.null(v3),
  !is.null(v4), !is.null(v5), !is.null(v6)
))
