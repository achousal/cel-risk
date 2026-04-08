#!/usr/bin/env Rscript
# fig05_confirmation.R -- V5: Seed-split confirmation
#
# Does the locked configuration hold up on held-out seeds?
# Two modes: per-seed data (selection vs confirmation AUROC) or fallback
# (variance ratio). Adapts automatically based on available CSV columns.
#
# Layout: A | B (12x6")

source(file.path("..", "_theme_factorial.R"))
library(patchwork)

# ── Load factorial data (for general context) ───────────────────────────────
df <- load_factorial()

# ── Read validation CSVs ───────────────────────────────────────────────────
v5_path <- file.path(FACTORIAL_TBL_DIR, "v5_confirmation.csv")
v4_path <- file.path(FACTORIAL_TBL_DIR, "v4_calibration.csv")

# Try parent tables dir too
if (!file.exists(v5_path)) {
  v5_path <- file.path(dirname(FACTORIAL_TBL_DIR), "tables", "v5_confirmation.csv")
}
if (!file.exists(v4_path)) {
  v4_path <- file.path(dirname(FACTORIAL_TBL_DIR), "tables", "v4_calibration.csv")
}

# ── Load v5 confirmation data ──────────────────────────────────────────────
if (file.exists(v5_path)) {
  v5_raw <- read.csv(v5_path, stringsAsFactors = FALSE)
  v5 <- as.list(v5_raw[1, ])
  message(sprintf("Loaded v5 confirmation: status = %s", v5$status))
} else {
  # Best-performing defaults from factorial data
  message("v5_confirmation.csv not found -- using best-performing defaults")
  best <- df %>%
    group_by(recipe_id, factorial_model) %>%
    summarise(auroc_mean = mean(summary_auroc_mean, na.rm = TRUE),
              auroc_std  = sd(summary_auroc_mean, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(desc(auroc_mean))

  winner <- best[1, ]
  runners <- best[2:min(4, nrow(best)), ]

  v5 <- list(
    status           = "confirmation_pending",
    winner_auroc_std = winner$auroc_std,
    runner_auroc_std = mean(runners$auroc_std, na.rm = TRUE),
    variance_ratio   = winner$auroc_std / mean(runners$auroc_std, na.rm = TRUE),
    stability_flag   = ifelse(winner$auroc_std / mean(runners$auroc_std, na.rm = TRUE) > 1.5,
                              "unstable", "stable")
  )
}

# ── Load v4 locked configuration (for caption context) ─────────────────────
locked_label <- "locked configuration"
if (file.exists(v4_path)) {
  v4_raw <- read.csv(v4_path, stringsAsFactors = FALSE)
  locked <- v4_raw[v4_raw$selected == TRUE, ]
  if (nrow(locked) > 0) {
    locked_label <- paste0(
      locked$factorial_model[1], " / ",
      locked$factorial_calibration[1], " / ",
      locked$factorial_weighting[1], " / ds=",
      locked$factorial_downsampling[1]
    )
    message(sprintf("Locked config: %s", locked_label))
  }
}

# ── Detect mode: per-seed vs fallback ──────────────────────────────────────
per_seed_mode <- all(c("selection_auroc", "confirmation_auroc") %in% names(v5))

# ── Panel A: AUROC comparison ──────────────────────────────────────────────
if (per_seed_mode) {
  bar_data <- data.frame(
    phase = c("Selection\n(seeds 100-119)", "Confirmation\n(seeds 120-129)"),
    auroc = c(v5$selection_auroc, v5$confirmation_auroc),
    se    = c(v5$selection_se, v5$selection_se)
  )
  bar_data$phase <- factor(bar_data$phase, levels = bar_data$phase)

  pa <- ggplot(bar_data, aes(x = phase, y = auroc, fill = phase)) +
    geom_col(width = 0.6, show.legend = FALSE) +
    geom_errorbar(aes(ymin = auroc - 1.96 * se, ymax = auroc + 1.96 * se),
                  width = 0.2) +
    geom_hline(yintercept = v5$selection_auroc - v5$selection_se,
               linetype = "dashed", color = "red", alpha = 0.6) +
    annotate("text",
             x = 2.3, y = v5$selection_auroc - v5$selection_se,
             label = "1 SE threshold", hjust = 0, color = "red", size = 3) +
    scale_fill_manual(values = c("#4C78A8", "#D95F02")) +
    coord_cartesian(ylim = c(min(bar_data$auroc) - 0.05,
                             max(bar_data$auroc) + 0.03)) +
    labs(title = "A. Selection vs Confirmation AUROC",
         x = NULL, y = "Mean AUROC") +
    theme_cel()
} else {
  bar_data <- data.frame(
    group     = c("Winner", "Runners-up (mean)"),
    auroc_std = c(v5$winner_auroc_std, v5$runner_auroc_std)
  )
  bar_data$group <- factor(bar_data$group, levels = bar_data$group)

  pa <- ggplot(bar_data, aes(x = group, y = auroc_std, fill = group)) +
    geom_col(width = 0.6, show.legend = FALSE) +
    scale_fill_manual(values = c("#4C78A8", "#999999")) +
    labs(title = "A. Winner Variance vs Runners-up",
         x = NULL, y = "Mean AUROC Std (across seeds)") +
    theme_cel()
}

# ── Panel B: Stability indicator ───────────────────────────────────────────
if (per_seed_mode) {
  drop_data <- data.frame(
    metric = "AUROC Drop\n(selection - confirmation)",
    drop   = v5$auroc_drop,
    ci_lo  = v5$auroc_drop - 1.96 * v5$selection_se,
    ci_hi  = v5$auroc_drop + 1.96 * v5$selection_se
  )

  pb <- ggplot(drop_data, aes(x = drop, y = metric)) +
    geom_pointrange(aes(xmin = ci_lo, xmax = ci_hi), size = 1.5) +
    geom_vline(xintercept = 0, linetype = "solid", color = "grey50") +
    geom_vline(xintercept = v5$selection_se, linetype = "dashed", color = "red") +
    annotate("text",
             x = v5$selection_se, y = 0.5,
             label = "1 SE", hjust = -0.2, color = "red", size = 3) +
    labs(title = "B. Confirmation Drop",
         x = "AUROC Difference", y = NULL) +
    theme_cel()
} else {
  ratio_data <- data.frame(
    metric    = "Variance Ratio\n(winner / runners-up)",
    ratio     = v5$variance_ratio,
    threshold = 1.5
  )

  point_color <- ifelse(v5$variance_ratio > 1.5, "red", "#4C78A8")

  pb <- ggplot(ratio_data, aes(x = ratio, y = metric)) +
    geom_point(size = 4, color = point_color) +
    geom_vline(xintercept = 1.0, linetype = "solid", color = "grey50") +
    geom_vline(xintercept = 1.5, linetype = "dashed", color = "red") +
    annotate("text",
             x = 1.5, y = 0.5,
             label = "Instability\nthreshold", hjust = -0.1, color = "red", size = 3) +
    xlim(0, max(2, v5$variance_ratio + 0.5)) +
    labs(title = "B. Variance Stability",
         x = "Std Ratio (winner / runners)", y = NULL) +
    theme_cel()
}

# ── Compose ────────────────────────────────────────────────────────────────
status_label <- switch(v5$status,
  "confirmed"            = "CONFIRMED -- winner generalizes to held-out seeds",
  "confirmation_drop"    = "WARNING -- confirmation AUROC drops > 1 SE",
  "confirmation_pending" = "PENDING -- per-seed data not yet available (variance fallback shown)",
  v5$status
)

mode_label <- ifelse(per_seed_mode, "Per-seed comparison", "Variance fallback")

p <- (pa | pb) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title    = "V5: Seed-Split Confirmation",
    subtitle = sprintf("Config: %s | Mode: %s", locked_label, mode_label),
    caption  = sprintf("Status: %s", status_label),
    theme    = theme_cel()
  )

factorial_save_fig(p, "fig05_confirmation", width = 12, height = 6)
message("Done: fig05_confirmation")
