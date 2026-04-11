#!/usr/bin/env Rscript
# fig_forest_upset.R
# Bootstrap CI forest plot (Fig 6) and feature overlap UpSet plot (Fig 7)
# for incident validation: LR_EN, SVM L1, SVM L2.
#
# Usage:
#   cd cel-risk/experiments/optimal-setup/incident-validation/analysis
#   Rscript fig_forest_upset.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
})

# ── UpSetR: install if missing ────────────────────────────────────────────────
if (!requireNamespace("UpSetR", quietly = TRUE)) {
  message("Installing UpSetR ...")
  install.packages("UpSetR", repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(UpSetR))

# ── Paths ─────────────────────────────────────────────────────────────────────
CEL_ROOT <- normalizePath("../../../..", mustWork = TRUE)
RESULTS  <- file.path(CEL_ROOT, "results")
OUT_DIR  <- file.path(dirname(normalizePath(".", mustWork = TRUE)), "analysis", "out")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

RUN_DIRS <- c(
  LR_EN  = file.path(RESULTS, "incident_validation"),
  SVM_L1 = file.path(RESULTS, "incident_validation_svm_l1"),
  SVM_L2 = file.path(RESULTS, "incident_validation_svm_l2")
)

stopifnot(all(dir.exists(RUN_DIRS)))

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_KEYS   <- c("LR_EN", "SVM_L1", "SVM_L2")
MODEL_LABELS <- c(LR_EN  = "Logistic (EN)",
                  SVM_L1 = "LinSVM (L1)",
                  SVM_L2 = "LinSVM (L2)")
MODEL_COLORS <- setNames(c("#4C78A8", "#1B9E77", "#E7298A"), MODEL_LABELS)

N_BOOT <- 1000L
BOOT_SEED <- 123L

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_cel <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor   = element_blank(),
      strip.background   = element_rect(fill = "#f0f0f0", colour = NA),
      plot.title         = element_text(face = "bold", size = rel(1.1)),
      plot.subtitle      = element_text(colour = "grey40")
    )
}

# ── Helpers ───────────────────────────────────────────────────────────────────
save_fig <- function(p, stem, width, height) {
  pdf_path <- file.path(OUT_DIR, paste0(stem, ".pdf"))
  png_path <- file.path(OUT_DIR, paste0(stem, ".png"))
  ggsave(pdf_path, p, width = width, height = height)
  ggsave(png_path, p, width = width, height = height, dpi = 300)
  message("  -> ", pdf_path)
}

# Trapezoidal AUPRC from vectors of y_true and y_prob
calc_auprc <- function(y_true, y_prob) {
  ord      <- order(y_prob, decreasing = TRUE)
  ys       <- y_true[ord]
  tp_cum   <- cumsum(ys == 1)
  fp_cum   <- cumsum(ys == 0)
  n_pos    <- sum(ys == 1)
  if (n_pos == 0L) return(NA_real_)
  prec <- tp_cum / (tp_cum + fp_cum)
  rec  <- tp_cum / n_pos
  sum(diff(c(0, rec)) * prec)
}

# AUROC via Wilcoxon rank-sum identity
calc_auroc <- function(y_true, y_prob) {
  pos  <- y_prob[y_true == 1]
  neg  <- y_prob[y_true == 0]
  n_pos <- length(pos)
  n_neg <- length(neg)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  wilcox.test(pos, neg, exact = FALSE)$statistic / (n_pos * n_neg)
}

# Bootstrap CIs: returns list(est, lo, hi)
bootstrap_metrics <- function(y_true, y_prob, n_boot = N_BOOT, seed = BOOT_SEED) {
  set.seed(seed)
  n        <- length(y_true)
  auprc_b  <- numeric(n_boot)
  auroc_b  <- numeric(n_boot)
  for (i in seq_len(n_boot)) {
    idx        <- sample(n, n, replace = TRUE)
    auprc_b[i] <- calc_auprc(y_true[idx], y_prob[idx])
    auroc_b[i] <- calc_auroc(y_true[idx], y_prob[idx])
  }
  auprc_b <- auprc_b[!is.na(auprc_b)]
  auroc_b <- auroc_b[!is.na(auroc_b)]
  list(
    auprc_est = calc_auprc(y_true, y_prob),
    auprc_lo  = quantile(auprc_b, 0.025, names = FALSE),
    auprc_hi  = quantile(auprc_b, 0.975, names = FALSE),
    auroc_est = calc_auroc(y_true, y_prob),
    auroc_lo  = quantile(auroc_b, 0.025, names = FALSE),
    auroc_hi  = quantile(auroc_b, 0.975, names = FALSE)
  )
}

# ── Load test predictions ─────────────────────────────────────────────────────
message("Loading test predictions ...")
test_list <- lapply(MODEL_KEYS, function(tag) {
  read_csv(file.path(RUN_DIRS[tag], "test_predictions.csv"),
           show_col_types = FALSE) |>
    mutate(model = tag)
})
names(test_list) <- MODEL_KEYS

# ── Load feature coefficients ─────────────────────────────────────────────────
message("Loading feature coefficients ...")
coef_list <- lapply(MODEL_KEYS, function(tag) {
  read_csv(file.path(RUN_DIRS[tag], "feature_coefficients.csv"),
           show_col_types = FALSE) |>
    mutate(model = tag)
})
names(coef_list) <- MODEL_KEYS

# ============================================================================
# Figure 6: Bootstrap CI Forest Plot
# ============================================================================
message("Computing bootstrap CIs (n=", N_BOOT, ", seed=", BOOT_SEED, ") ...")

ci_rows <- lapply(MODEL_KEYS, function(tag) {
  df  <- test_list[[tag]]
  res <- bootstrap_metrics(df$y_true, df$y_prob)
  data.frame(
    model    = tag,
    label    = MODEL_LABELS[tag],
    auprc_est = res$auprc_est,
    auprc_lo  = res$auprc_lo,
    auprc_hi  = res$auprc_hi,
    auroc_est = res$auroc_est,
    auroc_lo  = res$auroc_lo,
    auroc_hi  = res$auroc_hi,
    stringsAsFactors = FALSE
  )
})
ci_df <- bind_rows(ci_rows) |>
  mutate(label = factor(label, levels = rev(unname(MODEL_LABELS))))

# Prevalence for AUPRC reference line
prevalence <- mean(test_list[["LR_EN"]]$y_true)

# Panel A: AUPRC
p_auprc <- ggplot(ci_df, aes(x = auprc_est, y = label, color = label)) +
  geom_vline(xintercept = prevalence,
             linetype = "dashed", colour = "grey55", linewidth = 0.5) +
  geom_errorbar(aes(xmin = auprc_lo, xmax = auprc_hi),
                orientation = "y", width = 0.15, linewidth = 0.8) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.3f", auprc_est)),
            hjust = -0.35, size = 3.2, show.legend = FALSE) +
  scale_color_manual(values = MODEL_COLORS, guide = "none") +
  scale_x_continuous(
    limits = c(
      max(0,     min(ci_df$auprc_lo) - 0.04),
      min(1,     max(ci_df$auprc_hi) + 0.10)
    )
  ) +
  labs(title = "A  AUPRC",
       x = "AUPRC",
       y = NULL,
       caption = sprintf("Dashed = prevalence (%.4f)", prevalence)) +
  theme_cel() +
  theme(panel.grid.major.y = element_blank())

# Panel B: AUROC
p_auroc <- ggplot(ci_df, aes(x = auroc_est, y = label, color = label)) +
  geom_errorbar(aes(xmin = auroc_lo, xmax = auroc_hi),
                orientation = "y", width = 0.15, linewidth = 0.8) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.3f", auroc_est)),
            hjust = -0.35, size = 3.2, show.legend = FALSE) +
  scale_color_manual(values = MODEL_COLORS, guide = "none") +
  scale_x_continuous(
    limits = c(
      max(0,   min(ci_df$auroc_lo) - 0.02),
      min(1,   max(ci_df$auroc_hi) + 0.06)
    )
  ) +
  labs(title = "B  AUROC",
       x = "AUROC",
       y = NULL) +
  theme_cel() +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y  = element_blank(),
        axis.ticks.y = element_blank())

# Color legend strip (manual)
legend_df <- data.frame(
  label = factor(unname(MODEL_LABELS), levels = unname(MODEL_LABELS)),
  x = seq_along(MODEL_LABELS),
  y = 1
)
p_legend <- ggplot(legend_df, aes(x = label, y = y, color = label)) +
  geom_point(size = 3) +
  scale_color_manual(values = MODEL_COLORS, name = NULL) +
  theme_void() +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 10))

p_forest <- (p_auprc | p_auroc) +
  plot_annotation(
    title    = "Figure 6. Bootstrap 95% CI for test-set performance",
    subtitle = sprintf("Locked test set  |  %d bootstrap resamples, seed=%d",
                       N_BOOT, BOOT_SEED),
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(colour = "grey40", size = 10)
    )
  )

save_fig(p_forest, "fig6_bootstrap_forest", width = 10, height = 4)

# ============================================================================
# Figure 7: Feature Overlap UpSet Plot
# ============================================================================
message("Building UpSet plot ...")

active_sets <- lapply(MODEL_KEYS, function(tag) {
  df <- coef_list[[tag]]
  df$protein[abs(df$coefficient) > 1e-8]
})
# Use syntactically safe column names for UpSetR (spaces/parens break indexing)
SET_NAMES_SAFE <- c(LR_EN = "Logistic_EN", SVM_L1 = "LinSVM_L1", SVM_L2 = "LinSVM_L2")
names(active_sets) <- SET_NAMES_SAFE[MODEL_KEYS]

# Build binary membership matrix for UpSetR
all_proteins <- unique(unlist(active_sets))
upset_mat <- as.data.frame(
  lapply(active_sets, function(s) as.integer(all_proteins %in% s))
)
rownames(upset_mat) <- all_proteins

set_sizes <- vapply(active_sets, length, integer(1))
message(sprintf("  Active features: %s",
                paste(names(set_sizes), set_sizes, sep = "=", collapse = " | ")))

core_n <- length(Reduce(intersect, active_sets))
message(sprintf("  Core (all 3 models): %d proteins", core_n))

# Colors keyed to safe column names
set_colors <- c(Logistic_EN = "#4C78A8", LinSVM_L1 = "#1B9E77", LinSVM_L2 = "#E7298A")

# Helper to render UpSet to an open device
draw_upset <- function() {
  upset(
    upset_mat,
    sets           = names(active_sets),
    order.by       = "freq",
    keep.order     = FALSE,
    sets.bar.color = set_colors[names(active_sets)],
    main.bar.color = "grey30",
    matrix.color   = "grey20",
    point.size     = 3,
    line.size      = 0.8,
    text.scale     = c(1.2, 1.1, 1.0, 1.0, 1.1, 0.9),
    mainbar.y.label = "Intersection size",
    sets.x.label    = "Active features",
    mb.ratio        = c(0.60, 0.40)
  )
}

pdf(file.path(OUT_DIR, "fig7_feature_upset.pdf"), width = 8, height = 5)
draw_upset()
dev.off()
message("  -> ", file.path(OUT_DIR, "fig7_feature_upset.pdf"))

# PNG version -- UpSetR does not support ggsave, render via png() device
png(file.path(OUT_DIR, "fig7_feature_upset.png"),
    width = 8, height = 5, units = "in", res = 300)
draw_upset()
dev.off()
message("  -> ", file.path(OUT_DIR, "fig7_feature_upset.png"))

# ============================================================================
# Done
# ============================================================================
message("\nDone. Outputs in: ", OUT_DIR)
message("  fig6_bootstrap_forest.pdf / .png")
message("  fig7_feature_upset.pdf / .png")
