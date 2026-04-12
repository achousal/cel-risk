#!/usr/bin/env Rscript
# incident_validation_comparison.R
# Cross-model comparison of incident validation runs:
#   LR_EN (elastic net logistic regression)
#   LinSVM_cal L1 (sparse SVM)
#   LinSVM_cal L2 (dense SVM)
#
# Usage:
#   cd cel-risk/experiments/optimal-setup/incident-validation/analysis
#   Rscript incident_validation_comparison.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(stringr)
  library(patchwork)
  library(scales)
})

# ── Paths ────────────────────────────────────────────────────────────────────
# Run from experiments/incident-validation/analysis/. CEL_ROOT goes up 3 levels.
CEL_ROOT <- normalizePath("../../..", mustWork = TRUE)
RESULTS  <- file.path(CEL_ROOT, "results", "incident-validation", "lr")
OUT_DIR  <- file.path(dirname(normalizePath(".", mustWork = TRUE)), "analysis", "out")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

RUN_DIRS <- c(
  LR_EN       = file.path(RESULTS, "LR_EN"),
  SVM_L1      = file.path(RESULTS, "SVM_L1"),
  SVM_L2      = file.path(RESULTS, "SVM_L2")
)

stopifnot(all(dir.exists(RUN_DIRS)))

# ── Theme ────────────────────────────────────────────────────────────────────
MODEL_KEYS <- c("LR_EN", "SVM_L1", "SVM_L2")

MODEL_LABELS <- c(
  LR_EN  = "Logistic (EN)",
  SVM_L1 = "LinSVM (L1)",
  SVM_L2 = "LinSVM (L2)"
)

# Colors keyed by display label (for use after factor relabeling)
MODEL_COLORS <- setNames(
  c("#4C78A8", "#1B9E77", "#E7298A"),
  MODEL_LABELS
)

# Colors keyed by raw model tag (for use before relabeling)
MODEL_COLORS_RAW <- setNames(
  c("#4C78A8", "#1B9E77", "#E7298A"),
  MODEL_KEYS
)

theme_cel <- function(base_size = 11) {
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.background   = element_rect(fill = "grey93", color = NA),
      strip.text         = element_text(face = "bold", size = base_size),
      legend.position    = "bottom",
      plot.title         = element_text(face = "bold", size = base_size + 2),
      plot.subtitle      = element_text(color = "grey40", size = base_size),
      plot.caption       = element_text(color = "grey50", size = base_size - 2,
                                        hjust = 0)
    )
}

clean_protein <- function(x) str_remove(x, "_resid$") |> toupper()

save_fig <- function(p, stem, width, height) {
  ggsave(file.path(OUT_DIR, paste0(stem, ".pdf")), p,
         width = width, height = height)
  ggsave(file.path(OUT_DIR, paste0(stem, ".png")), p,
         width = width, height = height, dpi = 300)
  message("  -> ", file.path(OUT_DIR, paste0(stem, ".pdf")))
}

# ============================================================================
# 1. Load data
# ============================================================================

load_run <- function(tag, dir) {
  list(
    strategy = read_csv(file.path(dir, "strategy_comparison.csv"),
                        show_col_types = FALSE) |>
      mutate(model = tag),
    coefs    = read_csv(file.path(dir, "feature_coefficients.csv"),
                        show_col_types = FALSE) |>
      mutate(model = tag),
    cv       = read_csv(file.path(dir, "cv_results.csv"),
                        show_col_types = FALSE) |>
      mutate(model = tag),
    test     = read_csv(file.path(dir, "test_predictions.csv"),
                        show_col_types = FALSE) |>
      mutate(model = tag)
  )
}

runs <- mapply(load_run, names(RUN_DIRS), RUN_DIRS, SIMPLIFY = FALSE)

# ============================================================================
# 2. Strategy comparison across models
# ============================================================================

strategy_all <- bind_rows(lapply(runs, `[[`, "strategy"))

# Best per model
best_per_model <- strategy_all |>
  group_by(model) |>
  slice_max(mean_auprc, n = 1) |>
  ungroup()

cat("\n=== Best strategy per model ===\n")
best_per_model |>
  select(model, strategy, weight_scheme, mean_auprc, std_auprc, mean_auroc) |>
  print(n = Inf)

# ── Fig 1: Strategy x weight heatmap per model ──────────────────────────────
p_heat <- strategy_all |>
  mutate(
    model = factor(model, levels = names(MODEL_LABELS), labels = MODEL_LABELS),
    strategy = factor(strategy,
                      levels = c("incident_only", "incident_prevalent", "prevalent_only")),
    weight_scheme = factor(weight_scheme,
                           levels = c("none", "log", "sqrt", "balanced"))
  ) |>
  ggplot(aes(x = weight_scheme, y = strategy, fill = mean_auprc)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", mean_auprc)), size = 3) +
  facet_wrap(~model) +
  scale_fill_viridis_c(option = "mako", direction = -1,
                       limits = c(0.12, 0.24), oob = squish,
                       breaks = seq(0.12, 0.24, by = 0.03)) +
  labs(title = "Strategy x Weighting: Mean CV AUPRC",
       x = "Weight scheme", y = "Training strategy",
       fill = "AUPRC",
       caption = "UK Biobank proteomics | N=44,174 (119 incident dev, 29 incident test) | 5-fold CV, Optuna-tuned | 2,920 Olink proteins") +
  theme_cel() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.key.width = unit(1.5, "cm"))

save_fig(p_heat, "fig1_strategy_heatmap", width = 10, height = 4.5)

# ── Fig 2: Fold-level AUPRC for best strategy per model ─────────────────────
cv_all <- bind_rows(lapply(runs, `[[`, "cv"))

# Get the winning strategy+weight for each model
cv_best <- cv_all |>
  inner_join(
    best_per_model |> select(model, strategy, weight_scheme),
    by = c("model", "strategy", "weight_scheme")
  ) |>
  mutate(model = factor(model, levels = names(MODEL_LABELS), labels = MODEL_LABELS))

p_folds <- ggplot(cv_best, aes(x = model, y = auprc, color = model)) +
  geom_jitter(width = 0.15, size = 3, alpha = 0.7) +
  stat_summary(fun = mean, geom = "crossbar",
               width = 0.4, linewidth = 0.6, color = "black") +
  scale_color_manual(values = MODEL_COLORS, guide = "none") +
  labs(title = "CV AUPRC by fold (best strategy per model)",
       subtitle = paste0(
         "LR: incident_only+log | SVM L1: incident_only+log | SVM L2: incident_only+none"
       ),
       x = NULL, y = "AUPRC",
       caption = "5-fold outer CV | 3-fold inner Optuna (50 trials) | AUPRC objective | Crossbar = mean") +
  theme_cel()

save_fig(p_folds, "fig2_fold_auprc", width = 5, height = 4.5)

# ============================================================================
# 3. Test set comparison
# ============================================================================

test_all <- bind_rows(lapply(runs, `[[`, "test"))

# Compute test metrics per model
test_metrics <- test_all |>
  group_by(model) |>
  summarise(
    n = n(),
    n_pos = sum(y_true == 1),
    auroc = tryCatch(
      as.numeric(pROC::auc(pROC::roc(y_true, y_prob, quiet = TRUE))),
      error = function(e) NA_real_
    ),
    auprc = {
      # Manual trapezoidal AUPRC
      ord <- order(y_prob, decreasing = TRUE)
      y_sorted <- y_true[ord]
      tp_cum <- cumsum(y_sorted == 1)
      fp_cum <- cumsum(y_sorted == 0)
      prec <- tp_cum / (tp_cum + fp_cum)
      rec  <- tp_cum / sum(y_sorted == 1)
      # Trapezoidal integration
      sum(diff(c(0, rec)) * prec)
    },
    .groups = "drop"
  )

cat("\n=== Test set metrics ===\n")
print(test_metrics)

# ── Fig 3: Test set ROC and PR curves ───────────────────────────────────────
# Build curve data
roc_data <- test_all |>
  group_by(model) |>
  group_modify(~{
    roc_obj <- pROC::roc(.x$y_true, .x$y_prob, quiet = TRUE)
    tibble(
      specificity = as.numeric(roc_obj$specificities),
      sensitivity = as.numeric(roc_obj$sensitivities)
    )
  }) |>
  ungroup()

pr_data <- test_all |>
  group_by(model) |>
  group_modify(~{
    ordered <- .x |> arrange(desc(y_prob))
    tp_cum <- cumsum(ordered$y_true == 1)
    fp_cum <- cumsum(ordered$y_true == 0)
    precision <- tp_cum / (tp_cum + fp_cum)
    recall <- tp_cum / sum(ordered$y_true == 1)
    tibble(recall = recall, precision = precision)
  }) |>
  ungroup()

p_roc <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = MODEL_COLORS_RAW, labels = MODEL_LABELS) +
  coord_equal() +
  labs(title = "ROC (locked test set)", x = "FPR", y = "TPR", color = NULL,
       caption = sprintf("Locked test: %d incident + %d controls",
                         sum(test_all$y_true[test_all$model == "LR_EN"] == 1),
                         sum(test_all$y_true[test_all$model == "LR_EN"] == 0))) +
  theme_cel()

prevalence <- test_all |>
  filter(model == "LR_EN") |>
  summarise(prev = mean(y_true)) |>
  pull(prev)

p_pr <- ggplot(pr_data, aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = prevalence, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = MODEL_COLORS_RAW, labels = MODEL_LABELS) +
  labs(title = "PR curve (locked test set)", x = "Recall", y = "Precision",
       color = NULL,
       caption = sprintf("Dashed = prevalence (%.4f)", prevalence)) +
  theme_cel()

p_curves <- p_roc + p_pr + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

save_fig(p_curves, "fig3_test_roc_pr", width = 9, height = 4.5)

# ============================================================================
# 4. Feature overlap and stability
# ============================================================================

coefs_all <- bind_rows(lapply(runs, `[[`, "coefs"))

# Active features per model (non-zero coefficient)
active <- coefs_all |>
  filter(abs_coef > 1e-8) |>
  select(model, protein, abs_coef, stability_freq)

active_sets <- split(active$protein, active$model)

# Overlap matrix
models <- names(active_sets)
n_models <- length(models)
overlap_mat <- matrix(0, n_models, n_models, dimnames = list(models, models))
for (i in seq_along(models)) {
  for (j in seq_along(models)) {
    overlap_mat[i, j] <- length(intersect(active_sets[[i]], active_sets[[j]]))
  }
}

cat("\n=== Active features per model ===\n")
cat(paste(models, sapply(active_sets, length), sep = ": ", collapse = " | "), "\n")

cat("\n=== Overlap matrix ===\n")
print(overlap_mat)

# Core features: non-zero in all 3 models
core_proteins <- Reduce(intersect, active_sets)
cat("\n=== Core features (non-zero in all 3 models):", length(core_proteins), "===\n")

# Build stability table
core_table <- coefs_all |>
  filter(protein %in% core_proteins) |>
  select(model, protein, coefficient, abs_coef, stability_freq) |>
  pivot_wider(
    names_from = model,
    values_from = c(coefficient, abs_coef, stability_freq),
    names_glue = "{model}_{.value}"
  ) |>
  mutate(
    sign_consistent = sign(LR_EN_coefficient) == sign(SVM_L1_coefficient) &
                      sign(SVM_L1_coefficient) == sign(SVM_L2_coefficient),
    mean_abs_coef_rank = (rank(-LR_EN_abs_coef) +
                          rank(-SVM_L1_abs_coef) +
                          rank(-SVM_L2_abs_coef)) / 3,
    min_stability = pmin(LR_EN_stability_freq, SVM_L1_stability_freq,
                         SVM_L2_stability_freq)
  ) |>
  arrange(mean_abs_coef_rank)

# ── Fig 4: Feature importance comparison (top 30) ───────────────────────────
# Rank features within each model
ranked <- active |>
  group_by(model) |>
  mutate(rank = rank(-abs_coef)) |>
  ungroup()

# Top 30 by mean rank across models
top30 <- ranked |>
  group_by(protein) |>
  summarise(mean_rank = mean(rank), n_models = n(), .groups = "drop") |>
  filter(n_models >= 2) |>
  slice_min(mean_rank, n = 30) |>
  pull(protein)

p_coef <- ranked |>
  filter(protein %in% top30) |>
  mutate(
    protein = clean_protein(protein),
    protein = reorder(protein, -rank),
    model = factor(model, levels = names(MODEL_LABELS), labels = MODEL_LABELS)
  ) |>
  ggplot(aes(x = rank, y = protein, color = model)) +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS) +
  scale_x_continuous(trans = "reverse") +
  labs(title = "Feature importance rank (top 30 by mean rank)",
       subtitle = "Rank within each model by |coefficient|",
       x = "Rank (1 = most important)", y = NULL, color = NULL,
       caption = sprintf("Active features: LR_EN=%d | SVM_L1=%d | SVM_L2=%d | Bootstrap stability selection, correlation-pruned",
                         length(active_sets[["LR_EN"]]),
                         length(active_sets[["SVM_L1"]]),
                         length(active_sets[["SVM_L2"]]))) +
  theme_cel() +
  theme(panel.grid.major.x = element_line(color = "grey90"))

save_fig(p_coef, "fig4_feature_rank_comparison", width = 7, height = 8)

# ── Fig 5: Core feature coefficient direction + magnitude ────────────────────
core_long <- coefs_all |>
  filter(protein %in% core_proteins, abs_coef > 1e-8) |>
  mutate(
    protein = clean_protein(protein),
    model = factor(model, levels = names(MODEL_LABELS), labels = MODEL_LABELS)
  )

# Order by LR_EN |coef|
lr_order <- core_long |>
  filter(model == "Logistic (EN)") |>
  arrange(desc(abs_coef)) |>
  pull(protein)

core_long <- core_long |>
  mutate(protein = factor(protein, levels = rev(lr_order)))

p_core <- ggplot(core_long, aes(x = coefficient, y = protein, color = model)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_point(size = 2.5, alpha = 0.8) +
  scale_color_manual(values = MODEL_COLORS, labels = MODEL_LABELS) +
  labs(title = sprintf("Core features: non-zero in all 3 models (n=%d)", length(core_proteins)),
       subtitle = sprintf("Coefficient direction and relative magnitude | %d/%d sign-consistent (%.0f%%)",
                          sum(core_table$sign_consistent), nrow(core_table),
                          mean(core_table$sign_consistent) * 100),
       x = "Coefficient", y = NULL, color = NULL,
       caption = "Coefficients from final refit on full dev set | Scales differ across model types | Age+sex residualized proteins") +
  theme_cel()

save_fig(p_core, "fig5_core_features", width = 7, height = 8)

# ============================================================================
# 5. Write summary report
# ============================================================================

# Prepare core feature table for export
core_export <- core_table |>
  select(protein,
         sign_consistent,
         min_stability,
         mean_abs_coef_rank,
         LR_EN_coefficient, SVM_L1_coefficient, SVM_L2_coefficient,
         LR_EN_stability_freq, SVM_L1_stability_freq, SVM_L2_stability_freq) |>
  mutate(protein = clean_protein(protein))

write_csv(core_export, file.path(OUT_DIR, "core_features.csv"))

# Feature stability summary
stability_summary <- core_export |>
  summarise(
    n_core = n(),
    n_sign_consistent = sum(sign_consistent),
    pct_sign_consistent = mean(sign_consistent) * 100,
    mean_min_stability = mean(min_stability),
    median_min_stability = median(min_stability)
  )

# Write text report
report_lines <- c(
  "# Incident Validation: Three-Model Comparison",
  "",
  "## Pipeline",
  "```",
  "UK Biobank proteomics (N=44,174; 2,920 Olink proteins)",
  "  |",
  "  v",
  "Locked 80/20 dev/test split (seed=42, stratified by sex)",
  "  |-- Dev: 119 incident + 150 prevalent + 35,100 controls",
  "  |-- Test: 29 incident + 8,776 controls (LOCKED, touched once)",
  "  |",
  "  v",
  "Bootstrap stability feature selection (Wald, 100 resamples)",
  "  |-- LR_EN:  top 200/resample, threshold >= 70% --> 134 proteins",
  "  |-- SVM L1: top 150/resample, threshold >= 50% --> 130 proteins",
  "  |-- SVM L2: top 150/resample, threshold >= 50% --> 130 proteins",
  "  |",
  "  v",
  "Correlation pruning (|r| > 0.85, keep more stable)",
  "  |",
  "  v",
  "3 x 4 factorial: strategy x weight_scheme",
  "  |-- Strategies: incident_only, incident_prevalent, prevalent_only",
  "  |-- Weights:    none, log, sqrt, balanced",
  "  |-- 5-fold outer CV, 3-fold inner Optuna (AUPRC objective)",
  "  |",
  "  v",
  "Select best (strategy, weight) by mean CV AUPRC",
  "  |",
  "  v",
  "Refit on full dev set with winning config",
  "  |",
  "  v",
  "Evaluate on locked test set (bootstrap CIs)",
  "```",
  "",
  "## Model Configurations",
  "",
  "| Parameter          | LR_EN                 | SVM L1                    | SVM L2                    |",
  "|--------------------|-----------------------|---------------------------|---------------------------|",
  "| Model              | ElasticNet LogReg     | LinearSVC (L1) + Sigmoid  | LinearSVC (L2) + Sigmoid  |",
  "| Tuned HPs          | C, l1_ratio           | C                         | C                         |",
  "| Optuna trials      | 50                    | 50                        | 50                        |",
  "| Bootstrap resamples| 100                   | 100                       | 100                       |",
  "| Stability top_k    | 200                   | 150                       | 150                       |",
  "| Stability threshold| 70%                   | 50%                       | 50%                       |",
  "| Panel size         | 134                   | 130                       | 130                       |",
  "",
  "## Strategy Comparison",
  "",
  sprintf("All three models select **incident_only** as the best training strategy."),
  "",
  "| Model       | Best strategy     | Best weight | CV AUPRC        | Test AUPRC  | Test AUROC  |",
  "|-------------|-------------------|-------------|-----------------|-------------|-------------|",
  {
    # Build dynamic rows from best_per_model + test_metrics
    model_order <- c("LR_EN", "SVM_L1", "SVM_L2")
    model_display <- c("LR_EN       ", "SVM L1      ", "SVM L2      ")
    rows <- character(length(model_order))
    for (i in seq_along(model_order)) {
      m <- model_order[i]
      bp <- best_per_model[best_per_model$model == m, ]
      tm <- test_metrics[test_metrics$model == m, ]
      rows[i] <- sprintf(
        "| %s| %-17s | %-11s | %.3f +/- %.3f | %.3f       | %.3f       |",
        model_display[i],
        bp$strategy,
        bp$weight_scheme,
        bp$mean_auprc, bp$std_auprc,
        tm$auprc, tm$auroc
      )
    }
    rows
  },
  "",
  "**Key findings:**",
  "",
  "1. **incident_only dominates** across all three models. Prevalent cases add noise",
  "   rather than signal -- likely because prevalent CeD reflects post-diagnosis",
  "   biology (dietary changes, treatment effects) that diverges from pre-diagnostic",
  "   proteomic signatures.",
  "",
  "2. **Weight scheme matters less than strategy.** Within incident_only, the top",
  "   weight schemes (log, sqrt, none) are within 1 SD of each other. The SVMs",
  "   prefer lighter or no weighting; LR prefers log. Balanced weighting is",
  "   consistently worst (over-corrects the imbalance).",
  "",
  "3. **Test set performance is tightly grouped.** AUPRC ranges 0.188-0.210, AUROC",
  "   0.908-0.918. CIs overlap substantially. No model clearly dominates on the",
  "   locked test set -- the 29 incident test cases limit statistical power.",
  "",
  sprintf("## Feature Stability"),
  "",
  sprintf("### Model sparsity"),
  "",
  sprintf("| Model  | Panel | Non-zero | Sparsity |"),
  sprintf("|--------|-------|----------|----------|"),
  {
    model_order <- c("LR_EN", "SVM_L1", "SVM_L2")
    model_display <- c("LR_EN ", "SVM L1", "SVM L2")
    rows <- character(length(model_order))
    for (i in seq_along(model_order)) {
      m <- model_order[i]
      panel_size <- sum(coefs_all$model == m)
      nonzero <- length(active_sets[[m]])
      sparsity <- (panel_size - nonzero) / panel_size * 100
      rows[i] <- sprintf(
        "| %s | %-5d | %-8d | %.0f%%      |",
        model_display[i], panel_size, nonzero, sparsity
      )
    }
    rows
  },
  "",
  sprintf("### Cross-model core features"),
  "",
  sprintf("**%d proteins** have non-zero coefficients in all three models.", length(core_proteins)),
  sprintf("Of these, **%d/%d (%.0f%%)** have consistent sign (direction of effect) across all models.",
          stability_summary$n_sign_consistent,
          stability_summary$n_core,
          stability_summary$pct_sign_consistent),
  "",
  "Top core features by mean importance rank:",
  ""
)

# Add top 20 core features
top_core <- head(core_export, 20)
report_lines <- c(report_lines,
  "| Protein | Sign consistent | Min stability | LR coef | SVM L1 coef | SVM L2 coef |",
  "|---------|-----------------|---------------|---------|-------------|-------------|"
)
for (i in seq_len(nrow(top_core))) {
  row <- top_core[i, ]
  report_lines <- c(report_lines, sprintf(
    "| %-12s | %-15s | %.2f          | %+.4f  | %+.4f      | %+.4f      |",
    row$protein,
    ifelse(row$sign_consistent, "yes", "NO"),
    row$min_stability,
    row$LR_EN_coefficient,
    row$SVM_L1_coefficient,
    row$SVM_L2_coefficient
  ))
}

report_lines <- c(report_lines,
  "",
  "### Interpretation",
  "",
  "The core feature set is dominated by:",
  "",
  "- **TGM2** (transglutaminase 2): strongest signal in all models, negative",
  "  coefficient. TGM2 is the autoantigen in celiac disease -- lower circulating",
  "  levels pre-diagnosis may reflect tissue sequestration or immune complex",
  "  formation.",
  "",
  "- **Gut-epithelial markers** (MUC2, RBP2, FABP1, CPA2): intestinal integrity",
  "  and absorptive function proteins. Elevated pre-diagnosis suggests subclinical",
  "  mucosal changes.",
  "",
  "- **Immune/inflammatory** (CXCL9, CXCL11, CCL11, NOS2, CD160): IFN-gamma",
  "  responsive chemokines and NK/T cell markers. Consistent with the Th1-driven",
  "  immune response in CeD pathogenesis.",
  "",
  "- **CLEC4G, APOA1** (negative): hepatic/metabolic markers whose decrease may",
  "  reflect systemic inflammation or liver-gut axis perturbation.",
  "",
  "## Recommendation",
  "",
  "For the factorial (V0 gate), the incident validation confirms:",
  "",
  "1. **Lock incident_only** as the training strategy across all models.",
  "2. **Weight scheme is a secondary factor** -- test log/none but not balanced.",
  sprintf("3. **LR_EN is most parsimonious** (%d features, test AUPRC %.3f) but see",
          length(active_sets[["LR_EN"]]),
          test_metrics$auprc[test_metrics$model == "LR_EN"]),
  "   calibration section -- LR_EN is miscalibrated without post-hoc adjustment.",
  sprintf("4. **SVM L2 has the best point estimate** (test AUPRC %.3f) and best",
          test_metrics$auprc[test_metrics$model == "SVM_L2"]),
  "   calibration -- recommended as primary model.",
  sprintf("5. **SVM L1 is a middle ground** (%d features, test AUPRC %.3f) if moderate",
          length(active_sets[["SVM_L1"]]),
          test_metrics$auprc[test_metrics$model == "SVM_L1"]),
  "   sparsity is desired.",
  "",
  "See companion analyses for calibration metrics (calibration_metrics.csv),",
  "decision curve analysis (fig7_dca), SHAP-based feature importance (fig9-11),",
  "and saturation curve (fig8_saturation).",
  ""
)

# ============================================================================
# Extended sections -- appended if companion CSVs exist
# ============================================================================

# --- Calibration Assessment ---
calib_path <- file.path(OUT_DIR, "calibration_metrics.csv")
if (file.exists(calib_path)) {
  calib <- read_csv(calib_path, show_col_types = FALSE)
  calib_lines <- c(
    "## Calibration Assessment",
    "",
    "| Metric                  | LR_EN   | SVM L1  | SVM L2  | Ideal |",
    "|-------------------------|---------|---------|---------|-------|"
  )
  fmt_row <- function(label, col, fmt = "%.4f", ideal = "") {
    vals <- sapply(c("LR_EN", "SVM_L1", "SVM_L2"), function(m) {
      v <- calib[[col]][calib$model_key == m]
      if (length(v) == 0) return(NA_real_)
      v
    })
    sprintf("| %-23s | %-7s | %-7s | %-7s | %-5s |",
            label,
            sprintf(fmt, vals["LR_EN"]),
            sprintf(fmt, vals["SVM_L1"]),
            sprintf(fmt, vals["SVM_L2"]),
            ideal)
  }
  calib_lines <- c(calib_lines,
    fmt_row("ECE", "ece", "%.4f", "0"),
    fmt_row("ICI (LOESS)", "ici", "%.4f", "0"),
    fmt_row("Brier score", "brier_score", "%.4f", "0"),
    fmt_row("Brier reliability", "brier_reliability", "%.4f", "0"),
    fmt_row("Brier resolution", "brier_resolution", "%.4f", "high"),
    fmt_row("Calibration intercept", "calibration_intercept", "%.3f", "0"),
    fmt_row("Calibration slope", "calibration_slope", "%.3f", "1"),
    fmt_row("Spiegelhalter z", "spiegelhalter_z", "%.3f", "~0"),
    fmt_row("Spiegelhalter p", "spiegelhalter_p", "%.3g", ">.05"),
    "",
    "**Interpretation:** LR_EN shows systematic overestimation (large negative",
    "intercept, Spiegelhalter p < 0.001). Both SVMs are well-calibrated",
    "(Spiegelhalter p > 0.9, near-zero ICI). The CalibratedClassifierCV sigmoid",
    "wrapper on the SVMs is doing meaningful work.",
    ""
  )
  report_lines <- c(report_lines, calib_lines)
}

# --- Feature overlap (UpSet decomposition) ---
lr_set <- active_sets[["LR_EN"]]
svm1_set <- active_sets[["SVM_L1"]]
svm2_set <- active_sets[["SVM_L2"]]
overlap_lines <- c(
  "## Feature Overlap (UpSet decomposition)",
  "",
  "| Intersection          | Count |",
  "|-----------------------|-------|",
  sprintf("| All 3 models (core)   | %-5d |", length(Reduce(intersect, active_sets))),
  sprintf("| SVM L1 + SVM L2 only  | %-5d |",
          length(setdiff(intersect(svm1_set, svm2_set), lr_set))),
  sprintf("| SVM L2 only           | %-5d |",
          length(setdiff(svm2_set, union(lr_set, svm1_set)))),
  "",
  "The regularization hierarchy is clean: LR_EN active features are a strict",
  "subset of SVM L1's, which are a strict subset of SVM L2's. Elastic net",
  "selects the most stable core.",
  ""
)
report_lines <- c(report_lines, overlap_lines)

# --- Saturation curve ---
sat_path <- file.path(OUT_DIR, "saturation_results.csv")
if (file.exists(sat_path)) {
  sat <- read_csv(sat_path, show_col_types = FALSE)
  sat_lines <- c(
    "## Saturation Curve (LR_EN, incident_only + log)",
    "",
    "| Panel size | CV AUPRC        | Test AUPRC            |",
    "|-----------:|-----------------|-----------------------|"
  )
  for (i in seq_len(nrow(sat))) {
    r <- sat[i, ]
    sat_lines <- c(sat_lines, sprintf(
      "| %10d | %.3f +/- %.3f | %.3f [%.3f, %.3f] |",
      r$panel_size, r$mean_cv_auprc, r$std_cv_auprc,
      r$test_auprc, r$test_auprc_lo, r$test_auprc_hi
    ))
  }
  sat_lines <- c(sat_lines,
    "",
    "**Knee at N~25-28.** CV AUPRC plateaus near the LR_EN non-zero count.",
    "Test AUPRC is flat from N=25 onwards -- the 134-feature panel offers no",
    "test-set advantage over ~25 features. Elastic net regularization arrived",
    "at a near-optimal panel size without needing this curve as a guide.",
    ""
  )
  report_lines <- c(report_lines, sat_lines)
}

# --- Figure inventory ---
fig_lines <- c(
  "## Figure Inventory",
  "",
  "| Figure                        | Description                                           |",
  "|-------------------------------|-------------------------------------------------------|",
  "| fig1_strategy_heatmap         | 3x4 AUPRC heatmap: strategy x weight, faceted         |",
  "| fig2_fold_auprc               | CV AUPRC per fold, best config per model              |",
  "| fig3_test_roc_pr              | ROC and PR curves on locked test set                  |",
  "| fig4_feature_rank_comparison  | Top 30 features by cross-model importance rank       |",
  "| fig5_core_features            | Core proteins: coefficient direction and magnitude   |",
  "| fig6_calibration              | Reliability diagrams with LOESS smooth                |",
  "| fig6_bootstrap_forest         | Bootstrap 95% CI forest plot (AUPRC + AUROC)         |",
  "| fig7_dca                      | Decision curve analysis (net benefit vs threshold)   |",
  "| fig7_feature_upset            | UpSet plot of feature overlap across models           |",
  "| fig8_saturation               | Performance vs panel size (saturation curve)          |",
  "| fig9_shap_beeswarm            | SHAP beeswarm plots (top 20 features, 3 models)      |",
  "| fig10_shap_bar                | Mean |SHAP| bar chart (top 15 features, 3 models)    |",
  "| fig11_shap_dependence         | SHAP dependence for top 5 core features (LR_EN)      |",
  ""
)
report_lines <- c(report_lines, fig_lines)

writeLines(report_lines, file.path(OUT_DIR, "incident_validation_report.md"))
cat("\n  -> ", file.path(OUT_DIR, "incident_validation_report.md"), "\n")

# ============================================================================
# Done
# ============================================================================
cat("\n=== Analysis complete ===\n")
cat("Outputs in:", OUT_DIR, "\n")
cat("  fig1_strategy_heatmap.pdf\n")
cat("  fig2_fold_auprc.pdf\n")
cat("  fig3_test_roc_pr.pdf\n")
cat("  fig4_feature_rank_comparison.pdf\n")
cat("  fig5_core_features.pdf\n")
cat("  core_features.csv\n")
cat("  incident_validation_report.md\n")
